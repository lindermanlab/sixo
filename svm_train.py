"""Train a stochastic volatility model (SVM)."""
import os
import pathlib
from functools import partial
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

from jax._src.random import KeyArray as PRNGKey
from chex import Array, Scalar
import optax
import argparse
import wandb

from models import svm
import snax
import bounds
import smc
import datasets
from distutils.util import strtobool
import util

tfd = tfp.distributions

DEFAULT_DATA_DIR = pathlib.Path(__file__).parent.resolve() / "data"

parser = argparse.ArgumentParser(description='Train stochastic volatility model.')

parser.add_argument(
       '--data_dir', type=str,
       default=DEFAULT_DATA_DIR,
       help='Directory to load forex data from')
parser.add_argument(
        '--bound', type=str, choices=['fivo', 'iwae', 'elbo', 'sixo'],
        default='fivo',
        help="The bound to optimize")
parser.add_argument(
        '--rsamp_crit', type=str, choices=smc.RESAMPLING_CRITS.keys(),
        default='ess',
        help="The resampling criterion.")
parser.add_argument(
        '--rsamp_type', type=str, choices=smc.RESAMPLING_FNS.keys(),
        default='multinomial',
        help="The resampling type.")
parser.add_argument(
        '--train_num_particles', type=int,
        default=4,
        help="Number of particles in train bound.")
parser.add_argument(
        '--eval_num_particles', type=int,
        default=128,
        help="Number of particles in eval bound.")
parser.add_argument(
        '--num_train_steps', type=int,
        default=100_000,
        help="Number of steps to train for.")
parser.add_argument(
        '--lr', type=float,
        default=3e-4,
        help="Learning rate.")
parser.add_argument(
        '--init_scale', type=float,
        default=0.01,
        help="Scale for random initialization")
parser.add_argument(
        '--min_scale_diag', type=float,
        default=1e-8,
        help="Minimum scale of proposal.")
parser.add_argument(
        '--tilt_type', type=str,
        default='none', choices=['bwd_dre', 'quadrature', 'none'],
        help="Type of tilt.")
parser.add_argument(
        '--tilt_lr', type=float,
        default=1e-4,
        help="Tilt learning rate.")
parser.add_argument(
        '--tilt_batch_size', type=int,
        default=128,
        help="Batch size for tilt training.")
parser.add_argument(
        '--tilt_train_set_size', type=int,
        default=256,
        help="Number of samples to take from p for training the tilt.")
parser.add_argument(
        '--tilt_inner_steps', type=int,
        default=100,
        help="Number of steps to train tilt for each 'superstep'.")
parser.add_argument(
        '--model_inner_steps', type=int,
        default=100,
        help="Number of steps to train model for each 'superstep'.")
parser.add_argument(
        '--quad_tilt_window', type=int,
        default=1,
        help="Window for quadrature tilt.")
parser.add_argument(
        '--quad_tilt_deg', type=int,
        default=5,
        help="Degree of quadrature tilt.")
parser.add_argument(
        '--dre_tilt_rnn_hdims', type=str,
        default="128",
        help="Hidden dims for the DRE tilt RNN.")
parser.add_argument(
        '--dre_tilt_mlp_hdims', type=str,
        default="128",
        help="Hidden dims for the DRE tilt MLP.")
parser.add_argument(
        '--step_0_vsmc', type=strtobool,
        default="False",
        help="If true, the first step of the model matches the VSMC paper implementation."
             "Otherwise, it matches the VSMC paper description.")
parser.add_argument(
        '--parallelism', type=int,
        default=4,
        help="Number of XLA CPU devices to create.")
parser.add_argument(
        '--summarize_every', type=int,
        default=1_000,
        help="Steps between summaries.")
parser.add_argument(
        '--expensive_summary_mult', type=int,
        default=10,
        help="Run expensive summaries every summarize_every*expensive_summary_mult steps.")
parser.add_argument(
        '--eval_batch_size', type=int,
        default=1,
        help="The batch size to use when evaluating multiple bounds.")
parser.add_argument(
        '--checkpoint_dir', type=str,
        default='/tmp/fivo_svm',
        help="Where to store checkpoints.")
parser.add_argument(
        '--checkpoint_every', type=int,
        default=2000,
        help="Number of steps between checkpoints.")
parser.add_argument(
        '--checkpoints_to_keep', type=int,
        default=3,
        help="Number of checkpoints to keep.")
parser.add_argument(
        '--use_wandb', type=strtobool,
        default="False",
        help="If true, log to Weights and Biases.")
parser.add_argument(
        '--wandb_proj', type=str,
        default="svm",
        help="Weights and biases project name.")
parser.add_argument(
        '--wandb_run', type=str,
        default=None,
        help="Weights and biases run name.")
parser.add_argument(
        '--seed', type=int,
        default=0,
        help="Random seed.")


def make_model(
        key: PRNGKey,
        data_dim: int,
        num_timesteps: int,
        cfg):

  k1, k2, k3 = jax.random.split(key, num=3)
  model = svm.DiagCovSVMProposal(
          k1,
          svm.DiagCovSVM(
              k2,
              data_dim,
              init_scale=cfg.init_scale,
              min_scale_diag=cfg.min_scale_diag,
              step_0_vsmc=cfg.step_0_vsmc),
          num_timesteps,
          init_scale=cfg.init_scale,
          min_scale_diag=cfg.min_scale_diag)

  if cfg.tilt_type == 'quadrature':
    model = svm.QuadratureTiltSVMProposal(model, data_dim, cfg.quad_tilt_deg, cfg.quad_tilt_window)
  elif cfg.tilt_type == 'bwd_dre':
    k1, k2 = jax.random.split(k3)
    bwd_rnn_hdims = [int(x.strip()) for x in cfg.dre_tilt_rnn_hdims.split(",")]
    mlp_hdims = [int(x.strip()) for x in cfg.dre_tilt_mlp_hdims.split(",")]
    model = svm.BackwardsTilt(
              k1, model, snax.LSTM(k2, data_dim, bwd_rnn_hdims), mlp_hdims, data_dim)
  return model


def make_summarize(cfg):
  train_data = datasets.create_forex_dataset(cfg.data_dir)
  eval_data = datasets.create_forex_eval_dataset(cfg.data_dir)
  train_num_timesteps, data_dim = train_data.shape
  eval_num_timesteps, _ = eval_data.shape

  train_loss = partial(fivo_loss,
          cfg.bound, cfg.rsamp_crit, cfg.rsamp_type,
          data_dim, train_num_timesteps, train_data, cfg.train_num_particles)

  # eval_loss runs a bootstrap particle filter on the eval data.
  eval_loss = partial(fivo_loss,
          'fivo', 'ess', 'multinomial',
          data_dim, eval_num_timesteps, eval_data, cfg.eval_num_particles,
          propose_from_prior=True)

  # eval_loss runs a bootstrap particle filter on the train data.
  eval_loss_train = partial(fivo_loss,
          'fivo', 'ess', 'multinomial',
          data_dim, train_num_timesteps, train_data, cfg.eval_num_particles,
                      propose_from_prior=True)

  @jax.jit
  def eval_dre_tilt(key, model):
    sk1, sk2 = jax.random.split(key, num=2)
    neg_xs, neg_ys = model.prop.svm.sample_trajectory(sk1, train_num_timesteps)
    pos_xs, pos_ys = model.prop.svm.sample_trajectory(sk2, train_num_timesteps)
    pos_logits, neg_logits = model.tilt_seq(neg_xs, pos_xs, pos_ys)
    pos_bernoullis = tfd.Bernoulli(logits=pos_logits)
    neg_bernoullis = tfd.Bernoulli(logits=neg_logits)
    pos_lp = jnp.mean(pos_bernoullis.log_prob(1))
    neg_lp = jnp.mean(neg_bernoullis.log_prob(0))

    pos_preds = pos_logits > 0.
    neg_preds = neg_logits <= 0.

    pos_acc = jnp.mean(pos_preds)
    neg_acc = jnp.mean(neg_preds)
    # Said it was 1, it was 1.
    tp = jnp.sum(pos_preds)
    # Said it was 1, it was 0.
    fp = jnp.sum(1 - neg_preds)
    # Said it was 0, it was 1.
    fn = jnp.sum(1 - pos_preds)

    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = (2 * prec * rec) / (prec + rec)
    return ((pos_lp, neg_lp, (pos_lp + neg_lp) / 2),
            (pos_acc, neg_acc, (pos_acc + neg_acc) / 2),
            prec, rec, f1,
            (pos_xs, pos_ys),
            (neg_xs, neg_ys))

  def summarize(key, model, step):
    if cfg.tilt_type in ['none', 'quadrature']:
      model_local_step = step
    else:
      total_inner_steps = cfg.model_inner_steps + cfg.tilt_inner_steps
      global_epoch = step // total_inner_steps
      model_local_step = global_epoch * cfg.model_inner_steps

    if cfg.bound == 'sixo':
      prop = model.prop
    else:
      prop = model

    # Log the train bounds
    train_bound = train_loss(key, model_local_step, model)
    print(f"  train bound {cfg.train_num_particles}: {train_bound:0.4f}")
    if cfg.use_wandb:
      wandb.log({f"train_fivo_bound_{cfg.train_num_particles}": train_bound}, step=step)

    # Check if its an expensive summary step
    exp_summ_step = ((step % (cfg.summarize_every * cfg.expensive_summary_mult)) == 0)

    if cfg.tilt_type == 'bwd_dre':
      ((_, _, lps),
       (_, _, accs),
       precs, recs, f1s, _, _) = jax.vmap(eval_dre_tilt, in_axes=(0, None))(
               jax.random.split(key, num=cfg.eval_batch_size), model)
      lp = jnp.mean(lps)
      acc = jnp.mean(accs)
      prec = jnp.mean(precs)
      rec = jnp.mean(recs)
      f1 = jnp.mean(f1s)
      print(f"  tilt log prob: {lp:0.4f} f1: {f1:0.4f} acc: {acc:0.4f}"
            f" prec: {prec:0.4f} rec: {rec:0.4f}")
      if cfg.use_wandb:
          wandb.log({"tilt_lp": lp, "tilt_acc": acc, "tilt_prec": prec,
              "tilt_rec": rec, "tilt_f1": f1}, step=step)

    if exp_summ_step:
      # Log the eval bounds.
      eval_bound = eval_loss(key, model_local_step, prop)
      print(f"  eval bpf bound {cfg.eval_num_particles}: {eval_bound:0.4f}")
      if cfg.use_wandb:
        wandb.log({f"eval_bpf_bound_{cfg.eval_num_particles}": eval_bound}, step=step)

      train_bpf_bound = eval_loss_train(key, model_local_step, prop)
      print(f"  train bpf bound {cfg.eval_num_particles}: {train_bpf_bound:0.4f}")
      if cfg.use_wandb:
          wandb.log({f"train_bpf_bound_{cfg.eval_num_particles}": train_bpf_bound}, step=step)

  return summarize


def fivo_loss(
        bound: str,
        rsamp_crit_str: str,
        rsamp_type_str: str,
        data_dim: int,
        num_timesteps: int,
        data: Array,
        num_particles: int,
        key: PRNGKey,
        step: int,
        model: svm.SVMWithProposal,
        propose_from_prior: bool = False) -> Scalar:
  init_state = jnp.zeros([data_dim])
  rsamp_crit = smc.RESAMPLING_CRITS[rsamp_crit_str]
  rsamp_fn = smc.RESAMPLING_FNS[rsamp_type_str]
  if bound == 'sixo':
    assert isinstance(model, svm.TiltedSVMProposal)
    p_and_w = model.make_propose_and_weight(step, data, num_timesteps,
            propose_from_prior=propose_from_prior)
    _, _, log_Z_hat, _, _ = bounds.sixo(
            key,
            p_and_w,
            init_state,
            num_timesteps,
            num_timesteps,
            num_particles,
            observations=data,
            resampling_criterion=rsamp_crit,
            resampling_fn=rsamp_fn)
  else:
    p_and_w = partial(model.propose_and_weight,
            propose_from_prior=propose_from_prior)
    assert isinstance(model, svm.SVMWithProposal)
    if bound == 'fivo':
      _, _, log_Z_hat, _, _ = bounds.fivo(
              key,
              p_and_w,
              init_state,
              num_timesteps,
              num_timesteps,
              num_particles,
              observations=data,
              resampling_criterion=rsamp_crit,
              resampling_fn=rsamp_fn)
    elif bound == 'iwae':
      _, _, log_Z_hat, _, _ = bounds.iwae(
              key,
              p_and_w,
              init_state,
              num_timesteps,
              num_timesteps,
              num_particles,
              observations=data)
    else:
      _, _, log_Z_hat, _, _ = bounds.elbo(
              key,
              p_and_w,
              init_state,
              num_timesteps,
              num_timesteps,
              observations=data)
  return - log_Z_hat


def make_dre_tilt_train_step(cfg, num_timesteps):

  def tilt_loss(
            key: PRNGKey,
            step: int,
            model: svm.TiltedSVMProposal) -> Scalar:
    k1, k2, k3 = jax.random.split(key, num=3)
    neg_xs, _ = model.prop.svm.sample_trajectory(k1, num_timesteps)
    pos_xs, pos_ys = model.prop.svm.sample_trajectory(k2, num_timesteps)
    data = (neg_xs, pos_xs, pos_ys)
    data = jax.lax.stop_gradient(data)
    return model.tilt_loss(k3, step, data)

  tilt_loss_opt = util.make_masked_optimizer(
            optax.adam(cfg.tilt_lr), [(lambda m: m.prop, False)], mask_default=True)

  tilt_step = snax.train_lib.TrainStep(
          tilt_loss,
          tilt_loss_opt,
          num_inner_steps=cfg.tilt_inner_steps,
          name="tilt",
          parallelize=(cfg.parallelism > 1),
          batch_size=cfg.tilt_batch_size)

  return tilt_step


def train_svm(cfg):
  if cfg.tilt_type != 'none':
    assert cfg.bound == 'sixo', "Tilts should only be used with sixo"

  key = jax.random.PRNGKey(cfg.seed)

  # Create the dataset.
  key, subkey = jax.random.split(key)
  train_data = datasets.create_forex_dataset(cfg.data_dir)
  num_timesteps, data_dim = train_data.shape

  summarize = make_summarize(cfg)

  # Create the model.
  key, subkey = jax.random.split(key)
  model = make_model(subkey, data_dim, num_timesteps, cfg)

  fivo_train_loss = partial(fivo_loss,
          cfg.bound, cfg.rsamp_crit, cfg.rsamp_type,
          data_dim, num_timesteps, train_data, cfg.train_num_particles)

  # options
  # fivo: tilt = none, bound = fivo
  # sixo w/ quadrature tilt: tilt=quadrature, bound=sixo
  # sixo w/ dre tilt: tilt=bwd_dre bound=sixo
  if cfg.tilt_type in ['none', 'quadrature']:
    learned_params = snax.train_lib.train(
            key,
            fivo_train_loss,
            optax.adam(cfg.lr),
            model,
            num_steps=cfg.num_train_steps,
            summarize_every=cfg.summarize_every,
            summarize_fn=summarize,
            checkpoint_every=cfg.checkpoint_every,
            checkpoint_dir=cfg.checkpoint_dir,
            checkpoints_to_keep=cfg.checkpoints_to_keep,
            use_wandb=cfg.use_wandb)

  else:  # We're using a dre tilt
    fivo_loss_opt = util.make_masked_optimizer(
            optax.adam(cfg.lr), [(lambda m: m.prop, True)], mask_default=False)

    fivo_step = snax.train_lib.TrainStep(
            fivo_train_loss,
            fivo_loss_opt,
            num_inner_steps=cfg.model_inner_steps,
            name="sixo",
            parallelize=False)

    tilt_step = make_dre_tilt_train_step(cfg, num_timesteps)

    # Use the fivo and tilt train steps to train.
    learned_params = snax.train_lib.train_alternating(
            key,
            [fivo_step, tilt_step],
            model,
            num_steps=cfg.num_train_steps,
            summarize_every=cfg.summarize_every,
            summarize_fn=summarize,
            checkpoint_every=cfg.checkpoint_every,
            checkpoint_dir=cfg.checkpoint_dir,
            checkpoints_to_keep=cfg.checkpoints_to_keep,
            use_wandb=cfg.use_wandb)

  return learned_params


def main():
  args = parser.parse_args()
  util.print_args(args)
  if args.parallelism > 1:
    os.environ["XLA_FLAGS"] = f" --xla_force_host_platform_device_count={args.parallelism}"
    print(f"Set number of XLA devices to {args.parallelism},"
          f" JAX now sees {jax.local_device_count()} devices.")

  pathlib.Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

  if args.use_wandb:
    prev_run_id = util.get_wandb_id("fivo", args.wandb_proj, args.wandb_run)
    wandb.init(
      project=args.wandb_proj,
      name=args.wandb_run,
      config=args,
      id=prev_run_id,
      dir=args.checkpoint_dir,
      resume='allow',
    )
  _ = train_svm(args)


if __name__ == '__main__':
  main()
