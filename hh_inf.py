"""Performs inference (no model learning) in a fixed Hodgkin-Huxley model."""
import os
import pathlib
from functools import partial
import jax
import jax.numpy as jnp
import equinox as eqx
from jax._src.random import KeyArray as PRNGKey
from typing import Union
from chex import Array, Scalar
import optax
import argparse
import util
import wandb
import wandb.plot
import models.hh as hh
import snax
from snax import train_lib
import bounds
import datasets
from distutils.util import strtobool
from util import linear_ramp_schedule
from timeit import default_timer as dt
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

import logging
logger = logging.getLogger()


class CheckTypesFilter(logging.Filter):

    def filter(self, record):
        return "check_types" not in record.getMessage()


logger.addFilter(CheckTypesFilter())

parser = argparse.ArgumentParser(description='Perform inference in a Hodgkin-Huxley model.')

# Exp config.
parser.add_argument(
    '--bound', type=str, choices=['fivo', 'iwae', 'elbo', 'sixo'],
    default='fivo',
    help="The bound to optimize")
parser.add_argument(
    '--num_train_steps', type=int,
    default=500_000,
    help="Number of steps to train for.")
parser.add_argument(
    '--proposal_type', type=str,
    default='smoothing', choices=['filtering', 'smoothing', 'bootstrap'],
    help="Type of proposal to use.")
parser.add_argument(
    '--prop_mlp_hdims', type=str,
    default='16',
    help="The hidden dimensions of the proposal NN.")
parser.add_argument(
    '--prop_rnn_hdims', type=str,
    default='16',
    help="The hidden dimensions of the proposal NN.")
parser.add_argument(
    '--resq_type', type=str,
    default='mean', choices=['none', 'mean', 'full', 'mean_sg', 'full_sg'],
    help="Type of resq to use for the proposal.")

parser.add_argument(
    '--model_inner_steps', type=int,
    default=100,
    help="Number of steps to train model for each 'superstep'.")
parser.add_argument(
    '--tilt_inner_steps', type=int,
    default=100,
    help="Number of steps to train tilt for each 'superstep'.")
parser.add_argument(
    '--checkpoint_every', type=int,
    default=2000,
    help="Number of steps between checkpoints.")
parser.add_argument(
    '--summarize_every', type=int,
    default=1000,
    help="Steps between summaries.")

# Tilt arch parameters.
parser.add_argument(
    '--tilt_type', type=str,
    default='none', choices=['none', 'bwd_dre'],
    help="Parametric form of tilt.")
parser.add_argument(
    '--tilt_lr', type=float,
    default=1e-3,
    help="Tilt learning rate.")
parser.add_argument(
    '--tilt_lr_mult', type=float,
    default=1.,
    help="Multiplier for lowering the learning rate.")
parser.add_argument(
    '--tilt_lr_steps', type=str,
    default="0",
    help="List of steps to lower the learning rate on.")
parser.add_argument(
    '--tilt_batch_size', type=int,
    default=8,
    help="Batch size for tilt training.")
parser.add_argument(
    '--bwd_tilt_rnn_hdims', type=str,
    default="16",
    help="number of residual blocks in made tilt.")
parser.add_argument(
    '--bwd_tilt_mlp_hdims', type=str,
    default="16",
    help="number of residual blocks in made tilt.")

# Args that change how expensive evaluation is.
parser.add_argument(
    '--eval_num_particles', type=str,
    default="128",
    help="List of numbers of particles in eval bound.")
parser.add_argument(
    '--expensive_summary_mult', type=int,
    default=5,
    help="Run expensive summaries every summarize_every*expensive_summary_mult steps.")

# Data settings
parser.add_argument(
    '--train_dataset_size', type=int,
    default=1_000,
    help="Number of synthetic datasets to sample for training.")

# HH-specific parameters.
parser.add_argument(
    '--ode_int', type=str,
    default='single_step_euler', choices=hh.ODE_SOLVERS.keys(),
    help='ODE integrator to use.')
# 0.025 seems to be about the highest you can push this without getting wild results.
# You can reject those traces in generation, though. You can then iterate particles
# with a higher DT because they should be rejected before going seriously unstable.
parser.add_argument(
    '--dt', type=float,
    default=0.02,
    help="Step size to use for ODE integrators.")
parser.add_argument(
    '--num_compartments', type=int,
    default=1,
    help="Number of series to train on for synthetic data.")
parser.add_argument(
    '--data_seq_len', type=int,
    default=2_048,
    help="Length of series to train on for synthetic data.")
parser.add_argument(
    '--obs_subsample', type=int,
    default=50,
    help="Integration steps per observation.")
parser.add_argument(
    '--train_batch_size', type=int,
    default=4,
    help="Batch size.")
parser.add_argument(
    '--eval_batch_size', type=int,
    default=64,
    help="Evaluation batch size.")
parser.add_argument(
    '--num_evals', type=int,
    default=10,
    help="Evaluation batch size.")
parser.add_argument(
    '--train_num_particles', type=int,
    default=4,
    help="Number of particles in train bound.")
parser.add_argument(
    '--lr_mult', type=float,
    default=1.,
    help="Multiplier for lowering the learning rate.")
parser.add_argument(
    '--lr_steps', type=str,
    default="0",
    help="List of steps to lower the learning rate on.")
parser.add_argument(
    '--prop_lr', type=float,
    default=1e-3,
    help="Learning rate for the proposal.")
parser.add_argument(
    '--seed', type=int,
    default=12,
    help="Random seed.")
parser.add_argument(
    '--init_scale', type=float,
    default=0.1,
    help="Scale for random initialization")
parser.add_argument(
    '--tilt_anneal', type=strtobool,
    default="False",
    help="Whether or not to anneal")
parser.add_argument(
    '--tilt_anneal_zero_steps', type=int,
    default=0,
    help="Number of steps to train without tilt")
parser.add_argument(
    '--tilt_anneal_ramp_steps', type=int,
    default=1,
    help="Number of steps to ramp up tilt")
parser.add_argument(
    '--checkpoint_dir', type=str,
    default=None,
    help="Where to store checkpoints.")
parser.add_argument(
    '--checkpoints_to_keep', type=int,
    default=3,
    help="Number of checkpoints to keep.")
parser.add_argument(
    '--parallelism', type=int,
    default=1,
    help="Number of parallel devices.")

# WandB args.
parser.add_argument(
    '--use_wandb', type=strtobool,
    default="False",
    help="If true, log to Weights and Biases.")
parser.add_argument(
    '--wandb_proj', type=str,
    default="hh",
    help="Weights and biases project name.")
parser.add_argument(
    '--wandb_run', type=str,
    default=None,
    help="Weights and biases run name.")
parser.add_argument(
    '--wandb_notes', type=str,
    default="",
    help="Weights and biases run notes.")
parser.add_argument(
    '--wandb_id', type=str,
    default="",
    help="Weights and biases run id.")
parser.add_argument(
    '--p_grad_max_norm',
    type=float,
    default=-1.,
    help='Will clip gradients of p to this value, -1 does nothing')
parser.add_argument(
    '--noise_scale',
    type=float,
    default=5.,
    help='Observation noise.')


def gen_data(cfg, key: PRNGKey, num_datapoints: int):
  data, gen_model = datasets.create_synthetic_hh_dataset(
          key,
          num_datapoints,
          cfg.num_compartments,
          cfg.data_seq_len,
          cfg.obs_subsample,
          cfg.dt,
          cfg.ode_int,
          cfg.noise_scale)
  return data, gen_model


def make_model(model, key: PRNGKey, data_dim: int, cfg):
    k1, k2, k3 = jax.random.split(key, num=3)
    # Wrap the base model in a proposal.
    prop_mlp_hdims = [int(x.strip()) for x in cfg.prop_mlp_hdims.split(",")]
    prop_rnn_hdims = [int(x.strip()) for x in cfg.prop_rnn_hdims.split(",")]
    if cfg.proposal_type == 'filtering':
      model = hh.FilteringHHProposal(
                k1,
                model,
                snax.LSTM(k2, data_dim, prop_rnn_hdims),
                prop_mlp_hdims,
                pos_enc_dim=16,
                pos_enc_base=50.,
                resq_type=cfg.resq_type)

    elif cfg.proposal_type == 'smoothing':
      model = hh.SmoothingHHProposal(
                k1,
                model,
                snax.BiLSTM(k2, data_dim, prop_rnn_hdims),
                prop_mlp_hdims,
                pos_enc_dim=16,
                pos_enc_base=50.,
                resq_type=cfg.resq_type)
    elif cfg.proposal_type == 'bootstrap':
      model = hh.BootstrapHHProposal(model)
    else:
      raise ValueError("proposal_type '{cfg.proposal_type}'"
                       " not 'filtering', 'smoothing', or 'bootstrap'")

    # Construct the labels for model parameters.
    # Initialize all labels to be 'proposal'
    model_labels = jax.tree_util.tree_map(lambda _: 'proposal', model)
    # Replace the labels in the subtree 'hh' with 'model'
    model_labels = eqx.tree_at(lambda x: x.hh, model_labels, 'model')

    # If the bound is sixo, wrap the proposal in a tilt.
    if cfg.bound == 'sixo':
      assert cfg.tilt_type == 'bwd_dre'

      if cfg.tilt_anneal:
        anneal_sched = partial(linear_ramp_schedule,
                               cfg.tilt_anneal_zero_steps,
                               cfg.tilt_anneal_ramp_steps)
      else:
        anneal_sched = hh.no_annealing

      k1, k2 = jax.random.split(k3)
      bwd_rnn_hdims = [int(x.strip()) for x in cfg.bwd_tilt_rnn_hdims.split(",")]
      mlp_hdims = [int(x.strip()) for x in cfg.bwd_tilt_mlp_hdims.split(",")]

      model = hh.BackwardsTilt(
        k1,
        model,
        snax.LSTM(k2, data_dim, bwd_rnn_hdims),
        mlp_hdims,
        model.hh.state_dim,
        tilt_anneal_sched=anneal_sched)

      # Construct a pytree with all nodes labeled as tilt parameters.
      all_tilt = jax.tree_util.tree_map(lambda _: 'tilt', model)
      # Replace the subtree at 'prop' with the labels previously constructed.
      model_labels = eqx.tree_at(lambda x: x.prop, all_tilt, model_labels)

    return model, model_labels


def make_summarize(cfg, train_data):
  num_timesteps = train_data.shape[1]

  prop_lr_sched = make_schedule(cfg.prop_lr, cfg.lr_steps, cfg.lr_mult)
  tilt_lr_sched = make_schedule(cfg.tilt_lr, cfg.tilt_lr_steps, cfg.tilt_lr_mult)

  ds = snax.InMemDataset(train_data, cfg.eval_batch_size, shuffle=False)

  @partial(jax.jit)
  def summ_bound(key, step, model):
    return make_summary_bound(
            step, model, cfg.bound, num_timesteps, ds, cfg.train_num_particles)(key)

  def eval_dre_tilt_one_sample(key, model):
    sk1, sk2 = jax.random.split(key, num=2)
    neg_xs, _ = model.prop.hh.sample_trajectory(sk1, num_timesteps)
    pos_xs, pos_ys = model.prop.hh.sample_trajectory(sk2, num_timesteps)

    # Returns [num_timesteps] arrays.
    pos_logits, neg_logits = model.tilt_seq(neg_xs, pos_xs, pos_ys)
    pos_bernoullis = tfd.Bernoulli(logits=pos_logits)
    neg_bernoullis = tfd.Bernoulli(logits=neg_logits)

    pos_lp = jnp.mean(pos_bernoullis.log_prob(1))
    neg_lp = jnp.mean(neg_bernoullis.log_prob(0))
    lp = (pos_lp + neg_lp) / 2.

    # [num_timesteps]
    pos_preds = pos_logits > 0.
    neg_preds = neg_logits <= 0.

    pos_acc = jnp.mean(pos_preds)
    neg_acc = jnp.mean(neg_preds)
    acc = (pos_acc + neg_acc) / 2.
    # Said it was 1, it was 1.
    tp = jnp.sum(pos_preds)
    # Said it was 1, it was 0.
    fp = jnp.sum(1 - neg_preds)
    # Said it was 0, it was 1.
    fn = jnp.sum(1 - pos_preds)

    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return lp, acc, prec, rec, pos_preds, neg_preds, pos_xs[:, 0]

  @partial(jax.jit)
  def eval_dre_tilt(key, model):
    keys = jax.random.split(key, num=cfg.eval_batch_size)
    lps, accs, precs, recs, pos_preds, neg_preds, true_y = jax.vmap(
            eval_dre_tilt_one_sample, in_axes=(0, None))(keys, model)
    lp = jnp.mean(lps)
    acc = jnp.mean(accs)
    prec = jnp.mean(precs)
    rec = jnp.mean(recs)
    # pos_preds and neg_preds are [eval_batch_size, num_timesteps]
    per_t_pos_acc = jnp.mean(pos_preds, axis=0)
    per_t_neg_acc = jnp.mean(neg_preds, axis=0)
    per_t_acc = (per_t_pos_acc + per_t_neg_acc) / 2.
    per_t_accs = (per_t_pos_acc, per_t_neg_acc, per_t_acc)
    return lp, acc, prec, rec, per_t_accs, true_y

  def summarize(key, params, step):
    total_inner_steps = cfg.model_inner_steps + cfg.tilt_inner_steps
    global_epoch = step // total_inner_steps
    tilt_local_step = global_epoch * cfg.tilt_inner_steps
    model_local_step = global_epoch * cfg.model_inner_steps

    cur_prop_lr = prop_lr_sched(model_local_step)
    cur_tilt_lr = tilt_lr_sched(tilt_local_step)
    if cfg.use_wandb:
      wandb.log({"lrs/prop": cur_prop_lr, "lrs/tilt": cur_tilt_lr}, step=step)

    if cfg.bound == 'sixo':
      tilt = params
    else:
      tilt = None

    # Log the tilt annealing schedule if we're using it.
    if (cfg.tilt_anneal and
          cfg.use_wandb and
          cfg.bound == 'sixo' and
          cfg.tilt_type != 'none' and
          isinstance(tilt, hh.TiltedHHProposal)):
      wandb.log({"tilt/annealing_temp": tilt.tilt_anneal_sched(tilt_local_step)}, step=step)

    # Check if its an expensive summary step
    exp_summ_step = ((step % (cfg.summarize_every * cfg.expensive_summary_mult)) == 0)
    if exp_summ_step:

      st = dt()
      key, subkey = jax.random.split(key)
      bound_value = summ_bound(subkey, step, params).block_until_ready()
      print(f"  {cfg.bound}-{cfg.proposal_type} bound k={cfg.train_num_particles}:"
            f" {bound_value:0.4f} Time: {dt() - st: <4.1f}s.")
      if cfg.use_wandb:
        wandb.log(
            {f"bounds/{cfg.bound}_{cfg.proposal_type}_{cfg.train_num_particles}": bound_value})

      # Evaluate the DRE tilt.
      if cfg.tilt_type == 'bwd_dre':
        lp, acc, prec, rec, _, _ = eval_dre_tilt(key, tilt)
        print(f"  tilt log prob: {lp:0.4f} acc: {acc:0.4f} "
              f"prec: {prec:0.4f} rec: {rec:0.4f}")
        if cfg.use_wandb:
            wandb.log({"tilt/lp": lp, "tilt/acc": acc, "tilt/prec": prec, "tilt/rec": rec},
                      step=step)

  return summarize


def make_summary_bound(
        step: int,
        model,
        bound: str,
        num_timesteps: int,
        ds: snax.InMemDataset,
        num_particles: int):

  def loss_fn(_key: PRNGKey, _datapoint) -> float:
    return make_bound(bound, num_timesteps, num_particles, _key, step, model, _datapoint)

  def avg_bound(key: PRNGKey) -> float:
    return ds.batch_sum_reduce_with_key(key, loss_fn, init_acc=0.) / ds.num_data_points

  return avg_bound


def make_bound(
        bound: str,
        num_timesteps: int,
        num_particles: int,
        key: PRNGKey,
        step: int,
        model: Union[hh.TiltedHHProposal, hh.HHProposal],
        data: Array) -> Scalar:

    tilt = None
    prop = None

    if isinstance(model, hh.TiltedHHProposal):
      tilt = model
      prop = tilt.prop
    elif isinstance(model, hh.HHProposal):
      tilt = None
      prop = model
    else:
      raise ValueError(f"Model must be an HHProposal or TiltedHHProposal, is {model}")

    init_state = jnp.zeros([prop.hh.state_dim])

    if tilt is not None:
      p_and_w = tilt.make_propose_and_weight(step, data, num_timesteps)
    elif prop is not None:
      p_and_w = prop.make_propose_and_weight(step, data, num_timesteps)

    if bound == 'sixo':
        assert isinstance(model, hh.TiltedHHProposal)
        _, _, log_Z_hat, _, _ = bounds.sixo(
            key,
            p_and_w,
            init_state,
            num_timesteps,
            num_timesteps,
            num_particles,
            observations=data)
    elif bound == 'fivo':
        assert isinstance(model, hh.HHProposal)
        _, _, log_Z_hat, _, _ = bounds.fivo(
                key,
                p_and_w,
                init_state,
                num_timesteps,
                num_timesteps,
                num_particles,
                observations=data)
    elif bound == 'iwae':
        assert isinstance(model, hh.HHProposal)
        _, _, log_Z_hat, _, _ = bounds.iwae(
                key,
                p_and_w,
                init_state,
                num_timesteps,
                num_timesteps,
                num_particles,
                observations=data)
    elif bound == 'elbo':
        assert isinstance(model, hh.HHProposal)
        _, _, log_Z_hat, _, _ = bounds.elbo(
                key,
                p_and_w,
                init_state,
                num_timesteps,
                num_timesteps,
                observations=data)
    else:
        raise ValueError(f"Invalid bound name {bound}.")
    return - log_Z_hat


def make_tilt_train_step(cfg, num_timesteps, tilt_opt):

    def tilt_loss(key: PRNGKey, step: int, model: hh.TiltedHHProposal) -> Scalar:
        k1, k2, k3 = jax.random.split(key, num=3)
        neg_xs, _ = model.prop.hh.sample_trajectory(k1, num_timesteps)
        pos_xs, pos_ys = model.prop.hh.sample_trajectory(k2, num_timesteps)
        data = (neg_xs, pos_xs, pos_ys)
        data = jax.lax.stop_gradient(data)
        return model.tilt_loss(k3, step, data)

    tilt_step = train_lib.TrainStep(
        tilt_loss,
        tilt_opt,
        num_inner_steps=cfg.tilt_inner_steps,
        batch_size=cfg.tilt_batch_size,
        parallelize=(cfg.parallelism > 1),
        name="tilt")

    return tilt_step


def make_schedule(lr, steps, mult):
  i_steps = [int(x.strip()) for x in steps.split(",")]
  return optax.piecewise_constant_schedule(lr, {s: mult for s in i_steps})


def train_hh(cfg):
    if cfg.bound != 'sixo':
        assert cfg.tilt_type == 'none'

    assert cfg.obs_subsample < cfg.data_seq_len, "Subsampling too high."

    key = jax.random.PRNGKey(cfg.seed)

    train_key = jax.random.PRNGKey(1)
    trn_data, true_gen_model = gen_data(cfg, train_key, cfg.train_dataset_size)
    train_dataset_itr = datasets.in_mem_jax_dataset(trn_data, 1)

    _, num_timesteps, data_dim = trn_data.shape

    # Create the model
    key, subkey = jax.random.split(key)
    model, labels = make_model(true_gen_model, subkey, data_dim, cfg)

    # Make the summarization functions
    key, subkey = jax.random.split(key)
    summarize = make_summarize(cfg, trn_data)

    # Set up the functions for the proposal loss.
    def train_prop_loss(key, step, params):
      k1, k2 = jax.random.split(key)
      model_loss = partial(make_bound, cfg.bound, num_timesteps, cfg.train_num_particles)
      data = train_dataset_itr(k1)[0]
      return model_loss(k2, step, params, data)

    prop_lr_sched = make_schedule(cfg.prop_lr, cfg.lr_steps, cfg.lr_mult)
    tilt_lr_sched = make_schedule(cfg.tilt_lr, cfg.tilt_lr_steps, cfg.tilt_lr_mult)

    if cfg.p_grad_max_norm > 0:
      prop_opt = util.make_clipped_adam_optimizer(prop_lr_sched, cfg.p_grad_max_norm)
      tilt_opt = util.make_clipped_adam_optimizer(tilt_lr_sched, cfg.p_grad_max_norm)
    else:
      prop_opt = optax.adam(prop_lr_sched)
      tilt_opt = optax.adam(tilt_lr_sched)

    key, subkey = jax.random.split(key)
    if cfg.tilt_type == 'none' and cfg.proposal_type != 'bootstrap':
      # We are not learning a tilt or a model, so but are learning a proposal.
      opt = optax.multi_transform({'model': optax.set_to_zero(), 'proposal': prop_opt}, labels)
      learned_params = train_lib.train(
          key,
          train_prop_loss,
          opt,
          model,
          parallelize=(cfg.parallelism > 1),
          batch_size=cfg.train_batch_size,
          num_steps=cfg.num_train_steps,
          summarize_every=cfg.summarize_every,
          summarize_fn=summarize,
          checkpoint_every=cfg.checkpoint_every,
          checkpoint_dir=cfg.checkpoint_dir,
          checkpoints_to_keep=cfg.checkpoints_to_keep,
          use_wandb=cfg.use_wandb)

    elif cfg.tilt_type != 'none':
      # We are learning a tilt, and possibly a proposal.
      # Create the train step for updating model and proposal (not tilt).
      # Use the predefined model_opt for clipping but mask it to just the proposal and model.
      prop_opt = optax.multi_transform(
              {'model': optax.set_to_zero(), 'proposal': prop_opt, 'tilt': optax.set_to_zero()},
              labels)
      prop_step = train_lib.TrainStep(
              train_prop_loss,
              prop_opt,
              num_inner_steps=cfg.model_inner_steps,
              batch_size=cfg.train_batch_size,
              parallelize=(cfg.parallelism > 1),
              name="sixo")
      tilt_step_opt = optax.multi_transform(
              {'model': optax.set_to_zero(), 'proposal': optax.set_to_zero(), 'tilt': tilt_opt},
              labels)
      # Create the train step for the tilt.
      tilt_step = make_tilt_train_step(cfg, num_timesteps, tilt_step_opt)
      # Use the fivo and tilt train steps to train.
      key, subkey = jax.random.split(key)
      learned_params = train_lib.train_alternating(
          key,
          [prop_step, tilt_step],
          model,
          num_steps=cfg.num_train_steps,
          summarize_every=cfg.summarize_every,
          summarize_fn=summarize,
          checkpoint_every=cfg.checkpoint_every,
          checkpoint_dir=cfg.checkpoint_dir,
          checkpoints_to_keep=cfg.checkpoints_to_keep,
          use_wandb=cfg.use_wandb)
    elif cfg.tilt_type == 'none' and cfg.proposal_type == 'bootstrap':
      learned_params = model

    print("Training complete, evaluating model...")
    evaluate_model(cfg, subkey, cfg.num_train_steps, learned_params, trn_data)
    return learned_params


def evaluate_model(cfg, key, step, model, data):
  num_timesteps = data.shape[1]
  ds = snax.InMemDataset(data, cfg.eval_batch_size, shuffle=False)
  num_eval_particles = [cfg.train_num_particles] + \
          [int(x.strip()) for x in cfg.eval_num_particles.split(",")]
  bound_fns = []
  for n in num_eval_particles:
    bound_fns.append(jax.jit(make_summary_bound(step, model, cfg.bound, num_timesteps, ds, n)))

  for n, b_fn in zip(num_eval_particles, bound_fns):
    for i in range(cfg.num_evals):
      key, subkey = jax.random.split(key)
      st = dt()
      b_value = b_fn(subkey).block_until_ready()
      print(f"  {i} {cfg.bound}-{cfg.proposal_type} bound k={n}:"
            f" {b_value:0.4f} Time: {dt() - st: <4.1f}s.")
      if cfg.use_wandb:
        wandb.log({f"bounds/{cfg.bound}_{cfg.proposal_type}_{n}": b_value})


def main():
    args = parser.parse_args()
    util.print_args(args)
    if args.parallelism > 1:
        os.environ["XLA_FLAGS"] = f" --xla_force_host_platform_device_count={args.parallelism}"
        print(f"Set number of XLA devices to {args.parallelism},"
              f" JAX now sees {jax.local_device_count()} devices.")

    if args.checkpoint_dir is not None:
        pathlib.Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    if args.use_wandb:
        wandb.init(
            project=args.wandb_proj,
            name=args.wandb_run,
            notes=args.wandb_notes,
            config=args,
            id=args.wandb_id,
            dir=args.checkpoint_dir
        )

    _ = train_hh(args)


if __name__ == '__main__':
  main()
