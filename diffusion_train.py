"""Trains a simple Gaussian diffusion model.

Example calls:

FIVO training:
  python train_diffusion.py --train_steps 250000 --train_num_particles 10 --summarize_every 10000
                            --eval_num_particles 64 --bound fivo --syn_data_num_timesteps 10
                            --syn_data_num_series 10000 --lr 1e-4 --batch_size 32 --learn_q

FIVO training but with a tilt: (YES: sixo is correct)
  python train_diffusion.py --seed 1 --train_steps 500000 --train_num_particles 10
                            --eval_num_particles 128 --bound sixo --syn_data_num_timesteps 10
                            --syn_data_num_series 1000 --lr 1e-3 --batch_size 32
                            --resampling_gradient_mode none
                            --learn_p --learn_q --drift 1
"""
import os
import argparse
from functools import partial
import jax
import jax.numpy as jnp
from jax._src.random import KeyArray as PRNGKey
import optax
import wandb
from chex import Scalar

import bounds
import snax
import datasets
from util import abs_and_rel_diff, make_masked_optimizer
from smc import always_resample_criterion
from models.diffusion import (
  GaussianDiffusion, GaussianDiffusionWithLearntProposal, GaussianDiffusionWithPosteriorProposal,
  GaussianDiffusionWithProposal, StandardTilt, TiltedGaussianDiffusion, MLPTilt)

import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

parser = argparse.ArgumentParser(description='Train diffusion.')
parser.add_argument(
        '--syn_data_num_series', type=int,
        default=10,
        help='Number of series to train on for synthetic data.')
parser.add_argument(
        '--syn_data_num_timesteps', type=int,
        default=10,
        help='Length of series to train on for synthetic data.')
parser.add_argument(
        '--train_num_particles', type=int,
        default=4,
        help='Number of particles in train bound.')
parser.add_argument(
        '--train_steps', type=int,
        default=200_000,
        help='Number of steps to train for.')
parser.add_argument(
        '--lr', type=float,
        default=1e-4,
        help='Learning rate.')
parser.add_argument(
        '--batch_size', type=int,
        default=16,
        help='Batch size.')
parser.add_argument(
        '--eval_num_particles', type=int,
        default=128,
        help='Number of particles in eval bound.')
parser.add_argument(
        '--checkpoint_dir', type=str,
        default='/tmp/fivo_diff',
        help='Where to store checkpoints.')
parser.add_argument(
        '--checkpoints_to_keep', type=int,
        default=3,
        help='Number of checkpoints to keep.')
parser.add_argument(
        '--seed', type=int,
        default=0,
        help='Random seed.')
parser.add_argument(
        '--summarize_every', type=int,
        default=1_000,
        help='Steps between summaries.')
parser.add_argument(
        '--drift', type=float,
        default=0.,
        help='Generative model drift')
parser.add_argument(
        '--bound', type=str,
        choices=["sixo", "fivo", "iwae"],
        default='sixo')
parser.add_argument(
        '--resampling_gradient_mode', type=str,
        choices=["score_fn", "score_fn_rb", "none"],
        default='score_fn_rb')
parser.add_argument(
        '--train_inner_steps', type=int,
        default=400,
        help='Number of steps to train tilt for. Used only for DRE training.')
parser.add_argument(
        '--tilt_inner_steps', type=int,
        default=100,
        help='Number of steps to train tilt for. Used only for DRE training.')
parser.add_argument(
        '--tilt_lr', type=float,
        default=1e-4,
        help='Tilt learning rate.')
parser.add_argument(
        '--tilt_mlp_hdims', type=str,
        default="32",
        help="Comma-separated list of hidden dims for MLP tilt.")
parser.add_argument(
        '--tilt_train_method', type=str,
        choices=["dre", "unified", "none"],
        default='none')
parser.add_argument(
        '--tilt_batch_size', type=int,
        default=64,
        help='Tilt batch size.')
parser.add_argument(
        '--learn_p',
        dest='learn_p', action='store_true',
        default=False)
parser.add_argument(
        '--learn_q',
        dest='learn_q', action='store_true',
        default=False)
parser.add_argument(
        '--learn_r',
        dest='learn_r', action='store_true',
        default=False)
parser.add_argument(
        '--parallelism', type=int,
        default=1,
        help='Number of XLA devices to use.')
parser.add_argument(
        '--use_wandb',
        action='store_true')
parser.add_argument(
        '--wandb_proj',
        type=str,
        default=None)


def fivo_loss(
        bound: str,
        num_timesteps: int,
        resampling_gradient_mode: str,
        num_particles: int,
        key: PRNGKey,
        model,
        data: Scalar) -> Scalar:
  init_state = jnp.array(0.)
  if bound == 'sixo':
    assert isinstance(model, TiltedGaussianDiffusion)
    p_and_w = model.make_propose_and_weight(data, num_timesteps)
    _, _, log_Z_hat, _, _ = bounds.sixo(
            key,
            p_and_w,
            init_state,
            num_timesteps,
            num_timesteps,
            num_particles,
            resampling_criterion=always_resample_criterion,
            resampling_gradient_mode=resampling_gradient_mode)
  else:
    assert isinstance(model, GaussianDiffusionWithProposal)

    # p_and_w needs to wrap data
    def p_and_w(key, prev_z, t):
      return model.propose_and_weight(key, prev_z, data, t)

    if bound == 'fivo':
      _, _, log_Z_hat, _, _ = bounds.fivo(
              key,
              p_and_w,
              init_state,
              num_timesteps,
              num_timesteps,
              num_particles,
              resampling_criterion=always_resample_criterion,
              resampling_gradient_mode=resampling_gradient_mode)
    elif bound == 'iwae':
      _, _, log_Z_hat, _, _ = bounds.iwae(
              key,
              p_and_w,
              init_state,
              num_timesteps,
              num_timesteps,
              num_particles)
  return - log_Z_hat


def make_summarize(cfg, eval_loss, xs, true_drift):

  seq_len = cfg.syn_data_num_timesteps

  @jax.jit
  def eval_model(key, model, n_replicates=16):
    """
    Returns
    -------
    true_avg_ll: float
      ll of each observation *under current p* averaged over all observations in dataset

    model_avg_ll: float
      ll of each observation estimate returned by SMC call (with current p, q, r), using
      eval_num_particles
    """
    # This assumes we can run the entire dataset in one batch (at high number of particles)
    # FIXME: optionally run on a fraction of the data or something
    def compute_avg_ll_one_obs(key, model, obs):
      # For one observation get estimate of ELBO by averaging over keys
      return - jax.vmap(eval_loss, in_axes=(0, None, None))(
        jax.random.split(key, num=n_replicates), model, obs)

    avg_ll_across_keys_and_obs = jax.vmap(compute_avg_ll_one_obs, in_axes=(0, None, 0))

    # model_avg_ll is (n_replicates, n_timeseries)
    model_avg_ll = avg_ll_across_keys_and_obs(jax.random.split(key, num=len(xs)), model, xs)

    # Averaging over datapoints first
    # (naming not exactly right, averaging over axis=1 not too sensical, summing better)
    full_dataset_ll_per_seed = jnp.mean(model_avg_ll, axis=1)
    mu = jnp.mean(full_dataset_ll_per_seed)
    std = jnp.std(full_dataset_ll_per_seed)
    model_avg_ll = mu

    if cfg.bound == 'sixo':
      true_avg_ll = jnp.mean(model.model.model.marginal().log_prob(xs))
    else:
      true_avg_ll = jnp.mean(model.model.marginal().log_prob(xs))
    return true_avg_ll, model_avg_ll, std

  def summarize(key, model, step):
    """
    This method is obviously too long; most of the logic is for logging reasonably nicely
    to command line
    """
    # NOTE: `model_avg_ll` is computed with smoothed model params
    true_avg_ll, model_avg_ll, stddev = eval_model(key, model)

    gap = true_avg_ll - model_avg_ll
    print(f'  avg_ll under model: {true_avg_ll: 0.3f}, '
          f'observed avg_ll ({cfg.eval_num_particles: } particles): {model_avg_ll: 0.3f}, '
          f'gap: {gap: 0.3f}')
    if cfg.use_wandb:
      wandb.log({f'eval_bound_{cfg.eval_num_particles}': model_avg_ll}, step=step)
      wandb.log({'ll_under_model': model_avg_ll}, step=step)
      wandb.log({'ll_under_model_std': stddev}, step=step)
      wandb.log({f'avg_ll_gap_{cfg.eval_num_particles}': gap}, step=step)

    print('  Parameter statistics:')

    # These have proposal params
    if cfg.bound == 'sixo':
      _model = model.model
    else:
      _model = model

    if cfg.learn_q:
      q_prev_z_weights = _model.q_z_weights
      for _t in range(seq_len):
        # Use smoothed weights to get parameter diffs
        abs_diff, rel_diff = abs_and_rel_diff(
          q_prev_z_weights[_t], _model.model.q_prev_z_weight(_t))
        if _t == 0:
          print('     q_prev_z_weight abs_diff     N/A', end=' ')
        else:
          print(f'{abs_diff: 0.3f}', end=' ')
          if cfg.use_wandb:
            wandb.log({f'q_prev_z_weight_{_t}': q_prev_z_weights[_t]}, step=step)
            wandb.log({f'q_prev_z_weight_{_t}_abs_diff': abs_diff}, step=step)
            wandb.log({f'q_prev_z_weight_{_t}_rel_diff': rel_diff}, step=step)
      print()
      q_x_weights = _model.q_x_weights
      for _t in range(seq_len):
        abs_diff, rel_diff = abs_and_rel_diff(q_x_weights[_t], _model.model.q_x_weight(_t))
        if _t == 0:
          print('          q_x_weight abs_diff  ', end=' ')
        print(f'{abs_diff: 0.3f}', end=' ')
        if cfg.use_wandb:
          wandb.log({f'q_x_weight_{_t}': q_x_weights[_t]}, step=step)
          wandb.log({f'q_x_weight_{_t}_abs_diff': abs_diff}, step=step)
          wandb.log({f'q_x_weight_{_t}_rel_diff': rel_diff}, step=step)
      print()
      # This smoothing is happening at the log level
      q_vars = jnp.exp(_model.q_log_vars)
      for _t in range(seq_len):
        abs_diff, rel_diff = abs_and_rel_diff(q_vars[_t], _model.model.q_var(_t))
        if _t == 0:
          print('               q_var abs_diff  ', end=' ')
        print(f'{abs_diff: 0.3f}', end=' ')
        if cfg.use_wandb:
          wandb.log({f'q_var_{_t}': q_vars[_t]}, step=step)
          wandb.log({f'q_var_{_t}_abs_diff': abs_diff}, step=step)
          wandb.log({f'q_var_{_t}_rel_diff': rel_diff}, step=step)
      print()
      q_biases = _model.q_biases
      for _t in range(seq_len):
        # q_bias should always be 0 (regardless of drift and variance in model)
        if _t == 0:
          print('                       q_bias  ', end=' ')
        print(f'{q_biases[_t]: 0.3f}', end=' ')
        if cfg.use_wandb:
          wandb.log({f'q_bias_{_t}': q_biases[_t]}, step=step)
      print()

    # Print drift stats always just to check whether p is training
    print(f'                  model_drift   {_model.model.drift: .3f}')
    print(f'                   true_drift   {true_drift: .3f}')
    if cfg.use_wandb:
      wandb.log({'true_drift': true_drift}, step=step)
      wandb.log({'model_drift': _model.model.drift}, step=step)  # could make this one smoothed
      if cfg.learn_p:
        abs_diff, rel_diff = abs_and_rel_diff(_model.model.drift, true_drift)
        wandb.log({'model_drift_abs_diff': abs_diff}, step=step)
        wandb.log({'model_drift_rel_diff': rel_diff}, step=step)

    if cfg.bound == 'sixo' and isinstance(model.tilt, StandardTilt):
      r_biases = model.tilt.biases
      r_vars = jnp.exp(model.tilt.log_vars)

      for _t in range(seq_len):
        if _t == 0:
          print('                       r_bias  ', end=' ')
        if _t == seq_len - 1:
          print('N/A')
        else:
          print(f'{r_biases[_t]: 0.3f}', end=' ')
      for _t in range(seq_len):
        # NOTE: lookahead_bias will be 0 early on until model gets away from 0 drift
        # Just logging anyways for now
        abs_diff, rel_diff = abs_and_rel_diff(r_biases[_t], _model.model.lookahead_bias(_t))
        if _t == 0:
          print('              r_bias abs_diff  ', end=' ')
        if _t == seq_len - 1:
          print('N/A')
        else:
          print(f'{abs_diff: 0.3f}', end=' ')
          if cfg.use_wandb:
            wandb.log({f'r_bias_{_t}': r_biases[_t]}, step=step)
            wandb.log({f'r_bias_{_t}_abs_diff': abs_diff}, step=step)
            wandb.log({f'r_bias_{_t}_rel_diff': rel_diff}, step=step)
      print()
      for _t in range(seq_len):
        if _t == 0:
          print('                        r_var  ', end=' ')
        if _t == seq_len - 1:
          print('N/A')
        else:
          print(f'{r_vars[_t]: 0.3f}', end=' ')
      print()
      for _t in range(seq_len):
        abs_diff, rel_diff = abs_and_rel_diff(r_vars[_t], _model.model.lookahead_var(_t))
        if _t == 0:
          print('               r_var abs_diff  ', end=' ')
        if _t == seq_len - 1:
          print('N/A')
        else:
          print(f'{abs_diff: 0.3f}', end=' ')
          if cfg.use_wandb:
            wandb.log({f'r_var_{_t}': r_vars[_t]}, step=step)
            wandb.log({f'r_var_{_t}_abs_diff': abs_diff}, step=step)
            wandb.log({f'r_var_{_t}_rel_diff': rel_diff}, step=step)

  return summarize


def make_unified_train_step(cfg, model_and_proposal, train_fivo_loss, seq_len):
  if cfg.bound == 'sixo':
    tilt = None
    if cfg.learn_r:
      tilt = StandardTilt(seq_len, log_var_init=0.)
    model = TiltedGaussianDiffusion(model_and_proposal, tilt)
  else:
    model = model_and_proposal

  # We don't parallelize the unified train step because it is slower for some reason.
  # We also jit 100 steps at a time, again for speed reasons.
  train_step = snax.TrainStep(
          train_fivo_loss,
          optax.adam(cfg.lr),
          num_inner_steps=100,
          parallelize=False,
          batch_size=cfg.batch_size,
          name="loss")
  return model, [train_step]


def make_dre_train_step(cfg, key, model_and_proposal, train_fivo_loss, seq_len):

  hdims = [int(x) for x in cfg.tilt_mlp_hdims.split(",")]
  assert len(hdims) != 0, f"Must have a hidden layer for MLPTilt, passed {cfg.tilt_mlp_hdims}."
  key, subkey = jax.random.split(key)
  tilt = MLPTilt(subkey, seq_len, hdims)
  model = TiltedGaussianDiffusion(model_and_proposal, tilt)

  fivo_opt = make_masked_optimizer(
          optax.adam(cfg.lr), [(lambda m: m.model, True)], mask_default=False)

  fivo_train_step = snax.TrainStep(
          train_fivo_loss,
          fivo_opt,
          num_inner_steps=cfg.train_inner_steps,
          parallelize=(cfg.parallelism > 1),
          batch_size=cfg.batch_size,
          name="fivo")

  def dre_tilt_loss(key: PRNGKey, step: int, model: TiltedGaussianDiffusion) -> Scalar:
    del step
    k1, k2 = jax.random.split(key)
    zs_pos, xs = model.model.model.sample_trajectory(k1)
    zs_neg, _ = model.model.model.sample_trajectory(k2)
    data = (zs_pos, zs_neg, xs)
    assert model.tilt is not None
    return model.tilt.dre_tilt_loss(data)

  tilt_opt = make_masked_optimizer(
          optax.adam(cfg.tilt_lr), [(lambda m: m.model, False)], mask_default=True)

  dre_tilt_train_step = snax.TrainStep(
          dre_tilt_loss,
          tilt_opt,
          num_inner_steps=cfg.tilt_inner_steps,
          batch_size=cfg.tilt_batch_size,
          parallelize=(cfg.parallelism > 1),
          name="tilt")

  return model, [fivo_train_step, dre_tilt_train_step]


def train_diffusion(cfg):
  """
  Parameters
  ----------
  bound: str, default: 'sixo'
    If 'sixo' this method *always* trains r

  drift: float, default: 1.
    drift to use in Gaussian Diffusion model that generates training data

  Notes
  -----
    - Can train with fewer timeseries if we compute what the "true" empirical drift value is
    - Can initialize parameters to N(0,1). How important this is depends on convexity of problem

  TODO:
    - Only training r on w&b on a couple of `num_series`
    - Train on one datapoint and see how fivo vs. sixo does
  """
  if cfg.learn_r:
    assert cfg.bound == 'sixo', 'Cannot learn_r with bound other than sixo.'
  if cfg.bound == 'sixo' and not cfg.learn_r:
    print('WARNING: running sixo without learning r, make sure this is intentional')
    assert cfg.resampling_gradient_mode == 'none', \
            'This is likely a test of FIVO tilting with true r'

  seq_len = cfg.syn_data_num_timesteps
  key = jax.random.PRNGKey(cfg.seed)

  # Create the dataset
  key, subkey = jax.random.split(key)
  (_, xs), ds_itr, _, true_drift = datasets.create_synthetic_diffusion_dataset(
          subkey, seq_len, cfg.syn_data_num_series, 1)

  # Construct model and proposal (initializing p drift to 0. if training)
  # NOTE: if *not* learning p then drift is initialized to generative model drift
  drift = 0. if cfg.learn_p else cfg.drift
  diffusion_model = GaussianDiffusion(seq_len, drift=drift, train=cfg.learn_p)
  if cfg.learn_q:
    model_and_proposal = GaussianDiffusionWithLearntProposal(diffusion_model)
  else:
    model_and_proposal = GaussianDiffusionWithPosteriorProposal(diffusion_model)

  # Make the loss function
  loss_fn = partial(fivo_loss, cfg.bound, seq_len, cfg.resampling_gradient_mode)

  def train_fivo_loss(key, _, params):
    k1, k2 = jax.random.split(key)
    x = ds_itr(k1)[0]
    return loss_fn(cfg.train_num_particles, k2, params, x)

  eval_fivo_loss = partial(loss_fn, cfg.eval_num_particles)

  if cfg.tilt_train_method == 'dre':
    model, train_steps = make_dre_train_step(
            cfg, key, model_and_proposal, train_fivo_loss, seq_len)
  elif cfg.tilt_train_method in ['unified', 'none']:
    model, train_steps = make_unified_train_step(
            cfg, model_and_proposal, train_fivo_loss, seq_len)
  else:
    assert False, "tilt_train_method must be either 'dre', 'unified'," \
            f" or 'none', was {cfg.tilt_train_method}."

  snax.train_alternating(
    key,
    train_steps,
    model,
    num_steps=cfg.train_steps,
    summarize_every=cfg.summarize_every,
    summarize_fn=make_summarize(cfg, eval_fivo_loss, xs, true_drift),
    checkpoint_every=1000 * cfg.summarize_every,
    checkpoint_dir=cfg.checkpoint_dir,
    checkpoints_to_keep=cfg.checkpoints_to_keep)


def main(args):
  if args.parallelism > 1:
    os.environ["XLA_FLAGS"] = f" --xla_force_host_platform_device_count={args.parallelism}"
    print(f"Set number of XLA devices to {args.parallelism},"
          f" JAX now sees {jax.local_device_count()} devices.")

  if args.use_wandb:
    wandb.init(
      project=args.wandb_proj,
      config=args)

  train_diffusion(args)


if __name__ == '__main__':
  main(parser.parse_args())
