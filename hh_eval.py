"""A script for evaluating trained Hodgkin-Huxley models."""
import shutil
from pathlib import Path
import pickle
import jax
import jax.numpy as jnp
import numpy as onp
import equinox as eqx
from jax._src.random import KeyArray as PRNGKey
from typing import Union
from chex import Array, Scalar
import argparse
import util
import wandb
import wandb.plot
import models.hh as hh
import snax
import bounds
import datasets
from distutils.util import strtobool
from util import abs_and_rel_diff
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

import logging
logger = logging.getLogger()


class CheckTypesFilter(logging.Filter):

    def filter(self, record):
        return "check_types" not in record.getMessage()


logger.addFilter(CheckTypesFilter())

parser = argparse.ArgumentParser(description='Train HH model.')

# Exp config.
parser.add_argument(
    '--bound', type=str, choices=['fivo', 'iwae', 'elbo', 'sixo'],
    default='fivo',
    help="The bound to optimize")
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

# Tilt arch parameters.
parser.add_argument(
    '--tilt_type', type=str,
    default='none', choices=['none', 'bwd_dre'],
    help="Parametric form of tilt.")
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
    '--eval_num_particles', type=int,
    default=128,
    help="Number of particles in eval bound.")

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
    '--train_dataset_size', type=int,
    default=10_000,
    help="Number of data points in train dataset.")
parser.add_argument(
    '--val_dataset_size', type=int,
    default=64,
    help="Number of data points in validation dataset.")
parser.add_argument(
    '--test_dataset_size', type=int,
    default=64,
    help="Number of data points in test dataset.")

# Menial args.
parser.add_argument(
    '--eval_batch_size', type=int,
    default=64,
    help="Evaluation batch size.")
parser.add_argument(
    '--seed', type=int,
    default=12,
    help="Random seed.")
parser.add_argument(
    '--checkpoint_basedir', type=str,
    default=None,
    help="Where to store checkpoints.")
parser.add_argument(
    '--run_name', type=str,
    default=None,
    help="Name of the run, used for checkpointing.")
parser.add_argument(
    '--store_basedir', type=str,
    default=None,
    help="Where to copy the checkpoints for storage.")
parser.add_argument(
    '--num_evals', type=int,
    default=3,
    help="Number of evals to do.")

parser.add_argument(
    '--eval_train', type=strtobool,
    default="False",
    help="If true, evaluate on the training set.")
parser.add_argument(
    '--eval_val', type=strtobool,
    default="True",
    help="If true, evaluate on the validation set.")
parser.add_argument(
    '--eval_test', type=strtobool,
    default="True",
    help="If true, evaluate on the test set.")
parser.add_argument(
    '--plot', type=strtobool,
    default="True",
    help="If true, generate pickles for plotting.")

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
    '--i_ext_rel_error_init',
    type=float,
    default=None,
    help='Initial external current in HH model we are training'
)


def gen_data(cfg, key: PRNGKey, num_datapoints: int):
  data, gen_model = datasets.create_synthetic_hh_dataset(
          key,
          num_datapoints,
          cfg.num_compartments,
          cfg.data_seq_len,
          cfg.obs_subsample,
          cfg.dt,
          cfg.ode_int,
          5.)
  return data, gen_model


def make_model(key: PRNGKey, data_dim: int, cfg):
    k1, k2, k3 = jax.random.split(key, num=3)
    assert (cfg.i_ext_rel_error_init is None) or \
            (isinstance(cfg.i_ext_rel_error_init, float) and cfg.i_ext_rel_error_init > -1.)

    # Define the base model.
    model = hh.HodgkinHuxley(
            k1,
            cfg.num_compartments,
            cfg.dt,
            ode_solver=cfg.ode_int,
            obs_subsample=cfg.obs_subsample,
            i_ext_rel_error_init=cfg.i_ext_rel_error_init)

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

        k1, k2 = jax.random.split(k3)
        bwd_rnn_hdims = [int(x.strip()) for x in cfg.bwd_tilt_rnn_hdims.split(",")]
        mlp_hdims = [int(x.strip()) for x in cfg.bwd_tilt_mlp_hdims.split(",")]

        model = hh.BackwardsTilt(
          k1,
          model,
          snax.LSTM(k2, data_dim, bwd_rnn_hdims),
          mlp_hdims,
          model.hh.state_dim,
          tilt_anneal_sched=hh.no_annealing)

        # Construct a pytree with all nodes labeled as tilt parameters.
        all_tilt = jax.tree_util.tree_map(lambda _: 'tilt', model)
        # Replace the subtree at 'prop' with the labels previously constructed.
        model_labels = eqx.tree_at(lambda x: x.prop, all_tilt, model_labels)

    return model, model_labels


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
      raise ValueError("Model must be an HHProposal or TiltedHHProposal")

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


def eval_hh(cfg):
  if cfg.bound != 'sixo':
      assert cfg.tilt_type == 'none'
  assert cfg.obs_subsample < cfg.data_seq_len, "Subsampling too high."

  train_key = jax.random.PRNGKey(1)
  val_key = jax.random.PRNGKey(2)
  test_key = jax.random.PRNGKey(3)
  trn_data, true_gen_model = gen_data(cfg, train_key, cfg.train_dataset_size)
  train_ds = snax.InMemDataset(trn_data, cfg.eval_batch_size, shuffle=False)
  val_data, _ = gen_data(cfg, val_key, cfg.val_dataset_size)
  val_ds = snax.InMemDataset(val_data, cfg.eval_batch_size, shuffle=False)
  tst_data, _ = gen_data(cfg, test_key, cfg.test_dataset_size)
  test_ds = snax.InMemDataset(tst_data, cfg.eval_batch_size, shuffle=False)
  _, num_timesteps, data_dim = trn_data.shape

  # create the bounds to evaluate
  train_method = lambda k, m: make_summary_bound(
          global_step, m, cfg.bound, num_timesteps, train_ds, cfg.eval_num_particles)(k)
  val_method = lambda k, m: make_summary_bound(
          global_step, m, cfg.bound, num_timesteps, val_ds, cfg.eval_num_particles)(k)
  test_method = lambda k, m: make_summary_bound(
          global_step, m, cfg.bound, num_timesteps, test_ds, cfg.eval_num_particles)(k)
  train_method = jax.vmap(train_method, in_axes=(0, None))
  val_method = jax.vmap(val_method, in_axes=(0, None))
  test_method = jax.vmap(test_method, in_axes=(0, None))

  train_bpf = lambda k, m: make_summary_bound(
          global_step, m, "fivo", num_timesteps, train_ds, cfg.eval_num_particles)(k)
  val_bpf = lambda k, m: make_summary_bound(
          global_step, m, "fivo", num_timesteps, val_ds, cfg.eval_num_particles)(k)
  test_bpf = lambda k, m: make_summary_bound(
          global_step, m, "fivo", num_timesteps, test_ds, cfg.eval_num_particles)(k)
  train_bpf = jax.vmap(train_bpf, in_axes=(0, None))
  val_bpf = jax.vmap(val_bpf, in_axes=(0, None))
  test_bpf = jax.vmap(test_bpf, in_axes=(0, None))

  def eval_checkpoint(model, key, num):
    if cfg.bound == 'sixo':
      hh_model = model.prop.hh
    else:
      hh_model = model.hh
    bs_model = hh.BootstrapHHProposal(hh_model)

    wandb_outs = {}

    if cfg.eval_train:
      print("  Computing train method bounds")
      key, subkey = jax.random.split(key)
      keys = jax.random.split(subkey, num=cfg.num_evals)
      train_method_bounds = train_method(keys, model)
      print("   ", train_method_bounds)

      print("  Computing train bpf bounds")
      key, subkey = jax.random.split(key)
      keys = jax.random.split(subkey, num=cfg.num_evals)
      train_bpf_bounds = train_bpf(keys, bs_model)
      print("   ", train_bpf_bounds)
      for i in range(cfg.num_evals):
        wandb_outs[f"train_method_{cfg.eval_num_particles}_{num}_{i}"] = train_method_bounds[i]
        wandb_outs[f"train_bpf_{cfg.eval_num_particles}_{num}_{i}"] = train_bpf_bounds[i]

    if cfg.eval_val:
      print("  Computing val method bounds")
      key, subkey = jax.random.split(key)
      keys = jax.random.split(subkey, num=cfg.num_evals)
      val_method_bounds = val_method(keys, model)
      print("   ", val_method_bounds)

      print("  Computing val bpf bounds")
      key, subkey = jax.random.split(key)
      keys = jax.random.split(subkey, num=cfg.num_evals)
      val_bpf_bounds = val_bpf(keys, bs_model)
      print("   ", val_bpf_bounds)
      for i in range(cfg.num_evals):
        wandb_outs[f"val_method_{cfg.eval_num_particles}_{num}_{i}"] = val_method_bounds[i]
        wandb_outs[f"val_bpf_{cfg.eval_num_particles}_{num}_{i}"] = val_bpf_bounds[i]

    if cfg.eval_test:
      print("  Computing test method bounds")
      key, subkey = jax.random.split(key)
      keys = jax.random.split(subkey, num=cfg.num_evals)
      test_method_bounds = test_method(keys, model)
      print("   ", test_method_bounds)

      print("  Computing test bpf bounds")
      key, subkey = jax.random.split(key)
      keys = jax.random.split(subkey, num=cfg.num_evals)
      test_bpf_bounds = test_bpf(keys, bs_model)
      print("   ", test_bpf_bounds)
      for i in range(cfg.num_evals):
        wandb_outs[f"test_method_{cfg.eval_num_particles}_{num}_{i}"] = test_method_bounds[i]
        wandb_outs[f"test_bpf_{cfg.eval_num_particles}_{num}_{i}"] = test_bpf_bounds[i]

    if cfg.use_wandb:
      wandb.log(wandb_outs)

    print("  Parameter differences")
    abs_diff, rel_diff = abs_and_rel_diff(hh_model.const_i_ext, true_gen_model.const_i_ext)
    print(f"    const_i_ext abs diff: {abs_diff:0.5f} rel diff: {rel_diff:0.5f}")
    if cfg.use_wandb:
      wandb.log({f"const_i_ext_abs_diff_{num}": abs_diff, f"const_i_ext_rel_diff_{num}": rel_diff})

  # Create the model
  key = jax.random.PRNGKey(cfg.seed)
  key, subkey = jax.random.split(key)
  model, _ = make_model(subkey, data_dim, cfg)
  _, treedef = jax.tree_util.tree_flatten(model)

  # copy the checkpoints to storage
  checkpoint_dir = Path(cfg.checkpoint_basedir) / cfg.run_name
  checkpoint_paths = snax.checkpoint.get_checkpoints(checkpoint_dir)
  print(f"Found {len(checkpoint_paths)} checkpoints in {checkpoint_dir}.")
  storage_dir = Path(cfg.store_basedir) / cfg.run_name
  print(f"Copying checkpoints to {storage_dir}.")
  for p in checkpoint_paths:
    storage_chk_path = storage_dir / p.name
    if storage_chk_path.exists() and storage_chk_path.is_file():
      print(f"Skipping copying checkpoint at {p}, as it already exists in storage.")
      continue
    shutil.copy2(p, storage_dir)

  checkpoint_paths = snax.checkpoint.get_checkpoints(storage_dir)
  print(f"Found {len(checkpoint_paths)} checkpoints in storage at {storage_dir}.")
  for i, p in enumerate(checkpoint_paths):
    # Load checkpoints
    data, global_step = snax.checkpoint.load_checkpoint_from_path(p)
    loaded_leaves, _ = jax.tree_util.tree_flatten(data[0])
    loaded_leaves = [jnp.array(x) for x in loaded_leaves]
    restored_model = treedef.unflatten(loaded_leaves)
    print(f"Loaded checkpoint {i+1} from {p} at step {global_step}.")
    key, subkey = jax.random.split(key)
    eval_checkpoint(restored_model, subkey, i)

  if cfg.plot:
    print("Making plotting pickles")
    path = snax.checkpoint.get_latest_checkpoint_path(cfg.checkpoint_dir)
    data, global_step = snax.checkpoint.load_checkpoint_from_path(path)
    loaded_leaves, _ = jax.tree_util.tree_flatten(data[0])
    loaded_leaves = [jnp.array(x) for x in loaded_leaves]
    restored_model = treedef.unflatten(loaded_leaves)

    if cfg.bound == 'sixo':
      hh_model = restored_model.prop.hh
    else:
      hh_model = restored_model.hh

    init_state = jnp.zeros([hh_model.state_dim])
    bs_model = hh.BootstrapHHProposal(hh_model)

    ds = val_data

    def bpf_bound(key, dp):
      p_and_w = bs_model.make_propose_and_weight(0, dp, num_timesteps)
      bpf_states, _, _, _, bpf_resampled = bounds.fivo(
                key,
                p_and_w,
                init_state,
                num_timesteps,
                num_timesteps,
                cfg.eval_num_particles,
                observations=dp)
      return bpf_states, bpf_resampled

    key, subkey = jax.random.split(key)
    bpf_states, bpf_resampled = jax.vmap(bpf_bound)(
            jax.random.split(subkey, num=ds.shape[0]), ds)

    with open("bpf_sweep.p", "wb") as f:
      pickle.dump({"particles": onp.array(bpf_states), "resampled": onp.array(bpf_resampled)}, f)
    with open("obs.p", "wb") as f2:
      pickle.dump({"obs": onp.array(ds)}, f2)

    if cfg.bound != 'sixo':
      print("Can't plot sixo trajectories for non-sixo model, skipping")
    else:
      def sixo_bound(key, dp):
        sixo_p_and_w = restored_model.make_propose_and_weight(0, dp, num_timesteps)
        sixo_states, _, _, _, sixo_resampled = bounds.sixo(
                key,
                sixo_p_and_w,
                init_state,
                num_timesteps,
                num_timesteps,
                cfg.eval_num_particles,
                observations=dp)
        return sixo_states, sixo_resampled

      sixo_states, sixo_resampled = jax.vmap(sixo_bound)(
              jax.random.split(key, num=ds.shape[0]), ds)
      with open("smc_sweep.p", "wb") as f3:
        pickle.dump({
            "particles": onp.array(sixo_states),
            "resampled": onp.array(sixo_resampled)}, f3)


def main():
  args = parser.parse_args()
  util.print_args(args)
  storage_dir = Path(args.store_basedir) / args.run_name
  storage_dir.mkdir(parents=True, exist_ok=True)

  if args.use_wandb:
    wandb.init(
        project=args.wandb_proj,
        name=args.wandb_run,
        notes=args.wandb_notes,
        config=args,
        id=args.wandb_id,
        dir=str(storage_dir),
    )

  eval_hh(args)


if __name__ == '__main__':
  main()
