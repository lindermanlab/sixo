"""Dataset utilities."""
import os
import jax
import jax.numpy as jnp
import numpy as onp
import tensorflow as tf
from models import svm
from models import hh
from models import diffusion
from chex import ArrayTree, Array
from typing import TypeVar

from jax._src.random import KeyArray as PRNGKey

tf.config.experimental.set_visible_devices([], "GPU")

DEFAULT_PARALLELISM = 12
MIN_NOTE = 21
MAX_NOTE = 108
NUM_NOTES = MAX_NOTE - MIN_NOTE + 1


def in_mem_jax_dataset(data, batch_size, post_process_fn=None):
  data = jax.tree_util.tree_map(lambda x: jnp.array(x), data)

  leading_shapes, _ = jax.tree_util.tree_flatten(
          jax.tree_util.tree_map(lambda x: x.shape[0], data))
  N = leading_shapes[0]
  for s in leading_shapes[1:]:
    assert s == N, "All data must have same leading dimension."

  def next_fn(key: PRNGKey):
    inds = jax.random.randint(key, [batch_size], 0, N)
    batch = jax.tree_util.tree_map(lambda x: x[inds], data)
    if post_process_fn is not None:
      return jax.vmap(post_process_fn)(batch)
    else:
      return batch

  return next_fn


def load_forex_dataset(data_dir: str) -> Array:
  data_path = os.path.join(data_dir, 'forex_data.tsv')
  data = onp.genfromtxt(fname=data_path, delimiter="\t", skip_header=1)
  data = onp.log(data[1:, :]) - onp.log(data[:-1, :])
  return data


def create_forex_dataset(data_dir: str) -> Array:
  return load_forex_dataset(data_dir)[:119, :]


def create_forex_eval_dataset(data_dir: str) -> Array:
  return load_forex_dataset(data_dir)[119:, :]


def create_synthetic_svm_dataset(
        key: PRNGKey,
        data_dim: int,
        num_timesteps: int,
        init_scale: float = 1.):
  key, sk1, sk2 = jax.random.split(key, num=3)
  gen_model = svm.DiagCovSVM(
          sk1,
          data_dim,
          init_scale=init_scale)
  _, data = gen_model.sample_trajectory(sk2, num_timesteps)
  return data, gen_model


def create_synthetic_hh_dataset(
        key: PRNGKey,
        num_datapoints: int,
        data_dim: int,
        num_timesteps: int,
        obs_subsample: int,
        dt: float,
        ode_solver,
        noise_scale: float):

  def valid_seq(s):
    nobig = jnp.logical_not(jnp.nanmax(jnp.abs(s)) > 20.0)  # This value is normalized by `VS`.
    noinf = jnp.logical_not(jnp.any(jnp.isinf(s)))
    nonan = jnp.logical_not(jnp.any(jnp.isnan(s[::obs_subsample])))
    return nobig and noinf and nonan

  key, subkey = jax.random.split(key)

  gen_model = hh.HodgkinHuxley(
    subkey,
    data_dim,
    dt=dt,
    ode_solver=ode_solver,
    y_scale=noise_scale,
    obs_subsample=obs_subsample)

  sample = jax.jit(gen_model.sample_trajectory, static_argnums=1)

  # We will generate a sequence, test if it is valid, and append
  # until we have the desired number of sequences.
  valid_obs = []
  while len(valid_obs) < num_datapoints:
    key, subkey = jax.random.split(key)
    _, obs = sample(subkey, num_timesteps)
    if valid_seq(obs):
      valid_obs.append(obs)

  obs = jnp.asarray(valid_obs)
  return obs, gen_model


DataType = TypeVar('DataType', bound=ArrayTree)


def create_synthetic_diffusion_dataset(
        key: PRNGKey,
        seq_len: int,
        num_datapoints: int,
        batch_size: int,
        drift: float = 0.):
  model = diffusion.GaussianDiffusion(seq_len, drift=drift, train=False)
  zs, xs = jax.vmap(model.sample_trajectory)(jax.random.split(key, num=num_datapoints))

  # Compute empirical "true" drift (NOTE: this is subtle and maybe should be placed in
  # GaussianDiffusion class, depends on whether there's a drift in z1 and in emissions, etc.)
  true_drift = jnp.mean(xs) / (seq_len + 1)

  ds = in_mem_jax_dataset(xs, batch_size)

  return (zs, xs), ds, model, true_drift
