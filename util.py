"""Utility functions."""
import itertools
import io
import jax
import jax.numpy as jnp
import numpy as onp
from chex import Array, Shape, Scalar
from typing import Callable, Sequence, Tuple
import equinox as eqx
import PIL.Image as pil_image
import wandb
import optax


def L_from_raw(raw_L, diag_min=1e-6):
  """
  Convert an unconstrained matrix (log-terms on the diagonal + lower triangular)
  into a lower triangular matrix, for which L'L is the full precision matrix.
  :param raw_L:
  :param diag_min:
  :return:
  """
  scale_diag = jnp.maximum(jnp.exp(jnp.diag(raw_L)), diag_min)
  L = jnp.diag(scale_diag) + jnp.tril(raw_L, -1)
  return L


def relative_diff(a, b):
  """
  Compute the distance difference between a and b as a multiple of b.
  For scalar a and b, negative diffs imply that a is smaller than b.
  :param a:
  :param b:
  :return:
  """
  a = a.flatten()
  b = b.flatten()
  if len(a) == 1:
    sign = -1 if a < b else 1
  else:
    sign = 1
  return sign * jnp.linalg.norm(a - b) / jnp.linalg.norm(b)


def abs_and_rel_diff(a, b):
  """
  Compute the distance between a and b as a multiple of b.
  For scalar a and b, negative diffs imply that a is smaller than b.
  :param a:
  :param b:
  :return:
  """
  if hasattr(a, "flatten"):
    a = a.flatten()

  if hasattr(b, "flatten"):
    b = b.flatten()

  if len(a) == 1:
    sign = -1 if a < b else 1
  else:
    sign = 1
  abs_diff = jnp.linalg.norm(a - b)
  rel_diff = sign * jnp.linalg.norm(a - b) / jnp.linalg.norm(b)
  return abs_diff, rel_diff


def rolling_window(x: Array, x_shape: Shape, window_len: int):
  shape = list(x_shape)
  padding = [[0, 0]] * len(shape)
  padding[0][1] = window_len - 1
  x = jnp.pad(x, padding)

  def scan_fn(unused_carry, t):
    start_inds = [0] * len(shape)
    start_inds[0] = t
    slice_sizes = shape
    slice_sizes[0] = window_len
    return None, jax.lax.dynamic_slice(x, start_inds, slice_sizes)

  _, outs = jax.lax.scan(scan_fn, None, jnp.arange(x_shape[0]))
  return outs


def get_pytree_leading_dim(x):
  flat_x, _ = jax.tree_util.tree_flatten(x)
  leading_dims = jax.tree_util.tree_map(lambda a: a.shape[0], flat_x)
  assert all(d == leading_dims[0] for d in leading_dims), leading_dims
  return leading_dims[0]


def integrate_hg(deg: int, f: Callable[[Scalar], Scalar], mean: float, scale: float):
  points, weights = onp.polynomial.hermite.hermgauss(deg)
  eval_points = jnp.sqrt(2) * scale * points + mean
  evals = jax.vmap(f)(eval_points)
  return (1 / jnp.sqrt(jnp.pi)) * jnp.sum(weights * evals)


def log_integrate_hg(deg: int, log_f: Callable[[Scalar], Scalar], mean: float, scale: float):
  points, weights = onp.polynomial.hermite.hermgauss(deg)
  eval_points = jnp.sqrt(2) * scale * points + mean
  log_evals = jax.vmap(log_f)(eval_points)
  log_weights = jnp.log(weights)
  return jax.nn.logsumexp(log_weights + log_evals - 0.5 * jnp.log(jnp.pi))


def log_integrate_multid_hg(
        deg: int, log_f: Callable[[Array], Array], loc: Array, scale_L: Array):
  n = loc.shape[0]
  xs_1d, ws_1d = onp.polynomial.hermite.hermgauss(deg)
  # quadrature points and weights of shape [deg^n, deg]
  xs_nd = onp.array(list(itertools.product(xs_1d, repeat=n)))
  ws_nd = onp.array(list(itertools.product(ws_1d, repeat=n)))
  # Take the product over the second axis of the weights to
  # compute the per-point weight. New shape [deg^n].
  log_ws_nd = onp.sum(onp.log(ws_nd), axis=1)
  log_g = lambda x: log_f(jnp.sqrt(2.) * jnp.dot(scale_L, x.T).T + loc)
  log_gs = jax.vmap(log_g)(xs_nd)
  return jax.nn.logsumexp(log_ws_nd + log_gs - (n / 2.) * jnp.log(jnp.pi))


def replace_subtree(where, tree, replace_val):
  """Replaces all leaves of a subtree with a single value.

  Args:
    where: A function which accepts tree and returns the node or a sequence of nodes to replace.
    tree: The tree to change.
    replace_val: The value to replace all subtree leaves with.
  """
  subtree = jax.tree_util.tree_map(lambda _: replace_val, where(tree))
  result = eqx.tree_at(
      lambda x: jax.tree_util.tree_leaves(where(x)),
      tree,
      jax.tree_util.tree_leaves(subtree))
  return result


def wandb_im_from_fig(figure):
  buf = io.BytesIO()
  figure.savefig(buf, bbox_inches='tight')
  buf.seek(0)
  return wandb.Image(pil_image.open(buf))


def linear_ramp_schedule(zero_steps: int, ramp_steps: int, step: int) -> float:
  zero_factor = jnp.where(step <= zero_steps, 0., 1.)
  ramp_factor = jnp.clip((step - zero_steps) / float(ramp_steps), a_min=0., a_max=1.)
  return ramp_factor * zero_factor


def make_clipped_adam_optimizer(lr: float, clip: float):
  return optax.chain(
    optax.clip(clip),  # Clip by the gradient by the global norm.
    optax.scale_by_adam(),  # Use the updates from sgd.
    optax.scale(lr),  # Use the learning rate from the scheduler.
    # Scale updates by -1 since optax.apply_updates is additive and we want to descend on the loss.
    optax.scale(-1.0))


def make_masked_optimizer(
        optimizer: optax.GradientTransformation,
        query_fns: Sequence[Tuple[Callable, bool]],
        mask_default: bool = True):

  def loss_mask(updates):
    loss_mask = jax.tree_util.tree_map(lambda _: mask_default, updates)
    for query_fn, mask_val in query_fns:
      loss_mask = replace_subtree(query_fn, loss_mask, mask_val)
    return loss_mask

  new_opt = optax.multi_transform({
      True: optimizer,
      False: optax.set_to_zero()}, loss_mask)
  return new_opt


def chain_cov(diag):
  # rows are all same value
  tiled_diag = jnp.tile(diag[:, jnp.newaxis], [1, diag.shape[0]])
  upper_L = tiled_diag.T[jnp.tril_indices_from(tiled_diag, k=-1)]
  out = tiled_diag.at[jnp.tril_indices_from(tiled_diag, k=-1)].set(upper_L)
  return out


def print_args(cfg):
  print("Args")
  for arg in vars(cfg):
    print(' {}={}'.format(arg, getattr(cfg, arg)))


def get_wandb_id(entity, project, run_name):
  api = wandb.Api()

  try:
    runs = api.runs(
      path=f"{entity}/{project}",
      filters={"display_name": run_name}
    )
    assert len(runs) <= 1, "More than one runs found for {entity}/{project}/{run_name}"
    if len(runs) == 0:
      return None
    else:
      return runs[0].id
  except ValueError:
    return None


def pos_enc(length, d, base=50, period=None, flip=False):
  """Transformer position encoding."""
  if period is None:
    period = length
  if flip:
    ts = period - (jnp.arange(length) % period) - 1
  else:
    ts = jnp.arange(length) % period
  return jax.vmap(_pos_enc_t, in_axes=(0, None, None))(ts, d, base)


def _pos_enc_t(t, d, b):
  denom = jnp.power(b, (2. * jnp.arange(d)) / d)
  return jnp.concatenate([jnp.sin(t / denom), jnp.cos(t / denom)], axis=0)


def gaussian_prod(mu_a, log_s_a, mu_b, log_s_b):
  """Computes the product of two gaussian densities."""
  log_var_sum = jnp.logaddexp(2 * log_s_a, 2 * log_s_b)
  log_s_prod = log_s_a + log_s_b - 0.5 * log_var_sum
  mu_prod = (mu_a * jnp.exp(2 * log_s_b) + mu_b * jnp.exp(2 * log_s_a)) / jnp.exp(log_var_sum)
  return mu_prod, log_s_prod
