import jax
import jax.numpy as jnp
from jax._src.random import KeyArray as PRNGKey
from typing import Tuple, Callable, TypeVar, Optional, Union
from chex import Array, ArrayTree, Scalar

import smc
from smc import ResamplingCriterion, ResamplingFn, TransitionFn
from smc import ess_criterion, multinomial_resampling, never_resample_criterion
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

DEFAULT_RESAMPLING = ess_criterion


T = TypeVar('T', bound=ArrayTree)


def _ensure_leading_dim(xs: T, dim: int, tile: bool) -> T:
  """Ensure ArrayTree xs has leading dim `dim`, either by tiling or asserting."""
  if tile:
    def tile_fn(x: Array) -> Array:
      ndim = len(x.shape)
      reps = [dim] + [1] * ndim
      return jnp.tile(x[jnp.newaxis, ...], reps)

    result = jax.tree_util.tree_map(tile_fn, xs)
  else:

    def assert_fn(x):
      assert x.shape[0] == dim, "Initial states do not have leading dim num_particles"

    jax.tree_util.tree_map(assert_fn, xs)
    result = xs

  return result


StateType = TypeVar('StateType', bound=ArrayTree)
ObsType = TypeVar('ObsType', bound=ArrayTree)


def fivo(
    key: PRNGKey,
    propose_and_weight: TransitionFn[StateType, ObsType],
    initial_state: StateType,
    num_steps: int,
    max_num_steps: int,
    num_particles: int,
    observations: Optional[ObsType] = None,
    resampling_criterion: ResamplingCriterion = DEFAULT_RESAMPLING,
    resampling_fn: ResamplingFn = multinomial_resampling,
    resampling_gradient_mode: str = 'none',
    tile_initial_state=True) -> Tuple[StateType, Array, Scalar, Array, Array]:
  """Computes the FIVO lower bound on the log marginal probability.

  For more details see

  "Filtering Variational Objectives" by Maddison et al.
  https://arxiv.org/abs/1705.09279

  "Variational Sequential Monte Carlo" by Naessseth et al.
  https://arxiv.org/abs/1705.11140

  and "Auto-Encoding Sequential Monte Carlo" by Le et al.
  https://arxiv.org/abs/1705.10306.

  Args:
    key: A JAX PRNG key.
    propose_and_weight: A function which draws a sample from the proposal
      distribution and computes the incremental log weights.
      The function should accept a JAX PRNG key, the set of latents from the
      previous timestep, and the current timestep. The function should return
      a new latent sample from the proposal and the incremental log weight of the latent.
    initial_state: The initial latent state of the model, a PyTree.
    num_steps: The number of steps to run the model for, a scalar int. Must be
      less than or equal to max_num_steps.
    max_num_steps: An upper bound on the number of steps to run for. This
      argument determines an intermediate shape, so must be static when jitting.
    num_particles: The number of particles to use in the particle filter, a
      scalar int.
    observations: An ArrayTree of observations to pass to the model. Each leaf of the tree
      should have leading dimension [max_num_timesteps].
    resampling_criterion: Function that determines when the underlying particle
      filter should resample, see smc() for more information.
    resampling_fn: The resampling function to use for the particle
      filter. See smc() for more information.
    resampling_gradient_mode: str
      'none': drop score function gradient term (as in FIVO and VSMC)
      'score_fn': include score fn gradient term
      'score_fn_rb': Rao-Blackwellized score fn gradients
    tile_initial_state: A boolean indicating whether the initial state should be replicated
      `num_particles` times. If false, the leaves of initial_state must all have leading dim
      `num_particles`.
  Returns:
    states: The latent states sampled during the SMC sweep, an object of StateType with
      all leaf nodes having leading dimension max_num_timesteps.
    log_weights: The log weights of the SMC samples at each timestep, an Array of
      shape [max_num_timesteps, num_particles].
    Z_hat: FIVO's estimate of the log marginal probability of the
      observations, a scalar float.
    ancestors: The ancestor indices from resampling, an Array of dimension
      [max_num_timesteps, num_particles].
    resampled: A boolean array of shape [max_num_timesteps] indicating the steps on which
      resampling occurred.
  """
  initial_states = _ensure_leading_dim(initial_state, num_particles, tile_initial_state)

  states, log_weights, ancestors, log_Z_hat, resampled = smc.smc(
      key,
      initial_states,
      propose_and_weight,
      num_steps,
      max_num_steps,
      observations=observations,
      num_particles=num_particles,
      resampling_criterion=resampling_criterion,
      resampling_fn=resampling_fn,
      resampling_gradient_mode=resampling_gradient_mode)

  return states, log_weights, log_Z_hat, ancestors, resampled


TwistedProposeAndWeightWithObs = Callable[[PRNGKey, StateType, ObsType, int],
                                Tuple[StateType, Scalar, Scalar]]

TwistedProposeAndWeightNoObs = Callable[[PRNGKey, StateType, int],
                                Tuple[StateType, Scalar, Scalar]]

TwistedProposeAndWeight = Union[TwistedProposeAndWeightNoObs[StateType],
        TwistedProposeAndWeightWithObs[StateType, ObsType]]


def sixo(
    key: PRNGKey,
    propose_and_weight: TwistedProposeAndWeight[StateType, ObsType],
    initial_state: StateType,
    num_steps: int,
    max_num_steps: int,
    num_particles: int,
    observations: Optional[ObsType] = None,
    resampling_criterion: ResamplingCriterion = DEFAULT_RESAMPLING,
    resampling_fn: ResamplingFn[StateType] = multinomial_resampling,
    resampling_gradient_mode: str = 'none',
    tile_initial_state=True) -> Tuple[StateType, Array, Scalar, Array, Array]:
  """Computes the SIXO lower bound on the log marginal probability.

  Args:
    key: A JAX PRNG key.
    propose_and_weight: A function which draws a sample from the proposal
      distribution and computes the incremental log weights.
      The function should accept a JAX PRNG key, the set of latents from the
      previous timestep prev_zs, and the current timestep. The function should return
      a sample of z from the proposal and the incremental log weight of z.
    initial_state: The initial latent state of the model, a pytree.
    num_steps: The number of steps to run the model for, a scalar int. Must be
      less than or equal to max_num_steps.
    max_num_steps: An upper bound on the number of steps to run for. This
      argument determines an intermediate shape, so must be static when jitting.
    num_particles: The number of particles to use in the particle filter, a
      scalar int.
    observations: An ArrayTree of observations to pass to the model. Each leaf of the tree
      should have leading dimension [max_num_timesteps].
    resampling_criterion: Function that determines when the underlying particle
      filter should resample, see smc() for more information.
    resampling_fn: The resampling function to use for the particle
      filter. See smc() for more information.
    resampling_gradient_mode: str
      'none': drop score function gradient term (as in FIVO and VSMC)
      'score_fn': include score fn gradient term
      'score_fn_rb': Rao-Blackwellized score fn gradients
      'scibior': Scibior gradient trick
    tile_initial_state: A boolean indicating whether the initial state should be replicated
      `num_particles` times. If false, the leaves of initial_state must all have leading dim
      `num_particles`.
  Returns:
    states: The latent states sampled during the SMC sweep, an object of StateType with
      all leaf nodes having leading dimension max_num_timesteps.
    log_weights: The log weights of the SMC samples at each timestep, an Array of
      shape [max_num_timesteps, num_particles].
    Z_hat: SIXO's estimate of the log marginal probability of the
      observations, a scalar float.
    ancestors: The ancestor indices from resampling, an Array of dimension
      [max_num_timesteps, num_particles].
    resampled: A boolean array of shape [max_num_timesteps] indicating the steps on which
      resampling occurred.
  """
  if observations is not None:
    def p_and_w(
            key: PRNGKey,
            prev_state: ArrayTree,
            cur_obs: ArrayTree,
            t: int) -> Tuple[Tuple[ArrayTree, Scalar], Scalar]:
      prev_state, prev_log_r = prev_state
      new_state, new_log_weight, new_log_r = propose_and_weight(key, prev_state, cur_obs, t)
      # Starting at last time step (i.e. step with index ``num_steps-1``), new_log_r is 0.
      # This line also makes new_log_r and prev_log_r = 0 for all steps with index t >= T
      new_log_r = jnp.where(t < num_steps - 1, new_log_r, 0.)
      return (new_state, new_log_r), new_log_weight + new_log_r - prev_log_r
  else:
    def p_and_w(
            key: PRNGKey,
            prev_state: ArrayTree,
            t: int) -> Tuple[Tuple[ArrayTree, Scalar], Scalar]:
      prev_state, prev_log_r = prev_state
      new_state, new_log_weight, new_log_r = propose_and_weight(key, prev_state, t)
      # Starting at last time step (i.e. step with index ``num_steps-1``), new_log_r is 0.
      # This line also makes new_log_r and prev_log_r = 0 for all steps with index t >= T
      new_log_r = jnp.where(t < num_steps - 1, new_log_r, 0.)
      return (new_state, new_log_r), new_log_weight + new_log_r - prev_log_r

  if tile_initial_state:
    initial_states = _ensure_leading_dim(
            (initial_state, jnp.array(0.)), num_particles, True)
  else:
    initial_states = _ensure_leading_dim(
            (initial_state, jnp.zeros(num_particles)), num_particles, False)

  (states, _), log_weights, ancestors, log_Z_hat, resampled = smc.smc(
      key,
      initial_states,
      p_and_w,
      num_steps,
      max_num_steps,
      observations=observations,
      num_particles=num_particles,
      resampling_criterion=resampling_criterion,
      resampling_fn=resampling_fn,
      resampling_gradient_mode=resampling_gradient_mode)

  return states, log_weights, log_Z_hat, ancestors, resampled


def iwae(
    key: PRNGKey,
    propose_and_weight: TransitionFn[StateType, ObsType],
    initial_state: StateType,
    num_steps: int,
    max_num_steps: int,
    num_particles: int,
    observations: Optional[ObsType] = None,
    tile_initial_state=True) -> Tuple[StateType, Array, Scalar, Array, Array]:
  """Computes the IWAE lower bound on the log marginal probability.

  Args:
    key: A JAX PRNG key.
    propose_and_weight: A function which draws a sample from the proposal
      distribution and computes the incremental log weights.
      The function should accept a JAX PRNG key, the set of latents from the
      previous timestep prev_zs, and the current timestep. The function should return
      a sample of z from the proposal and the incremental log weight of z.
    initial_state: The initial latent state of the model, a pytree.
    num_steps: The number of steps to run the model for, a scalar int. Must be
      less than or equal to max_num_steps.
    max_num_steps: An upper bound on the number of steps to run for. This
      argument determines an intermediate shape, so must be static when jitting.
    num_particles: The number of particles to use in the particle filter, a
      scalar int.
    observations: An ArrayTree of observations to pass to the model. Each leaf of the tree
      should have leading dimension [max_num_timesteps].
    tile_initial_state: A boolean indicating whether the initial state should be replicated
      `num_particles` times. If false, the leaves of initial_state must all have leading dim
      `num_particles`.
  Returns:
    states: The latent states sampled during the sweep, an object of StateType with
      all leaf nodes having leading dimension max_num_timesteps.
    log_weights: The log weights of the samples at each timestep, an Array of
      shape [max_num_timesteps, num_particles].
    Z_hat: IWAE's estimate of the log marginal probability of the
      observations, a scalar float.
    ancestors: Returned for convenience, the ancestor indices are not relevant for IWAE.
    resampled: Returned for convenience, resampling is relevant for IWAE.
  """
  return fivo(
          key,
          propose_and_weight,
          initial_state,
          num_steps,
          max_num_steps,
          num_particles,
          observations=observations,
          resampling_criterion=never_resample_criterion,
          resampling_gradient_mode='none',
          tile_initial_state=tile_initial_state)


def elbo(
    key: PRNGKey,
    propose_and_weight: TransitionFn[StateType, ObsType],
    initial_state: StateType,
    num_steps: int,
    max_num_steps: int,
    observations: Optional[ObsType] = None,
    tile_initial_state=True) -> Tuple[StateType, Array, Scalar, Array, Array]:
  """Computes the ELBO lower bound on the log marginal probability.

  Args:
    key: A JAX PRNG key.
    propose_and_weight: A function which draws a sample from the proposal
      distribution and computes the incremental log weights.
      The function should accept a JAX PRNG key, the set of latents from the
      previous timestep prev_zs, and the current timestep. The function should return
      a sample of z from the proposal and the incremental log weight of z.
    initial_state: The initial latent state of the model, a pytree.
    num_steps: The number of steps to run the model for, a scalar int. Must be
      less than or equal to max_num_steps.
    max_num_steps: An upper bound on the number of steps to run for. This
      argument determines an intermediate shape, so must be static when jitting.
    tile_initial_state: A boolean indicating whether the initial state should be replicated
      `num_particles` times. If false, the leaves of initial_state must all have leading dim
      `num_particles`.
  Returns:
    states: The latent states sampled during the sweep, an object of StateType with
      all leaf nodes having leading dimension max_num_timesteps.
    log_weights: The log weights of the samples at each timestep, an Array of
      shape [max_num_timesteps, 1].
    Z_hat: ELBO's estimate of the log marginal probability of the
      observations, a scalar float.
    ancestors: Returned for convenience, the ancestor indices are not relevant for ELBO.
    resampled: Returned for convenience, resampling is not relevant for ELBO.
  """
  return iwae(
          key,
          propose_and_weight,
          initial_state,
          num_steps,
          max_num_steps,
          1,
          observations=observations,
          tile_initial_state=tile_initial_state)
