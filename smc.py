from typing import Callable, Tuple, Optional, Union, TypeVar, Protocol, runtime_checkable

import jax
from jax import vmap
import jax.numpy as jnp
import jax.scipy as jscipy
from jax._src.random import KeyArray as PRNGKey

from chex import Array, ArrayTree, Scalar
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

StateType = TypeVar('StateType', bound=ArrayTree)
ObsType_contra = TypeVar('ObsType_contra', bound=ArrayTree, contravariant=True)

ResamplingCriterion = Callable[[Array, int], Array]
ResamplingFn = Callable[[PRNGKey, Array, StateType], Tuple[StateType, Array]]


@runtime_checkable
class TransFnNoObs(Protocol[StateType]):

  def __call__(self,
          __key: PRNGKey,
          __prev_state: StateType,
          __t: int) -> Tuple[StateType, Scalar]:
      ...


@runtime_checkable
class TransFnWithObs(Protocol[StateType, ObsType_contra]):

  def __call__(self,
          __key: PRNGKey,
          __prev_state: StateType,
          __cur_obs: ObsType_contra,
          __t: int) -> Tuple[StateType, Scalar]:
      ...


TransitionFn = Union[TransFnWithObs[StateType, ObsType_contra], TransFnNoObs[StateType]]


def ess_criterion(log_weights: Array, unused_t: int) -> Array:
  """A criterion that resamples based on effective sample size."""
  del unused_t
  num_particles = log_weights.shape[0]
  ess_num = 2 * jscipy.special.logsumexp(log_weights)
  ess_denom = jscipy.special.logsumexp(2 * log_weights)
  log_ess = ess_num - ess_denom
  return log_ess <= jnp.log(num_particles / 2.0)


def never_resample_criterion(log_weights: Array, t: int) -> Array:
  """A criterion that never resamples."""
  del log_weights
  del t
  return jnp.array(False)


def always_resample_criterion(log_weights: Array, t: int) -> Array:
  """A criterion that always resamples."""
  del log_weights
  del t
  return jnp.array(True)


def multinomial_resampling(
    key: PRNGKey, log_weights: Array, states: StateType) -> Tuple[StateType, Array]:
  """Resample states with multinomial resampling.

  Args:
    key: A JAX PRNG key.
    log_weights: A [num_particles] ndarray, the log weights for each particle.
    states: A pytree of [num_particles, ...] ndarrays that
      will be resampled.
  Returns:
    resampled_states: A pytree of [num_particles, ...] ndarrays resampled via
      multinomial sampling.
    parents: A [num_particles] array containing index of parent of each state
  """
  num_particles = log_weights.shape[0]
  cat = tfd.Categorical(logits=log_weights)
  parents = cat.sample(sample_shape=(num_particles,), seed=key)
  assert isinstance(parents, jnp.ndarray)
  return (jax.tree_util.tree_map(lambda item: item[parents], states), parents)


def stratified_resampling(
        key: PRNGKey, log_weights: Array, states: StateType) -> Tuple[StateType, Array]:
  """Resample states with stratified resampling.
  Args:
    key: A JAX PRNG key.
    log_weights: A [num_particles] ndarray, the log weights for each particle.
    states: A pytree of [num_particles, ...] ndarrays that
      will be resampled.
  Returns:
    resampled_states: A pytree of [num_particles, ...] ndarrays resampled via
      multinomial sampling.
    parents: A [num_particles] array containing index of parent of each state
  """
  num_particles = log_weights.shape[0]
  us = jax.random.uniform(key, shape=[num_particles])
  us = (jnp.arange(num_particles) + us) / num_particles
  norm_log_weights = log_weights - jax.nn.logsumexp(log_weights)
  bins = jnp.cumsum(jnp.exp(norm_log_weights))
  inds = jax.lax.stop_gradient(jnp.digitize(us, bins))
  return (jax.tree_util.tree_map(lambda x: x[inds], states), inds)


def resampling_loss(
        num_steps: int,
        log_weights: Array,
        log_Z_hat: Array,
        resampled: Array,
        ancestors: Array,
        resampling_gradient_mode: str) -> Scalar:
  """Pseudo-loss to incorporate score function resampling gradients
  NOTE: this method leads to unbiased gradients only under a fixed resampling schedule,
  otherwise gradients are still biased

  Args:
    log_weights: Array of shape [max_num_steps, num_particles]
      containing the log weights at each timestep of the particle filter
    log_Z_hat: Estimate of log normalizing constant from SMC run
    resampled: An boolean array of shape [max_num_steps] indicating
      which timesteps the filter resampled on
    ancestors: Array of shape [max_num_steps, num_particles], where
      index of parent of states[step, n] is ancestors[step, n]. Entries are
      in {0, 1, ..., num_particles - 1}
    resampling_gradient_mode: str
      'score_fn': include score fn gradient term
      'score_fn_rb': Rao-Blackwellized score fn gradients
  Returns:
    loss: pseudo-loss term whose gradients will be score function gradients
  """
  assert resampling_gradient_mode in ('score_fn', 'score_fn_rb')

  _, num_particles = log_weights.shape
  log_p_hats = jscipy.special.logsumexp(log_weights, axis=1) - jnp.log(num_particles)
  # Normalized log weights (still [max_num_steps, num_particles])
  normalizers = jscipy.special.logsumexp(log_weights, axis=1, keepdims=True)
  log_weights_normalized = log_weights - normalizers
  # Compute the log probability of each resampling operation, [max_num_steps, num_particles]
  resampling_log_probs = jax.vmap(lambda x, i: x[i])(log_weights_normalized, ancestors)
  # Sum across the num_particles dimension, resulting in [max_num_steps]
  resampling_log_probs = jnp.sum(resampling_log_probs, axis=1)
  # Compute resampling indicators ensuring the last timestep is 0.
  resampled_without_last_step = resampled.at[num_steps].set(0.)
  if resampling_gradient_mode == 'score_fn':
    # Compute the sum of the resampling log probs.
    # Does not include the last step (if there was a resampling event) because resampling
    # on the last step does not affect the marginal likelihood.
    sum_log_probs = jnp.sum(resampled_without_last_step * resampling_log_probs)
    # Compute the 'rewards' which are the sum of log_p_hats on resampling steps
    # after the first resampling. The log_p_hat computed on the first resampling step
    # is not included because resampling does not affect that one.
    first_resampling_step = jnp.argmax(resampled)
    resampled_without_first = resampled.at[first_resampling_step].set(0.)
    reward = jnp.sum(log_p_hats * resampled_without_first)
    return sum_log_probs * jax.lax.stop_gradient(reward)
  else:
    # Compute the resampling log probs.
    # Does not include the last step (if there was a resampling event) because resampling
    # on the last step does not affect the marginal likelihood.
    log_probs = resampling_log_probs * resampled_without_last_step
    # Compute the returns. First, mask out the log_p_hats on non-resampling steps.
    rewards = log_p_hats * resampled
    # Compute the sum of future log_p_hats on each resampling step.
    returns = resampled * (log_Z_hat - jnp.cumsum(rewards))
    return jnp.sum(log_probs * jax.lax.stop_gradient(returns))


def smc(
    key: PRNGKey,
    initial_states: StateType,
    transition_fn: TransitionFn[StateType, ObsType_contra],
    num_steps: int,
    max_num_steps: int,
    observations: Optional[ObsType_contra] = None,
    num_particles: int = 1,
    resampling_criterion: ResamplingCriterion = ess_criterion,
    resampling_fn: ResamplingFn[StateType] = multinomial_resampling,
    resampling_gradient_mode: str = 'none') -> Tuple[StateType, Array, Array, Scalar, Array]:
  """Run sequential Monte Carlo (SMC).

  Note that this implementation is not meant to work with non-Markovian
  transition functions. As such, the transition function must only depend
  on the previous state. States further into the past are not provided to
  the transition function.

  Furthermore, the state trajectories returned are *not* the particles that
  make up SMC's posterior approximation. Computing SMC's particle
  approximation to the posterior requires re-resampling all previous states
  every time a resampling operation occurs. We opt to only resample the
  current state for performance reasons. This is possible because we assume
  the transition function is Markov and so only depends on the previous state.

  If you wish to transform the states returned from this function into SMC's
  weighted particle approximation to the posterior, use either unwind_states
  or smc_posterior_dist directly.

  Args:
    key: A JAX PRNG key.
    initial_states: The intial states of the particles, a pytree with
      leaf nodes of shape [num_particles, ...].
    transition_fn: A callable that propogates a single particle one step.
      Must accept as arguments a JAX PRNG key, a pytree containing
      a single particle state, and the current timestep as an int. Must return
      the particle state one timestep in the future and the incremental
      log weight (possibly unnormalized) of the particle.
    num_steps: A scalar int, the number of steps to run SMC for.
    max_num_steps: A Python int, an upper bound on the number of steps to run for.
      This argument determines an intermediate shape, so must be static when jitting.
    num_particles: A scalar int, the number of particles.
    resampling_criterion: The resampling criterion. Must accept the
      [num_particles] vector of log weights and current timestep and return a
      boolean indicating whether resampling should be performed. See ess_criterion
      for an example. When resampling_criterion is never_resample_criterion,
      resampling_fn is ignored and never called.
    resampling_fn: A callable that performs the resampling operation. Must
      accept as arguments a JAX PRNG key, the [num_particles] log weights
      vector, and a pytree of particle states, and return a pytree of the
      resampled particle states. See multinomial_resampling for an example.
    resampling_gradient_mode: str
      'none': drop score function gradient term (as in FIVO and VSMC)
      'score_fn': include score fn gradient term
      'score_fn_rb': Rao-Blackwellized score fn gradients
  Returns:
    states: A pytree with array leaves of shape
      [max_num_steps, num_particles, ...] representing the states of the
      particles at each timestep. See above for disclaimer -- these ARE NOT
      the atomic support of SMC's posterior approximation.
    log_weights: An array of shape [max_num_steps, num_particles]
      containing the log weights at each timestep of the particle filter.
    ancestors: Array of shape [max_num_steps, num_particles], where
      index of parent of states[step, n] is ancestors[step, n]. Entries are
      in {0, 1, ..., num_particles - 1}
    log_Z_hat: An estimate of the log normalizing constant, a scalar float.
    resampled: An boolean array of shape [max_num_steps] indicating
      which timesteps the algorithm resampled on.
  """
  assert resampling_gradient_mode in ('none', 'score_fn', 'score_fn_rb')

  def resample(args) -> Tuple[StateType, Array, Array]:
    key, log_weights, states = args
    states, inds = resampling_fn(key, log_weights, states)
    return states, inds, jnp.zeros_like(log_weights)

  def dont_resample(args) -> Tuple[StateType, Array, Array]:
    _, log_weights, states = args
    return states, jnp.arange(num_particles), log_weights

  def smc_step(carry, state_slice):
    key, states, log_ws = carry
    key, sk1, sk2 = jax.random.split(key, num=3)
    t, observation = state_slice

    # Propagate the particle states
    if observations is not None:
      assert isinstance(transition_fn, TransFnWithObs)
      new_states, incr_log_ws = vmap(transition_fn, (0, 0, None, None))(
          jax.random.split(sk1, num=num_particles), states, observation, t)
    else:
      assert isinstance(transition_fn, TransFnNoObs)
      new_states, incr_log_ws = vmap(transition_fn, (0, 0, None))(
          jax.random.split(sk1, num=num_particles), states, t)

    # Update the log weights.
    log_ws += incr_log_ws

    # Resample the particles if resampling_criterion returns True and we haven't
    # exceeded the supplied number of steps.
    should_resample = jax.lax.stop_gradient(
            jnp.logical_and(resampling_criterion(log_ws, t), t < num_steps))

    resampled_states, parents, resampled_log_ws = jax.lax.cond(
        should_resample,
        resample,
        dont_resample,
        (sk2, log_ws, new_states)
    )

    return ((key, resampled_states, resampled_log_ws),
            (new_states, log_ws, parents, should_resample))

  _, (states, log_weights, ancestors, resampled) = jax.lax.scan(
      smc_step,
      (key, initial_states, jnp.zeros([num_particles])),
      (jnp.arange(max_num_steps), observations))

  # Average along particle dimension
  log_p_hats = jscipy.special.logsumexp(log_weights, axis=1) - jnp.log(num_particles)
  # Sum in time dimension on resampling steps.
  # Note that this does not include any steps past num_steps because
  # the resampling criterion doesn't allow resampling past num_steps steps.
  log_Z_hat = jnp.sum(log_p_hats * resampled)
  # If we didn't resample on the last timestep, add in the missing log_p_hat
  log_Z_hat += jnp.where(resampled[num_steps - 1], 0., log_p_hats[num_steps - 1])

  if resampling_gradient_mode in ('score_fn', 'score_fn_rb'):
    resampling_term = resampling_loss(
      num_steps, log_weights, log_Z_hat, resampled, ancestors, resampling_gradient_mode)
    log_Z_hat += resampling_term - jax.lax.stop_gradient(resampling_term)

  return states, log_weights, ancestors, log_Z_hat, resampled


def unwind_states(
        states: StateType,
        ancestor_inds: Array,
        resampled: Array,
        num_timesteps: int) -> StateType:
  """Computes the resampled SMC states.

  Args:
    states: A PyTree with leaves of leading dimensions [max_num_timesteps, num_particles].
    ancestor_inds: The ancestor indices returned by smc, an Array of shape
      [max_num_timesteps, num_particles].
    resampled: An Array of shape [max_num_timesteps] indicating if smc resampled on each timestep.
    num_timesteps: The number of timesteps smc ran for, can be less than max_num_timesteps.
  Returns:
    A Pytree with the same structure and leaf node shape of states containing the resampled
      states.
  """

  def scan_fn(inds, inputs):
    ancestors, resampled = inputs
    new_inds = jax.lax.cond(
            resampled == 0,
            lambda i: i,
            lambda i: ancestors[i],
            inds)
    return new_inds, new_inds

  init_inds = jnp.arange(ancestor_inds.shape[1])
  _, ins = jax.lax.scan(
      scan_fn,
      init_inds,
      (ancestor_inds[:-1], resampled[:-1]),
      reverse=True)
  ins = jnp.concatenate([ins, init_inds[jnp.newaxis]])

  def map_fn(x):
    extra_state_dims = x.ndim - 2
    new_inds_shape = [ins.shape[0], ins.shape[1]] + [1] * extra_state_dims
    inds = jnp.reshape(ins, new_inds_shape)
    return jnp.take_along_axis(x, inds, axis=1)

  return jax.tree_util.tree_map(map_fn, states)


def make_posterior_dist(states, ancestors, resampled, num_steps, log_weights):
  # [num_timesteps, num_particles, state_dim]
  unwound_states = unwind_states(states, ancestors, resampled, num_steps)
  assert isinstance(unwound_states, jnp.ndarray)
  perm = list(range(unwound_states.ndim))
  perm[0] = 1
  perm[1] = 0
  # [num_particles, num_timesteps, state_dim]
  state_locs = jnp.transpose(unwound_states, axes=perm)
  return tfd.MixtureSameFamily(
      tfd.Categorical(logits=log_weights[num_steps - 1]),
      components_distribution=tfd.Independent(
          tfd.VectorDeterministic(state_locs), reinterpreted_batch_ndims=1))


def smc_posterior_dist(
    key: PRNGKey,
    initial_states: StateType,
    transition_fn: TransitionFn[StateType, ObsType_contra],
    num_steps: int,
    max_num_steps: int,
    observations: Optional[ObsType_contra] = None,
    num_particles: int = 1,
    resampling_criterion: ResamplingCriterion = ess_criterion,
    resampling_fn: ResamplingFn[StateType] = multinomial_resampling) -> tfd.Distribution:
  states, log_weights, ancestors, _, resampled = smc(
      key,
      initial_states,
      transition_fn,
      num_steps,
      max_num_steps,
      observations=observations,
      num_particles=num_particles,
      resampling_criterion=resampling_criterion,
      resampling_fn=resampling_fn)

  return make_posterior_dist(states, ancestors, resampled, num_steps, log_weights)


RESAMPLING_FNS = {
  "multinomial": multinomial_resampling,
  "stratified": stratified_resampling
}
RESAMPLING_CRITS = {
  "always": always_resample_criterion,
  "never": never_resample_criterion,
  "ess": ess_criterion
}
