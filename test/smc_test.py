import pytest

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions

import sys
sys.path.append('../sixo')
import smc


def test_basic_smc_with_obs():

  def transition_fn(key, state, obs, t):
    return obs + state + 1, 0.

  states, log_weights, _, Z_hat, resampled = smc.smc(
      jax.random.PRNGKey(0),
      jnp.array([0., 0., 0.]),
      transition_fn, 10, 12, observations=jnp.ones([12]), num_particles=3)
  true_states = jnp.tile((jnp.arange(1, stop=11, step=1) * 2.)[:, jnp.newaxis], (1, 3))
  assert jnp.allclose(true_states, states[:10])


def test_basic_smc_no_obs():
  def transition_fn(key, state, t):
    return state + 1, 0.

  states, log_weights, _, Z_hat, resampled = smc.smc(
      jax.random.PRNGKey(0),
      jnp.array([0., 0., 0.]),
      transition_fn, 10, 12, num_particles=3)
  true_states = jnp.tile((jnp.arange(1, stop=11, step=1))[:, jnp.newaxis], (1, 3))
  assert jnp.allclose(true_states, states[:10])


def test_smc_no_resampling(num_particles=3, num_steps=10, max_num_steps=12):
  """Test that SMC correctly produces log_Z_hat when no resampling occurs."""

  def transition_fn(key, state, t):
    """A transition function that always returns 2 for the log weight."""
    return state + 1, 2.

  states, log_weights, _, log_Z_hat, resampled = smc.smc(
          jax.random.PRNGKey(0),
          jnp.full([num_particles], 0.),
          transition_fn,
          num_steps,
          max_num_steps,
          num_particles=num_particles,
          resampling_criterion=smc.never_resample_criterion)

  assert log_Z_hat == 20.
  assert jnp.allclose(resampled, jnp.zeros_like(resampled))


def test_no_resampling_last_step(num_particles=3, num_steps=10, max_num_steps=12):
  """Test that SMC correctly produces log_Z_hat when no resampling occurs."""

  def resampling_criterion(unused_log_weights, t):
    return t < num_steps - 1

  def transition_fn(key, state, t):
    """A transition function that always returns 2 for the log weight."""
    return state + 1, 2.

  states, log_weights, _, log_Z_hat, resampled = smc.smc(
          jax.random.PRNGKey(0),
          jnp.full([num_particles], 0.),
          transition_fn,
          num_steps,
          max_num_steps,
          num_particles=num_particles,
          resampling_criterion=resampling_criterion)

  assert log_Z_hat == 20.
  assert jnp.allclose(resampled[:num_steps - 2], jnp.ones_like(resampled[:num_steps - 2]))
  assert jnp.allclose(resampled[num_steps - 1], jnp.zeros_like(resampled[num_steps - 1]))


def test_some_neg_inf_weights(num_particles=4, num_steps=3, max_num_steps=3):
  """Test that SMC correctly handles weights of negative infinity."""

  def transition_fn(key, state, t):
    """Decrement the state and returns -inf weight when it dips below zero."""
    return state - 1, jnp.where(state < 1, x=-jnp.inf, y=0.)

  states, log_weights, ancestors, log_Z_hat, _ = smc.smc(
          jax.random.PRNGKey(0),
          jnp.arange(num_particles),
          transition_fn,
          num_steps,
          max_num_steps,
          num_particles=num_particles,
          resampling_criterion=smc.always_resample_criterion)
  # assert that the weights of the negative states are -inf
  assert jnp.all(jnp.equal(jnp.isfinite(log_weights), jnp.greater_equal(states, 0.)))

  # assert that the weights don't all become negative inf at once.
  for i in range(num_steps):
    assert jnp.any(jnp.isfinite(log_weights[i]))

  # assert that -inf weights are never selected for resampling
  for i in range(num_steps):
    neg_inf_inds = jnp.nonzero(jnp.logical_not(jnp.isfinite(log_weights[i])))
    ancestor_inds = ancestors[i]

    if jnp.any(jnp.isfinite(log_weights[i])):
      for ind in neg_inf_inds:
        if ind.size > 0:
          assert ind[0] not in ancestor_inds

  # assert that the bound is not negative inf
  assert jnp.isfinite(log_Z_hat)


def test_all_neg_inf_weights(num_particles=4, num_steps=5, max_num_steps=5):
  """Test that SMC correctly handles weights of negative infinity."""

  def transition_fn(key, state, t):
    """Decrement the state and returns -inf weight when it dips below zero."""
    return state - 1, jnp.where(state < 1, x=-jnp.inf, y=0.)

  states, log_weights, ancestors, log_Z_hat, _ = smc.smc(
          jax.random.PRNGKey(0),
          jnp.arange(num_particles),
          transition_fn,
          num_steps,
          max_num_steps,
          num_particles=num_particles,
          resampling_criterion=smc.always_resample_criterion)
  # assert that the weights of the negative states are -inf
  assert jnp.all(jnp.equal(jnp.isfinite(log_weights), jnp.greater_equal(states, 0.)))

  # assert that the weights eventually all become negative inf at once.
  assert jnp.any(jnp.logical_not(jnp.any(jnp.isfinite(log_weights), axis=1)))

  # assert that -inf weights are never selected for resampling
  for i in range(num_steps):
    neg_inf_inds = jnp.nonzero(jnp.logical_not(jnp.isfinite(log_weights[i])))
    ancestor_inds = ancestors[i]
    # if the weights are not all -inf
    if jnp.any(jnp.isfinite(log_weights[i])):
      for ind in neg_inf_inds:
        if ind.size > 0:
          assert ind[0] not in ancestor_inds

  # assert that the bound is negative inf
  assert not jnp.isfinite(log_Z_hat)


def test_some_neg_inf_weights_no_resampling(num_particles=4, num_steps=3, max_num_steps=3):
  """Test that SMC correctly handles weights of negative infinity without resampling."""

  def transition_fn(key, state, t):
    """Decrement the state and returns -inf weight when it dips below zero."""
    return state - 1, jnp.where(state < 1, x=-jnp.inf, y=0.)

  states, log_weights, _, log_Z_hat, _ = smc.smc(
          jax.random.PRNGKey(0),
          jnp.arange(num_particles),
          transition_fn,
          num_steps,
          max_num_steps,
          num_particles=num_particles,
          resampling_criterion=smc.never_resample_criterion)
  # assert that the weights of the negative states are -inf
  assert jnp.all(jnp.equal(jnp.isfinite(log_weights), jnp.greater_equal(states, 0.)))

  # assert that the weights don't all become negative inf at once.
  for i in range(num_steps):
    assert jnp.any(jnp.isfinite(log_weights[i]))

  # assert that the bound is not negative inf
  assert jnp.isfinite(log_Z_hat)


def test_all_neg_inf_weights_no_resampling(num_particles=4, num_steps=5, max_num_steps=5):
  """Test that SMC correctly handles weights of negative infinity without resampling."""

  def transition_fn(key, state, t):
    """Decrement the state and returns -inf weight when it dips below zero."""
    return state - 1, jnp.where(state < 1, x=-jnp.inf, y=0.)

  states, log_weights, _, log_Z_hat, _ = smc.smc(
          jax.random.PRNGKey(0),
          jnp.arange(num_particles),
          transition_fn,
          num_steps,
          max_num_steps,
          num_particles=num_particles,
          resampling_criterion=smc.never_resample_criterion)
  # assert that the weights of the negative states are -inf
  assert jnp.all(jnp.equal(jnp.isfinite(log_weights), jnp.greater_equal(states, 0.)))

  # assert that the weights eventually all become negative inf at once.
  assert jnp.any(jnp.logical_not(jnp.any(jnp.isfinite(log_weights), axis=1)))

  # assert that the bound is negative inf
  assert not jnp.isfinite(log_Z_hat)

# def test_some_neg_inf_weights_grads(num_particles=4, num_steps=4, max_num_steps=4):
#  """Test that SMC correctly handles weights of negative infinity without resampling."""
#
#  def transition_fn(b, key, state, t):
#    """Decrement the state and returns -inf weight when it dips below zero."""
#    new_state = -jnp.power(b, jnp.abs(state))
#    new_state = jnp.where(
#            jnp.isfinite(new_state),
#            lambda x: x,
#            lambda x: -1e6,
#            new_state)
#    return new_state, new_state
#
#  def bound(b):
#    states, log_ws, _, log_Z_hat, _ = smc.smc(
#          jax.random.PRNGKey(0),
#          jnp.arange(num_particles, dtype=jnp.float32),
#          functools.partial(transition_fn, b),
#          num_steps,
#          max_num_steps,
#          num_particles=num_particles,
#          resampling_criterion=smc.never_resample_criterion)
#    #print(states)
#    #print(log_ws)
#    return log_Z_hat
#
#  print(bound(2.))
#  print(jax.grad(bound)(2.))


def test_unwind_states_2d():
  states = jnp.arange(3 * 3).reshape([3, 3])
  ancestor_inds = jnp.array([[0, 1, 2],
                             [2, 2, 1],
                             [0, 1, 2]])
  resampled = jnp.array([0, 1, 1])

  outs = smc.unwind_states(states, ancestor_inds, resampled, 3)
  assert jnp.all(jnp.equal(outs, jnp.array([[2, 2, 1], [5, 5, 4], [6, 7, 8]]))), outs

  ancestor_inds = jnp.array([[0, 1, 2],
                             [2, 2, 1],
                             [0, 1, 2]])
  resampled = jnp.array([0, 0, 1])

  outs = smc.unwind_states(states, ancestor_inds, resampled, 3)
  assert jnp.all(jnp.equal(outs, jnp.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])))

  ancestor_inds = jnp.array([[2, 1, 0],
                             [2, 2, 1],
                             [2, 0, 1]])
  resampled = jnp.array([1, 0, 1])
  outs = smc.unwind_states(states, ancestor_inds, resampled, 3)
  assert jnp.all(jnp.equal(outs, jnp.array([[2, 1, 0], [3, 4, 5], [6, 7, 8]])))

  ancestor_inds = jnp.array([[1, 1, 0],
                             [2, 2, 1],
                             [2, 0, 1]])
  resampled = jnp.array([1, 1, 1])
  outs = smc.unwind_states(states, ancestor_inds, resampled, 3)
  assert jnp.all(jnp.equal(outs, jnp.array([[0, 0, 1], [5, 5, 4], [6, 7, 8]])))


def test_unwind_states_3d():
  states = jnp.arange(3 * 3 * 2).reshape([3, 3, 2])

  ancestor_inds = jnp.array([[0, 1, 2],
                             [2, 2, 1],
                             [2, 2, 1]])
  resampled = jnp.array([0, 1, 0])
  outs = smc.unwind_states(states, ancestor_inds, resampled, 3)
  assert jnp.all(jnp.equal(outs, jnp.array([[[4, 5], [4, 5], [2, 3]],
                                            [[10, 11], [10, 11], [8, 9]],
                                            [[12, 13], [14, 15], [16, 17]]]))), outs


def test_unwind_states_4d():
  states = jnp.arange(3 * 3 * 2 * 2).reshape([3, 3, 2, 2])
  ancestor_inds = jnp.array([[0, 1, 2],
                             [2, 2, 1],
                             [2, 2, 1]])
  resampled = jnp.array([0, 1, 0])
  outs = smc.unwind_states(states, ancestor_inds, resampled, 3)
  assert jnp.all(jnp.equal(outs, jnp.array([[[[8, 9],
                                              [10, 11]],
                                             [[8, 9],
                                              [10, 11]],
                                             [[4, 5],
                                              [6, 7]]],
                                            [[[20, 21],
                                              [22, 23]],
                                             [[20, 21],
                                              [22, 23]],
                                             [[16, 17],
                                              [18, 19]]],
                                            [[[24, 25],
                                              [26, 27]],
                                             [[28, 29],
                                              [30, 31]],
                                             [[32, 33],
                                              [34, 35]]]])))


def test_basic_smc_posterior_dist():

  def transition_fn(key, state, obs, t):
    return obs + state + 1, 0.

  dist = smc.smc_posterior_dist(
      jax.random.PRNGKey(0),
      jnp.array([[0.], [0.], [0.]]),
      transition_fn, 10, 12, observations=jnp.ones([12]), num_particles=3)
  x = dist.sample(seed=jax.random.PRNGKey(0))
  assert x.shape[0] == 12


@pytest.mark.parametrize("resampling_criterion",
        [smc.always_resample_criterion, smc.ess_criterion])
@pytest.mark.parametrize("resampling_fn", smc.RESAMPLING_FNS.values())
def test_smc_norm_const(resampling_criterion, resampling_fn):
  """Check that the normalizing constant estimate of SMC is correct."""
  q_std = 1.25
  num_steps = 25
  num_particles = 7_500

  def transition_fn(key, state, t):
    q_dist = tfd.Normal(0., q_std)
    new_x = q_dist.sample(seed=key)
    log_q = q_dist.log_prob(new_x)
    log_p = - jnp.square(new_x) / 2
    return new_x, log_p - log_q

  _, _, _, log_Z_hat, _ = smc.smc(
      jax.random.PRNGKey(0),
      jnp.zeros([num_particles]),
      transition_fn,
      num_steps,
      num_steps,
      num_particles=num_particles,
      resampling_criterion=resampling_criterion,
      resampling_fn=resampling_fn)

  log_Z_expected = (num_steps / 2.) * (jnp.log(2) + jnp.log(jnp.pi))
  assert jnp.allclose(log_Z_hat, log_Z_expected, atol=1e-2), \
          f"Normalizing constant should be approx log ((2pi)^(t/2)) = {log_Z_expected}," \
          f" is {log_Z_hat} instead."
