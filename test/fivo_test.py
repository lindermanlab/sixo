import sys
import jax
import jax.numpy as jnp

from scipy.special import logsumexp
from numpy import log

sys.path.append('../sixo')
import bounds
import smc


def test_basic_fivo_with_obs():

  def propose_and_weight(key, state, obs, t):
    return obs + state + 1, 0.

  states, _, Z_hat, _, _ = bounds.fivo(
      jax.random.PRNGKey(0),
      propose_and_weight,
      jnp.array(0.), 10, 12, observations=jnp.ones([12]), num_particles=3)
  true_states = jnp.tile((jnp.arange(1, stop=11, step=1) * 2.)[:, jnp.newaxis], (1, 3))
  assert jnp.allclose(true_states, states[:10])


def test_basic_fivo_no_obs():

  def propose_and_weight(key, state, t):
    return state + 1, 0.

  states, _, Z_hat, _, _ = bounds.fivo(
      jax.random.PRNGKey(0),
      propose_and_weight,
      jnp.array(0.),
      10, 12, num_particles=3)
  true_states = jnp.tile((jnp.arange(1, stop=11, step=1))[:, jnp.newaxis], (1, 3))
  assert jnp.allclose(true_states, states[:10])


def test_sixo():

  def propose_and_weight(key, state, cur_obs, t):
    return state + 1, 0., 0.

  states, _, Z_hat, _, _ = bounds.sixo(
      jax.random.PRNGKey(0),
      propose_and_weight,
      jnp.zeros(3),
      10, 12,
      observations=jnp.ones([12]),
      num_particles=3,
      tile_initial_state=False)
  true_states = jnp.tile((jnp.arange(1, stop=11, step=1))[:, jnp.newaxis], (1, 3))
  assert jnp.allclose(true_states, states[:10])


def test_sixo_no_resample():

  def propose_and_weight(key, state, cur_obs, t):
    return state + 1, 0., t

  states, _, Z_hat, _, _ = bounds.sixo(
      jax.random.PRNGKey(0),
      propose_and_weight,
      jnp.zeros(3),
      10, 12,
      observations=jnp.ones([12]),
      resampling_criterion=smc.never_resample_criterion,
      num_particles=3,
      tile_initial_state=False)
  true_states = jnp.tile((jnp.arange(1, stop=11, step=1))[:, jnp.newaxis], (1, 3))
  assert Z_hat == 0.
  assert jnp.allclose(true_states, states[:10])


def test_sixo_resample_once():

  def propose_and_weight(key, state, cur_obs, t):
    return state + 1, 0., t

  def resample_once(unused_log_weights, t):
    return jnp.equal(t, 4)

  states, _, Z_hat, _, _ = bounds.sixo(
      jax.random.PRNGKey(0),
      propose_and_weight,
      jnp.zeros(3),
      10, 12,
      observations=jnp.ones([12]),
      resampling_criterion=resample_once,
      num_particles=3,
      tile_initial_state=False)
  true_states = jnp.tile((jnp.arange(1, stop=11, step=1))[:, jnp.newaxis], (1, 3))
  assert Z_hat == 0.
  assert jnp.allclose(true_states, states[:10])


def test_sixo_resample_once_r_in_bound(log_r=1.):

  def propose_and_weight(key, state, cur_obs, t):
    # weights are 0
    return state, 0., state

  def resample_once(unused_log_weights, t):
    return jnp.equal(t, 4)

  states, log_weights, log_Z_hat, _, _ = bounds.sixo(
      jax.random.PRNGKey(0),
      propose_and_weight,
      jnp.array([-log_r, log_r]),
      10, 12,
      observations=jnp.ones([12]),
      resampling_criterion=resample_once,
      num_particles=2,
      tile_initial_state=False)
  true_states = jnp.array([
      [-log_r, log_r],
      [-log_r, log_r],
      [-log_r, log_r],
      [-log_r, log_r],
      [-log_r, log_r],
      [log_r, log_r],
      [log_r, log_r],
      [log_r, log_r],
      [log_r, log_r],
      [log_r, log_r]])
  assert jnp.allclose(true_states, states[:10])
  log_p_hat_1 = logsumexp([log_r, -log_r]) - log(2.)
  log_p_hat_2 = logsumexp([-log_r, -log_r]) - log(2.)
  expected_log_Z_hat = log_p_hat_1 + log_p_hat_2
  assert jnp.allclose(log_Z_hat, expected_log_Z_hat)
  # For each timestep, these are the expected prev_log_rs and new_log_rs.
  # Note that we cannot actually expect the log weights to be correct after timestep 10,
  # but the extra timesteps are included here for completeness.
  #
  #     prev_log_r  new_log_r  resample  log_w_change  log_weight
  #     ----------  ---------  --------  ------------  ----------
  # 0)    0,  0      -lr, lr      no       -lr, lr      -lr, lr
  # 1)   -lr, lr     -lr, lr      no         0, 0       -lr, lr
  # 2)   -lr, lr     -lr, lr      no         0, 0       -lr, lr
  # 3)   -lr, lr     -lr, lr      no         0, 0       -lr, lr
  # 4)   -lr, lr     -lr, lr      yes        0, 0       -lr, lr
  # 5)    lr, lr      lr, lr      no         0, 0         0, 0
  # 6)    lr, lr      lr, lr      no         0, 0         0, 0
  # 7)    lr, lr      lr, lr      no         0, 0         0, 0
  # 8)    lr, lr      lr, lr      no         0, 0         0, 0
  # 9)    lr, lr       0, 0       no       -lr, -lr     -lr, -lr
  # 10)    0, 0        0, 0       no         0, 0       -lr, -lr
  # 11)    0, 0        0, 0       no         0, 0       -lr, -lr

  true_log_ws = jnp.array([
      [-log_r, log_r],
      [-log_r, log_r],
      [-log_r, log_r],
      [-log_r, log_r],
      [-log_r, log_r],
      [0., 0.],
      [0., 0.],
      [0., 0.],
      [0., 0.],
      [-log_r, -log_r],
      [-log_r, -log_r],
      [-log_r, -log_r]])
  assert jnp.allclose(true_log_ws, log_weights)


def test_sixo_resample_on_last_timestep(log_r=1.):

  def propose_and_weight(key, state, cur_obs, t):
    # weights are 0
    return state, 0., state

  def resample_once(unused_log_weights, t):
    return jnp.equal(t, 3)

  states, log_weights, log_Z_hat, _, _ = bounds.sixo(
      jax.random.PRNGKey(0),
      propose_and_weight,
      jnp.array([-log_r, log_r]),
      4, 6,
      observations=jnp.ones([6]),
      resampling_criterion=resample_once,
      num_particles=2,
      tile_initial_state=False)
  true_states = jnp.array([
      [-log_r, log_r],
      [-log_r, log_r],
      [-log_r, log_r],
      [-log_r, log_r],
      [log_r, log_r],
      [log_r, log_r]])
  assert jnp.allclose(true_states[:4], states[:4])
  assert jnp.allclose(log_Z_hat, 0.)
  # For each timestep, these are the expected prev_log_rs and new_log_rs.
  # Note that we cannot actually expect the log weights to be correct after timestep 3,
  # but the extra timesteps are included here for completeness.
  #
  #     prev_log_r  new_log_r  resample  log_w_change  log_weight
  #     ----------  ---------  --------  ------------  ----------
  # 0)    0,  0      -lr, lr      no       -lr, lr      -lr, lr
  # 1)   -lr, lr     -lr, lr      no         0, 0       -lr, lr
  # 2)   -lr, lr     -lr, lr      no         0, 0       -lr, lr
  # 3)   -lr, lr       0, 0       yes       lr, -lr       0, 0
  # 4)     0, 0        0, 0       no         0, 0         0, 0
  # 5)     0, 0        0, 0       no         0, 0         0, 0

  true_log_ws = jnp.array([
      [-log_r, log_r],
      [-log_r, log_r],
      [-log_r, log_r],
      [0., 0.],
      [0., 0.],
      [0., 0.]])
  assert jnp.allclose(true_log_ws, log_weights)
