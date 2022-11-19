from functools import partial
import scipy.stats
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

import sys
sys.path.append('../sixo')
import bounds
from smc import always_resample_criterion
import pytest


def generate_data(key, T, alpha, mu0=0., std0=1., transition_std=1., obs_std=1.):
  """Generate data for a simple Gaussian drift model with an observation per timestep
  """
  def step(carry, dummy):
    key, x_prev, t = carry
    key, k1, k2 = jax.random.split(key, num=3)
    x = jax.lax.cond(
      t > 0,
      lambda _: x_prev + alpha + tfd.Normal(0., transition_std).sample(seed=k1),
      lambda _: tfd.Normal(mu0, std0).sample(seed=k1),
      None
    )
    y = tfd.Normal(x, obs_std).sample(seed=k2)
    return ((key, x, t + 1), (x, y))  # (t+1) is a bit unnecessary
  _, (x, y) = jax.lax.scan(step,
                           (key, 0., 0),
                           None,
                           length=T)
  return (x[:, jnp.newaxis], y[:, jnp.newaxis])


def propose_and_weight(params, key, prev_state, observation, t):
  """Bootstrap Particle Filter (BPF) propose_and_weight for Gaussian drift
  NOTE: Only param is alpha_p and is shared by transition model and proposal
  """
  # alpha_p is a scalar, prev_state has shape (1,), observation has shape (1,)
  alpha_p = params[0]
  q = jax.lax.cond(
      t > 0,
      lambda _: tfd.Normal(prev_state[0] + alpha_p, 1.),
      lambda _: tfd.Normal(0., 1.),
      None
  )
  f = q
  new_state = q.sample(seed=key)
  g = tfd.Normal(new_state, 1.)
  incremental_log_weight = (
          f.log_prob(new_state) + g.log_prob(observation[0]) - q.log_prob(new_state))
  return jnp.array([new_state]), incremental_log_weight


def loss(resampling_gradient_mode, params, key, init_state, observations):
  T = observations.shape[0]
  _, _, log_Z_hat, _, _ = bounds.fivo(
      key,
      partial(propose_and_weight, params),
      init_state,
      T,
      T,
      64,
      observations=observations,
      resampling_criterion=always_resample_criterion,
      resampling_gradient_mode=resampling_gradient_mode
  )
  return log_Z_hat


@pytest.mark.parametrize("init_param", [-5., 0., 5., 10.])
def test_bias(init_param):
  # FIXME: directly link init_state to data generation (both independently start at 0.)
  init_state = jnp.zeros(1)

  T = 3
  alpha = 5.

  # Interesting to try parameters closer to and farther from true `alpha`
  init_params = jnp.array([init_param])

  # Generate data
  key = jax.random.PRNGKey(0)
  _, observations = generate_data(key, T, alpha)

  # Was using 100_000 but tests should be faster than that
  N = 10_000

  grad_fn = jax.grad(partial(loss, 'score_fn'))
  keys = jax.random.split(key, num=N)
  grads = jax.vmap(
      grad_fn, in_axes=(None, 0, None, None))(
          init_params, keys, init_state, observations)
  print(f'Easy score fn grads mean: {grads.mean():0.3f}, std: {grads.std():0.3f}')

  grad_fn = jax.grad(partial(loss, 'score_fn_rb'))
  grads_rb = jax.vmap(
      grad_fn, in_axes=(None, 0, None, None))(
          init_params, keys, init_state, observations)
  print(("Rao-Blackwellized score fn grads mean:"
         f" {grads_rb.mean():0.3f}, std: {grads_rb.std():0.3f}"))

  _, p_val = scipy.stats.ttest_rel(
          grads, grads_rb, alternative='two-sided', nan_policy='raise')
  assert p_val >= 0.1
  print("Paired, two-sided t-test p value", p_val)

  diffs = grads - grads_rb
  diff_mean = jnp.mean(diffs)
  se_diff = jnp.std(diffs, ddof=1) / jnp.sqrt(N)
  for alpha in [0.05, 0.01, 0.001]:
    t_crit = scipy.stats.t.ppf(1. - (alpha / 2.), N - 1)
    conf_lower = diff_mean - (t_crit * se_diff)
    conf_upper = diff_mean + (t_crit * se_diff)
    print(f"{100*(1-alpha):0.1f}% confidence interval @ : ({conf_lower:0.2f}, {conf_upper:0.2f})")
