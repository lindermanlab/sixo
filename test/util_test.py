import jax
import jax.numpy as jnp

import sys
sys.path.append('../sixo')
import util


def test_rolling_window():
  x = jnp.arange(1, 6)
  x_shape = [5]
  fn = jax.jit(lambda v: util.rolling_window(v, x_shape, 2))
  out = fn(x)
  assert jnp.all(jnp.equal(out, jnp.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 0]])))


def test_rolling_window2():
  x = jnp.arange(1, 6)
  x_shape = [5]
  fn = jax.jit(lambda v: util.rolling_window(v, x_shape, 3))
  out = fn(x)
  assert jnp.all(jnp.equal(out, jnp.array([[1, 2, 3],
                                           [2, 3, 4],
                                           [3, 4, 5],
                                           [4, 5, 0],
                                           [5, 0, 0]])))


def test_integrate_hg():
  num_mc_samples = 1_000_000
  mean = 1.
  scale = 2.
  f = lambda x: x**2
  key = jax.random.PRNGKey(0)
  mc_xs = (scale * jax.random.normal(key, shape=[num_mc_samples])) + mean
  mc_est = jnp.mean(jax.vmap(f)(mc_xs))
  quad_est = util.integrate_hg(5, lambda x: x**2, 1., 2.)
  assert jnp.allclose(mc_est, quad_est, atol=1e-2)


def test_annealing_sched():
  assert util.linear_ramp_schedule(100, 50, 0) == 0.
  assert util.linear_ramp_schedule(100, 50, 100) == 0.
  assert util.linear_ramp_schedule(100, 50, 150) == 1.
