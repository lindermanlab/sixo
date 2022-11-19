import jax
import jax.numpy as jnp
import sys
sys.path.append("../sixo")
from models import svm
import pytest
import equinox as eqx
import snax


@pytest.mark.parametrize("svm_cls", [svm.DiagCovSVM])
def test_basic_svm(svm_cls, data_dim=5, num_timesteps=27):
  key = jax.random.PRNGKey(0)
  k1, k2 = jax.random.split(key)
  model = svm_cls(k1, data_dim)
  xs, ys = model.sample_trajectory(k2, num_timesteps)
  lp = model.log_prob(xs, ys)
  assert jnp.all(jnp.isfinite(xs))
  assert jnp.all(jnp.isfinite(ys))
  assert jnp.all(jnp.isfinite(lp))


@pytest.mark.parametrize("svm_cls,proposal_cls", [(svm.DiagCovSVM, svm.DiagCovSVMProposal)])
def test_svm_with_proposal(svm_cls, proposal_cls, data_dim=5, num_timesteps=27):
  key = jax.random.PRNGKey(0)
  model = svm_cls(
          key,
          data_dim=data_dim)
  model_with_proposal = proposal_cls(
          key,
          model,
          num_timesteps)
  init_state = jnp.zeros([data_dim])
  new_state, log_weight = model_with_proposal.propose_and_weight(
          jax.random.PRNGKey(0), init_state, jnp.ones([5]), 0)
  new_state, log_weight = model_with_proposal.propose_and_weight(
          jax.random.PRNGKey(0), new_state, jnp.ones([5]), 1)


def test_quadrature_tilted_svm_with_proposal(data_dim=5, num_timesteps=27):
  key = jax.random.PRNGKey(0)
  model = svm.DiagCovSVM(
          key,
          data_dim=data_dim)
  model_with_proposal = svm.DiagCovSVMProposal(
          key,
          model,
          num_timesteps)
  tilt = svm.QuadratureTiltSVMProposal(
          model_with_proposal,
          data_dim,
          5, 1)
  obs = jnp.ones([num_timesteps, data_dim])
  p_and_w = tilt.make_propose_and_weight(1, obs, num_timesteps)
  init_state = jnp.zeros([data_dim])
  new_state, log_weight, log_r = p_and_w(
          jax.random.PRNGKey(0), init_state, obs[0], 0)
  assert jnp.all(jnp.isfinite(log_weight))
  assert jnp.all(jnp.isfinite(log_r))
  new_state, log_weight, log_r = p_and_w(
          jax.random.PRNGKey(0), new_state, obs[1], 1)
  assert jnp.all(jnp.isfinite(log_weight))
  assert jnp.all(jnp.isfinite(log_r))


def test_bwd_tilted_svm_with_proposal(data_dim=5, num_timesteps=27):
  key = jax.random.PRNGKey(0)
  model = svm.DiagCovSVM(
          key,
          data_dim=data_dim)
  model_with_proposal = svm.DiagCovSVMProposal(
          key,
          model,
          num_timesteps)
  tilt = svm.BackwardsTilt(
          key,
          model_with_proposal,
          snax.LSTM(key, data_dim, [4]),
          [4],
          data_dim)
  obs = jnp.ones([num_timesteps, data_dim])
  p_and_w = tilt.make_propose_and_weight(1, obs, num_timesteps)
  init_state = jnp.zeros([data_dim])
  new_state, log_weight, log_r = p_and_w(
          jax.random.PRNGKey(0), init_state, obs[0], 0)
  assert jnp.all(jnp.isfinite(log_weight))
  assert jnp.all(jnp.isfinite(log_r))
  new_state, log_weight, log_r = p_and_w(
          jax.random.PRNGKey(0), new_state, obs[1], 1)
  assert jnp.all(jnp.isfinite(log_weight))
  assert jnp.all(jnp.isfinite(log_r))


class WindowedTestTilt(svm.WindowedTiltSVMProposal):

  num_timesteps: int = eqx.static_field()

  def __init__(self, prop, window, num_timesteps):
    super().__init__(prop, 1, window)
    self.num_timesteps = num_timesteps

  def score_window(self, latent, obs, t, mask):
    conds = []
    obs = jnp.reshape(obs, [self.window + 1])
    expected_mask = jnp.arange(t + 1, self.window + t + 1) < self.num_timesteps
    expected_obs = jnp.arange(t, t + 1 + self.window)
    expected_obs = expected_obs * (expected_obs < self.num_timesteps)
    conds.append(jnp.allclose(obs, expected_obs))
    conds.append(jnp.allclose(mask, expected_mask))
    conds.append(jnp.any(mask > 0.))
    conds.append(t >= 0)
    conds.append(t < self.num_timesteps - 1)
    allconds = jnp.all(jnp.array(conds))
    return jnp.where(allconds, 1., jnp.nan)


def test_windowed_tilt(num_timesteps=27, window=5):
  key = jax.random.PRNGKey(0)
  model = svm.DiagCovSVM(
          key,
          data_dim=1)
  model_with_proposal = svm.DiagCovSVMProposal(
          key,
          model,
          num_timesteps)
  tilt = WindowedTestTilt(model_with_proposal, window, num_timesteps)

  latents = jnp.arange(num_timesteps, dtype=jnp.float32)[:, jnp.newaxis]
  obs = jnp.arange(num_timesteps, dtype=jnp.float32)[:, jnp.newaxis]

  p_and_w = tilt.make_propose_and_weight(1, obs, num_timesteps)

  for t in range(num_timesteps):
    _, _, out = p_and_w(key, latents[t], obs[t + 1], t)
    assert jnp.isfinite(out)
