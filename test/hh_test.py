import jax
import jax.numpy as jnp
import snax
import sys
sys.path.append("../sixo")
from models import hh
import pytest


@pytest.mark.parametrize("ode_int", hh.ODE_SOLVERS.keys())
def test_hh(ode_int, num_timesteps=100):
  key = jax.random.PRNGKey(0)
  key, subkey = jax.random.split(key)
  model = hh.HodgkinHuxley(key, data_dim=1, ode_solver=ode_int)
  (states, obs) = model.sample_trajectory(subkey, num_timesteps=num_timesteps)
  lp = model.log_prob(states, obs)
  assert jnp.all(jnp.isfinite(lp))


@pytest.mark.parametrize("tilt_type", ['none', 'bwd_dre'])
@pytest.mark.parametrize("prop_type", ['bootstrap', 'filtering', 'smoothing'])
@pytest.mark.parametrize("obs_subsample", [1, 2])
@pytest.mark.parametrize("resq_type", ['full'])
def test_hh_proposal(tilt_type, prop_type, obs_subsample, resq_type, num_timesteps=7, data_dim=1):
  rnn_hdims = [3]
  mlp_hdims = [3]
  key = jax.random.PRNGKey(0)
  k1, k2, k3, k4, k5 = jax.random.split(key, num=5)
  model = hh.HodgkinHuxley(k1, data_dim=data_dim, obs_subsample=obs_subsample)
  if prop_type != 'bootstrap':
    if prop_type == 'filtering':
      cls = hh.FilteringHHProposal
      rnn_cls = snax.LSTM
    elif prop_type == 'smoothing':
      cls = hh.SmoothingHHProposal
      rnn_cls = snax.BiLSTM
    prop = cls(
            k2,
            model,
            rnn_cls(k3, data_dim, rnn_hdims),
            mlp_hdims,
            pos_enc_dim=1,
            resq_type=resq_type)
  else:
    prop = hh.BootstrapHHProposal(model)

  if tilt_type == 'bwd_dre':
    prop = hh.BackwardsTilt(
            k4,
            prop,
            snax.LSTM(k3, data_dim, rnn_hdims),
            mlp_hdims,
            model.state_dim,
            pos_enc_dim=1)

  xs, ys = model.sample_trajectory(k5, num_timesteps)
  state = xs[0]
  p_and_w = jax.jit(prop.make_propose_and_weight(0, ys, num_timesteps))
  for t in range(num_timesteps):
    key, subkey = jax.random.split(key)
    outs = p_and_w(subkey, state, ys[t], t)
    assert jnp.all(jnp.isfinite(outs[0]))  # new state
    assert jnp.all(jnp.isfinite(outs[1]))  # weight
    if tilt_type == 'bwd_dre':
      assert jnp.all(jnp.isfinite(outs[2]))  # log r
