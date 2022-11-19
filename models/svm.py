from abc import abstractmethod
from jax._src.random import KeyArray as PRNGKey
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
import equinox as eqx

import snax
from chex import Array, Scalar, ArrayTree
from typing import Tuple, List
from models import base
from util import L_from_raw, log_integrate_multid_hg, chain_cov

tf = tfp.tf2jax
tfd = tfp.distributions
tfd_e = tfp.experimental.distributions


class SVM(base.SSM[Array, Array]):

  mu: Array
  raw_phi: Array
  log_beta: Array

  data_dim: int = eqx.static_field()
  min_scale_diag: float = eqx.static_field()
  step_0_vsmc: bool = eqx.static_field()

  def __init__(
          self,
          key: PRNGKey,
          data_dim: int,
          min_scale_diag: float = 1e-4,
          init_scale=0.3,
          step_0_vsmc: bool = False):
    self.data_dim = data_dim
    self.min_scale_diag = min_scale_diag
    self.step_0_vsmc = step_0_vsmc
    keys = jax.random.split(key, num=3)
    self.mu = tfd.Normal(0, init_scale).sample(seed=keys[0], sample_shape=[data_dim])
    # Set phis to 0.1 initially.
    self.raw_phi = tfd.Normal(jnp.arctanh(0.1), init_scale).sample(
            seed=keys[1], sample_shape=[data_dim])
    self.log_beta = tfd.Normal(0., init_scale).sample(
         seed=keys[2], sample_shape=[data_dim])

  def dynamics_mean(self, prev_state: Array, t: int) -> Array:
    if self.step_0_vsmc:
      phi = jnp.tanh(self.raw_phi)
    else:
      phi = jax.lax.cond(
              jnp.equal(t, 0),
              lambda _: jnp.zeros_like(self.raw_phi),
              lambda _: jnp.tanh(self.raw_phi),
              None)
    loc = self.mu + phi * (prev_state - self.mu)
    return loc

  def initial_state_prior(self) -> tfd.Distribution:
    return self.dynamics_dist(jnp.zeros([self.data_dim]), 0)

  def emission_dist(self, cur_state: Array, t: int) -> tfd.Distribution:
    del t
    loc = jnp.zeros_like(cur_state)
    scale_diag = jnp.exp(self.log_beta + (cur_state / 2.))
    return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)


class DiagCovSVM(SVM):

  raw_scale_diag: Array

  def __init__(
          self,
          key: PRNGKey,
          data_dim: int,
          min_scale_diag: float = 1e-4,
          init_scale=0.3,
          step_0_vsmc: bool = False):
    key, subkey = jax.random.split(key)
    super().__init__(subkey, data_dim, min_scale_diag=min_scale_diag, init_scale=init_scale,
                     step_0_vsmc=step_0_vsmc)
    self.raw_scale_diag = tfd.Normal(jnp.log(1.), init_scale).sample(
              seed=key, sample_shape=[data_dim])

  def dynamics_scale_diag(self):
    return jnp.maximum(jnp.exp(self.raw_scale_diag), self.min_scale_diag)

  def dynamics_dist(self, prev_state: Array, t: int) -> tfd.Distribution:
    loc = self.dynamics_mean(prev_state, t)
    scale_diag = self.dynamics_scale_diag()
    return tfd.MultivariateNormalDiag(loc, scale_diag=scale_diag)


class FullRankCovSVM(SVM):

  raw_scale_L: Array

  def __init__(
          self,
          key: PRNGKey,
          data_dim: int,
          min_scale_diag: float = 1e-4,
          init_scale=0.3,
          step_0_vsmc: bool = False):
    key, subkey = jax.random.split(key)
    super().__init__(subkey, data_dim, min_scale_diag=min_scale_diag, init_scale=init_scale,
                     step_0_vsmc=step_0_vsmc)
    raw_scale_L = jnp.eye(data_dim) * jnp.log(1.)
    self.raw_scale_L = raw_scale_L + tfd.Normal(1., init_scale).sample(
            seed=key, sample_shape=[data_dim, data_dim])

  def dynamics_dist(self, prev_state: Array, t: int) -> tfd.Distribution:
    loc = self.dynamics_mean(prev_state, t)
    scale_L = tf.linalg.LinearOperatorLowerTriangular(
            L_from_raw(self.raw_scale_L, diag_min=self.min_scale_diag))
    return tfd.MultivariateNormalLinearOperator(loc, scale=scale_L)


class SVMWithProposal(eqx.Module):

  svm: SVM

  def __init__(self, svm: SVM):
    self.svm = svm

  @abstractmethod
  def proposal_dist(self, prev_state: Array, t: int) -> tfd.Distribution:
    raise NotImplementedError("proposal_dist not yet implemented.")

  def propose_and_weight(self, key: PRNGKey, prev_latent: Array, cur_obs: Array, t: int,
          propose_from_prior=False) -> Tuple[Array, Scalar]:
    if propose_from_prior:
      q_dist = self.svm.dynamics_dist(prev_latent, t)
    else:
      q_dist = self.proposal_dist(prev_latent, t)
    new_x = q_dist.sample(seed=key)
    log_q_x = q_dist.log_prob(new_x)
    log_p_x = self.svm.dynamics_dist(prev_latent, t).log_prob(new_x)
    log_p_y_given_x = self.svm.emission_dist(new_x, t).log_prob(cur_obs)
    return new_x, log_p_x + log_p_y_given_x - log_q_x


class DiagCovSVMProposal(SVMWithProposal):

  mus: Array
  raw_scale_diags: Array
  min_scale_diag: float = eqx.static_field()
  num_timesteps: int = eqx.static_field()

  def __init__(
          self,
          key: PRNGKey,
          svm: DiagCovSVM,
          num_timesteps: int,
          min_scale_diag=1e-4,
          init_scale=0.3):
    super().__init__(svm)
    self.min_scale_diag = min_scale_diag
    self.num_timesteps = num_timesteps
    k1, k2 = jax.random.split(key)
    self.mus = tfd.Normal(0, init_scale).sample(seed=k1, sample_shape=[num_timesteps, svm.data_dim])
    self.raw_scale_diags = tfd.Normal(jnp.log(1.), init_scale).sample(
            seed=k2, sample_shape=[num_timesteps, svm.data_dim])

  def proposal_dist(self, prev_state: Array, t: int) -> tfd.Distribution:
    prior_scale_diag = self.svm.dynamics_scale_diag()
    prop_scale_diag = jnp.maximum(jnp.exp(self.raw_scale_diags[t]), self.min_scale_diag)
    prior_var = jnp.square(prior_scale_diag)
    prop_var = jnp.square(prop_scale_diag)
    sum_vars = prior_var + prop_var
    scale_diag = (prior_scale_diag * prop_scale_diag) / jnp.sqrt(sum_vars)
    prior_loc = self.svm.dynamics_mean(prev_state, t)
    prop_loc = self.mus[t]
    loc = (prior_loc * prop_var + prop_loc * prior_var) / sum_vars
    return tfd.MultivariateNormalDiag(loc, scale_diag=scale_diag)


class TiltedSVMProposal(eqx.Module):

  prop: DiagCovSVMProposal

  def __init__(self,
          prop: DiagCovSVMProposal):
    self.prop = prop

  @abstractmethod
  def preprocess_obs(self, obs: Array) -> Array:
    pass

  @abstractmethod
  def tilt(self, latent: Array, obs: Array, t: int, num_timesteps: int) -> Scalar:
    pass

  def tilt_loss(self, key: PRNGKey, step: int, data: ArrayTree) -> Scalar:
    return 0.

  def make_propose_and_weight(self, step, obs, num_timesteps, propose_from_prior=False):
    tilt_obs = self.preprocess_obs(obs)

    def propose_and_weight(
            key: PRNGKey,
            prev_latent: Array,
            cur_obs: Array,
            t: int) -> Tuple[Array, Scalar, Scalar]:
      if not propose_from_prior:
        q_dist = self.prop.proposal_dist(prev_latent, t)
      else:
        q_dist = self.prop.svm.dynamics_dist(prev_latent, t)
      new_latent = q_dist.sample(seed=key)
      log_q_latent = q_dist.log_prob(new_latent)

      def run_tilt():
        log_r = self.tilt(new_latent, tilt_obs, t, num_timesteps)
        return log_r

      # Should zero out log_r after num_steps, but we anyways avoid computing it past T-1
      log_r = jax.lax.cond(
        t < num_timesteps - 1,
        lambda _: run_tilt(),
        lambda _: 0.,
        None
      )
      log_p_x = self.prop.svm.dynamics_dist(prev_latent, t).log_prob(new_latent)
      log_p_y_given_x = self.prop.svm.emission_dist(new_latent, t).log_prob(cur_obs)
      log_p_joint = log_p_x + log_p_y_given_x
      return new_latent, log_p_joint - log_q_latent, log_r

    return propose_and_weight


class WindowedTiltSVMProposal(TiltedSVMProposal):

  data_dim: int = eqx.static_field()
  window: int = eqx.static_field()

  def __init__(
          self,
          prop: DiagCovSVMProposal,
          data_dim: int,
          window: int):
    super().__init__(prop)
    self.window = window
    self.data_dim = data_dim

  def preprocess_obs(self, obs: Array) -> Array:
    # obs is [max_num_timesteps, data_dim]
    # we pad it to [max_num_timesteps + window + 1, data_dim] so that dynamic_slice does
    # not hit the end of the sequence.
    return jnp.pad(obs, ((0, self.window + 1), (0, 0)))

  @abstractmethod
  def score_window(self, latent: Array, obs: Array, t: int, mask: Array) -> Scalar:
    """Score the observations given latent. Computes p(x_{t+1:t+window} | z_t, x_t).

    Args:
      latent: The latent z_t, an Array of shape [latent_dim].
      obs: The observations x_{t:t+window}, and Array of shape [window + 1, data_dim],
      t: The timestep being scored, an int.
      mask: A 0/1 mask indicating which observations are not to be scored.
    Returns:
      The score, log p(x_{t+1:t+window} | z_t, x_t).
    """
    raise NotImplementedError("score_window not yet implemented.")

  def tilt(self, latent: Array, obs: Array, t: int, num_timesteps: int) -> Scalar:
    # confusingly, this selects the observations t, t+1, ... , t + window
    # which is of shape [window + 1, data_dim].
    obs_window = jax.lax.dynamic_slice(obs, (t, 0), (self.window + 1, self.data_dim))
    # mask must be made like this so that arange args are concrete values (can't include t).
    mask = jnp.arange(0, self.window) < num_timesteps - (t + 1)
    return self.score_window(latent, obs_window, t, mask)

  def tilt_loss(self, key: PRNGKey, step: int, data: Tuple[Array, Array, Array]) -> Scalar:
    del key
    del step
    latent, obs, length = data
    mask = jnp.arange(self.window) < length
    lp = self.score_window(latent, obs, 1, mask)
    return -lp / length


class QuadratureTiltSVMProposal(WindowedTiltSVMProposal):

  quad_degree: int = eqx.static_field()

  def __init__(
          self,
          prop: DiagCovSVMProposal,
          data_dim: int,
          quad_degree: int,
          window: int):
    super().__init__(prop, data_dim, window)
    self.quad_degree = quad_degree

  def preprocess_obs(self, obs: Array) -> Tuple[Array, Array]:
    # compute scales for obs window
    variance = self.prop.svm.dynamics_scale_diag() ** 2.
    latent_vars = [variance]
    phi_sq = jnp.tanh(self.prop.svm.raw_phi) ** 2.
    for i in range(2, 1 + self.window):
      latent_vars.append(phi_sq * latent_vars[-1] + variance)

    assert len(latent_vars) == self.window
    # [22, window]
    latent_var_diags = jnp.array(latent_vars).T
    # [22, window, window]
    latent_covs = jax.vmap(chain_cov)(latent_var_diags)
    latent_scale_Ls = jax.vmap(jnp.linalg.cholesky)(latent_covs)
    return latent_scale_Ls, super().preprocess_obs(obs)

  def tilt(self, latent: Array, obs: Array, t: int, num_timesteps: int) -> Scalar:
    latent_scale_Ls, obs = obs
    # confusingly, this selects the observations t, t+1, ... , t + window
    # which is of shape [window + 1, data_dim].
    obs_window = jax.lax.dynamic_slice(obs, (t, 0), (self.window + 1, self.data_dim))
    # mask must be made like this so that arange args are concrete values (can't include t).
    mask = jnp.arange(0, self.window) < num_timesteps - (t + 1)
    return self.score_window(latent, (latent_scale_Ls, obs_window), t, mask)

  def score_window(self, latent: Array, obs: Tuple[Array, Array], t: int, mask: Array) -> Scalar:
    """Score the observations given latent. Computes p(x_{t+1} | z_t, x_t).

    Args:
      latent: The latent z_t, an Array of shape [latent_dim].
      obs: The observations x_{t:t+1}, and Array of shape [2, data_dim],
      mask: A 0/1 mask indicating which observations are not to be scored.
    Returns:
      The score, log p(x_{t+1} | z_t, x_t).
    """
    latent_scale_Ls, obs_window = obs

    def int_ll(means, stddevs, obs, log_beta):
      # Accepts the means and stddevs of the latents in the window as [window] vectors.
      # Obs is also a [window] vector.
      # log_beta is [1]

      def emission_ll(z):
        emission_scale = jnp.exp(log_beta + (z / 2.))
        emission_dist = tfd.MultivariateNormalDiag(loc=0., scale_diag=emission_scale)
        return emission_dist.log_prob(obs)

      return log_integrate_multid_hg(self.quad_degree, emission_ll, means, stddevs)

    # compute latent means and scales for z_t+1
    latent_means = [self.prop.svm.dynamics_mean(latent, t + 1)]
    for i in range(2, 1 + self.window):
      latent_means.append(self.prop.svm.dynamics_mean(latent_means[-1], t + i))

    assert len(latent_means) == self.window
    # [22, window]
    latent_means = jnp.array(latent_means).T
    # [22, window]
    obs_window = obs_window[1:].T

    outs = jax.vmap(int_ll)(latent_means, latent_scale_Ls, obs_window, self.prop.svm.log_beta)
    outs = jnp.sum(outs, axis=0)
    return jnp.sum(jnp.nan_to_num(outs) * mask)


class BackwardsTilt(TiltedSVMProposal):

  rev_rnn: snax.RNN
  mlp: snax.MLP

  def __init__(
          self,
          key: PRNGKey,
          prop: DiagCovSVMProposal,
          rev_rnn: snax.RNN,
          mlp_hidden_dims: List[int],
          latent_dim):
    super().__init__(prop)
    self.rev_rnn = rev_rnn
    self.mlp = snax.MLP(
            key,
            self.rev_rnn.out_dim + latent_dim,
            mlp_hidden_dims + [1],
            act_fn=jax.nn.relu,
            final_act_fn=lambda x: x)

  def preprocess_obs(self, obs: Array) -> Array:
    """Preprocesses the observations by running RNN backwards over them.
    Args:
      obs: A [num_timesteps, ...] Array of observations
    Returns:
      outs: A [num_timesteps, ...] Array of outputs from the RNN.
    """
    # Runs an RNN reversed over the inputs, but returns the output in the original forward
    # order.
    _, outs = self.rev_rnn(obs, reverse=True)
    # We want h_{t+1} available at step t, so we shift outputs left by 1 and pad with zeros
    shifted_outs = jnp.pad(outs[1:], ((0, 1), (0, 0)))
    return shifted_outs

  def tilt(self, latent: Array, rev_rnn_out: Array, t: int, num_timesteps: int) -> Scalar:
    """
    Args:
      latent: A [data_dim] array containing the current latent state.
      rev_rnn_out: A [num_timesteps, rev_rnn.out_dim] Array containing the outputs of the
        reverse rnn in forward-time order. Should be outputs as returned from preprocess_obs.
      t: current timestep
      num_timesteps: Number of timetsteps.
    """
    del num_timesteps
    mlp_inputs = jnp.concatenate([rev_rnn_out[t], latent])
    mlp_out = self.mlp(mlp_inputs)
    return jnp.reshape(mlp_out, [])

  def tilt_seq(self, neg_xs: Array, pos_xs: Array, pos_ys: Array) -> Tuple[Array, Array]:
    """Computes the tilt logits for a full series of positive and negative examples.

    Args:
      neg_xs: A [num_timesteps, data_dim] array containing the 'negative latents'
        sampled from p(x_{1:T}).
      pos_xs: A [num_timesteps, data_dim] array containing the 'positive latents'
        sampled from p(x_{1:T}).
      pos_ys: A [num_timesteps, data_dim] array containing the positive observations sampled
        from p(y_{1:T}| x_{1:T} = pos_xs).

    Returns:
      pos_logits: A [num_timesteps] array of the tilt logits produced for the positive examples.
      neg_logits: A [num_timesteps] array of the tilt logits produced for the negative examples.
    """
    # [num_timesteps, rnn_out_dim]
    rnn_outs = self.preprocess_obs(pos_ys)
    # [num_timesteps, rnn_out_dim + latent_dim]
    pos_mlp_inputs = jnp.concatenate([rnn_outs, pos_xs], axis=1)
    neg_mlp_inputs = jnp.concatenate([rnn_outs, neg_xs], axis=1)

    # [num_timesteps, 1]
    pos_logits = jax.vmap(self.mlp)(pos_mlp_inputs)
    neg_logits = jax.vmap(self.mlp)(neg_mlp_inputs)
    return pos_logits, neg_logits

  def tilt_loss(self, key: PRNGKey, step: int, data: Tuple[Array, Array, Array]) -> Scalar:
    del key
    del step
    neg_xs, pos_xs, pos_ys = data
    pos_logits, neg_logits = self.tilt_seq(neg_xs, pos_xs, pos_ys)
    pos_lps = tfd.Bernoulli(logits=pos_logits).log_prob(1)
    neg_lps = tfd.Bernoulli(logits=neg_logits).log_prob(0)
    return - (jnp.mean(pos_lps) + jnp.mean(neg_lps)) / 2.  # definitely minus
