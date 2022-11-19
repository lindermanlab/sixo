from abc import abstractmethod
from typing import Tuple, List, Optional
import jax
import jax.numpy as jnp
from jax._src.random import KeyArray as PRNGKey

import equinox as eqx
import snax
from chex import Array, Scalar
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions


class GaussianDiffusion(eqx.Module):

  drift: Scalar
  variance: Scalar = eqx.static_field()
  num_time_steps: int = eqx.static_field()
  train: bool = eqx.static_field()

  def __init__(
          self,
          num_time_steps,
          drift=0.,
          variance=1.,
          train=False):
    self.drift = drift
    self.variance = variance  # change to log_var if going to train
    self.num_time_steps = num_time_steps
    self.train = train

  def b(self):
    """
    NOTE: it is important to understand that if `p` is training, then will have gradients through
    ``self.drift`` for any method using ``self.drift``, e.g. tilt `r` using ``self.lookahead``.
    This is likely the desired behavior in terms of, under fixed `r`, getting the parameters of
    `p` and `q` to be as good as possible (which likely means using "correct" gradients - up to
    dropping high variance terms)
    """
    if self.train:
      return self.drift
    else:
      return jax.lax.stop_gradient(self.drift)

  def dynamics_dist(self, prev_z, t):
    # `prev_z` at t=0 is enforced to be 0
    loc = jnp.where(t == 0, 0., prev_z)
    return tfd.Normal(loc=loc + self.b(), scale=jnp.sqrt(self.variance))

  def emission_dist(self, z):
    return tfd.Normal(loc=z + self.b(), scale=jnp.sqrt(self.variance))

  def sample_trajectory(self, key):
    """Samples a trajectory from the model.
    Args:
      key: A JAX PRNGKey.
    Returns:
      zs: latents, an Array of shape [num_timesteps].
      xs: observations, an Scalar
    """
    def scan_fn(carry, t):
      key, prev_state = carry
      key, subkey = jax.random.split(key)
      new_z = self.dynamics_dist(prev_state, t).sample(seed=subkey)
      return (key, new_z), new_z
    _, zs = jax.lax.scan(scan_fn, (key, 0.), jnp.arange(self.num_time_steps))
    x = self.emission_dist(zs[-1]).sample(seed=key)
    return zs, x

  def log_prob(self, zs: Array, x: Scalar):
    """Compute the joint prob of a sequence of latents and observations, p(z_{1:T}, x)."""
    def scan_fn(carry, inputs):
      cur_z, t = inputs
      prev_z, log_p = carry
      cur_log_p = self.dynamics_dist(prev_z, t).log_prob(cur_z)
      return (prev_z, log_p + cur_log_p), None
    (_, log_p), _ = jax.lax.scan(
        scan_fn,
        (0., 0.),  # initialize "z_{-1}" and cumulative log_p to 0.
        (zs, jnp.arange(self.num_time_steps)))
    log_p += self.emission_dist(zs[-1]).log_prob(x)
    return log_p

  def marginal(self):
    T = self.num_time_steps
    return tfd.Normal(loc=self.b() * (T + 1), scale=jnp.sqrt(self.variance * (T + 1)))

  def lookahead_bias(self, t):
    T = self.num_time_steps
    return self.b() * (T - t)

  def lookahead_var(self, t):
    T = self.num_time_steps
    return self.variance * (T - t)

  def lookahead(self, t, z):
    # t \in {0, ..., T - 1}
    return tfd.Normal(loc=z + self.lookahead_bias(t), scale=jnp.sqrt(self.lookahead_var(t)))

  def q_prev_z_weight(self, t):
    """NOTE: for t=0, there is no z previous so we could return anything
    """
    T = self.num_time_steps
    return (T - t) / (1 + T - t)

  def q_x_weight(self, t):
    T = self.num_time_steps
    return 1. / (1 + T - t)

  def q_var(self, t):
    T = self.num_time_steps
    return (T - t) / (1 + T - t) * self.variance

  def posterior(self, t, z_prev, x):
    z_prev = jnp.where(t == 0, 0., z_prev)
    q_mean = self.q_prev_z_weight(t) * z_prev + self.q_x_weight(t) * x
    q_std = jnp.sqrt(self.q_var(t))
    return tfd.Normal(q_mean, q_std)


class GaussianDiffusionWithProposal(eqx.Module):

  model: GaussianDiffusion

  def __init__(self, model: GaussianDiffusion):
    self.model = model

  @abstractmethod
  def proposal_dist(self, prev_z, x, t):
    raise NotImplementedError

  def propose_and_weight(self, key: PRNGKey, prev_z: Scalar, x: Scalar, t: int):
    q_dist = self.proposal_dist(prev_z, x, t)
    new_z = q_dist.sample(seed=key)
    log_q_z = q_dist.log_prob(new_z)
    log_p_z = self.model.dynamics_dist(prev_z, t).log_prob(new_z)
    log_p_x_given_z = jnp.where(t == self.model.num_time_steps - 1,
                                self.model.emission_dist(new_z).log_prob(x),
                                0.)
    log_p_z_x = log_p_z + log_p_x_given_z
    return new_z, log_p_z_x - log_q_z


class GaussianDiffusionWithLearntProposal(GaussianDiffusionWithProposal):
  """
  This version is used to actually learn the parameters
  """
  q_z_weights: Array
  q_x_weights: Array
  q_biases: Array
  q_log_vars: Array

  def __init__(self, model: GaussianDiffusion):
    super().__init__(model)
    # NOTE: first q_z_weight (i.e. weight on latent prior to z_0) is meaningless
    # but keeping all of these to have same length (and assumption is that z_{-1} = 0)
    self.q_z_weights = jnp.zeros(model.num_time_steps)
    self.q_x_weights = jnp.zeros(model.num_time_steps)
    self.q_biases = jnp.zeros(model.num_time_steps)
    self.q_log_vars = jnp.zeros(model.num_time_steps)

  def proposal_dist(self, prev_z, x, t):
    q_mean = self.q_z_weights[t] * prev_z + self.q_x_weights[t] * x + self.q_biases[t]
    return tfd.Normal(loc=q_mean, scale=jnp.sqrt(jnp.exp(self.q_log_vars[t])))


class GaussianDiffusionWithPosteriorProposal(GaussianDiffusionWithProposal):
  """
  This version will propose using the "true" posterior parameters
  i.e. q(z_t | z_{t-1}, x) = p(z_t | z_{t-1}, x)

  NOTE: this maybe relies on `drift` and `variance` parameters in
  ``GaussianDiffusion`` being static
  """
  def __init__(self, model: GaussianDiffusion):
    super().__init__(model)

  def proposal_dist(self, prev_z, x, t):
    return self.model.posterior(t, prev_z, x)


class GaussianDiffusionTilt(eqx.Module):
  """Simple super class for Diffusion tilts
  """
  num_time_steps: int = eqx.static_field()

  def __init__(self, num_time_steps: int):
    self.num_time_steps = num_time_steps

  @abstractmethod
  def __call__(self, x, z_t, t: int):
    raise NotImplementedError


class StandardTilt(GaussianDiffusionTilt):
  """
  """
  biases: Array
  log_vars: Array

  def __init__(self, num_time_steps: int, log_var_init: float = 0.):
    super().__init__(num_time_steps)
    # NOTE: Only first T-1 params are actually used (maybe should make these length T-1)
    # self.log_vars = jnp.zeros(num_time_steps)
    self.log_vars = jnp.ones(num_time_steps) * log_var_init
    self.biases = jnp.zeros(num_time_steps)

    self.num_time_steps = num_time_steps

  def score(self, x, z_t, t):
    return tfd.Normal(
      loc=z_t + self.biases[t], scale=jnp.sqrt(jnp.exp(self.log_vars[t]))).log_prob(x)

  def __call__(self, x, z_t, t: int):
    return self.score(x, z_t, t)


class TiltedGaussianDiffusion(eqx.Module):

  model: GaussianDiffusionWithProposal
  tilt: Optional[GaussianDiffusionTilt]

  def __init__(self, model: GaussianDiffusionWithProposal,
          tilt: Optional[GaussianDiffusionTilt]):
    self.model = model
    self.tilt = tilt

  def make_propose_and_weight(self, obs: Array, num_time_steps: int):
    def propose_and_weight(
            key: PRNGKey,
            prev_latent: Array,
            t: int) -> Tuple[Array, Scalar, Scalar]:
      # NOTE: likely written elsewhere, but it is *critical* that `prev_latent`=0 at t=0
      q_dist = self.model.proposal_dist(prev_latent, obs, t)
      new_latent = q_dist.sample(seed=key)
      log_q_latent = q_dist.log_prob(new_latent)

      if self.tilt is not None:
        log_r = jax.lax.cond(
          t < num_time_steps - 1,
          lambda _: self.tilt(obs, new_latent, t),
          lambda _: 0.,
          None
        )
      else:
        log_r = jax.lax.cond(
          t < num_time_steps - 1,
          lambda _: self.model.model.lookahead(t, new_latent).log_prob(obs),
          lambda _: 0.,
          None
        )
      p_latent = self.model.model.dynamics_dist(prev_latent, t).log_prob(new_latent)
      p_obs = jax.lax.cond(t == num_time_steps - 1,
                           lambda _: self.model.model.emission_dist(new_latent).log_prob(obs),
                           lambda _: 0.,
                           None)
      log_p_joint = p_latent + p_obs
      return new_latent, log_p_joint - log_q_latent, log_r
    return propose_and_weight


class MLPTilt(GaussianDiffusionTilt):

  mlp: snax.MLP

  def __init__(self, key: PRNGKey, num_timesteps: int, hdims: List[int]):
    super().__init__(num_timesteps)
    self.mlp = snax.MLP(key, 2, hdims + [3], jax.nn.relu)

  def __call__(self, x, z_t, t: int):
    inputs = jnp.stack([x, t])
    outs = self.mlp(inputs)
    a, b, c = jnp.split(outs, 3, axis=0)
    return jnp.squeeze(a * jnp.square(z_t) + b * z_t + c)

  def dre_tilt_loss(self, data: Tuple[Array, Array, Array]) -> Scalar:
    zs_pos, zs_neg, x = data
    loss = 0.
    for t in range(self.num_time_steps - 1):
      pos_logit = self.__call__(x, zs_pos[t], t)
      neg_logit = self.__call__(x, zs_neg[t], t)
      loss += tfd.Bernoulli(logits=pos_logit).log_prob(1)
      loss += tfd.Bernoulli(logits=neg_logit).log_prob(0)
    return - loss / (self.num_time_steps)
