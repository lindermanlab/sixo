import math
from functools import partial
from jax._src.random import KeyArray as PRNGKey
import jax
import jax.numpy as jnp
import equinox as eqx
import tensorflow_probability.substrates.jax as tfp
from typing import Callable, List, Union, Tuple
from chex import Array, Scalar, ArrayTree
from diffrax import diffeqsolve, Dopri5, ODETerm, PIDController, ConstantStepSize, Euler
from jax.scipy.special import logit
from util import pos_enc, gaussian_prod
from abc import abstractmethod
import snax

tf = tfp.tf2jax
tfd = tfp.distributions
tfd_e = tfp.experimental.distributions

# Set the integration requirements for the ODE solver.
RTOL = 1e-4
ATOL = 1e-4
MAX_STEPS = 100
VS = 10  # Voltage scaling.
# NOTE - the higher scaled g-gate value in Paninski allows bigger currents.
GS = 100  # Scaling of g_s parameter.
HH_EPS = 0.01
TAYLOR_EPS = 1e-01
HH_GATE_EPS = 1e-05

AnnealingScheduleFn = Callable[[int], float]
no_annealing = lambda _: 1.


def ode_solve(solver, stepsize_controller,
        t0: Scalar, t1: Scalar, state: Array, ode_term: ODETerm,
        initial_stepsize: float = None, max_steps: int = MAX_STEPS) -> Array:
  if initial_stepsize is None:
    initial_stepsize = (t1 - t0) / 10.
  return diffeqsolve(
            ode_term, solver,
            t0=t0, t1=t1, dt0=initial_stepsize,
            y0=state,
            stepsize_controller=stepsize_controller,
            max_steps=max_steps).ys[0]


dopri_solve = partial(ode_solve, Dopri5(), PIDController(rtol=RTOL, atol=ATOL))
multi_step_euler_solve = partial(ode_solve, Euler(), ConstantStepSize(compile_steps=False))


def single_step_euler_solve(t0, t1, state, ode_term):
  stepsize = t1 - t0
  return state + (stepsize * ode_term.vf(t0, state, None))


ODE_SOLVERS = {
        'dopri': dopri_solve,
        'multi_step_euler': multi_step_euler_solve,
        'single_step_euler': single_step_euler_solve
}


class HodgkinHuxley(eqx.Module):
  r"""A Hodgkin Huxley model.

  # NOTES:
  - Multiple compartments are specified through `data_dim > 1`.
    - No interaction between compartments.
  - Assumes that the `p`/model process variances are not being learned.
    - To change this requires some re-plumbing.
  - Models the noise process as diagonal.
    - This should probably be some block diagonal definition though precision matrices.
    - Affects the definition of `state_distribution`.

  Follows the HH definition and nomenclature in
  https://hodgkin-huxley-tutorial.readthedocs.io/en/latest/_static/Hodgkin%20Huxley.html

  Taken channel parameters from Huys & Paninski.

  The HH model defines a four-dimensional state space per compartment.
  The four states correspond to:
      $v \in R$:     Membrane potential [mV].
      $n \in [0, 1]: Relative potassium channel activation [no units].
      $m \in [0, 1]: Relative sodium channel activation [no units].
      $h \in [0, 1]: Relative sodium channel inactivation [no units].

  The potential changes as a function of the:
      $i_l \in R$: Leakage current from the potential difference between
                     intra- and extra-cellular potential [mA].
      $i_k \in R$: Ionic current through the potassium gates [mA].
      $i_n \in R$: Ionic current through the sodium gates [mA].

  The channel activations then change as an ODE based on the current potential,
  current activation, and a steady-state value.

  Due to the bounded nature of {n, m, h}, we transform and store their values in
  "inverse-sigmoid" space, i.e. we map from [0, 1] -> R. This means that we can add Gaussian
  noise to the quantity without worrying about straying into invalid space. This does mean
  that we are adding heteroscedastic noise in constrained space, but even this might be
  a boon -- since adding a bucketload of noise at particularly the lower end of the allowable
  range might not be a good idea. Typically the $n$ and $h$ gates are halfway open, and so
  their noise scale will be roughly the same, and (very-)roughly linear.
  The $m$ gate can have huge swings, and so this hetereoscedasticity may be a problem in
  certain places, but at least it won't cause the sampler to break.

  The (process) noise model we use is a combination of "current noise" and "subunit noise",
  as named by: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002247.
  There is Gaussian noise added to the potential at each step, which is equivalent to adding
  some unknown current at each point in time (except you get to tune this in the more
  tractable potential space). There is then Gaussian noise added to the channel activations
  at each time step (where this is done in the unconstrained/inverse-sigmoid space to avoid
  having to worry about truncated Gaussians or invalid states). This is equivilant to saying
  that the gate opens or closes in proportion to a non-linear transform of a further
  noise-corrupted membrane potential (not an entirely unreasonable intuitive model).

  The emission noise model is Gaussian white noise centred on the
  true underlying potential value.
  """

  # Learnable parameters.
  g_n: Array = eqx.static_field()
  e_n: Array = eqx.static_field()
  g_k: Array = eqx.static_field()
  e_k: Array = eqx.static_field()
  g_l: Array = eqx.static_field()
  e_l: Array = eqx.static_field()
  const_i_ext: Array

  # Dimensions.
  state_dim: int = eqx.static_field()
  n_compartments: int = eqx.static_field()

  # Means.
  v_0_mu: float = eqx.static_field()
  invsig_n_0_mu: float = eqx.static_field()
  invsig_m_0_mu: float = eqx.static_field()
  invsig_h_0_mu: float = eqx.static_field()

  # Variances.
  v_0_lvar: float = eqx.static_field()
  invsig_n_0_lvar: float = eqx.static_field()
  invsig_m_0_lvar: float = eqx.static_field()
  invsig_h_0_lvar: float = eqx.static_field()

  # Transition kernel widths.
  v_lvar: float = eqx.static_field()
  invsig_n_lvar: float = eqx.static_field()
  invsig_m_lvar: float = eqx.static_field()
  invsig_h_lvar: float = eqx.static_field()

  # Emission density width.
  y_lvar: float = eqx.static_field()

  c: float = eqx.static_field()
  dt: float = eqx.static_field()
  ode_solver_fn: Callable = eqx.static_field()

  process_raw_prec_L: Array = eqx.static_field()
  process_scale: Array = eqx.static_field()

  obs_subsample: int = eqx.static_field()

  def __init__(
          self,
          key: PRNGKey,
          data_dim: int,
          dt: float = 0.02,
          # was 0.1 for most things all g's. Right thing is to sample around true values for
          # the parameters we're training
          scale: Union[Array, float] = 0.5,
          y_scale=5.,
          ode_solver: str = 'dopri',
          obs_subsample: int = 1,
          i_ext_rel_error_init: float = None):
    """Construct a Hodgkin Huxley model.

      Args:
        key: A JAX PRNGKey
        data_dim: Number of observed (independent) neurons, `n_compartments / 4`
          since we assume all are observed.
        dt: A float, stepsize in HMM -- i.e. observation interval.
        scale: A float, the scale for the parameter initialization Gaussian distributions.
    """
    n_learnable_params = 1
    self.dt = dt
    # Four states per compartment.
    # This is required for all neurons to be observed.
    # Will need to pass in a `state_dim` or `n_compartments` parameter otherwise.
    self.state_dim = data_dim * 4
    self.n_compartments = data_dim
    n_comp = self.n_compartments
    self.obs_subsample = obs_subsample

    # Define a default potential so that we can initialize the
    # channel (in)activation states to reasonable quantities.
    default_v_0_mu = jnp.asarray(-65.0 / VS)  # [mV].

    # Means.
    self.v_0_mu = jnp.asarray(default_v_0_mu)                             # [mV / VS].
    self.invsig_n_0_mu = jnp.asarray(logit(_n_inf(VS * default_v_0_mu)))  # [].
    self.invsig_m_0_mu = jnp.asarray(logit(_m_inf(VS * default_v_0_mu)))  # [].
    self.invsig_h_0_mu = jnp.asarray(logit(_h_inf(VS * default_v_0_mu)))  # [].

    # Variances.
    self.v_0_lvar = jnp.asarray(jnp.log(jnp.square(25.0 / VS)))    # [mV / VS].
    self.invsig_n_0_lvar = jnp.asarray(jnp.log(jnp.square(1.0)))   # [].
    self.invsig_m_0_lvar = jnp.asarray(jnp.log(jnp.square(1.0)))   # [].
    self.invsig_h_0_lvar = jnp.asarray(jnp.log(jnp.square(1.0)))   # [].

    # Transition kernel widths.
    # [VS * mV] - applied every `dt`. Was 3 thurs morning
    self.v_lvar = jnp.log(jnp.square(3.0 * jnp.sqrt(dt) / VS))

    # NOTE: small transition noises here is comparable to effectively deterministic
    # transitions in the g's in Paninski
    # [] - applied every `dt`.
    self.invsig_n_lvar = jnp.asarray(jnp.log(jnp.square(0.1 * jnp.sqrt(dt))))
    # [] - applied every `dt`.
    self.invsig_m_lvar = jnp.asarray(jnp.log(jnp.square(0.1 * jnp.sqrt(dt))))
    # [] - applied every `dt`.
    self.invsig_h_lvar = jnp.asarray(jnp.log(jnp.square(0.1 * jnp.sqrt(dt))))

    # Emission density width.
    self.y_lvar = jnp.asarray(jnp.log(jnp.square(y_scale / VS)))  # [mV / VS].

    # Fix this to one since it is the ratio that is important.
    self.c = 1.0  # Get the units of this...

    # Build up the process scales and unconstrained precision matricies.
    self.process_scale = jnp.concatenate(
            (jnp.repeat(jnp.sqrt(jnp.exp(self.v_lvar)), n_comp),
             jnp.repeat(jnp.sqrt(jnp.exp(self.invsig_n_lvar)), n_comp),
             jnp.repeat(jnp.sqrt(jnp.exp(self.invsig_m_lvar)), n_comp),
             jnp.repeat(jnp.sqrt(jnp.exp(self.invsig_h_lvar)), n_comp)))

    assert self.process_scale.ndim == 1, "Scale is assumed to be diagonal."

    self.process_raw_prec_L = (jnp.eye(self.state_dim) *
            jnp.log(1.0 / jnp.square(self.process_scale)))

    g_n_mu = jnp.asarray(120.0)         # [mS/cm^2].
    e_n_mu = jnp.asarray(50.0)          # [mV].
    g_k_mu = jnp.asarray(20.0)          # [mS/cm^2].
    e_k_mu = jnp.asarray(-80.0)         # [mV].
    g_l_mu = jnp.asarray(30.0)          # [GS * mS/cm^2].  The GS is divided in `i_l`.
    e_l_mu = jnp.asarray(-55.4)         # [mV].
    const_i_ext_mu = jnp.asarray(13.0)  # [micro-A].

    # Sometimes we just want to pass a scale around which is the same for all.
    if type(scale) is not Array:
      scale = jnp.full([n_learnable_params], scale)

    assert isinstance(scale, Array)

    self.g_n = g_n_mu
    self.e_n = e_n_mu
    self.g_k = g_k_mu
    self.e_k = e_k_mu
    self.g_l = g_l_mu
    self.e_l = e_l_mu

    if i_ext_rel_error_init is not None:
      self.const_i_ext = jnp.asarray(const_i_ext_mu * (1 + i_ext_rel_error_init))
    else:
      self.const_i_ext = const_i_ext_mu

    self.ode_solver_fn = ODE_SOLVERS[ode_solver]

  def dynamics_scale_diag(self):
    return self.process_scale

  def initial_state_prior(self) -> tfd.Distribution:
    """Define the initial state distribution."""
    # Assume a Gaussian prior over the initial states.
    init_v_mean = self.v_0_mu
    init_v_scale = jnp.sqrt(jnp.exp(self.v_0_lvar))

    # Assume a Gaussian prior over the initial states in inverse-sigmoid space.
    init_invsig_n_mean = self.invsig_n_0_mu
    init_invsig_n_scale = jnp.sqrt(jnp.exp(self.invsig_n_0_lvar))

    # Assume a Gaussian prior over the initial states in inverse-sigmoid space.
    init_invsig_m_mean = self.invsig_m_0_mu
    init_invsig_m_scale = jnp.sqrt(jnp.exp(self.invsig_m_0_lvar))

    # Assume a Gaussian prior over the initial states in inverse-sigmoid space.
    init_invsig_h_mean = self.invsig_h_0_mu
    init_invsig_h_scale = jnp.sqrt(jnp.exp(self.invsig_h_0_lvar))

    # Repeat these.
    # Because we store the state as [v1, v2, ... ,vNC, n1, n2, ..., nNC, mXXX, hXXX],
    # repeat each of the scales. We need to duplicate the scales accordingly.
    init_v_mean_rep = jnp.full([self.n_compartments], init_v_mean)
    init_v_scale_rep = jnp.full([self.n_compartments], init_v_scale)
    init_invsig_n_mean_rep = jnp.full([self.n_compartments], init_invsig_n_mean)
    init_invsig_n_scale_rep = jnp.full([self.n_compartments], init_invsig_n_scale)
    init_invsig_m_mean_rep = jnp.full([self.n_compartments], init_invsig_m_mean)
    init_invsig_m_scale_rep = jnp.full([self.n_compartments], init_invsig_m_scale)
    init_invsig_h_mean_rep = jnp.full([self.n_compartments], init_invsig_h_mean)
    init_invsig_h_scale_rep = jnp.full([self.n_compartments], init_invsig_h_scale)

    # Compact these into an object.
    init_mean_rep = jnp.concatenate(
            (init_v_mean_rep,
             init_invsig_n_mean_rep,
             init_invsig_m_mean_rep,
             init_invsig_h_mean_rep))
    init_scale_rep = jnp.concatenate(
            (init_v_scale_rep,
             init_invsig_n_scale_rep,
             init_invsig_m_scale_rep,
             init_invsig_h_scale_rep))

    return tfd.MultivariateNormalDiag(init_mean_rep, init_scale_rep)

  def _dynamics_dist(self, prev_state: Array, t: int) -> tfd.Distribution:
    # Pull the unconstrained state apart.
    v, invsig_n, invsig_m, invsig_h = jnp.split(prev_state, 4)

    # Convert into linear space.
    n = jax.nn.sigmoid(invsig_n)
    m = jax.nn.sigmoid(invsig_m)
    h = jax.nn.sigmoid(invsig_h)

    # Recapitulate the constrained state.
    constrained_state = jnp.concatenate((v, n, m, h))

    # Build the integrator.
    ode_term = ODETerm(lambda t, y, _: self._dynamics_grads(t, y))
    new_state = self.ode_solver_fn(t, t + self.dt, constrained_state, ode_term)
    v_new, n_new, m_new, h_new = jnp.split(new_state, 4)

    #  Lets just enforce the constraint, i.e. clip(constrained_state, [0, 1]).
    n_new = jnp.clip(n_new, a_min=0.0 + HH_GATE_EPS, a_max=1.0 - HH_GATE_EPS)
    m_new = jnp.clip(m_new, a_min=0.0 + HH_GATE_EPS, a_max=1.0 - HH_GATE_EPS)
    h_new = jnp.clip(h_new, a_min=0.0 + HH_GATE_EPS, a_max=1.0 - HH_GATE_EPS)

    # Recapitulate the unconstrained state and the variance.
    state_mean = jnp.concatenate((v_new, logit(n_new), logit(m_new), logit(h_new)))

    # Pack these puppies into a tfd distribution.
    state_distribution = tfd.MultivariateNormalDiag(
            state_mean, self.process_scale)
    return state_distribution

  def dynamics_dist(self, prev_state: Array, t: int) -> tfd.Distribution:
    r"""
    Define `p(x_t | x_{t-1}, \theta)`.

    `prev_state` is passed in unconstrained space, converted into constrained space, and then
    converted back to unconstrained space. Noise is defined in unconstrained space.

    :param params:
    :param prev_state:  (Array, (4*N,)):    Unconstrained state.
    :param t:
    :return:
    """

    # If we are on t=0, call the `step_one` distribution.
    # This is essentially a no-op compared to the iterate function.
    state_distribution = jax.lax.cond(jnp.any(jnp.equal(t, 0)),
                                      lambda _: self.initial_state_prior(),
                                      lambda _: self._dynamics_dist(prev_state, t),
                                      None)
    return state_distribution

  def emission_dist(self, cur_state: Array) -> tfd.Distribution:
    r"""
    `v` is stored in linear-space, so no transform is required.
    :param cur_state:
    :param t:           not used.
    :return:
    """
    # Because `n_compartments == data_dim`, this picks off the `data_dim` potentials.
    v = cur_state[:self.n_compartments]  # Removed scaling in the emission distribution.  * VS
    y_dist = tfd.MultivariateNormalDiag(
            loc=v,
            scale_identity_multiplier=jnp.sqrt(jnp.exp(self.y_lvar)))
    return y_dist

  def emission_log_prob(self, state: Array, obs: Array, t: int) -> Scalar:
    return jax.lax.cond(
            jnp.any(t % self.obs_subsample == 0),
            lambda so: self.emission_dist(so[0]).log_prob(so[1]),
            lambda _: 0.,
            (state, obs))

  def one_step_dist(self, prev_state: Array, t: int) -> tfd.Distribution:
    joint = tfd.JointDistributionNamed(dict(
        x=self.dynamics_dist(prev_state, t),
        y=lambda x: self.emission_dist(x)
    ))
    return joint

  def sample_trajectory(
          self,
          key: PRNGKey,
          num_timesteps: int,
          obs_subsample: int = None) -> Tuple[Array, Array]:
    key, subkey = jax.random.split(key)
    init_state = self.initial_state_prior().sample(seed=subkey)

    # This codepath will be executed most times.
    if obs_subsample is None:
      obs_subsample = self.obs_subsample

    def scan_fn(carry, t):
      key, prev_state = carry
      key, subkey = jax.random.split(key)
      samples = self.one_step_dist(prev_state, t).sample(seed=subkey)

      masked_obs = jax.lax.cond(jnp.any(t % obs_subsample == 0),
                                lambda _: samples['y'],
                                lambda _: samples['y'] * jnp.nan,  # Mask out those observations.
                                None)

      return (key, samples['x']), (samples['x'], masked_obs)

    _, samples = jax.lax.scan(scan_fn,
            (key, init_state),
            jnp.arange(num_timesteps),
            length=num_timesteps)

    xs, ys = samples

    return (xs, ys)

  def log_prob(self, xs: Array, ys: Array) -> Scalar:
    seq_len = xs.shape[0]
    # NOTE - this is a dummy variable. It is ignored by dynamics.
    init_x = jnp.zeros([self.state_dim])

    def scan_fn(carry, cur_state):
      prev_x, log_p = carry
      cur_x, cur_y, t = cur_state

      def _score_no_obs():
        p_x_dist = self.dynamics_dist(prev_x, t)
        log_p_x = p_x_dist.log_prob(cur_x)
        return log_p_x

      def _score_obs():
        p_x_y_dist = self.one_step_dist(prev_x, t)
        log_p_x_y = p_x_y_dist.log_prob({'x': cur_x, 'y': cur_y})
        return log_p_x_y

      # If there is a NaN in the observation then
      # it is a missing observation as so we don't reweight.
      cur_log_p = jax.lax.cond(jnp.any(jnp.isnan(cur_y)),
                               lambda *_: _score_no_obs(),
                               lambda *_: _score_obs(),
                               None)

      return (cur_x, log_p + cur_log_p), None

    (_, log_p), _ = jax.lax.scan(
        scan_fn,
        (init_x, 0.),
        (xs, ys, jnp.arange(seq_len)),
        length=seq_len)

    return log_p

  def _dynamics_grads(self, t, state):
    # Pull the state apart
    v_raw, n, m, h = jnp.split(state, 4)

    # We need to convert v back in to mV.
    v = v_raw * VS

    # Compute the gradients of each of the states.
    dv = (self.const_i_ext - _i_n(self.g_n, self.e_n, v, m, h)
                           - _i_k(self.g_k, self.e_k, v, n)
                           - _i_l(self.g_l, self.e_l, v)) / self.c
    dn = (_alpha_n(v) * (1.0 - n)) - (_beta_n(v) * n)
    dm = (_alpha_m(v) * (1.0 - m)) - (_beta_m(v) * m)
    dh = (_alpha_h(v) * (1.0 - h)) - (_beta_h(v) * h)

    # Convert dv back to raw.
    dv_raw = dv / VS

    return jnp.concatenate((dv_raw, dn, dm, dh))


def _n_inf(v: float) -> float:
  """Steady state n activation."""
  return _alpha_n(v) / (_alpha_n(v) + _beta_n(v))


def _m_inf(v: float) -> float:
  """Steady state m activation."""
  return _alpha_m(v) / (_alpha_m(v) + _beta_m(v))


def _h_inf(v: float) -> float:
  """Steady state h activation."""
  return _alpha_h(v) / (_alpha_h(v) + _beta_h(v))


def _alpha_m(v: float) -> float:
  """Channel gating kinetics. Functions of membrane voltage."""

  def ratio_fn(_v):
    den = (1.0 - jnp.exp(-(_v + 35.0) / 10.0))
    ratio = (_v + 35.0) / den
    return ratio

  def taylor_ratio_fn(_v):
    limit_pnt_at_35 = 10.0
    limit_grd_at_35 = 0.5
    return limit_pnt_at_35 + (limit_grd_at_35 * (_v - (- 35.0)))

  ratio = jax.lax.cond(jnp.any(jnp.abs(v + 35.0) < TAYLOR_EPS),
                       lambda _v: taylor_ratio_fn(_v),
                       lambda _v: ratio_fn(_v),
                       v)

  return 0.1 * ratio


def _alpha_n(v: float) -> float:
  """Channel gating kinetics. Functions of membrane voltage."""

  def ratio_fn(_v):
    den = (1.0 - jnp.exp(-(v + 55.0) / 10.0))
    ratio = (v + 55.0) / den
    return ratio

  def taylor_ratio_fn(_v):
    limit_pnt_at_55 = 10.0
    limit_grd_at_55 = 0.5
    return limit_pnt_at_55 + (limit_grd_at_55 * (_v - (- 55.0)))

  ratio = jax.lax.cond(jnp.any(jnp.abs(v + 55.0) < TAYLOR_EPS),
                       lambda _v: taylor_ratio_fn(_v),
                       lambda _v: ratio_fn(_v),
                       v)

  return 0.01 * ratio


def _alpha_h(v: float) -> float:
  """Channel gating kinetics. Functions of membrane voltage."""
  return 0.07 * jnp.exp(-(v + 50.0) / 20.0)


def _beta_m(v: float) -> float:
  """Channel gating kinetics. Functions of membrane voltage."""
  return 4.0 * jnp.exp(-(v + 65.0) / 18.0)


def _beta_h(v: float) -> float:
  """Channel gating kinetics. Functions of membrane voltage."""
  den = (jnp.exp(-(v + 35.0) / 10.0) + 1.0)
  return 1.0 / den


def _beta_n(v: float) -> float:
  """Channel gating kinetics. Functions of membrane voltage."""
  return 0.125 * jnp.exp(-(v + 65.0) / 80.0)


def _i_n(g_n: float, e_n: float, v: float, m: float, h: float) -> float:
  """Membrane current (in uA/cm^2). Sodium (Na = element name)."""
  return g_n * (m**3) * h * (v - e_n)


def _i_k(g_k: float, e_k: float, v: float, n: float) -> float:
  """Membrane current (in uA/cm^2). Potassium (K = element name)."""
  return g_k * (n**4) * (v - e_k)


def _i_l(g_l: float, e_l: float, v: float) -> float:
  """Membrane current (in uA/cm^2). Leak."""
  return (g_l / GS) * (v - e_l)


class HHProposal(eqx.Module):

  hh: HodgkinHuxley

  def __init__(self, hh: HodgkinHuxley):
    self.hh = hh

  @abstractmethod
  def make_propose_and_weight(
          self,
          step: int,
          obs: Array,
          num_timesteps: int):
    pass


class BootstrapHHProposal(HHProposal):
  """HodgkinHuxley model that proposes from the prior."""

  def make_propose_and_weight(
          self,
          step: int,
          obs: Array,
          num_timesteps: int):

    def propose_and_weight(
            key: PRNGKey,
            prev_state: Array,
            cur_obs: Array,
            t: int) -> Tuple[Array, Scalar]:
      x_dist = self.hh.dynamics_dist(prev_state, t)
      new_x = x_dist.sample(seed=key)
      log_weight = self.hh.emission_log_prob(new_x, cur_obs, t)
      return new_x, log_weight

    return propose_and_weight


class ParametricHHProposal(HHProposal):

  proposal_mlp: snax.MLP

  min_prop_scale_diag: float = eqx.static_field()
  pos_enc_dim: int = eqx.static_field()
  pos_enc_base: float = eqx.static_field()
  resq_type: str = eqx.static_field()

  def __init__(
          self,
          key: PRNGKey,
          hh: HodgkinHuxley,
          mlp_hdims: List[int],
          obs_encoder_dim: int,
          pos_enc_dim: int = 16,
          pos_enc_base: float = 10.,
          activation_fn=jax.nn.relu,
          W_init=jax.nn.initializers.glorot_uniform(),
          b_init=jax.nn.initializers.zeros,
          resq_type='mean'):
    super().__init__(hh)
    self.pos_enc_dim = pos_enc_dim
    self.pos_enc_base = pos_enc_base
    self.min_prop_scale_diag = 1e-8  # Currently fixed (no parameter).
    self.resq_type = resq_type
    self.proposal_mlp = snax.MLP(
      key,
      hh.state_dim + obs_encoder_dim + (2 * pos_enc_dim),  # HH state + observation.
      mlp_hdims + [2 * hh.state_dim],
      activation_fn, W_init=W_init, b_init=b_init)

  @abstractmethod
  def encode_obs(self, obs: Array, num_timesteps: int) -> Array:
    pass

  def make_propose_and_weight(
          self,
          step: int,
          obs: Array,
          num_timesteps: int):

    real_obs = obs[::self.hh.obs_subsample]
    encoded_obs = self.encode_obs(real_obs, real_obs.shape[0])
    pos_encs = pos_enc(
            num_timesteps,
            self.pos_enc_dim,
            base=self.pos_enc_base,
            period=self.hh.obs_subsample,
            flip=True)

    def propose_and_weight(
            key: PRNGKey,
            prev_state: Array,
            cur_obs: Array,
            t: int) -> Tuple[Array, Scalar]:
      # Run the prior.
      prior_dist = self.hh.dynamics_dist(prev_state, t)
      prior_loc = prior_dist.mean()
      prior_var_diag = prior_dist.variance()
      prior_scale_diag = jnp.sqrt(prior_var_diag)
      # Construct the proposal MLP input which includes:
      #  * position encoding
      #  * current encoded obs
      #  * previous state
      cur_pos_enc = pos_encs[t]
      cur_enc_obs = encoded_obs[t // self.hh.obs_subsample]
      mlp_in = jnp.concatenate([prev_state, cur_enc_obs, cur_pos_enc], axis=0)
      # Run the MLP
      mlp_out = self.proposal_mlp(mlp_in)
      prop_loc, prop_log_scale = jnp.split(mlp_out, 2)
      prop_scale_diag = jnp.maximum(jnp.exp(prop_log_scale), self.min_prop_scale_diag)
      if "_sg" in self.resq_type:
        prior_loc = jax.lax.stop_gradient(prior_loc)
        prior_scale_diag = jax.lax.stop_gradient(prior_scale_diag)
      # Construct the proposal distribution using the selected resq method
      if self.resq_type == 'none':
        prop_dist = tfd.MultivariateNormalDiag(
                loc=prop_loc,
                scale_diag=prop_scale_diag)
      elif self.resq_type in ['mean', 'mean_sg']:
        prop_dist = tfd.MultivariateNormalDiag(
                loc=prop_loc + prior_loc,
                scale_diag=prop_scale_diag)
      elif self.resq_type in ['full', 'full_sg']:
        loc, log_scale_diag = gaussian_prod(
                prior_loc, jnp.log(prior_scale_diag), prop_loc, prop_log_scale)
        prop_dist = tfd.MultivariateNormalDiag(
                loc=loc,
                scale_diag=jnp.exp(log_scale_diag))
      else:
        raise ValueError(
          "resq_type was {self.resq_type}, must be one of 'none', 'mean(_sg)', or 'full(_sg)'")

      # Sample from proposal, compute importance weights
      new_x = prop_dist.sample(seed=key)
      log_q_x = prop_dist.log_prob(new_x)
      log_p_x = prior_dist.log_prob(new_x)
      # Only compute emission prob if there is actually an observation.
      log_p_y_given_x = self.hh.emission_log_prob(new_x, cur_obs, t)
      log_weight = log_p_x + log_p_y_given_x - log_q_x
      return new_x, log_weight

    return propose_and_weight

  def initial_distribution(self):
    return self.hh.initial_state_prior()


class FilteringHHProposal(ParametricHHProposal):
  """HodgkinHuxley model with filtering proposal."""

  obs_encoder_rnn: snax.RNN

  def __init__(
          self,
          key: PRNGKey,
          hh: HodgkinHuxley,
          obs_encoder_rnn: snax.RNN,
          mlp_hdims: List[int],
          pos_enc_dim: int = 16,
          pos_enc_base: float = 10.,
          activation_fn=jax.nn.relu,
          W_init=jax.nn.initializers.glorot_uniform(),
          b_init=jax.nn.initializers.zeros,
          resq_type='mean'):
    self.obs_encoder_rnn = obs_encoder_rnn
    super().__init__(
            key, hh, mlp_hdims, self.obs_encoder_rnn.out_dim,
            pos_enc_dim=pos_enc_dim,
            pos_enc_base=pos_enc_base,
            activation_fn=activation_fn,
            W_init=W_init,
            b_init=b_init,
            resq_type=resq_type)

  def encode_obs(self, obs: Array, unused_num_timesteps: int) -> Array:
    _, outs = self.obs_encoder_rnn(obs)
    return outs


class SmoothingHHProposal(ParametricHHProposal):
  """HodgkinHuxley model with filtering proposal."""

  obs_encoder_rnn: snax.BiRNN

  def __init__(
          self,
          key: PRNGKey,
          hh: HodgkinHuxley,
          obs_encoder_rnn: snax.BiRNN,
          mlp_hdims: List[int],
          pos_enc_dim: int = 16,
          pos_enc_base: float = 10.,
          activation_fn=jax.nn.relu,
          W_init=jax.nn.initializers.glorot_uniform(),
          b_init=jax.nn.initializers.zeros,
          resq_type='mean'):
    self.obs_encoder_rnn = obs_encoder_rnn
    super().__init__(
            key, hh, mlp_hdims, self.obs_encoder_rnn.out_dim,
            pos_enc_dim=pos_enc_dim,
            pos_enc_base=pos_enc_base,
            activation_fn=activation_fn,
            W_init=W_init,
            b_init=b_init,
            resq_type=resq_type)

  def encode_obs(self, obs: Array, num_timesteps: int) -> Array:
    _, outs = self.obs_encoder_rnn(obs, num_timesteps)
    return outs


class TiltedHHProposal(eqx.Module):
  """
  Abstract base class.

  Implements the
  """

  prop: HHProposal
  tilt_anneal_sched: AnnealingScheduleFn = eqx.static_field()

  def __init__(self,
               prop: HHProposal,
               tilt_anneal_sched: AnnealingScheduleFn = no_annealing):
    self.prop = prop
    self.tilt_anneal_sched = tilt_anneal_sched

  @abstractmethod
  def preprocess_obs(self, obs: Array) -> Array:
    pass

  @abstractmethod
  def tilt(self, latent: Array, obs: Array, t: int, num_timesteps: int) -> Scalar:
    pass

  def tilt_loss(self, key: PRNGKey, step: int, data: ArrayTree) -> Scalar:
    raise NotImplementedError("Must be implemented by child")

  def make_propose_and_weight(self, step, obs, num_timesteps):
    tilt_obs = self.preprocess_obs(obs)
    prop_p_and_w = self.prop.make_propose_and_weight(step, obs, num_timesteps)
    num_obs = math.ceil(num_timesteps / self.prop.hh.obs_subsample)
    last_obs_ind = (num_obs - 1) * self.prop.hh.obs_subsample

    def propose_and_weight(
            key: PRNGKey,
            prev_latent: Array,
            cur_obs: Array,
            t: int) -> Tuple[Array, Scalar, Scalar]:

      new_latent, log_weight = prop_p_and_w(key, prev_latent, cur_obs, t)

      def run_tilt():
        log_r = self.tilt(new_latent, tilt_obs, t, num_timesteps)
        return log_r * self.tilt_anneal_sched(step)

      # Zero out the tilt starting at the last observation
      log_r = jax.lax.cond(
        jnp.any(t < last_obs_ind),
        lambda _: run_tilt(),
        lambda _: 0.,
        None
      )

      return new_latent, log_weight, log_r

    return propose_and_weight


class BackwardsTilt(TiltedHHProposal):

  rev_rnn: snax.RNN
  mlp: snax.MLP
  pos_enc_dim: int = eqx.static_field()
  pos_enc_base: float = eqx.static_field()

  def __init__(
          self,
          key: PRNGKey,
          prop: HHProposal,
          rev_rnn: snax.RNN,
          mlp_hidden_dims: List[int],
          latent_dim,
          tilt_anneal_sched: AnnealingScheduleFn = no_annealing,
          pos_enc_dim: int = 4,
          pos_enc_base: float = 4.):
    super().__init__(prop, tilt_anneal_sched=tilt_anneal_sched)
    self.rev_rnn = rev_rnn
    self.pos_enc_dim = pos_enc_dim
    self.pos_enc_base = pos_enc_base

    # w_init = lambda *args: (0.01 * jax.nn.initializers.glorot_normal()(*args))

    mlp_in_dim = self.rev_rnn.out_dim + latent_dim + (pos_enc_dim * 2)
    self.mlp = snax.MLP(
      key,
      mlp_in_dim,
      mlp_hidden_dims + [1],
      act_fn=jax.nn.relu,
      final_act_fn=lambda x: x)
    # W_init=w_init)

  def preprocess_obs(self, obs: Array) -> Array:
    """Preprocesses the observations by running RNN backwards over them.
    Args:
      obs: A [num_timesteps, ...] Array of observations
    Returns:
      outs: A [num_timesteps, ...] Array of outputs from the RNN.
    """
    # We are going to linearly interpolate between states.

    # Data:       [num_timesteps, obs_dim].
    # valid_data: [num_valid_obs, obs_dim].
    valid_data = obs[::self.prop.hh.obs_subsample]

    # Explicitly pull out shapes.
    obs_subsamp = self.prop.hh.obs_subsample
    # num_valid_obs = len(valid_data)
    # encoder_outdim = self.rev_rnn.cells[-1].out_dim

    # Runs an RNN reversed over the inputs, but returns the output in the original forward
    # order.
    # [num_valid_obs, encoder_outdim].
    _, outs = self.rev_rnn(valid_data, reverse=True)

    # We want to discard the first encoding as we don't use it.
    # [num_valid_obs - 1, encoder_outdim].
    nxt_output = outs[1:]

    # Repeat this representation.
    # [num_timesteps // obs_subsamp, encoder_outdim].
    repeated_output = jnp.repeat(nxt_output, obs_subsamp, axis=0)
    # Append zeros for timesteps past the last observation.
    enc_outdim = repeated_output.shape[0]
    num_timesteps = obs.shape[0]
    padding_size = num_timesteps - enc_outdim
    out = jnp.pad(repeated_output, ((0, padding_size), (0, 0)))

    pe = pos_enc(num_timesteps, self.pos_enc_dim,
            base=self.pos_enc_base, period=obs_subsamp, flip=True)
    out = jnp.concatenate([out, pe], axis=1)

    return out

  def tilt(self, latent: Array, rev_rnn_out: Array, t: int, num_timesteps: int) -> Scalar:
    """
    Args:
      latent: A [data_dim] array containing the current latent state.
      rev_rnn_out: A [num_timesteps, rev_rnn.out_dim] Array containing the outputs of the
        reverse rnn in forward-time order. Should be outputs as returned from preprocess_obs.
      t: current timestep
      num_timesteps: Number of timetsteps.
    """
    # FIXME: it seems that MLP needs to know how many timesteps away the observation is
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

    num_timesteps = pos_logits.shape[0]
    obs_subsamp = self.prop.hh.obs_subsample
    num_obs = math.ceil(num_timesteps / obs_subsamp)
    last_obs_ind = (num_obs - 1) * obs_subsamp
    return pos_logits[:last_obs_ind], neg_logits[:last_obs_ind]

  def tilt_loss(self, key: PRNGKey, step: int, data: Tuple[Array, Array, Array]) -> Scalar:
    del key
    del step
    neg_xs, pos_xs, pos_ys = data
    pos_logits, neg_logits = self.tilt_seq(neg_xs, pos_xs, pos_ys)
    pos_lps = tfd.Bernoulli(logits=pos_logits).log_prob(1)
    neg_lps = tfd.Bernoulli(logits=neg_logits).log_prob(0)
    return - (jnp.mean(pos_lps) + jnp.mean(neg_lps)) / 2.  # definitely minus
