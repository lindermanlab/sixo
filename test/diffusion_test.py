import jax
import jax.numpy as jnp

import sys
sys.path.append("../sixo")
from models import diffusion


def test_diffusion():
    model = diffusion.GaussianDiffusion(10)
    zs, x = model.sample_trajectory(jax.random.PRNGKey(0))
    lp = model.log_prob(zs, x)
    model.posterior(0, 0., x)
    model.lookahead(0, zs[0])
    model.marginal()
    assert jnp.all(jnp.isfinite(zs)) and jnp.isfinite(x) and jnp.isfinite(lp)
