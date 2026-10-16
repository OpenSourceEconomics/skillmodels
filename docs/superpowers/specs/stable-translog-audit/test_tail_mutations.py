"""Native tail/adjacent-float generators for F2. Not run by this reviewer."""
from __future__ import annotations
import math
import sys
from pathlib import Path
import pytest
if sys.version_info[:2]!=(3,14):
    raise RuntimeError('Native tests require the declared Python 3.14 environment.')
import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
from skillmodels.common import transition_functions as tf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit_oracles.reference import clip
jax.config.update('jax_enable_x64',True)


@pytest.mark.parametrize('scale,power',[(.5,4),(3.,4),(9.,8)])
def test_finite_power_overflow_neighborhood(scale,power):
    limit=np.finfo(np.float64).max
    threshold=scale*math.exp(math.log(limit)/power)
    samples=[0.,scale,np.nextafter(scale,0.),np.nextafter(scale,np.inf),50.,
             threshold/2,np.nextafter(threshold,0.),threshold,
             np.nextafter(threshold,np.inf),threshold*2,1e80,1e150,1e300]
    samples=sorted(set(v for x in samples for v in (x,-x) if np.isfinite(v)))
    x=jnp.array(samples)
    helper=jax.jit(lambda z:tf._soft_clip(z,scale,power))
    got=helper(x)
    expected=np.array([float(clip(v,scale,power)) for v in samples])
    assert bool(jnp.all(jnp.isfinite(got)))
    assert bool(jnp.all(jnp.diff(got)>=-2e-13*scale))
    assert_allclose(got,expected,rtol=8e-14,atol=8e-14*scale)
    f=jax.jit(tf.stable_translog,static_argnames=('scale','power'))
    p=jnp.array([0.,1.,0.])
    vals=jax.vmap(lambda z:f(jnp.array([z]),p,scale=scale,power=power))(x)
    assert_allclose(vals,expected**2,rtol=2e-13,atol=2e-13*scale**2)
    # Losing the nonlinear term must not be hidden by an enormous linear part.
    slopes=jax.vmap(jax.grad(lambda z:f(jnp.array([z]),jnp.array([1.,0.,0.]),
                                      scale=scale,power=power)))(x)
    assert bool(jnp.all(jnp.isfinite(slopes)))
    assert_allclose(slopes,1.,rtol=0,atol=2e-13)
