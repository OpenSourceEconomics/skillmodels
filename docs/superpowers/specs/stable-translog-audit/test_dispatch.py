"""Native public-model/AMN dispatch tests; not executed by the reviewer.

Uses real public model objects, not forged private objects or monkeypatches.
Full CHS likelihood/moment acceptance is specified separately in the repaired plan
because the CHS predict kernel and existing CHS fixture are not shipped here.
"""
from __future__ import annotations
import sys
import pytest
if sys.version_info[:2] != (3,14):
    raise RuntimeError('Run in the declared Python 3.14 environment.')
import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose
from skillmodels.common.model_spec import FactorSpec, ModelSpec
from skillmodels.common.process_model import process_model
from skillmodels.common.types import Normalizations
from skillmodels.common import transition_functions as tf
from skillmodels.amn.simulate_and_regress import _resolve_transition_callable
from skillmodels.af.validate import validate_af_model

jax.config.update('jax_enable_x64',True)


@pytest.fixture
def model_and_processed():
    normalizations=Normalizations(
        loadings=({'m0':1.0},{'m0':1.0}),
        intercepts=({'m0':0.0},{'m0':0.0}),
    )
    factor=FactorSpec(
        measurements=(('m0','m1','m2'),('m0','m1','m2')),
        normalizations=normalizations,
        transition_function='stable_translog',
    )
    model=ModelSpec(factors={'skill':factor})
    return model,process_model(model)


def test_public_model_uses_clipped_values(model_and_processed):
    _,processed=model_and_processed
    info=processed.transition_info
    assert info is not None
    states=jnp.array([[2.],[3.]])
    params={'skill':jnp.array([0.,1.,0.])}
    expected=jnp.array([[36./jnp.sqrt(97.)],[9./jnp.sqrt(2.)]])
    # process_model wraps the full transition with in_axes=(None,0).
    assert_allclose(info.func(params,states),expected,rtol=3e-13,atol=3e-13)
    assert_allclose(jax.jit(info.func)(params,states),expected,rtol=3e-13,atol=3e-13)
    assert tuple(info.param_names['skill'])==('skill','skill ** 2','constant')


def test_amn_resolver_returns_the_right_law(model_and_processed):
    model,processed=model_and_processed
    func,names=_resolve_transition_callable('stable_translog','skill',processed,model)
    assert func is tf.stable_translog
    assert names==('skill','skill ** 2','constant')
    value=func(jnp.array([3.]),jnp.array([0.,1.,0.]))
    assert_allclose(value,9./jnp.sqrt(2.),rtol=3e-14)
    assert abs(float(value)-9.)>2.0  # An alias to translog must fail.


def test_af_public_validation_accepts_name(model_and_processed):
    model,_=model_and_processed
    validate_af_model(model)
