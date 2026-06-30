"""Guards that make the AF calendar adapter fail loudly on unsupported models.

The shared integrand assembles the latent vector as all dynamic-state factors followed
by all reconstructed-endogenous factors, and the adapter is only defined for endogenous
factors that are reconstructed (no initial distribution). Two configurations would
otherwise be estimated silently against the wrong model:

- an endogenous factor that still carries an initial distribution taking the
  source-investment calendar (it is not a reconstructed investment);
- a public factor order that interleaves an endogenous factor before a dynamic-state
  factor, which transposes loading columns relative to the assembled latent vector.

Both must raise rather than estimate the wrong model.
"""

import pytest

from skillmodels.af.step_layout import AFFactorRole
from skillmodels.af.transition_period import (
    _factor_infos_from_spec,
    _fail_if_endogenous_precedes_state,
    _fail_if_unsupported_adapter_measurements,
)
from skillmodels.common.measurement_models import ProbitMeasurement
from skillmodels.common.model_spec import FactorSpec, ModelSpec, Normalizations


def _spec(*, has_initial_distribution: bool) -> ModelSpec:
    return ModelSpec(
        factors={
            "skills": FactorSpec(
                measurements=(("s1",), ("s1",)),
                normalizations=Normalizations(
                    loadings=({"s1": 1}, {"s1": 1}), intercepts=({"s1": 0}, {"s1": 0})
                ),
                transition_function="linear",
            ),
            "investment": FactorSpec(
                measurements=(("i1",), ("i1",)),
                normalizations=Normalizations(
                    loadings=({"i1": 1}, {"i1": 1}), intercepts=({"i1": 0}, {"i1": 0})
                ),
                transition_function="linear",
                is_endogenous=True,
                has_initial_distribution=has_initial_distribution,
            ),
        },
    )


def test_factor_infos_marks_reconstructed_endogenous() -> None:
    infos = _factor_infos_from_spec(
        _spec(has_initial_distribution=False), endogenous_factors=("investment",)
    )
    roles = {info.name: info.role for info in infos}
    assert roles["investment"] == AFFactorRole.ENDOGENOUS
    assert roles["skills"] == AFFactorRole.DYNAMIC


def test_factor_infos_rejects_endogenous_with_initial_distribution() -> None:
    with pytest.raises(ValueError, match="has_initial_distribution"):
        _factor_infos_from_spec(
            _spec(has_initial_distribution=True), endogenous_factors=("investment",)
        )


def test_fail_if_endogenous_precedes_state_accepts_state_first() -> None:
    # State before endogenous: matches the assembled [theta_states, inv_endog] order.
    _fail_if_endogenous_precedes_state(("skills", "investment"), ("investment",))


def test_fail_if_endogenous_precedes_state_rejects_interleaving() -> None:
    with pytest.raises(ValueError, match="precede"):
        _fail_if_endogenous_precedes_state(("investment", "skills"), ("investment",))


def test_adapter_measurements_accept_plain_gaussian_non_crossloaded() -> None:
    # The clean CNLSY-style model (Gaussian, one factor per measurement) is accepted.
    _fail_if_unsupported_adapter_measurements(_spec(has_initial_distribution=False))


def test_adapter_measurements_reject_cross_loaded_measurement() -> None:
    # "shared" loads on both skills and investment -> not yet supported by the adapter.
    model = ModelSpec(
        factors={
            "skills": FactorSpec(
                measurements=(("s1", "shared"), ("s1",)),
                normalizations=Normalizations(
                    loadings=({"s1": 1}, {"s1": 1}), intercepts=({"s1": 0}, {"s1": 0})
                ),
                transition_function="linear",
            ),
            "investment": FactorSpec(
                measurements=(("i1", "shared"), ("i1",)),
                normalizations=Normalizations(
                    loadings=({"i1": 1}, {"i1": 1}), intercepts=({"i1": 0}, {"i1": 0})
                ),
                transition_function="linear",
                is_endogenous=True,
                has_initial_distribution=False,
            ),
        },
    )
    with pytest.raises(ValueError, match="cross-load"):
        _fail_if_unsupported_adapter_measurements(model)


def test_adapter_measurements_reject_non_gaussian_family() -> None:
    model = ModelSpec(
        factors={
            "skills": FactorSpec(
                measurements=(("s1",), ("s1",)),
                normalizations=Normalizations(
                    loadings=({"s1": 1}, {"s1": 1}), intercepts=({"s1": 0}, {"s1": 0})
                ),
                transition_function="linear",
            ),
            "investment": FactorSpec(
                measurements=(("i1",), ("i1",)),
                normalizations=Normalizations(
                    loadings=({"i1": 1}, {"i1": 1}), intercepts=({"i1": 0}, {"i1": 0})
                ),
                transition_function="linear",
                is_endogenous=True,
                has_initial_distribution=False,
            ),
        },
        measurement_models={"i1": ProbitMeasurement()},
    )
    with pytest.raises(ValueError, match="Gaussian"):
        _fail_if_unsupported_adapter_measurements(model)
