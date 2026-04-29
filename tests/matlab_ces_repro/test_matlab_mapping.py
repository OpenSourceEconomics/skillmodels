"""Unit tests for the MATLAB result parser."""

from pathlib import Path

import numpy as np
import pytest

from .matlab_mapping import (
    ces_to_skillmodels_gammas,
    load_matlab_results,
    translate_matlab_ces_production,
)

_DEFAULT_RESULTS_DIR = Path("/home/hmg/sciebo/Skill estimation/Application/Results")


def test_ces_to_skillmodels_gammas_sums_to_one() -> None:
    gamma_skills, gamma_inv, _ = ces_to_skillmodels_gammas(delta=0.7, phi=0.3)
    assert np.isclose(gamma_skills + gamma_inv, 1.0)
    assert np.isclose(gamma_skills, 0.7)
    assert np.isclose(gamma_inv, 0.3)


def test_ces_to_skillmodels_gammas_rejects_non_positive_sum() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        ces_to_skillmodels_gammas(delta=-0.3, phi=0.2)


def test_translate_matlab_ces_production_roundtrip() -> None:
    """At test points, skillmodels' log_ces must equal MATLAB's CES.

    Evaluate both forms at several ``(theta, X)`` test points and assert
    they differ by exactly the ``level_shift`` returned by the helper.
    """
    delta, phi, rho = 0.4, 0.7, 1.3
    gamma_skills, gamma_inv, phi_skm, level_shift = translate_matlab_ces_production(
        delta=delta, phi=phi, rho=rho
    )
    # ``f_skm`` below is skillmodels' log_ces output (normalised form) and
    # ``f_matlab`` is MATLAB's CES output (unnormalised). The helper's
    # ``level_shift`` is what you have to add to ``f_skm`` to recover
    # ``f_matlab``.
    for theta, x in [(0.1, 0.2), (-0.5, 1.0), (1.5, -0.3), (0.0, 0.0)]:
        f_skm = (1.0 / phi_skm) * np.log(
            gamma_skills * np.exp(rho * theta) + gamma_inv * np.exp(rho * x)
        )
        f_matlab = (1.0 / rho) * np.log(
            delta * np.exp(rho * theta) + phi * np.exp(rho * x)
        )
        np.testing.assert_allclose(f_matlab, f_skm + level_shift, rtol=0, atol=1e-12)


def test_translate_matlab_ces_production_rejects_non_positive_sum() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        translate_matlab_ces_production(delta=-0.5, phi=0.2, rho=1.0)


def test_translate_matlab_ces_production_carries_a_constant() -> None:
    # With delta + phi = 1 the ``(1 / rho) * log(delta + phi)`` term is
    # zero, so the returned ``level_shift`` equals ``a_const`` exactly.
    _, _, _, level_shift = translate_matlab_ces_production(
        delta=0.3, phi=0.7, rho=1.0, a_const=0.5
    )
    assert np.isclose(level_shift, 0.5)


@pytest.mark.skipif(
    not (_DEFAULT_RESULTS_DIR / "Results_AF_One_Normal_CES.mat").exists(),
    reason="MATLAB CES result file not available",
)
def test_load_matlab_results_ces() -> None:
    res = load_matlab_results(
        _DEFAULT_RESULTS_DIR / "Results_AF_One_Normal_CES.mat",
        variant="ces",
    )
    assert res.n_obs == 1403
    assert res.n_halton_nodes == 20000
    assert res.initial.var_diag.shape == (4,)
    assert res.initial.correlations.shape == (6,)
    assert res.initial.mu_mc.shape == (6,)
    assert res.transition_01.lambda_skills_next.shape == (3,)
    assert res.transition_01.variant == "ces"
    # The converged period-1->2 production shock SD is pinned at zero in the
    # MATLAB CES run (see `est_12[25]` in Results_AF_One_Normal_CES.mat).
    assert np.isclose(res.transition_12.sigma_eta_prod, 0.0)


@pytest.mark.skipif(
    not (_DEFAULT_RESULTS_DIR / "Results_AF_One_Normal_Translog.mat").exists(),
    reason="MATLAB translog result file not available",
)
def test_load_matlab_results_translog() -> None:
    res = load_matlab_results(
        _DEFAULT_RESULTS_DIR / "Results_AF_One_Normal_Translog.mat",
        variant="translog",
    )
    assert res.n_obs == 1403
    assert res.transition_01.variant == "translog"
    # Translog transition vectors are 25 elements; `phi_prod` is not present.
    assert np.isnan(res.transition_01.phi_prod)
