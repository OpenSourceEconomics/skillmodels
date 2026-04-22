"""Unit tests for the MATLAB result parser."""

from pathlib import Path

import numpy as np
import pytest

from .matlab_mapping import (
    ces_to_skillmodels_gammas,
    load_matlab_results,
)

_DEFAULT_RESULTS_DIR = Path("/home/hmg/sciebo/Skill estimation/Results")


def test_ces_to_skillmodels_gammas_sums_to_one() -> None:
    gamma_skills, gamma_inv, _ = ces_to_skillmodels_gammas(delta=0.7, phi=0.3)
    assert np.isclose(gamma_skills + gamma_inv, 1.0)
    assert np.isclose(gamma_skills, 0.7)
    assert np.isclose(gamma_inv, 0.3)


def test_ces_to_skillmodels_gammas_rejects_non_positive_sum() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        ces_to_skillmodels_gammas(delta=-0.3, phi=0.2)


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
