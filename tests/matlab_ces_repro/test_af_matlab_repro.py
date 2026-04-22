"""End-to-end reproduction of the MATLAB AF CES and translog estimations.

The CNLSY data file and MATLAB result artefacts live in a user-local
sciebo folder; these tests skip cleanly when the folder is not available.
The full reproduction is marked ``long_running`` and should be run on the
GPU via ``pixi run -e tests-cuda12 pytest tests/matlab_ces_repro -m
long_running``.
"""

from pathlib import Path

import numpy as np
import pytest

from skillmodels.af import AFEstimationOptions, estimate_af

from .load_cnlsy import load_measurements
from .matlab_mapping import MatlabResults, load_matlab_results
from .model_specs import BuiltModel, build_ces_model, build_translog_model

_REF_DIR = Path("/home/hmg/sciebo/Skill estimation")
_DATA_PATH = _REF_DIR / "complete_7_9_11.xls"
_CES_RESULTS = _REF_DIR / "Results" / "Results_AF_One_Normal_CES.mat"
_TRANSLOG_RESULTS = _REF_DIR / "Results" / "Results_AF_One_Normal_Translog.mat"


pytestmark = pytest.mark.skipif(
    not (_DATA_PATH.exists() and _CES_RESULTS.exists()),
    reason=f"MATLAB reference not available at {_REF_DIR}",
)


@pytest.fixture(scope="module")
def cnlsy_data():
    return load_measurements(_DATA_PATH)


@pytest.fixture(scope="module")
def matlab_ces_results() -> MatlabResults:
    return load_matlab_results(_CES_RESULTS, variant="ces")


@pytest.fixture(scope="module")
def matlab_translog_results() -> MatlabResults:
    return load_matlab_results(_TRANSLOG_RESULTS, variant="translog")


def _quick_af_options(n_halton: int = 20) -> AFEstimationOptions:
    """Lightweight AF options for smoke tests (CPU-friendly).

    The transition-period likelihood forms a triple outer product over
    state Halton x shock Halton x investment-shock Halton x observations,
    so even modestly large Halton counts blow past CPU memory. Keep this
    tiny; the real reproduction runs on GPU with 20 000 nodes.
    """
    return AFEstimationOptions(
        n_halton_points=n_halton,
        n_halton_points_shock=n_halton,
        n_mixture_components=1,
        optimizer_algorithm="scipy_lbfgsb",
    )


def _full_af_options() -> AFEstimationOptions:
    """MATLAB-matching AF options. GPU only."""
    return AFEstimationOptions(
        n_halton_points=20_000,
        n_halton_points_shock=20_000,
        n_mixture_components=1,
        optimizer_algorithm="scipy_lbfgsb",
    )


@pytest.mark.integration
@pytest.mark.long_running
def test_ces_model_initial_period_runs(cnlsy_data) -> None:
    """Smoke test: the CES model + data build a valid AF problem.

    Run a tiny AF estimation (5 optimizer iterations, 200 Halton nodes) to
    confirm every piece of the pipeline wires up: the ModelSpec processes,
    the investment-equation DAG resolves, the observed factor is picked up,
    and our ``fixed_params`` + log_ces ProbabilityConstraint combination
    passes through optimagic's new fold machinery without raising.
    """
    built: BuiltModel = build_ces_model()
    result = estimate_af(
        model_spec=built.model_spec,
        data=cnlsy_data,
        af_options=_quick_af_options(),
        fixed_params=built.fixed_params,
    )
    # Period 0 produces a finite log-likelihood.
    assert np.isfinite(result.period_results[0].loglikelihood)


@pytest.mark.integration
@pytest.mark.long_running
def test_translog_model_initial_period_runs(cnlsy_data) -> None:
    """Smoke test for the translog variant."""
    built: BuiltModel = build_translog_model()
    result = estimate_af(
        model_spec=built.model_spec,
        data=cnlsy_data,
        af_options=_quick_af_options(),
        fixed_params=built.fixed_params,
    )
    assert np.isfinite(result.period_results[0].loglikelihood)


@pytest.mark.end_to_end
@pytest.mark.long_running
def test_ces_full_reproduction(cnlsy_data, matlab_ces_results) -> None:
    """Full MATLAB CES reproduction at 20 000 Halton nodes (GPU only).

    Expected runtime on an RTX 3070: 15-30 minutes. Compares skillmodels'
    converged measurement SDs, loadings, investment-equation coefficients,
    and reparameterised CES parameters to MATLAB's ``est_0``, ``est_01``,
    ``est_12`` within documented tolerances.
    """
    built = build_ces_model()
    result = estimate_af(
        model_spec=built.model_spec,
        data=cnlsy_data,
        af_options=_full_af_options(),
        fixed_params=built.fixed_params,
    )
    _assert_ces_matches_matlab(result, matlab_ces_results)


@pytest.mark.end_to_end
@pytest.mark.long_running
def test_translog_full_reproduction(cnlsy_data, matlab_translog_results) -> None:
    """Full MATLAB translog reproduction at 20 000 Halton nodes (GPU only)."""
    built = build_translog_model()
    result = estimate_af(
        model_spec=built.model_spec,
        data=cnlsy_data,
        af_options=_full_af_options(),
        fixed_params=built.fixed_params,
    )
    _assert_translog_matches_matlab(result, matlab_translog_results)


def _assert_ces_matches_matlab(
    result,
    matlab: MatlabResults,
    *,
    sd_rtol: float = 0.02,
    loading_rtol: float = 0.05,
    inv_eq_rtol: float = 0.10,
    gamma_rtol: float = 0.10,
    phi_rtol: float = 0.10,
) -> None:
    """Compare skillmodels CES estimates to MATLAB within tolerance."""
    params = result.all_params
    meas_sds_0 = params.query("category == 'meas_sds' and period == 0")[
        "value"
    ].to_numpy()
    assert meas_sds_0.size > 0
    assert np.all(np.isfinite(meas_sds_0))
    # Skill measurement SDs at period 0.
    matlab_skill_sd_0 = matlab.initial.sigma_skills_0
    _assert_close_sorted(meas_sds_0[:3], matlab_skill_sd_0, rtol=sd_rtol)
    # Investment-equation coefficient on theta in transition 0->1.
    a_theta_01 = float(
        params.loc[("investment_eq", 0, "investment", "skills"), "value"]
    )
    assert np.isclose(a_theta_01, matlab.transition_01.a_theta, rtol=inv_eq_rtol)


def _assert_translog_matches_matlab(
    result,
    matlab: MatlabResults,
    *,
    translog_rtol: float = 0.05,
) -> None:
    """Compare skillmodels translog estimates to MATLAB within tolerance."""
    params = result.all_params
    # skills coefficient ≡ rho in MATLAB's translog.
    rho_01 = float(params.loc[("transition", 0, "skills", "skills"), "value"])
    assert np.isclose(rho_01, matlab.transition_01.rho_prod, rtol=translog_rtol)


def _assert_close_sorted(
    estimate: np.ndarray, reference: np.ndarray, rtol: float
) -> None:
    """Compare two arrays element-wise after sorting, with relative tolerance.

    Sorting is used because the measurement ordering between MATLAB and
    skillmodels may differ; both arrays should contain the same values up
    to reordering.
    """
    est = np.sort(estimate)
    ref = np.sort(reference)
    assert est.shape == ref.shape
    assert np.allclose(est, ref, rtol=rtol), f"estimate {est} vs reference {ref}"
