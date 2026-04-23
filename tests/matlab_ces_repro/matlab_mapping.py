"""Parse the MATLAB AF estimation result vectors into named fields.

The MATLAB scripts (`AF_Application_One_Normal_CES.m` and
`AF_Application_One_Normal_Translog.m`) serialise their optimisation output
as flat float arrays:

- ``est_0``: 44 values for the initial period (shared across CES and translog).
- ``est_01``, ``est_12``: 26 values (CES) or 25 values (translog) per
  transition period.

The helpers below parse those arrays into a `MatlabResults` dataclass with
explicit fields per parameter block, so comparison code reads ``res.rho_01``
instead of ``est_01[22]``.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.io import loadmat

from .load_cnlsy import (
    INCOME_MEASURE,
    MC_MEASURES,
    MN_MEASURES,
    SKILL_MEASURES,
)

# skillmodels' joint factor ordering in the initial distribution for our
# 4-latent + 1-observed model; MATLAB's 4-dim distribution covers
# (skills, MC, MN, log_income). ``investment`` is a skillmodels latent
# without a MATLAB analogue and is treated as independent of the other
# factors in the initial distribution.
_SKM_JOINT_ORDER: tuple[str, ...] = (
    "skills",
    "MC",
    "MN",
    "investment",
    INCOME_MEASURE,
)
_MATLAB_TO_SKM_INITIAL_INDEX: dict[int, int] = {
    0: 0,  # skills
    1: 1,  # MC
    2: 2,  # MN
    3: 4,  # log_income (index 4 in skillmodels because of investment at 3)
}


@dataclass(frozen=True)
class MatlabInitialResults:
    """Layout of MATLAB ``est_0``."""

    mu_log_income: float
    """Mean of the log_income latent factor (``mu_Omega(4)``)."""
    var_diag: NDArray[np.float64]
    """Variances of (skills, MC, MN, log_income); shape (4,)."""
    correlations: NDArray[np.float64]
    """Off-diagonal correlations in Sigma_Omega, ordering
    (skills,MC), (skills,MN), (skills,Y), (MC,MN), (MC,Y), (MN,Y); shape (6,).
    """
    mu_skills_0: NDArray[np.float64]
    """Measurement intercepts for skills at period 0; shape (3,)."""
    lambda_skills_0_free: NDArray[np.float64]
    """Free skill loadings at period 0 (first loading fixed to 1); shape (2,)."""
    sigma_skills_0: NDArray[np.float64]
    """Measurement SDs for skills at period 0; shape (3,)."""
    mu_mc: NDArray[np.float64]
    """Measurement intercepts for MC; shape (6,)."""
    lambda_mc_free: NDArray[np.float64]
    """Free MC loadings (first fixed to 1); shape (5,)."""
    sigma_mc: NDArray[np.float64]
    """Measurement SDs for MC; shape (6,)."""
    mu_mn: NDArray[np.float64]
    """Measurement intercepts for MN (3 aggregated items); shape (3,)."""
    lambda_mn_free: NDArray[np.float64]
    """Free MN loadings (first fixed to 1); shape (2,)."""
    sigma_mn: NDArray[np.float64]
    """Measurement SDs for MN; shape (3,)."""


@dataclass(frozen=True)
class MatlabTransitionResults:
    """Layout of MATLAB ``est_01`` / ``est_12``.

    CES transitions have 26 fields; translog transitions have 25 (no
    separate ``A`` constant because it is absorbed in ``rho``). The parser
    populates ``rho_prod``, ``delta_prod``, ``phi_prod`` as the production
    parameters; their interpretation depends on the variant string.
    """

    variant: str
    """Either ``"ces"`` or ``"translog"``."""
    mu_skills_next_free: NDArray[np.float64]
    """Free intercepts for skills at period t+1 (first tied to
    ``mu_skills_norm_0`` from the initial period); shape (2,).
    """
    lambda_skills_next: NDArray[np.float64]
    """Skill loadings at period t+1; shape (3,)."""
    sigma_skills_next: NDArray[np.float64]
    """Skill measurement SDs at period t+1; shape (3,)."""
    mu_inv: NDArray[np.float64]
    """Investment measurement intercepts at period t; shape (3,)."""
    lambda_inv: NDArray[np.float64]
    """Investment measurement loadings at period t; shape (3,)."""
    sigma_inv: NDArray[np.float64]
    """Investment measurement SDs at period t; shape (3,)."""
    a_theta: float
    """Investment-equation coefficient on ``theta_t``."""
    a_mc: float
    """Investment-equation coefficient on ``MC``."""
    a_mn: float
    """Investment-equation coefficient on ``MN``."""
    a_log_income: float
    """Investment-equation coefficient on ``log_income_t``."""
    sigma_eta_inv: float
    """Investment shock SD."""
    rho_prod: float
    """CES ``rho`` or translog ``rho``."""
    delta_prod: float
    """CES ``delta`` or translog ``delta``."""
    phi_prod: float
    """CES ``phi`` or translog ``phi``."""
    sigma_eta_prod: float
    """Production shock SD."""


@dataclass(frozen=True)
class MatlabResults:
    """Full MATLAB AF result set."""

    initial: MatlabInitialResults
    transition_01: MatlabTransitionResults
    transition_12: MatlabTransitionResults
    n_obs: int
    n_halton_nodes: int


def load_matlab_results(path: Path, variant: str) -> MatlabResults:
    """Load a MATLAB ``.mat`` file and parse into named fields.

    Args:
        path: Path to ``Results_AF_One_Normal_CES.mat`` or
            ``Results_AF_One_Normal_Translog.mat``.
        variant: ``"ces"`` or ``"translog"``.

    Return:
        ``MatlabResults`` with initial-period and transition-period blocks
        parsed into structured fields.
    """
    if variant not in {"ces", "translog"}:
        msg = f"variant must be 'ces' or 'translog', got {variant!r}"
        raise ValueError(msg)

    raw = loadmat(str(path))
    est_0 = np.asarray(raw["est_0"]).ravel()
    est_01 = np.asarray(raw["est_01"]).ravel()
    est_12 = np.asarray(raw["est_12"]).ravel()

    expected_initial_len = 44
    if est_0.size != expected_initial_len:
        msg = f"est_0 has {est_0.size} elements; expected {expected_initial_len}"
        raise ValueError(msg)
    expected_transition_len = 26 if variant == "ces" else 25
    for name, arr in (("est_01", est_01), ("est_12", est_12)):
        if arr.size != expected_transition_len:
            msg = (
                f"{name} has {arr.size} elements; expected "
                f"{expected_transition_len} for {variant}"
            )
            raise ValueError(msg)

    initial = _parse_initial(est_0)
    t01 = _parse_transition(est_01, variant)
    t12 = _parse_transition(est_12, variant)

    return MatlabResults(
        initial=initial,
        transition_01=t01,
        transition_12=t12,
        n_obs=int(raw["n"].item()),
        n_halton_nodes=int(raw["number_of_nodes_0"].item()),
    )


def _parse_initial(est: NDArray[np.float64]) -> MatlabInitialResults:
    """Parse the 44-element initial-period MATLAB vector."""
    return MatlabInitialResults(
        mu_log_income=float(est[0]),
        var_diag=est[1:5].copy(),
        correlations=est[5:11].copy(),
        mu_skills_0=est[11:14].copy(),
        lambda_skills_0_free=est[14:16].copy(),
        sigma_skills_0=est[16:19].copy(),
        mu_mc=est[19:25].copy(),
        lambda_mc_free=est[25:30].copy(),
        sigma_mc=est[30:36].copy(),
        mu_mn=est[36:39].copy(),
        lambda_mn_free=est[39:41].copy(),
        sigma_mn=est[41:44].copy(),
    )


def _parse_transition(
    est: NDArray[np.float64], variant: str
) -> MatlabTransitionResults:
    """Parse a transition-period MATLAB vector (26 CES / 25 translog)."""
    # Common measurement + investment-equation layout runs through index 21.
    if variant == "ces":
        rho_prod = float(est[22])
        delta_prod = float(est[23])
        phi_prod = float(est[24])
        sigma_eta_prod = float(est[25])
    else:  # translog
        rho_prod = float(est[22])
        delta_prod = float(est[23])
        phi_prod = float("nan")
        sigma_eta_prod = float(est[24])
    return MatlabTransitionResults(
        variant=variant,
        mu_skills_next_free=est[0:2].copy(),
        lambda_skills_next=est[2:5].copy(),
        sigma_skills_next=est[5:8].copy(),
        mu_inv=est[8:11].copy(),
        lambda_inv=est[11:14].copy(),
        sigma_inv=est[14:17].copy(),
        a_theta=float(est[17]),
        a_mc=float(est[18]),
        a_mn=float(est[19]),
        a_log_income=float(est[20]),
        sigma_eta_inv=float(est[21]),
        rho_prod=rho_prod,
        delta_prod=delta_prod,
        phi_prod=phi_prod,
        sigma_eta_prod=sigma_eta_prod,
    )


def ces_to_skillmodels_gammas(delta: float, phi: float) -> tuple[float, float, float]:
    """Convert MATLAB ``(delta, phi)`` to skillmodels' normalised gammas.

    Kept for backward compatibility with existing tests. Use
    `translate_matlab_ces_production` when you need the full translation
    (including the level shift that must be absorbed into the period-t+1
    skill intercepts).
    """
    gamma_skills, gamma_inv, _, _ = translate_matlab_ces_production(
        delta=delta, phi=phi, rho=float("nan"), a_const=0.0
    )
    return gamma_skills, gamma_inv, float("nan")


@dataclass(frozen=True)
class SkillmodelsCesTranslation:
    """Parameters of skillmodels' normalised ``log_ces`` derived from MATLAB.

    skillmodels' ``log_ces`` evaluates
    ``f_skm = (1 / phi_skm) * logsumexp(log(gamma) + states * phi_skm)``
    with ``gamma`` on the simplex. MATLAB's unnormalised form is
    ``f_m = A + (1 / rho) * log(delta * theta**rho + phi * X**rho)``.

    The two are related by ``f_m(theta, X) = f_skm(theta, X) + level_shift``
    where
    ``level_shift = A + (1 / rho) * log(delta + phi)``. Because the
    level shift is an additive constant that appears in every
    period-t+1 skill value, it is absorbed into the period-t+1 skill
    measurement intercepts (``mu_skills_next``).

    Attributes:
        gamma_skills: Normalised weight on skills in skillmodels'
            ``log_ces``; equals ``delta / (delta + phi)``.
        gamma_inv: Normalised weight on investment; equals
            ``phi / (delta + phi)``.
        phi_skm: The ``phi`` parameter skillmodels expects, equal to
            MATLAB's ``rho``.
        level_shift: The additive constant to add to every period-t+1
            skill measurement intercept to compensate for skillmodels'
            normalisation of the gammas.
    """

    gamma_skills: float
    gamma_inv: float
    phi_skm: float
    level_shift: float


def translate_matlab_ces_production(
    *,
    delta: float,
    phi: float,
    rho: float,
    a_const: float = 0.0,
) -> tuple[float, float, float, float]:
    """Translate MATLAB CES params into skillmodels' normalised form.

    Args:
        delta: MATLAB ``delta`` (unnormalised coefficient on skills).
        phi: MATLAB ``phi`` (unnormalised coefficient on investment).
        rho: MATLAB ``rho`` (elasticity exponent). Equals skillmodels'
            ``phi_skm`` directly.
        a_const: MATLAB ``A`` constant term. MATLAB sets this to ``0`` in
            both the CES and translog application scripts; accept it as
            a kwarg for completeness.

    Return:
        Tuple ``(gamma_skills, gamma_inv, phi_skm, level_shift)``.

    Raises:
        ValueError: If ``delta + phi`` is not positive.
    """
    total = delta + phi
    if not total > 0:
        msg = f"delta + phi must be positive; got {total}"
        raise ValueError(msg)
    gamma_skills = delta / total
    gamma_inv = phi / total
    phi_skm = rho
    level_shift = a_const + (1.0 / rho) * math.log(total)
    return gamma_skills, gamma_inv, phi_skm, level_shift


def _build_matlab_4x4_cov(initial: MatlabInitialResults) -> NDArray[np.float64]:
    """Reconstruct MATLAB's 4x4 initial covariance from variances + correlations."""
    var = initial.var_diag
    corr = initial.correlations
    cov = np.diag(var).astype(np.float64)
    cov[1, 0] = corr[0] * math.sqrt(var[0] * var[1])  # (skills, MC)
    cov[2, 0] = corr[1] * math.sqrt(var[0] * var[2])  # (skills, MN)
    cov[3, 0] = corr[2] * math.sqrt(var[0] * var[3])  # (skills, Y)
    cov[2, 1] = corr[3] * math.sqrt(var[1] * var[2])  # (MC, MN)
    cov[3, 1] = corr[4] * math.sqrt(var[1] * var[3])  # (MC, Y)
    cov[3, 2] = corr[5] * math.sqrt(var[2] * var[3])  # (MN, Y)
    cov[0, 1] = cov[1, 0]
    cov[0, 2] = cov[2, 0]
    cov[0, 3] = cov[3, 0]
    cov[1, 2] = cov[2, 1]
    cov[1, 3] = cov[3, 1]
    cov[2, 3] = cov[3, 2]
    return cov


def _embed_matlab_cov_in_skillmodels(
    initial: MatlabInitialResults,
    *,
    investment_sd: float = 1.0,
) -> NDArray[np.float64]:
    """Build the 5x5 skillmodels initial covariance from MATLAB's 4x4 one.

    ``investment`` (skillmodels dim 3) is placed as independent of the
    other four factors with variance ``investment_sd**2``. The returned
    matrix is ordered ``(skills, MC, MN, investment, log_income)``.
    """
    cov4 = _build_matlab_4x4_cov(initial)
    cov5 = np.zeros((5, 5), dtype=np.float64)
    for i_matlab, i_skm in _MATLAB_TO_SKM_INITIAL_INDEX.items():
        for j_matlab, j_skm in _MATLAB_TO_SKM_INITIAL_INDEX.items():
            cov5[i_skm, j_skm] = cov4[i_matlab, j_matlab]
    cov5[3, 3] = investment_sd**2
    return cov5


def _skillmodels_cholcov_entries(cov: NDArray[np.float64]) -> dict[str, float]:
    """Map a 5x5 covariance to skillmodels' ``initial_cholcovs`` entries.

    Keys are ``{factor_row}-{factor_col}`` matching the MultiIndex
    ``name2`` level built by ``get_initial_period_params_index``.
    """
    chol = np.linalg.cholesky(cov)
    entries: dict[str, float] = {}
    for row, f_row in enumerate(_SKM_JOINT_ORDER):
        for col in range(row + 1):
            f_col = _SKM_JOINT_ORDER[col]
            entries[f"{f_row}-{f_col}"] = float(chol[row, col])
    return entries


def fill_initial_params_from_matlab(
    params_template: pd.DataFrame,
    initial: MatlabInitialResults,
    *,
    period: int = 0,
    component: str = "mixture_0",
    investment_initial_sd: float = 1.0,
) -> pd.DataFrame:
    """Populate skillmodels' initial-period entries from MATLAB's ``est_0``.

    Overwrites the ``mixture_weights``, ``initial_states``,
    ``initial_cholcovs``, ``controls`` (measurement intercepts),
    ``loadings``, and ``meas_sds`` entries that correspond to the MATLAB
    initial-period vector. Entries that don't have a MATLAB counterpart
    (investment measurement model in period 0, investment in the joint
    initial distribution) are filled with placeholder values.

    Args:
        params_template: skillmodels AF initial-period params DataFrame
            with MultiIndex (category, period, name1, name2).
        initial: Parsed MATLAB initial-period block.
        period: Calendar period of the initial distribution (typically 0).
        component: Name of the mixture component (MATLAB uses a single
            Gaussian; default matches skillmodels' ``mixture_0``).
        investment_initial_sd: Placeholder SD for investment in the joint
            initial distribution (MATLAB has no investment dimension).

    Return:
        Modified copy of ``params_template`` with the MATLAB-derived values
        written in.
    """
    params = params_template.copy()

    # Mixture weights (single component → weight = 1).
    params.loc[("mixture_weights", period, component, "-"), "value"] = 1.0

    # Initial means: MATLAB has zero mean for skills, MC, MN and
    # ``mu_log_income`` for the 4th factor. Investment gets 0.
    means_skm = [0.0, 0.0, 0.0, 0.0, initial.mu_log_income]
    for factor, mean in zip(_SKM_JOINT_ORDER, means_skm, strict=True):
        params.loc[("initial_states", period, component, factor), "value"] = mean

    # Initial Cholesky covariances: 5x5 Cholesky of the embedded MATLAB cov.
    cov5 = _embed_matlab_cov_in_skillmodels(
        initial, investment_sd=investment_initial_sd
    )
    chol_entries = _skillmodels_cholcov_entries(cov5)
    for name2, value in chol_entries.items():
        params.loc[("initial_cholcovs", period, component, name2), "value"] = value

    # Measurement model for skills at period 0.
    _fill_block(
        params,
        period=period,
        measures=SKILL_MEASURES,
        mu=initial.mu_skills_0,
        lambdas_free=initial.lambda_skills_0_free,
        sigmas=initial.sigma_skills_0,
        factor="skills",
    )

    # Measurement model for MC at period 0.
    _fill_block(
        params,
        period=period,
        measures=MC_MEASURES,
        mu=initial.mu_mc,
        lambdas_free=initial.lambda_mc_free,
        sigmas=initial.sigma_mc,
        factor="MC",
    )

    # Measurement model for MN at period 0.
    _fill_block(
        params,
        period=period,
        measures=MN_MEASURES,
        mu=initial.mu_mn,
        lambdas_free=initial.lambda_mn_free,
        sigmas=initial.sigma_mn,
        factor="MN",
    )

    return params


def _fill_block(
    params: pd.DataFrame,
    *,
    period: int,
    measures: tuple[str, ...],
    mu: NDArray[np.float64],
    lambdas_free: NDArray[np.float64],
    sigmas: NDArray[np.float64],
    factor: str,
) -> None:
    """Write a measurement block (intercept, loadings, SDs) into params."""
    # Intercepts: first is normalised to 0, rest come from ``mu``.
    for i, measure in enumerate(measures):
        params.loc[("controls", period, measure, "constant"), "value"] = float(mu[i])
    # First measurement has intercept normalised to 0.
    params.loc[("controls", period, measures[0], "constant"), "value"] = 0.0

    # Loadings: first is normalised to 1, rest come from ``lambdas_free``.
    params.loc[("loadings", period, measures[0], factor), "value"] = 1.0
    for j, measure in enumerate(measures[1:]):
        params.loc[("loadings", period, measure, factor), "value"] = float(
            lambdas_free[j]
        )

    # Measurement SDs.
    for i, measure in enumerate(measures):
        params.loc[("meas_sds", period, measure, "-"), "value"] = float(sigmas[i])
