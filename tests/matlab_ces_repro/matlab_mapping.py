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
from numpy.typing import NDArray
from scipy.io import loadmat


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
