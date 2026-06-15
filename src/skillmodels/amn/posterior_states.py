"""Per-individual posterior latent-factor estimates from an AMN fit.

AMN does not Kalman-filter or quadrature-integrate; it fits a mixture
of normals on the augmented measure vector. The natural per-individual
factor estimate is therefore the mixture-Schur conditional posterior
``E[theta | Y_i]`` evaluated under the fitted reduced-form parameters,
mirrored across the K components weighted by per-individual mixture
responsibilities.

For every observation `i` and every mixture component `k`:

    mu_{theta|Y}(k, i) = mu_theta(k)
                       + Cov(theta, Y)(k) Cov(Y)(k)^{-1} (Y_i - mu_Y(k))

where ``mu_Y(k) = A + Lambda mu_theta(k)``,
``Cov(Y)(k) = Lambda Omega(k) Lambda^T + diag(sigma^2)``, and
``Cov(theta, Y)(k) = Omega(k) Lambda^T``. The mixture responsibility is the
standard Bayes posterior of `k` given `Y_i`, and
``E[theta | Y_i] = sum_k r(k|i) mu_{theta|Y}(k, i)``.

The function returns a dict matching the CHS / AF
`get_individual_states` shape (an ``"unanchored_states"`` entry only —
AMN does not produce anchored states without an explicit anchoring
post-step).
"""

from typing import Any

import numpy as np
import pandas as pd
from beartype import beartype

from skillmodels._beartype_conf import ESTIMATION_CONF
from skillmodels.amn.mixture_em import build_augmented_measure_matrix
from skillmodels.amn.types import AMNEstimationResult
from skillmodels.common.process_model import process_model
from skillmodels.common.state_ranges import create_state_ranges


@beartype(conf=ESTIMATION_CONF)
def get_amn_posterior_states(  # noqa: C901, PLR0912, PLR0915
    amn_result: AMNEstimationResult,
    data: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Compute the per-observation latent factor posteriors.

    Args:
        amn_result: The fitted AMN result.
        data: Same panel dataset used for the original fit.

    Return:
        Nested dict with the CHS-compatible
        ``{"unanchored_states": {"states": DataFrame, "state_ranges": ...}}``
        layout (no ``"anchored_states"`` key — AMN does not anchor).

    """
    processed_model = process_model(amn_result.model_spec)
    layout = amn_result.stages.mixture.layout
    augmented = build_augmented_measure_matrix(data, processed_model, layout)
    n_aug = augmented.shape[1]

    mixture = amn_result.stages.mixture
    structural = amn_result.stages.structural

    # Build Lambda and intercepts in the original AMN structural basis.
    n_components = mixture.weights.shape[0]
    factor_slots = structural.factor_period_slots
    n_factor = len(factor_slots)

    # Reconstruct Lambda from the loadings DataFrame + observed-factor
    # / control passthrough.
    lambda_mat = np.zeros((n_aug, n_factor))
    slot_to_id = {sp: i for i, sp in enumerate(factor_slots)}

    for aug_idx, (period, factor, meas) in zip(
        layout.measurement_slots, layout.measurement_meta, strict=True
    ):
        col = slot_to_id.get((period, factor))
        if col is None:
            continue
        try:
            loading = structural.loadings.loc[(period, meas, factor), "loading"]
        except KeyError:
            loading = 1.0
        lambda_mat[aug_idx, col] = float(loading)

    for aug_idx, (period, of_name) in zip(
        layout.observed_factor_slots, layout.observed_factor_meta, strict=True
    ):
        col = slot_to_id.get((period, of_name))
        if col is not None:
            lambda_mat[aug_idx, col] = 1.0

    for aug_idx, ctrl in zip(layout.control_slots, layout.control_meta, strict=True):
        col = slot_to_id.get((-1, ctrl))
        if col is not None:
            lambda_mat[aug_idx, col] = 1.0

    intercept = np.zeros(n_aug)
    for aug_idx, (period, _factor, meas) in zip(
        layout.measurement_slots, layout.measurement_meta, strict=True
    ):
        try:
            intercept[aug_idx] = float(
                structural.measurement_intercepts.loc[(period, meas), "intercept"]
            )
        except KeyError:
            intercept[aug_idx] = 0.0

    sigma2 = np.zeros(n_aug)
    for aug_idx, (period, _factor, meas) in zip(
        layout.measurement_slots, layout.measurement_meta, strict=True
    ):
        try:
            sd = float(structural.measurement_sds.loc[(period, meas), "sd"])
        except KeyError:
            sd = 0.0
        sigma2[aug_idx] = sd * sd

    diag_sigma = np.diag(sigma2)

    # Drop rows with any NaN in the augmented vector (listwise; matches
    # Stage 1's complete-case behaviour). Posterior is reported only
    # for complete-case observations.
    complete_mask = ~np.isnan(augmented).any(axis=1)
    y_complete = augmented[complete_mask]

    # Precompute per-component pieces.
    mu_theta = structural.factor_mixture_means
    omegas = structural.factor_mixture_covariances
    mu_y_per = np.empty((n_components, n_aug))
    cov_y_inv = np.empty((n_components, n_aug, n_aug))
    cov_theta_y = np.empty((n_components, n_factor, n_aug))
    log_det = np.empty(n_components)
    for k in range(n_components):
        mu_y_per[k] = intercept + lambda_mat @ mu_theta[k]
        cov_y = lambda_mat @ omegas[k] @ lambda_mat.T + diag_sigma
        cov_y = 0.5 * (cov_y + cov_y.T) + 1e-10 * np.eye(n_aug)
        cov_y_inv[k] = np.linalg.inv(cov_y)
        cov_theta_y[k] = omegas[k] @ lambda_mat.T
        sign, logdet = np.linalg.slogdet(cov_y)
        log_det[k] = logdet if sign > 0 else np.inf

    # Per-obs log-pdf in each component (up to a constant).
    log_pi = np.log(np.clip(mixture.weights, 1e-300, None))
    diffs = y_complete[:, None, :] - mu_y_per[None, :, :]  # (n_complete, K, n_aug)
    quad = np.einsum("ikj,kjl,ikl->ik", diffs, cov_y_inv, diffs)
    log_probs = log_pi[None, :] - 0.5 * (log_det[None, :] + quad)
    log_probs -= log_probs.max(axis=1, keepdims=True)
    probs = np.exp(log_probs)
    responsibilities = probs / probs.sum(axis=1, keepdims=True)

    # Per-component conditional mean of theta given Y_i.
    cond_means = np.empty((y_complete.shape[0], n_components, n_factor))
    for k in range(n_components):
        cond_means[:, k, :] = (
            mu_theta[k] + (cov_theta_y[k] @ cov_y_inv[k] @ diffs[:, k, :].T).T
        )

    # Mixture-averaged posterior mean of theta.
    posterior = np.einsum("ik,ikj->ij", responsibilities, cond_means)

    # Stuff into a (id, period) -> (factor, ...) DataFrame.
    case_level = str(data.index.names[0])
    caseids = data.index.get_level_values(case_level).unique()
    complete_caseids = caseids[np.asarray(complete_mask, dtype=bool)]

    latent_factors = processed_model.labels.latent_factors
    periods = processed_model.labels.periods
    rows = []
    for row_idx, caseid in enumerate(complete_caseids):
        for period in periods:
            row: dict[str, Any] = {"id": caseid, "period": int(period)}
            for factor in latent_factors:
                col_idx = slot_to_id.get((int(period), factor))
                row[factor] = (
                    float(posterior[row_idx, col_idx])
                    if col_idx is not None
                    else np.nan
                )
            rows.append(row)
    states_df = pd.DataFrame(rows)

    state_ranges = create_state_ranges(
        filtered_states=states_df,
        factors=latent_factors,
    )

    return {
        "unanchored_states": {
            "states": states_df,
            "state_ranges": state_ranges,
        },
    }
