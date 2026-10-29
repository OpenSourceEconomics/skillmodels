"""CHS measurement update for probit, Tobit and ordered-probit (limited) measurements.

For one person and mixture component let the predicted state be `theta ~ N(m, P)` with
`P = U'U`, and let the latent index be `y* = a + lambda'theta + eps`,
`eps ~ N(0, sigma^2)`, with `a` the control part. Write `v = U lambda`,
`s^2 = sigma^2 + v'v` and `mu = a + lambda'm`.

A half-line event `direction * (y* - threshold) > 0` -- probit `y = 1` / `y = 0` with
threshold 0 and `sigma = 1`, or a Tobit value at its upper / lower bound -- has the
exact probability `Phi(t)`, `t = direction * (mu - threshold) / s`, and the exact
posterior moments

- `m+ = m + direction * U'w * r`, `w = v / s`,
- `P+ = P - P lambda lambda' P * d / s^2`,

with `r = phi(t) / Phi(t)` and `d = r (r + t)`. The posterior is replaced by the
Gaussian with these moments (assumed-density filtering): the likelihood contribution
is exact given the Gaussian predicted state, the Gaussian shape of the filtered state is
an approximation. A Tobit value strictly inside its bounds is an ordinary Gaussian
observation, which is the same update with `d = 1` and the standardized innovation
`(y - mu) / s` in place of `direction * r`.

An interval event `lo < y* <= hi` -- an ordered-probit category between two finite
cutpoints, or a Tobit observation known only by an interval with two finite ends -- is
the same update with the moments of `z = (y* - mu) / s` truncated to
`(alpha, beta] = ((lo - mu) / s, (hi - mu) / s]`: probability
`Z = Phi(beta) - Phi(alpha)`, `E[z]` in place of `direction * r`, and
`d = 1 - Var[z]` (`normal_interval_terms`). An observed interval is evaluated from
its source ends, `mu` and `s` (`source_interval_terms`), not from two separately
rounded standardized ends, which can lose the location or the span of the interval:
`lo - mu` can overflow, and two ends that round to the same float can enclose a
mass that is not a narrow-density integral.

The covariance factor is updated as the R factor of `U - b w (w'U)` with
`b = d / (1 + c)` and `c = sqrt((sigma / s)^2 + (1 - d) w'w)`, which squares to the
posterior covariance. No step divides by `d`, takes `sqrt(d)`, or divides by
`lambda'P lambda`, so the update stays finite (and becomes the identity) when the event
is certain or the loadings are zero.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import Array

from skillmodels.chs.kalman_filters import reset_negligible_mixture_components
from skillmodels.common.measurement_models import (
    MeasurementFamily,
    MeasurementModel,
    ObservationEvent,
    OrderedProbitMeasurement,
    TobitMeasurement,
    measurement_family_arrays,
)
from skillmodels.common.normal_terms import (
    LOG_SQRT_2PI,
    normal_interval_terms,
    normal_tail_terms,
    source_interval_terms,
)

# Scan inputs that only the limited-measurement update consumes.
LIMITED_LOOP_KEYS = (
    "events",
    "is_limited",
    "cut_table",
    "codes_from_values",
    "interval_bounds",
)


@dataclass(frozen=True)
class LimitedUpdates:
    """Per-update family information of a model with limited measurements.

    Each update row has a table of latent cut values `T`; an observation with code
    `k` lies in `(T[k], T[k + 1]]`:

    - probit: `T = [0, 0, 0]`, codes `y` in `{0, 1}`;
    - ordered probit with `K` categories: `T = [0, c_1, ..., c_{K-1}, 0]` with the
      cutpoints from the parameter vector, codes `y` in `0..K-1`;
    - Tobit: `T = [0, lower, upper]`, code 0 at the lower and 2 at the upper bound.

    Entries that no event reads are 0. Unused trailing columns are 0.
    """

    is_limited: np.ndarray
    """Boolean array of length n_updates; True for probit, Tobit and ordered-probit
    updates."""
    is_unit_scale: np.ndarray
    """Boolean array of length n_updates; True for probit and ordered-probit updates,
    whose latent error SD is 1."""
    codes_from_values: np.ndarray
    """Boolean array of length n_updates; True if the observed value is the code
    (probit and ordered probit)."""
    base_table: np.ndarray
    """Array of shape (n_updates, n_columns) with the fixed entries of each table."""
    cut_positions: np.ndarray
    """Integer array like `base_table` with the position of each cutpoint in the
    parameter vector, -1 for fixed entries."""
    has_intervals: bool
    """Whether `INTERVAL` events can occur: an ordered-probit measurement has more
    than two categories, or a Tobit measurement has `interval_columns`."""


def get_limited_updates(
    update_info: pd.DataFrame,
    measurement_models: Mapping[str, MeasurementModel],
    params_index: pd.MultiIndex | None = None,
) -> LimitedUpdates | None:
    """Resolve the measurement family of every update row, in update order.

    Return None if every update is Gaussian. Anchoring rows carry generated names
    and are Gaussian. `params_index` locates the cutpoints of ordered-probit
    measurements; it is required if there are any.
    """
    variables = tuple(update_info.index.get_level_values("variable"))
    families, lowers, uppers = measurement_family_arrays(measurement_models, variables)
    is_limited = families != int(MeasurementFamily.GAUSSIAN)
    if not is_limited.any():
        return None
    is_tobit = families == int(MeasurementFamily.TOBIT)
    is_ordered = families == int(MeasurementFamily.ORDERED_PROBIT)
    models = [measurement_models.get(var) for var in variables]
    has_tobit_intervals = any(
        isinstance(m, TobitMeasurement) and m.interval_columns is not None
        for m in models
    )
    n_categories = [
        m.n_categories if isinstance(m, OrderedProbitMeasurement) else 2 for m in models
    ]
    n_columns = max(4, max(n_categories) + 1)
    base_table = np.zeros((len(families), n_columns))
    base_table[:, 1] = np.where(is_tobit & np.isfinite(lowers), lowers, 0.0)
    base_table[:, 2] = np.where(is_tobit & np.isfinite(uppers), uppers, 0.0)
    cut_positions = np.full((len(families), n_columns), -1, dtype=np.int64)
    for row, ((aug_period, var), model) in enumerate(
        zip(update_info.index, models, strict=True)
    ):
        if isinstance(model, OrderedProbitMeasurement):
            if params_index is None:
                msg = "Ordered-probit updates need the params index."
                raise ValueError(msg)
            for k in range(1, model.n_categories):
                cut_positions[row, k] = params_index.get_loc(
                    ("cutpoints", aug_period, var, f"cutpoint_{k}")
                )
    return LimitedUpdates(
        is_limited=is_limited,
        is_unit_scale=(families == int(MeasurementFamily.PROBIT)) | is_ordered,
        codes_from_values=(families == int(MeasurementFamily.PROBIT)) | is_ordered,
        base_table=base_table,
        cut_positions=cut_positions,
        has_intervals=has_tobit_intervals
        or bool(np.any(is_ordered & (np.array(n_categories) > 2))),
    )


def get_cut_table(limited_updates: LimitedUpdates, params: Array) -> Array:
    """Return the cut-value tables of all update rows with the current cutpoints."""
    positions = jnp.asarray(limited_updates.cut_positions)
    return jnp.where(
        positions >= 0,
        params[jnp.maximum(positions, 0)],
        jnp.asarray(limited_updates.base_table),
    )


def get_event_thresholds(
    measurements: Array,
    events: Array,
    cut_table: Array,
    codes_from_values: Array,
    interval_bounds: Array | None = None,
) -> Array:
    """Return the `[upper, lower]` latent ends of each observation's event.

    Args:
        measurements: 1d array of length n_obs with the observed values.
        events: 1d int8 array of length n_obs with `ObservationEvent` codes.
        cut_table: 1d cut-value table `T` of the update row.
        codes_from_values: Whether the observed value is the code.
        interval_bounds: Array of shape (n_obs, 2) with the `[lower, upper]` ends of
            the observations known only by an interval, NaN elsewhere; None if the
            model has no such observations.

    Return:
        Array of shape (n_obs, 2) with `T[k + 1]` and `T[k]` for code `k`, or the
        reversed interval ends of an observation known only by an interval.

    """
    value_codes = jnp.where(events == ObservationEvent.MISSING, 0.0, measurements)
    tail_codes = jnp.where(events == ObservationEvent.UPPER_TAIL, 2, 0)
    codes = jnp.where(codes_from_values, value_codes.astype(jnp.int32), tail_codes)
    thresholds = jnp.stack([cut_table[codes + 1], cut_table[codes]], axis=1)
    if interval_bounds is None:
        return thresholds
    given = ~jnp.isnan(interval_bounds[:, :1])
    return jnp.where(given, interval_bounds[:, ::-1], thresholds)


def limited_update_from_loop_kwargs(
    kwargs: Mapping[str, Array], *, has_intervals: bool
) -> tuple[Array, Array, Array, Array, dict[str, Array]]:
    """Run `kalman_update_limited` on the update arguments of one scan iteration.

    `kwargs` holds the Gaussian update arguments plus the `LIMITED_LOOP_KEYS`
    entries of the update row.
    """
    update_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k not in ("is_limited", "cut_table", "codes_from_values", "interval_bounds")
    }
    thresholds = get_event_thresholds(
        measurements=kwargs["measurements"],
        events=kwargs["events"],
        cut_table=kwargs["cut_table"],
        codes_from_values=kwargs["codes_from_values"],
        interval_bounds=kwargs.get("interval_bounds"),
    )
    return kalman_update_limited(
        **update_kwargs,
        thresholds=thresholds,
        has_intervals=has_intervals,
        source_intervals="interval_bounds" in kwargs,
    )


def kalman_update_limited(
    states: Array,
    upper_chols: Array,
    loadings: Array,
    control_params: Array,
    meas_sd: Array,
    measurements: Array,
    events: Array,
    thresholds: Array,
    controls: Array,
    log_mixture_weights: Array,
    *,
    has_intervals: bool = False,
    source_intervals: bool = False,
) -> tuple[Array, Array, Array, Array, dict[str, Array]]:
    """Condition each person's Gaussian mixture on one limited measurement.

    Args:
        states: Array of shape (n_obs, n_mixtures, n_states) with pre-update means.
        upper_chols: Array of shape (n_obs, n_mixtures, n_states, n_states) with
            upper Cholesky factors `U` of the pre-update covariances, `P = U'U`.
        loadings: 1d array of length n_states with factor loadings.
        control_params: 1d array of length n_controls.
        meas_sd: Measurement-error SD `sigma` (1 for a probit measurement).
        measurements: 1d array of length n_obs; only `DENSITY` entries are read.
        events: 1d int8 array of length n_obs with `ObservationEvent` codes.
        thresholds: Array of shape (2,) or (n_obs, 2) with the upper and the lower
            end of the latent interval of the event: `LOWER_TAIL` (`y* <= upper`)
            reads the first, `UPPER_TAIL` (`y* > lower`) the second and `INTERVAL`
            both. Probit uses `[0, 0]`, Tobit its bounds `[lower, upper]` or the
            reversed ends of an observed interval, ordered probit the cutpoints
            around each observed category. Unread entries may be infinite.
        controls: Array of shape (n_obs, n_controls).
        log_mixture_weights: Array of shape (n_obs, n_mixtures).
        has_intervals: Whether `INTERVAL` events can occur. Static; without them
            the interval moments are not evaluated.
        source_intervals: Whether an `INTERVAL` event is evaluated from its source
            ends, `mu` and `s` (`source_interval_terms`) instead of from the two
            standardized ends. Static; True for models with observed intervals,
            False keeps the fixed-threshold update unchanged.

    Return:
        - new_states: Same shape as `states`.
        - new_upper_chols: Same shape as `upper_chols`.
        - new_log_mixture_weights: Same shape as `log_mixture_weights`.
        - new_loglikes: 1d array of length n_obs; 0 for missing observations.
        - debug_info: `residuals` and `residual_sds` of shape (n_obs, n_mixtures)
          with `y - mu` and `s` for `DENSITY` observations and NaN otherwise.

    """
    n_obs, n_mixtures, n_states = states.shape
    not_missing = events != ObservationEvent.MISSING
    is_density = (events == ObservationEvent.DENSITY)[:, None]
    is_upper = (events == ObservationEvent.UPPER_TAIL)[:, None]
    is_tail = is_upper | (events == ObservationEvent.LOWER_TAIL)[:, None]

    # Placeholders keep every intermediate and its derivative finite for persons
    # without an observation; their results are discarded below.
    safe_controls = jnp.where(not_missing[:, None], controls, 0.0)
    safe_chols = jnp.where(
        not_missing.reshape(n_obs, 1, 1, 1),
        upper_chols,
        jnp.eye(n_states, dtype=upper_chols.dtype),
    )

    v = jnp.einsum("nkij,j->nki", safe_chols, loadings)
    # Accumulating hypot from sigma > 0 keeps s differentiable at v = 0.
    s = jnp.broadcast_to(meas_sd, (n_obs, n_mixtures))
    for i in range(n_states):
        s = jnp.hypot(s, v[..., i])
    w = v / s[..., None]
    alpha = jnp.sum(w * w, axis=-1)
    mu = jnp.dot(states, loadings) + jnp.dot(safe_controls, control_params)[:, None]

    safe_y = jnp.where(is_density, measurements[:, None], mu)
    innovation = (safe_y - mu) / s
    bounds = jnp.broadcast_to(thresholds, (n_obs, 2))
    upper_end, lower_end = bounds[:, :1], bounds[:, 1:]
    direction = jnp.where(is_upper, 1.0, -1.0)
    threshold = jnp.where(is_tail, jnp.where(is_upper, lower_end, upper_end), 0.0)
    t = jnp.where(is_tail, direction * (mu - threshold) / s, 0.0)
    log_cdf, mills, contraction, complement = normal_tail_terms(t)

    coefficient = jnp.where(is_tail, direction * mills, innovation)
    d = jnp.where(is_tail, contraction, 1.0)
    one_minus_d = jnp.where(is_tail, complement, 0.0)
    loglikes_per_dist = jnp.where(
        is_tail,
        log_cdf,
        -0.5 * innovation * innovation - jnp.log(s) - LOG_SQRT_2PI,
    )
    if has_intervals:
        coefficient, d, one_minus_d, loglikes_per_dist = _with_interval_events(
            terms=(coefficient, d, one_minus_d, loglikes_per_dist),
            is_interval=(events == ObservationEvent.INTERVAL)[:, None],
            lower_end=lower_end,
            upper_end=upper_end,
            mu=mu,
            s=s,
            source_intervals=source_intervals,
        )

    _new_states = (
        states + jnp.einsum("nkji,nkj->nki", safe_chols, w) * coefficient[..., None]
    )
    c = jnp.sqrt((meas_sd / s) ** 2 + one_minus_d * alpha)
    b = d / (1.0 + c)
    w_u = jnp.einsum("nkj,nkji->nki", w, safe_chols)
    factor = safe_chols - b[..., None, None] * w[..., :, None] * w_u[..., None, :]
    _new_upper_chols = jnp.linalg.qr(factor)[1]

    if n_mixtures >= 2:
        weighted = loglikes_per_dist + log_mixture_weights
        _loglikes = jax.scipy.special.logsumexp(weighted, axis=1)
        _new_log_mixture_weights = weighted - _loglikes[:, None]
    else:
        _loglikes = loglikes_per_dist[:, 0]
        _new_log_mixture_weights = log_mixture_weights

    new_states = jnp.where(not_missing.reshape(n_obs, 1, 1), _new_states, states)
    new_upper_chols = jnp.where(
        not_missing.reshape(n_obs, 1, 1, 1), _new_upper_chols, upper_chols
    )
    new_loglikes = jnp.where(not_missing, _loglikes, 0.0)
    new_log_mixture_weights = jnp.where(
        not_missing[:, None], _new_log_mixture_weights, log_mixture_weights
    )
    if n_mixtures >= 2:
        new_states, new_upper_chols = reset_negligible_mixture_components(
            states=new_states,
            upper_chols=new_upper_chols,
            log_mixture_weights=new_log_mixture_weights,
        )

    observed_density = is_density & not_missing[:, None]
    debug_info = {
        "residuals": jnp.where(observed_density, safe_y - mu, jnp.nan),
        "residual_sds": jnp.where(observed_density, s, jnp.nan),
        "log_mixture_weights": new_log_mixture_weights,
    }
    return (
        new_states,
        new_upper_chols,
        new_log_mixture_weights,
        new_loglikes,
        debug_info,
    )


def _with_interval_events(
    terms: tuple[Array, Array, Array, Array],
    is_interval: Array,
    lower_end: Array,
    upper_end: Array,
    mu: Array,
    s: Array,
    *,
    source_intervals: bool,
) -> tuple[Array, Array, Array, Array]:
    """Replace the update terms of `INTERVAL` observations by their interval moments.

    `terms` are the mean coefficient, `d`, `1 - d` and the log likelihood of each
    observation and component. With `source_intervals`, the ends, `mu` and `s` go to
    `source_interval_terms` as they are; otherwise the standardized ends go to
    `normal_interval_terms`. Rows without an interval get finite placeholder
    arguments before any arithmetic, so their discarded terms have finite values
    and derivatives.
    """
    if source_intervals:
        log_z, mean_z, contraction_z, var_z = source_interval_terms(
            lower=jnp.where(is_interval, lower_end, -1.0),
            upper=jnp.where(is_interval, upper_end, 1.0),
            location=jnp.where(is_interval, mu, 0.0),
            scale=jnp.where(is_interval, s, 1.0),
            log_scale=jnp.where(is_interval, jnp.log(s), 0.0),
        )
    else:
        lower_std = (jnp.where(is_interval, lower_end, mu - s) - mu) / s
        upper_std = (jnp.where(is_interval, upper_end, mu + s) - mu) / s
        log_z, mean_z, contraction_z, var_z = normal_interval_terms(
            lower_std, upper_std
        )
    coefficient, d, one_minus_d, loglikes = terms
    return (
        jnp.where(is_interval, mean_z, coefficient),
        jnp.where(is_interval, contraction_z, d),
        jnp.where(is_interval, var_z, one_minus_d),
        jnp.where(is_interval, log_z, loglikes),
    )
