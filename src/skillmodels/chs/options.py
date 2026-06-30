"""CHS-specific estimation options."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from skillmodels._beartype_conf import OPTIONS_CONF, beartype_init
from skillmodels.common.types import ensure_containers_are_immutable


@beartype_init(OPTIONS_CONF)
@dataclass(frozen=True)
class CHSEstimationOptions:
    """Tuning parameters for the CHS Kalman-MLE estimator."""

    robust_bounds: bool = True
    """Whether to use robust bounds."""
    bounds_distance: float = 1e-3
    """Distance for bounds. Zeroed out if `robust_bounds` is False."""
    sigma_points_scale: float = 2
    """Scaling factor for sigma points in unscented transform."""
    clipping_lower_bound: float = -1e30
    """Lower bound for soft clipping."""
    clipping_upper_bound: float | None = None
    """Upper bound for soft clipping (None for no upper bound)."""
    clipping_lower_hardness: float = 1
    """Hardness of lower clipping."""
    clipping_upper_hardness: float = 1
    """Hardness of upper clipping."""
    start_params_strategy: Literal["none", "constant", "spearman", "amn"] = "amn"
    """How to populate the `value` column of the `params_template`.

    Canonical name shared with `AFEstimationOptions`; the literal set is
    unified across the two likelihood estimators.

    `"amn"` (default) runs the full Attanasio-Meghir-Nix (2020)
    three-stage estimator and uses its parameter estimates as starting
    values for the downstream MLE. `"spearman"` seeds free entries
    from Spearman cross-covariance / Bartlett-OLS moments only (fast
    but less accurate on non-Gaussian factor distributions). `"none"`
    leaves free entries as `NaN` so the caller can fill them.
    `"constant"` is accepted for cross-estimator symmetry and, for the
    Kalman template, behaves identically to `"none"` (no moment seeding;
    the template's default values stand).
    """

    optimizer_algorithm: str = "scipy_lbfgsb"
    """`optimagic` algorithm name for the maximisation.

    `estimate_chs` forwards it as `estimate_ml`'s
    `optimize_options["algorithm"]` (e.g. `"scipy_lbfgsb"`, `"fides"`)."""

    optimizer_options: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Algorithm-specific options (`optimize_options["algo_options"]`).

    Forwarded by `estimate_chs` to `estimate_ml` (e.g. convergence
    tolerances, trust-region settings for `fides`)."""

    estimate_ml_options: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Extra keyword arguments forwarded verbatim to `estimagic.estimate_ml`.

    The generic estimagic pass-through for everything beyond the optimiser
    knobs above — e.g. `logging` (an `optimagic` log-options object),
    `hessian`, `jacobian`, `design_info`. `estimate_chs` defaults
    `hessian=False` (OPG/jacobian-based standard errors, since the numerical
    Hessian is prohibitively expensive); override it here for the sandwich
    covariance. `estimate_chs` manages `loglike`, `params`, `bounds`,
    `constraints`, and `optimize_options` itself, so do not set those here."""

    def __post_init__(self) -> None:  # noqa: D105
        if not self.robust_bounds:
            object.__setattr__(self, "bounds_distance", 0.0)
        object.__setattr__(
            self,
            "optimizer_options",
            ensure_containers_are_immutable(dict(self.optimizer_options)),
        )
        object.__setattr__(
            self,
            "estimate_ml_options",
            ensure_containers_are_immutable(dict(self.estimate_ml_options)),
        )
