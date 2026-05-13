"""CHS-specific estimation options."""

from dataclasses import dataclass
from typing import Literal


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
    start_params_strategy: Literal["none", "spearman", "amn"] = "amn"
    """How to populate the `value` column of the `params_template`.

    `"amn"` (default) runs the full Attanasio-Meghir-Nix (2020)
    three-stage estimator and uses its parameter estimates as starting
    values for the downstream MLE. `"spearman"` seeds free entries
    from Spearman cross-covariance / Bartlett-OLS moments only (fast
    but less accurate on non-Gaussian factor distributions). `"none"`
    leaves free entries as `NaN` so the caller can fill them.
    """

    def __post_init__(self) -> None:  # noqa: D105
        if not self.robust_bounds:
            object.__setattr__(self, "bounds_distance", 0.0)
