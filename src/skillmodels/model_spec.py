"""Strongly-typed model specification dataclasses.

This module provides frozen dataclasses for defining model specifications
in a type-safe, immutable manner. All collections use immutable types
(tuples, frozendict) to ensure the specification cannot be accidentally modified.
"""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Self

from frozendict import frozendict

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class Normalizations:
    """Normalizations for factor identification.

    Attributes:
        loadings: Per-period loading normalizations. Each element is a mapping
            from variable name to fixed loading value.
        intercepts: Per-period intercept normalizations. Each element is a mapping
            from variable name to fixed intercept value.

    """

    loadings: tuple[frozendict[str, float], ...]
    intercepts: tuple[frozendict[str, float], ...]

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        """Create Normalizations from a dictionary specification."""
        return cls(
            loadings=tuple(frozendict(x) for x in d["loadings"]),
            intercepts=tuple(frozendict(x) for x in d["intercepts"]),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for backwards compatibility."""
        return {
            "loadings": [dict(x) for x in self.loadings],
            "intercepts": [dict(x) for x in self.intercepts],
        }


@dataclass(frozen=True)
class FactorSpec:
    """Specification for a single latent factor.

    Attributes:
        measurements: Per-period measurement variables. Each element is a tuple
            of variable names measured in that period.
        normalizations: Identification normalizations for this factor.
        is_endogenous: Whether this factor is endogenous.
        is_correction: Whether this factor is a correction factor.
        transition_function: Optional transition function for this factor.
            Can be a string (referencing built-in functions) or a callable.

    """

    measurements: tuple[tuple[str, ...], ...]
    normalizations: Normalizations | None = None
    is_endogenous: bool = False
    is_correction: bool = False
    transition_function: str | Callable | None = None

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        """Create FactorSpec from a dictionary specification."""
        normalizations = None
        if "normalizations" in d:
            normalizations = Normalizations.from_dict(d["normalizations"])

        return cls(
            measurements=tuple(tuple(m) for m in d["measurements"]),
            normalizations=normalizations,
            is_endogenous=d.get("is_endogenous", False),
            is_correction=d.get("is_correction", False),
            transition_function=d.get("transition_function"),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for backwards compatibility."""
        result: dict = {
            "measurements": [list(m) for m in self.measurements],
            "is_endogenous": self.is_endogenous,
            "is_correction": self.is_correction,
        }
        if self.normalizations is not None:
            result["normalizations"] = self.normalizations.to_dict()
        if self.transition_function is not None:
            result["transition_function"] = self.transition_function
        return result

    def with_transition_function(self, func: str | Callable) -> Self:
        """Return a new FactorSpec with the given transition function."""
        return type(self)(
            measurements=self.measurements,
            normalizations=self.normalizations,
            is_endogenous=self.is_endogenous,
            is_correction=self.is_correction,
            transition_function=func,
        )

    def with_normalizations(self, normalizations: Normalizations) -> Self:
        """Return a new FactorSpec with the given normalizations."""
        return type(self)(
            measurements=self.measurements,
            normalizations=normalizations,
            is_endogenous=self.is_endogenous,
            is_correction=self.is_correction,
            transition_function=self.transition_function,
        )


@dataclass(frozen=True)
class EstimationOptionsSpec:
    """Options for model estimation.

    Attributes:
        robust_bounds: Whether to use robust bounds.
        bounds_distance: Distance for bounds.
        n_mixtures: Number of mixture components.
        sigma_points_scale: Scaling factor for sigma points in unscented transform.
        clipping_lower_bound: Lower bound for soft clipping.
        clipping_upper_bound: Upper bound for soft clipping (None for no upper bound).
        clipping_lower_hardness: Hardness of lower clipping.
        clipping_upper_hardness: Hardness of upper clipping.

    """

    robust_bounds: bool = True
    bounds_distance: float = 1e-3
    n_mixtures: int = 1
    sigma_points_scale: float = 2
    clipping_lower_bound: float = -1e30
    clipping_upper_bound: float | None = None
    clipping_lower_hardness: float = 1
    clipping_upper_hardness: float = 1

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        """Create EstimationOptionsSpec from a dictionary specification."""
        return cls(
            robust_bounds=d.get("robust_bounds", True),
            bounds_distance=d.get("bounds_distance", 1e-3),
            n_mixtures=d.get("n_mixtures", 1),
            sigma_points_scale=d.get("sigma_points_scale", 2),
            clipping_lower_bound=d.get("clipping_lower_bound", -1e30),
            clipping_upper_bound=d.get("clipping_upper_bound"),
            clipping_lower_hardness=d.get("clipping_lower_hardness", 1),
            clipping_upper_hardness=d.get("clipping_upper_hardness", 1),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for backwards compatibility."""
        result = {
            "robust_bounds": self.robust_bounds,
            "bounds_distance": self.bounds_distance,
            "n_mixtures": self.n_mixtures,
            "sigma_points_scale": self.sigma_points_scale,
            "clipping_lower_bound": self.clipping_lower_bound,
            "clipping_lower_hardness": self.clipping_lower_hardness,
            "clipping_upper_hardness": self.clipping_upper_hardness,
        }
        if self.clipping_upper_bound is not None:
            result["clipping_upper_bound"] = self.clipping_upper_bound
        return result


def _default_empty_frozendict() -> frozendict[str, str]:
    return frozendict({})


@dataclass(frozen=True)
class AnchoringSpec:
    """Specification for anchoring latent factors to outcomes.

    Attributes:
        outcomes: Mapping from factor names to outcome variable names.
        free_controls: Whether control coefficients are free in anchoring equations.
        free_constant: Whether the constant is free in anchoring equations.
        free_loadings: Whether loadings are free in anchoring equations.
        ignore_constant_when_anchoring: Whether to ignore constant when anchoring.

    """

    outcomes: frozendict[str, str] = field(default_factory=_default_empty_frozendict)
    free_controls: bool = False
    free_constant: bool = False
    free_loadings: bool = False
    ignore_constant_when_anchoring: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        """Create AnchoringSpec from a dictionary specification."""
        outcomes = d.get("outcomes", {})
        ignore_constant = d.get("ignore_constant_when_anchoring", False)
        return cls(
            outcomes=frozendict(outcomes),
            free_controls=d.get("free_controls", False),
            free_constant=d.get("free_constant", False),
            free_loadings=d.get("free_loadings", False),
            ignore_constant_when_anchoring=ignore_constant,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for backwards compatibility."""
        return {
            "outcomes": dict(self.outcomes),
            "free_controls": self.free_controls,
            "free_constant": self.free_constant,
            "free_loadings": self.free_loadings,
            "ignore_constant_when_anchoring": self.ignore_constant_when_anchoring,
        }


@dataclass(frozen=True, init=False)
class ModelSpec:
    """Complete model specification.

    This is the main strongly-typed container for model specifications.
    All fields are immutable to prevent accidental modifications.

    Attributes:
        factors: Mapping from factor name to FactorSpec.
        observed_factors: Tuple of observed factor variable names.
        controls: Tuple of control variable names.
        stagemap: Stage mapping for transition functions.
        anchoring: Anchoring specification.
        estimation_options: Estimation tuning parameters.

    """

    _factors: MappingProxyType[str, FactorSpec]
    observed_factors: tuple[str, ...] = ()
    controls: tuple[str, ...] = ()
    stagemap: tuple[int, ...] | None = None
    anchoring: AnchoringSpec | None = None
    estimation_options: EstimationOptionsSpec | None = None

    def __init__(
        self,
        factors: dict[str, FactorSpec] | MappingProxyType[str, FactorSpec],
        observed_factors: tuple[str, ...] = (),
        controls: tuple[str, ...] = (),
        stagemap: tuple[int, ...] | None = None,
        anchoring: AnchoringSpec | None = None,
        estimation_options: EstimationOptionsSpec | None = None,
    ) -> None:
        """Create ModelSpec, wrapping factors dict in MappingProxyType."""
        if isinstance(factors, MappingProxyType):
            object.__setattr__(self, "_factors", factors)
        else:
            object.__setattr__(self, "_factors", MappingProxyType(factors))
        object.__setattr__(self, "observed_factors", observed_factors)
        object.__setattr__(self, "controls", controls)
        object.__setattr__(self, "stagemap", stagemap)
        object.__setattr__(self, "anchoring", anchoring)
        object.__setattr__(self, "estimation_options", estimation_options)

    @property
    def factors(self) -> MappingProxyType[str, FactorSpec]:
        """Immutable mapping of factor names to specifications."""
        return self._factors

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        """Create ModelSpec from a dictionary specification.

        Args:
            d: Dictionary with keys 'factors', and optionally 'observed_factors',
                'controls', 'stagemap', 'anchoring', and 'estimation_options'.

        Returns:
            Immutable ModelSpec instance.

        """
        factors = {
            name: FactorSpec.from_dict(spec) for name, spec in d["factors"].items()
        }
        observed = d.get("observed_factors", [])
        controls = d.get("controls", [])
        stagemap = d.get("stagemap")
        anchoring = None
        if "anchoring" in d:
            anchoring = AnchoringSpec.from_dict(d["anchoring"])
        estimation = None
        if "estimation_options" in d:
            estimation = EstimationOptionsSpec.from_dict(d["estimation_options"])

        return cls(
            factors=MappingProxyType(factors),
            observed_factors=tuple(observed),
            controls=tuple(controls),
            stagemap=tuple(stagemap) if stagemap is not None else None,
            anchoring=anchoring,
            estimation_options=estimation,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for backwards compatibility with skillmodels.

        Returns:
            Mutable dictionary in the format expected by skillmodels.

        """
        result: dict = {
            "factors": {name: spec.to_dict() for name, spec in self.factors.items()},
            "observed_factors": list(self.observed_factors),
        }
        if self.controls:
            result["controls"] = list(self.controls)
        if self.stagemap is not None:
            result["stagemap"] = list(self.stagemap)
        if self.anchoring is not None:
            result["anchoring"] = self.anchoring.to_dict()
        if self.estimation_options is not None:
            result["estimation_options"] = self.estimation_options.to_dict()
        return result

    def with_transition_functions(
        self,
        transition_functions: dict[str, str | Callable],
    ) -> Self:
        """Return a new ModelSpec with transition functions added to factors.

        Args:
            transition_functions: Mapping from factor name to transition function.
                Can be strings (referencing built-in functions) or callables.

        Returns:
            New ModelSpec with transition functions set on factors.

        Raises:
            ValueError: If transition_functions keys don't match factor names.

        """
        if set(transition_functions.keys()) != set(self.factors.keys()):
            msg = (
                f"Transition function keys {set(transition_functions.keys())} "
                f"do not match factor keys {set(self.factors.keys())}"
            )
            raise ValueError(msg)

        new_factors = {
            name: spec.with_transition_function(transition_functions[name])
            for name, spec in self.factors.items()
        }
        return type(self)(
            factors=MappingProxyType(new_factors),
            observed_factors=self.observed_factors,
            controls=self.controls,
            stagemap=self.stagemap,
            anchoring=self.anchoring,
            estimation_options=self.estimation_options,
        )

    def with_added_factor(
        self,
        name: str,
        spec: FactorSpec,
    ) -> Self:
        """Return a new ModelSpec with an additional factor.

        Args:
            name: Name of the new factor.
            spec: Specification for the new factor.

        Returns:
            New ModelSpec with the additional factor.

        """
        new_factors = dict(self.factors)
        new_factors[name] = spec
        return type(self)(
            factors=MappingProxyType(new_factors),
            observed_factors=self.observed_factors,
            controls=self.controls,
            stagemap=self.stagemap,
            anchoring=self.anchoring,
            estimation_options=self.estimation_options,
        )

    def with_added_observed_factors(
        self,
        *names: str,
    ) -> Self:
        """Return a new ModelSpec with additional observed factors.

        Args:
            *names: Names of additional observed factors.

        Returns:
            New ModelSpec with the additional observed factors.

        """
        return type(self)(
            factors=self.factors,
            observed_factors=self.observed_factors + names,
            controls=self.controls,
            stagemap=self.stagemap,
            anchoring=self.anchoring,
            estimation_options=self.estimation_options,
        )

    def with_estimation_options(
        self,
        estimation_options: EstimationOptionsSpec,
    ) -> Self:
        """Return a new ModelSpec with the given estimation options.

        Args:
            estimation_options: New estimation options.

        Returns:
            New ModelSpec with the updated estimation options.

        """
        return type(self)(
            factors=self.factors,
            observed_factors=self.observed_factors,
            controls=self.controls,
            stagemap=self.stagemap,
            anchoring=self.anchoring,
            estimation_options=estimation_options,
        )

    def with_anchoring(
        self,
        anchoring: AnchoringSpec,
    ) -> Self:
        """Return a new ModelSpec with the given anchoring specification.

        Args:
            anchoring: New anchoring specification.

        Returns:
            New ModelSpec with the updated anchoring.

        """
        return type(self)(
            factors=self.factors,
            observed_factors=self.observed_factors,
            controls=self.controls,
            stagemap=self.stagemap,
            anchoring=anchoring,
            estimation_options=self.estimation_options,
        )

    def with_controls(
        self,
        controls: tuple[str, ...],
    ) -> Self:
        """Return a new ModelSpec with the given controls.

        Args:
            controls: New control variable names.

        Returns:
            New ModelSpec with the updated controls.

        """
        return type(self)(
            factors=self.factors,
            observed_factors=self.observed_factors,
            controls=controls,
            stagemap=self.stagemap,
            anchoring=self.anchoring,
            estimation_options=self.estimation_options,
        )

    def with_stagemap(
        self,
        stagemap: tuple[int, ...],
    ) -> Self:
        """Return a new ModelSpec with the given stagemap.

        Args:
            stagemap: New stage mapping.

        Returns:
            New ModelSpec with the updated stagemap.

        """
        return type(self)(
            factors=self.factors,
            observed_factors=self.observed_factors,
            controls=self.controls,
            stagemap=stagemap,
            anchoring=self.anchoring,
            estimation_options=self.estimation_options,
        )
