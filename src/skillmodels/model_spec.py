"""Strongly-typed model specification dataclasses.

This module provides frozen dataclasses for defining model specifications
in a type-safe, immutable manner. All collections use immutable types
(tuples, MappingProxyType) to ensure the specification cannot be accidentally
modified.
"""

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Self


@dataclass(frozen=True)
class Normalizations:
    """Normalizations for factor identification.

    Attributes:
        loadings: Per-period loading normalizations. Each element is a mapping
            from variable name to fixed loading value.
        intercepts: Per-period intercept normalizations. Each element is a mapping
            from variable name to fixed intercept value.

    """

    loadings: tuple[MappingProxyType[str, float], ...]
    intercepts: tuple[MappingProxyType[str, float], ...]

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
        return replace(self, transition_function=func)

    def with_normalizations(self, normalizations: Normalizations) -> Self:
        """Return a new FactorSpec with the given normalizations."""
        return replace(self, normalizations=normalizations)


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


def _default_empty_mapping_proxy() -> MappingProxyType[str, str]:
    return MappingProxyType({})


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

    outcomes: MappingProxyType[str, str] = field(
        default_factory=_default_empty_mapping_proxy,
    )
    free_controls: bool = False
    free_constant: bool = False
    free_loadings: bool = False
    ignore_constant_when_anchoring: bool = False

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

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        """Create a ModelSpec from a dictionary (e.g. loaded from YAML).

        Args:
            d: A dictionary with keys like "factors", "observed_factors",
                "controls", "stagemap", "anchoring", "estimation_options".

        Returns:
            A ModelSpec instance.

        """
        factors = {}
        for name, spec in d["factors"].items():
            normalizations = None
            if "normalizations" in spec:
                nd = spec["normalizations"]
                if "intercepts" not in nd:
                    n_periods = len(nd.get("loadings", []))
                    nd["intercepts"] = [{} for _ in range(n_periods)]
                normalizations = Normalizations(
                    loadings=tuple(MappingProxyType(x) for x in nd["loadings"]),
                    intercepts=tuple(MappingProxyType(x) for x in nd["intercepts"]),
                )
            factors[name] = FactorSpec(
                measurements=tuple(tuple(m) for m in spec["measurements"]),
                normalizations=normalizations,
                is_endogenous=spec.get("is_endogenous", False),
                is_correction=spec.get("is_correction", False),
                transition_function=spec.get("transition_function"),
            )

        anchoring = None
        if "anchoring" in d:
            ad = d["anchoring"]
            anchoring = AnchoringSpec(
                outcomes=MappingProxyType(ad.get("outcomes", {})),
                free_controls=ad.get("free_controls", False),
                free_constant=ad.get("free_constant", False),
                free_loadings=ad.get("free_loadings", False),
                ignore_constant_when_anchoring=ad.get(
                    "ignore_constant_when_anchoring", False
                ),
            )

        estimation = None
        if "estimation_options" in d:
            ed = d["estimation_options"]
            estimation = EstimationOptionsSpec(
                robust_bounds=ed.get("robust_bounds", True),
                bounds_distance=ed.get("bounds_distance", 1e-3),
                n_mixtures=ed.get("n_mixtures", 1),
                sigma_points_scale=ed.get("sigma_points_scale", 2),
                clipping_lower_bound=ed.get("clipping_lower_bound", -1e30),
                clipping_upper_bound=ed.get("clipping_upper_bound"),
                clipping_lower_hardness=ed.get("clipping_lower_hardness", 1),
                clipping_upper_hardness=ed.get("clipping_upper_hardness", 1),
            )

        stagemap = d.get("stagemap")

        return cls(
            factors=factors,
            observed_factors=tuple(d.get("observed_factors", [])),
            controls=tuple(d.get("controls", [])),
            stagemap=tuple(stagemap) if stagemap is not None else None,
            anchoring=anchoring,
            estimation_options=estimation,
        )

    @property
    def factors(self) -> MappingProxyType[str, FactorSpec]:
        """Immutable mapping of factor names to specifications."""
        return self._factors

    def _replace(self, **changes: Any) -> Self:
        """Return a new ModelSpec with the specified fields replaced."""
        return type(self)(
            factors=changes.get("factors", self.factors),
            observed_factors=changes.get("observed_factors", self.observed_factors),
            controls=changes.get("controls", self.controls),
            stagemap=changes.get("stagemap", self.stagemap),
            anchoring=changes.get("anchoring", self.anchoring),
            estimation_options=changes.get(
                "estimation_options", self.estimation_options
            ),
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
        return self._replace(factors=MappingProxyType(new_factors))

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
        return self._replace(factors=MappingProxyType(new_factors))

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
        return self._replace(
            observed_factors=self.observed_factors + names,
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
        return self._replace(estimation_options=estimation_options)

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
        return self._replace(anchoring=anchoring)

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
        return self._replace(controls=controls)

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
        return self._replace(stagemap=stagemap)
