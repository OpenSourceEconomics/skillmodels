"""Strongly-typed model specification dataclasses.

This module provides frozen dataclasses for defining model specifications
in a type-safe, immutable manner. All collections use immutable types
(tuples, MappingProxyType) to ensure the specification cannot be accidentally
modified.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Self

from skillmodels._beartype_conf import MODEL_SPEC_CONF, beartype_init
from skillmodels.common.types import (
    Normalizations,
    ensure_containers_are_immutable,
)


@beartype_init(MODEL_SPEC_CONF)
@dataclass(frozen=True)
class FactorSpec:
    """Specification for a single latent factor."""

    measurements: tuple[tuple[str, ...], ...]
    """Per-period measurement variables. Each element is a tuple of variable names."""
    normalizations: Normalizations | None = None
    """Identification normalizations for this factor."""
    is_endogenous: bool = False
    """Whether this factor is endogenous."""
    is_correction: bool = False
    """Whether this factor is a correction factor."""
    transition_function: str | Callable | None = None
    """Transition function name (e.g. `"linear"`, `"log_ces"`) or a callable."""
    has_production_shock: bool = True
    """Whether transitions add a stochastic shock for this factor.

    When `False`, the AF transition integrates the factor deterministically:
    no shock SD parameter, no shock dimension in the joint Halton draw, and
    the transition output is used as-is. Set this to `False` for
    time-invariant factors (combined with an identity transition pinned via
    `fixed_params`) to cut integration dimensionality.
    """
    has_initial_distribution: bool = True
    """Whether this factor is drawn from the AF period-0 mixture distribution.

    When `False`, the factor is not included in the initial joint mixture
    (no mean / Cholesky entries for it) and is instead reconstructed
    deterministically per Halton draw. Currently only supported in
    conjunction with `is_endogenous=True`: the factor's period-0 value is
    computed from its investment equation at period 0 plus an investment
    shock, with investment-equation and shock parameters estimated as part
    of the initial step. The transition function must not depend on the
    factor's own lag.
    """

    def with_transition_function(self, func: str | Callable) -> Self:
        """Return a new FactorSpec with the given transition function."""
        return replace(self, transition_function=func)

    def with_normalizations(self, normalizations: Normalizations) -> Self:
        """Return a new FactorSpec with the given normalizations."""
        return replace(self, normalizations=normalizations)


@beartype_init(MODEL_SPEC_CONF)
@dataclass(frozen=True)
class AnchoringSpec:
    """Specification for anchoring latent factors to outcomes."""

    outcomes: Mapping[str, str] = field(default_factory=dict)
    """Mapping from factor names to outcome variable names."""
    free_controls: bool = False
    """Whether control coefficients are free in anchoring equations."""
    free_constant: bool = False
    """Whether the constant is free in anchoring equations."""
    free_loadings: bool = False
    """Whether loadings are free in anchoring equations."""
    ignore_constant_when_anchoring: bool = False
    """Whether to ignore constant when anchoring."""

    def __post_init__(self) -> None:  # noqa: D105
        object.__setattr__(
            self, "outcomes", ensure_containers_are_immutable(self.outcomes)
        )


@beartype_init(MODEL_SPEC_CONF)
@dataclass(frozen=True, init=False)
class ModelSpec:
    """Complete model specification.

    All fields are immutable to prevent accidental modifications.
    """

    _factors: MappingProxyType[str, FactorSpec]
    observed_factors: tuple[str, ...] = ()
    """Observed factor variable names."""
    controls: tuple[str, ...] = ()
    """Control variable names."""
    stagemap: tuple[int, ...] | None = None
    """Stage mapping for transition functions."""
    anchoring: AnchoringSpec | None = None
    """Anchoring specification."""
    n_mixtures: int = 1
    """Number of Gaussian-mixture components in the latent-factor distribution."""

    def __init__(
        self,
        factors: Mapping[str, FactorSpec],
        observed_factors: tuple[str, ...] = (),
        controls: tuple[str, ...] = (),
        stagemap: tuple[int, ...] | None = None,
        anchoring: AnchoringSpec | None = None,
        n_mixtures: int = 1,
    ) -> None:
        """Create ModelSpec, wrapping factors dict in MappingProxyType."""
        object.__setattr__(self, "_factors", ensure_containers_are_immutable(factors))
        object.__setattr__(self, "observed_factors", observed_factors)
        object.__setattr__(self, "controls", controls)
        object.__setattr__(self, "stagemap", stagemap)
        object.__setattr__(self, "anchoring", anchoring)
        object.__setattr__(self, "n_mixtures", n_mixtures)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        """Create a ModelSpec from a dictionary (e.g. loaded from YAML).

        Args:
            d: A dictionary with keys like "factors", "observed_factors",
                "controls", "stagemap", "anchoring", "n_mixtures".

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
                    loadings=tuple(nd["loadings"]),
                    intercepts=tuple(nd["intercepts"]),
                )
            factors[name] = FactorSpec(
                measurements=tuple(tuple(m) for m in spec["measurements"]),
                normalizations=normalizations,
                is_endogenous=spec.get("is_endogenous", False),
                is_correction=spec.get("is_correction", False),
                transition_function=spec.get("transition_function"),
                has_production_shock=spec.get("has_production_shock", True),
                has_initial_distribution=spec.get("has_initial_distribution", True),
            )

        anchoring = None
        if "anchoring" in d:
            anchoring = AnchoringSpec(**d["anchoring"])

        stagemap = d.get("stagemap")

        return cls(
            factors=factors,
            observed_factors=tuple(d.get("observed_factors", [])),
            controls=tuple(d.get("controls", [])),
            stagemap=tuple(stagemap) if stagemap is not None else None,
            anchoring=anchoring,
            n_mixtures=d.get("n_mixtures", 1),
        )

    @property
    def factors(self) -> MappingProxyType[str, FactorSpec]:
        """Immutable mapping of factor names to specifications."""
        return self._factors

    def _replace(self, **changes: Any) -> Self:  # noqa: ANN401
        """Return a new ModelSpec with the specified fields replaced."""
        return type(self)(
            factors=changes.get("factors", self.factors),
            observed_factors=changes.get("observed_factors", self.observed_factors),
            controls=changes.get("controls", self.controls),
            stagemap=changes.get("stagemap", self.stagemap),
            anchoring=changes.get("anchoring", self.anchoring),
            n_mixtures=changes.get("n_mixtures", self.n_mixtures),
        )

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
        return self._replace(factors=new_factors)

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
        return self._replace(factors=new_factors)

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
