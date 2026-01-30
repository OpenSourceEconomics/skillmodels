"""Utility functions for manipulating model specifications and parameters."""

import warnings
from dataclasses import replace

import numpy as np
import pandas as pd

from skillmodels.model_spec import (
    FactorSpec,
    ModelSpec,
    Normalizations,
)
from skillmodels.params_index import get_params_index
from skillmodels.process_model import (
    get_dimensions,
    get_has_endogenous_factors,
    process_model,
)


def extract_factors(
    factors: str | list[str],
    model_spec: ModelSpec,
    params: pd.DataFrame | None = None,
) -> ModelSpec | tuple[ModelSpec, pd.DataFrame]:
    """Reduce a specification to a model with fewer latent factors.

    If provided, a params DataFrame is also reduced correspondingly.

    Args:
        factors: Name(s) of the factor(s) to extract.
        model_spec: The model specification. See: :ref:`model_specs`.
        params: The params DataFrame for the full model.

    Returns:
        ModelSpec: The reduced model specification
        pandas.DataFrame: The reduced parameter DataFrame (only if params is not None)

    """
    if isinstance(factors, str):
        factors = [factors]

    to_remove = list(set(model_spec.factors).difference(factors))
    return remove_factors(factors=to_remove, model_spec=model_spec, params=params)


def update_parameter_values(
    params: pd.DataFrame,
    others: pd.DataFrame | list[pd.DataFrame],
) -> pd.DataFrame:
    """Update the "value" column of params with values from other.

    Args:
        params: The params DataFrame for the full model.
        others: Another DataFrame with parameters or list
            of thereof. The values from other are used to update the value column
            of ``params``. If other is a list, the updates will be in order, i.e.
            later elements overwrite earlier ones.

    Returns:
        pandas.DataFrame: Updated copy of params.

    """
    if isinstance(others, pd.DataFrame):
        others = [others]

    out = params.copy(deep=True)

    # Create a temporary Series to hold the updated values
    temp_series = out["value"].copy()

    # Update the temporary Series
    for other in others:
        temp_series.update(other["value"])

    # Assign the updated Series back to the DataFrame
    out["value"] = temp_series

    return out


def remove_factors(
    factors: str | list[str],
    model_spec: ModelSpec,
    params: pd.DataFrame | None = None,
) -> ModelSpec | tuple[ModelSpec, pd.DataFrame]:
    """Remove factors from a model specification.

    If provided, a params DataFrame is also reduced correspondingly.

    It is possible that the reduced model has fewer periods than the original one.
    This happens if the remaining factors do not have measurements in later periods.

    Args:
        factors: Name(s) of the factor(s) to remove.
        model_spec: The model specification. See: :ref:`model_specs`.
        params: The params DataFrame for the full model.

    Returns:
        ModelSpec: The reduced model specification
        pandas.DataFrame: The reduced parameter DataFrame (only if params is not None)

    """
    if isinstance(factors, str):
        factors = [factors]

    # We need this for the full model when endogenous factors are present.
    has_endogenous_factors = get_has_endogenous_factors(model_spec.factors)

    new_factors = {k: v for k, v in model_spec.factors.items() if k not in factors}

    # adjust anchoring
    new_anchoring = model_spec.anchoring
    if new_anchoring is not None:
        new_outcomes = {
            k: v for k, v in new_anchoring.outcomes.items() if k not in factors
        }
        if new_outcomes:
            new_anchoring = replace(new_anchoring, outcomes=new_outcomes)
        else:
            new_anchoring = None

    out = model_spec._replace(
        factors=new_factors,
        anchoring=new_anchoring,
    )

    # Remove periods if necessary, but only if no endogenous factors are present.
    # (else we would mess up the mapping between raw periods model periods)
    if not has_endogenous_factors:
        new_n_periods = get_dimensions(
            out, has_endogenous_factors=has_endogenous_factors
        ).n_periods
        reduced = reduce_n_periods(model_spec=out, new_n_periods=new_n_periods)
        if not isinstance(reduced, ModelSpec):
            msg = "Expected ModelSpec from reduce_n_periods without params"
            raise TypeError(msg)
        out = reduced

    if params is not None:
        out_params = _reduce_params(
            params,
            out,
            has_endogenous_factors=has_endogenous_factors,
        )
        return (out, out_params)

    return out


def remove_measurements(
    measurements: str | list[str],
    model_spec: ModelSpec,
    params: pd.DataFrame | None = None,
) -> ModelSpec | tuple[ModelSpec, pd.DataFrame]:
    """Remove measurements from a model specification.

    If provided, a params DataFrame is also reduced correspondingly.

    Args:
        measurements: Name(s) of the measurement(s) to remove.
        model_spec: The model specification. See: :ref:`model_specs`.
        params: The params DataFrame for the full model.

    Returns:
        ModelSpec: The reduced model specification
        pandas.DataFrame: The reduced parameter DataFrame (only if params is not None)

    """
    if isinstance(measurements, str):
        measurements = [measurements]

    new_factors: dict[str, FactorSpec] = {}
    for factor, fspec in model_spec.factors.items():
        new_meas = tuple(
            tuple(m for m in period_meas if m not in measurements)
            for period_meas in fspec.measurements
        )

        new_normalizations = fspec.normalizations
        if new_normalizations is not None:
            new_loadings = tuple(
                {k: v for k, v in d.items() if k not in measurements}
                for d in new_normalizations.loadings
            )
            new_intercepts = tuple(
                {k: v for k, v in d.items() if k not in measurements}
                for d in new_normalizations.intercepts
            )
            if new_loadings != new_normalizations.loadings or (
                new_intercepts != new_normalizations.intercepts
            ):
                warnings.warn(
                    "Your removed a normalized measurement from a model. Make sure "
                    "there are enough normalizations left to ensure identification.",
                    stacklevel=2,
                )
            new_normalizations = Normalizations(
                loadings=new_loadings,
                intercepts=new_intercepts,
            )

        new_factors[factor] = replace(
            fspec, measurements=new_meas, normalizations=new_normalizations
        )

    out = model_spec._replace(factors=new_factors)

    if params is not None:
        # This likely won't work if we have endogenous factors.
        out_params = _reduce_params(params, out, has_endogenous_factors=False)
        return (out, out_params)

    return out


def remove_controls(
    controls: str | list[str],
    model_spec: ModelSpec,
    params: pd.DataFrame | None = None,
) -> ModelSpec | tuple[ModelSpec, pd.DataFrame]:
    """Remove control variables from a model specification.

    If provided, a params DataFrame is also reduced correspondingly.

    Args:
        controls: Name(s) of the contral variable(s) to remove.
        model_spec: The model specification. See: :ref:`model_specs`.
        params: The params DataFrame for the full model.

    Returns:
        ModelSpec: The reduced model specification
        pandas.DataFrame: The reduced parameter DataFrame (only if params is not None)

    """
    if isinstance(controls, str):
        controls = [controls]

    new_controls = tuple(c for c in model_spec.controls if c not in controls)
    out = model_spec._replace(controls=new_controls)

    if params is not None:
        # This likely won't work if we have endogenous factors.
        out_params = _reduce_params(params, out, has_endogenous_factors=False)
        return (out, out_params)

    return out


def switch_translog_to_linear(
    model_spec: ModelSpec,
    params: pd.DataFrame | None = None,
) -> ModelSpec | tuple[ModelSpec, pd.DataFrame]:
    """Switch all translog production functions to linear.

    If provided, a params DataFrame is also reduced correspondingly.

    Args:
        model_spec: The model specification. See: :ref:`model_specs`.
        params: The params DataFrame for the full model.

    Returns:
        ModelSpec: The reduced model specification
        pandas.DataFrame: The reduced parameter DataFrame (only if params is not None)

    """
    new_factors: dict[str, FactorSpec] = {}
    for name, fspec in model_spec.factors.items():
        if fspec.transition_function == "translog":
            new_factors[name] = fspec.with_transition_function("linear")
        else:
            new_factors[name] = fspec
    out = model_spec._replace(factors=new_factors)

    if params is not None:
        # This likely won't work if we have endogenous factors.
        out_params = _reduce_params(params, out, has_endogenous_factors=False)
        return (out, out_params)

    return out


def switch_linear_to_translog(
    model_spec: ModelSpec,
    params: pd.DataFrame | None = None,
) -> ModelSpec | tuple[ModelSpec, pd.DataFrame]:
    """Switch all linear production functions to translog.

    If provided, a params DataFrame is also extended correspondingly. The fill value
    for the additional terms is 0.05 because experience showed that estimating a
    translog model with start parameters obtained from a linear model is faster when
    the additional parameters are not initialized at zero.

    Args:
        model_spec: The model specification. See: :ref:`model_specs`.
        params: The params DataFrame for the full model.

    Returns:
        ModelSpec: The reduced model specification
        pandas.DataFrame: The reduced parameter DataFrame (only if params is not None)

    """
    new_factors: dict[str, FactorSpec] = {}
    for name, fspec in model_spec.factors.items():
        if fspec.transition_function == "linear":
            new_factors[name] = fspec.with_transition_function("translog")
        else:
            new_factors[name] = fspec
    out = model_spec._replace(factors=new_factors)

    if params is not None:
        out_params = _extend_params(params=params, model_spec=out, fill_value=0.05)
        return (out, out_params)

    return out


def reduce_n_periods(
    model_spec: ModelSpec,
    new_n_periods: int,
    params: pd.DataFrame | None = None,
) -> ModelSpec | tuple[ModelSpec, pd.DataFrame]:
    """Remove all periods after n_periods.

    Args:
        model_spec: The model specification. See: :ref:`model_specs`.
        new_n_periods: The new number of periods.
        params: The params DataFrame for the full model.

    Returns:
        ModelSpec: The reduced model specification
        pandas.DataFrame: The reduced parameter DataFrame (only if params is not None)

    """
    new_factors: dict[str, FactorSpec] = {}
    for name, fspec in model_spec.factors.items():
        new_meas = fspec.measurements[:new_n_periods]
        new_normalizations = fspec.normalizations
        if new_normalizations is not None:
            new_normalizations = Normalizations(
                loadings=new_normalizations.loadings[:new_n_periods],
                intercepts=new_normalizations.intercepts[:new_n_periods],
            )
        new_factors[name] = replace(
            fspec, measurements=new_meas, normalizations=new_normalizations
        )

    new_stagemap = model_spec.stagemap
    if new_stagemap is not None and len(new_stagemap) > new_n_periods - 1:
        new_stagemap = new_stagemap[: new_n_periods - 1]

    out = model_spec._replace(
        factors=new_factors,
        stagemap=new_stagemap,
    )

    if params is not None:
        out_params = _extend_params(params=params, model_spec=out, fill_value=0.05)
        return (out, out_params)

    return out


def _reduce_params(
    params: pd.DataFrame,
    model_spec: ModelSpec,
    *,
    has_endogenous_factors: bool,
) -> pd.DataFrame:
    """Reduce a parameter DataFrame from a larger model to a reduced model.

    The reduced model must be nested in the original model for which the params
    DataFrame was constructed.

    Args:
        params: The params DataFrame for the full model.
        model_spec: The model specification. See: :ref:`model_specs`.
        has_endogenous_factors: Whether the model has endogenous factors.

    Returns:
        pandas.DataFrame: The reduced parameters DataFrame.

    """
    index = _get_params_index(model_spec)
    # If we have endogenous factors, we need to keep the periods from params.
    if has_endogenous_factors:
        df = pd.merge(
            left=params.reset_index(),
            right=index.to_frame(index=False)[
                ["category", "name1", "name2"]
            ].drop_duplicates(),
            on=["category", "name1", "name2"],
            how="right",
        )
        index = pd.MultiIndex.from_frame(df[params.index.names])
    return params.loc[index]


def _extend_params(
    params: pd.DataFrame,
    model_spec: ModelSpec,
    fill_value: float,
) -> pd.DataFrame:
    index = _get_params_index(model_spec)
    out = params.reindex(index)
    out["value"] = out["value"].fillna(fill_value)
    if "lower_bound" in out:
        out["lower_bound"] = out["lower_bound"].fillna(-np.inf)

    if "upper_bound" in out:
        out["upper_bound"] = out["upper_bound"].fillna(np.inf)

    return out


def _get_params_index(
    model_spec: ModelSpec,
) -> pd.MultiIndex:
    mod = process_model(model_spec)
    return get_params_index(
        update_info=mod.update_info,
        labels=mod.labels,
        dimensions=mod.dimensions,
        transition_info=mod.transition_info,
        endogenous_factors_info=mod.endogenous_factors_info,
    )
