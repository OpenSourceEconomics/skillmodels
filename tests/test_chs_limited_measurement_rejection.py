"""CHS must reject non-Gaussian measurements rather than treat them as continuous.

The CHS likelihood runs a Gaussian Kalman update for every measurement, so a probit
or Tobit measure would be silently estimated as a continuous Gaussian one.
"""

import numpy as np
import pandas as pd
import pytest

from skillmodels.chs.maximization_inputs import get_maximization_inputs
from skillmodels.chs.options import CHSEstimationOptions
from skillmodels.common.measurement_models import (
    ProbitMeasurement,
    TobitMeasurement,
)
from skillmodels.common.model_spec import (
    FactorSpec,
    ModelSpec,
    Normalizations,
)


def _model(measurement_models) -> ModelSpec:
    return ModelSpec(
        factors={
            "skills": FactorSpec(
                measurements=(("y1", "y2", "y3"),) * 2,
                normalizations=Normalizations(
                    loadings=({"y1": 1},) * 2,
                    intercepts=({"y1": 0},) * 2,
                ),
                transition_function="linear",
            ),
        },
        measurement_models=measurement_models,
    )


def _tiny_data() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = [
        {
            "caseid": caseid,
            "period": period,
            **dict(zip(("y1", "y2", "y3"), rng.normal(size=3), strict=True)),
        }
        for caseid in range(20)
        for period in (0, 1)
    ]
    return pd.DataFrame(rows).set_index(["caseid", "period"])


@pytest.mark.parametrize("strategy", ["amn", "none"])
@pytest.mark.parametrize(
    "models",
    [
        {"y2": ProbitMeasurement()},
        {"y3": TobitMeasurement(lower=0.0)},
    ],
)
def test_get_maximization_inputs_rejects_non_gaussian_measurements(
    models, strategy
) -> None:
    """Rejected whatever the start-value strategy, not only via AMN seeding."""
    with pytest.raises(NotImplementedError, match="CHS supports only Gaussian"):
        get_maximization_inputs(
            model_spec=_model(models),
            data=_tiny_data(),
            chs_options=CHSEstimationOptions(start_params_strategy=strategy),
        )
