"""AMN must reject non-Gaussian measurements rather than treat them as continuous.

AMN's mixture-EM and minimum-distance stages recover loadings/SDs from the
cross-covariance of multi-indicator measurements -- a continuous-Gaussian moment
map. A probit/Tobit measure routed through it would be silently treated as
continuous, so `estimate_amn` rejects any non-Gaussian `measurement_models` entry.
A future working-linear seeding path will translate such measures before AMN runs.
"""

import numpy as np
import pandas as pd
import pytest

from skillmodels.amn.estimate import estimate_amn
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
    rows = []
    for caseid in range(20):
        for period in (0, 1):
            rows.append(
                {
                    "caseid": caseid,
                    "period": period,
                    "y1": rng.normal(),
                    "y2": rng.normal(),
                    "y3": rng.normal(),
                }
            )
    return pd.DataFrame(rows).set_index(["caseid", "period"])


@pytest.mark.parametrize(
    "models",
    [
        {"y2": ProbitMeasurement()},
        {"y3": TobitMeasurement(lower=0.0)},
    ],
)
def test_estimate_amn_rejects_non_gaussian_measurements(models) -> None:
    with pytest.raises(NotImplementedError, match="Gaussian"):
        estimate_amn(_model(models), _tiny_data())


def test_estimate_amn_rejects_non_gaussian_even_for_start_values() -> None:
    # The seeding path must hand AMN a working-linear (Gaussian) spec; a raw
    # probit/Tobit spec is rejected even with for_start_values=True.
    with pytest.raises(NotImplementedError, match="Gaussian"):
        estimate_amn(
            _model({"y2": ProbitMeasurement()}),
            _tiny_data(),
            for_start_values=True,
        )
