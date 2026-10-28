# ruff: noqa: INP001, T201
"""Measure the cost of the CHS likelihood with and without limited measurements.

One call measures one configuration in a fresh process, so the first evaluation
compiles from scratch:

- `compile_seconds`: the first `loglike_and_gradient` call minus the median warm call;
- `warm_seconds`: the median of the warm calls;
- `peak_device_bytes`: `peak_bytes_in_use` of the default device after all calls
  (`None` on CPU).

The model is MODEL2 from the test data. Its 4000 persons are copied `--copies` times
under new ids to reach application-sized samples. Variants:

- `gaussian`: MODEL2 on its data;
- `twin`: MODEL2 on the data of `limited`;
- `limited`: `y5` as a probit (dichotomized at its median) and `y6` as a Tobit
  (censored below at its 20% quantile);
- `mixed_twin`: MODEL2 on the data of `mixed`;
- `mixed`: five of the nine measurements limited, as in an application with about
  half of its updates limited: `y2`, `y5` and `y8` as four-category ordered probits
  (quartile codes, cutpoints 0, 0.67, 1.35), `y3` as a probit and `y6` as a Tobit.

The script imports whichever `skillmodels` is first on the path, so the same file
measures a baseline checkout through `PYTHONPATH`. Each result is appended to
`--output` as one JSON line.
"""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np
import pandas as pd

import skillmodels
from skillmodels.chs.maximization_inputs import get_maximization_inputs
from skillmodels.common.config import TEST_DATA_DIR
from skillmodels.common.measurement_models import (
    OrderedProbitMeasurement,
    ProbitMeasurement,
    TobitMeasurement,
)
from skillmodels.test_data.model2 import MODEL2, MODEL2_CHS_OPTIONS

_N_WARM = 5
_CUTPOINTS = (0.0, 0.67, 1.35)


def main() -> None:
    """Measure one configuration and append the result to `--output`."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant", choices=["gaussian", "twin", "limited", "mixed_twin", "mixed"]
    )
    parser.add_argument("--copies", type=int, default=1)
    parser.add_argument("--params", type=Path, help="one_stage_anchoring.csv")
    parser.add_argument("--label", default="")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    model, data = _model_and_data(args.variant, copies=args.copies)
    options = replace(MODEL2_CHS_OPTIONS, start_params_strategy="none")
    inputs = get_maximization_inputs(model, data, chs_options=options)
    params = pd.read_csv(args.params).set_index(
        ["category", "period", "name1", "name2"]
    )
    params = params.reindex(inputs["params_template"].index)
    for k, cut in enumerate(_CUTPOINTS, start=1):
        is_cut = params.index.get_level_values("name2") == f"cutpoint_{k}"
        params.loc[is_cut, "value"] = cut
    is_sd = params.index.get_level_values("category") == "meas_sds"
    params.loc[is_sd, "value"] = params.loc[is_sd, "value"].clip(lower=0.3)

    start = time.perf_counter()
    value, gradient = inputs["loglike_and_gradient"](params)
    first = time.perf_counter() - start
    warm = []
    for _ in range(_N_WARM):
        start = time.perf_counter()
        inputs["loglike_and_gradient"](params)
        warm.append(time.perf_counter() - start)
    warm_seconds = float(np.median(warm))
    stats = jax.devices()[0].memory_stats() or {}
    result = {
        "label": args.label,
        "variant": args.variant,
        "n_persons": int(data.index.get_level_values("caseid").nunique()),
        "skillmodels": skillmodels.__file__,
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "x64": bool(jax.config.read("jax_enable_x64")),
        "loglike": float(value),
        "gradient_finite": bool(np.isfinite(gradient).all()),
        "first_seconds": first,
        "warm_seconds": warm_seconds,
        "warm_all": warm,
        "compile_seconds": first - warm_seconds,
        "peak_device_bytes": stats.get("peak_bytes_in_use"),
    }
    with args.output.open("a") as f:
        f.write(json.dumps(result) + "\n")
    print(json.dumps(result))


def _model_and_data(variant: str, *, copies: int) -> tuple:
    data = pd.read_stata(TEST_DATA_DIR / "model2_simulated_data.dta")
    width = int(data["caseid"].max()) + 1
    data = pd.concat(
        [data.assign(caseid=data["caseid"] + k * width) for k in range(copies)]
    ).set_index(["caseid", "period"])
    if variant == "gaussian":
        return MODEL2, data
    lower = float(data["y6"].quantile(0.2))
    data["y6"] = data["y6"].clip(lower=lower)
    probit = "y5" if variant in ("twin", "limited") else "y3"
    data[probit] = (
        (data[probit] > data[probit].median()).astype(float).where(data[probit].notna())
    )
    models = {probit: ProbitMeasurement(), "y6": TobitMeasurement(lower=lower)}
    if variant in ("mixed_twin", "mixed"):
        for name in ("y2", "y5", "y8"):
            data[name] = pd.qcut(data[name], 4, labels=False).astype(float)
            models[name] = OrderedProbitMeasurement(n_categories=4)
    if variant in ("twin", "mixed_twin"):
        return MODEL2, data
    return MODEL2._replace(measurement_models=models), data


if __name__ == "__main__":
    main()
