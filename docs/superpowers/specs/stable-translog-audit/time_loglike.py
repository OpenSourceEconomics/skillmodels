"""Warm likelihood-evaluation time, translog vs stable_translog, estimation fixture."""
import sys, time, json, resource, platform; sys.path.insert(0, "tests")
import jax, numpy as np
from test_missing_observed_factors import MOMENTS, OPTIONS, _data, _model, _params
from skillmodels.chs.maximization_inputs import get_maximization_inputs
law, calls = sys.argv[1], int(sys.argv[2])
inputs = get_maximization_inputs(_model(2, law, MOMENTS), _data(), chs_options=OPTIONS)
params = _params(inputs["params_template"], nonlinear=0.05)
f = inputs["loglike_and_gradient"]
t0 = time.perf_counter(); ll, g = f(params); float(ll); first = time.perf_counter() - t0
t0 = time.perf_counter()
for _ in range(calls):
    ll, g = f(params); float(ll)
warm = (time.perf_counter() - t0) / calls
print("RESULT=" + json.dumps({"law": law, "first_call_s": first, "warm_s_per_call": warm, "calls": calls,
  "loglike": float(ll), "peak_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
  "python": platform.python_version(), "jax": jax.__version__}))
