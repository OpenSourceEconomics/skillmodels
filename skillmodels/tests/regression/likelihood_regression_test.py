import json
import pickle

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from skillmodels import SkillModel
from skillmodels.estimation.likelihood_function import log_likelihood_contributions

model_names = [
    "test_model_no_stages_anchoring",
    "test_model_one_stage",
    "test_model_one_stage_anchoring",
    "test_model_two_stages_anchoring",
]

model_dicts = []
for name in model_names:
    with open(f"skillmodels/tests/regression/{name}.json") as j:
        model_dicts.append(json.load(j))


start_params = []
for name in model_names:
    params_df = pd.read_csv(f"skillmodels/tests/regression/{name}.csv")
    params_df["name2"].fillna("", inplace=True)
    params_df["name1"].replace("0", 0, inplace=True)
    params_df.set_index(["category", "period", "name1", "name2"], inplace=True)
    start_params.append(params_df)


data = pd.read_stata("skillmodels/tests/estimation/chs_test_ex2.dta")
data.set_index(["id", "period"], inplace=True)

test_cases = []
for model, par, model_name in zip(model_dicts, start_params, model_names):
    test_cases.append((model, par, data, model_name))


@pytest.mark.parametrize("model, params, data, model_name", test_cases)
def test_likelihood_value(model, params, data, model_name):
    mod = SkillModel(model_dict=model, dataset=data)
    full_params = mod.generate_full_start_params(params)["value"]
    args = mod.likelihood_arguments_dict()
    log_like_contributions = log_likelihood_contributions(full_params, **args)
    like_contributions = np.exp(log_like_contributions)
    small = 1e-250
    like_vec = np.prod(like_contributions, axis=0)
    like_vec[like_vec < small] = small
    res = np.log(like_vec)
    in_path = f"skillmodels/tests/regression/{model_name}_result.pickle"
    with open(in_path, "rb") as p:
        last_result = pickle.load(p)
    aaae(res, last_result)
