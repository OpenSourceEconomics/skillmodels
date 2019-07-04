import pickle
import json
import pandas as pd
from skillmodels import SkillModel
from skillmodels.estimation.likelihood_function import likelihood_contributions
import numpy as np

from numpy.testing import assert_array_almost_equal as aaae
import pytest

model_names = [
    'test_model_no_stages_anchoring',
    'test_model_one_stage',
    'test_model_one_stage_anchoring',
    'test_model_two_stages_anchoring',
]

model_dicts = []
for name in model_names:
    with open('skillmodels/tests/regression/{}.json'.format(name)) as j:
        model_dicts.append(json.load(j))


start_params = []
for name in model_names:
    params_df = pd.read_csv('skillmodels/tests/regression/{}.csv'.format(name))
    params_df["name2"].fillna("", inplace=True)
    params_df["name1"].replace("0", 0, inplace=True)
    params_df.set_index(["category", "period", "name1", "name2"], inplace=True)
    start_params.append(params_df)


data = pd.read_stata("skillmodels/tests/estimation/chs_test_ex2.dta")
data.set_index(['id', 'period'], inplace=True)

test_cases = []
for model, par, model_name in zip(model_dicts, start_params, model_names):
    test_cases.append((model, par, data, model_name))


@pytest.mark.parametrize('model, params, data, model_name', test_cases)
def test_likelihood_value(model, params, data, model_name):
    mod = SkillModel(model_dict=model, dataset=data)
    full_params = mod.generate_full_start_params(params)['value']

    args = mod.likelihood_arguments_dict()
    # free, fixed = mod.start_params_helpers()
    # free.to_csv('skillmodels/tests/regression/{}.csv'.format(model_name))
    # assert False

    like_contributions = likelihood_contributions(full_params, **args)
    small = 1e-250
    like_vec = np.prod(like_contributions, axis=0)
    like_vec[like_vec < small] = small
    res = np.log(like_vec)

    in_path = "skillmodels/tests/regression/{}_result.pickle".format(model_name)
    with open(in_path, "rb") as p:
        last_result = pickle.load(p)
    aaae(res, last_result)

    # update the regression test
    # do not uncomment this if you are not absolutely sure that you want to
    # update the regression test!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # with open(in_path, 'wb') as p:
    #     pickle.dump(res, p)
