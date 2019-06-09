import pickle
import json
import pandas as pd
from skillmodels import SkillModel
from skillmodels.estimation.likelihood_function import log_likelihood_per_individual

from numpy.testing import assert_array_almost_equal as aaae


def test_likelihood_value():
    df = pd.read_stata("skillmodels/tests/estimation/chs_test_ex2.dta")
    df.set_index(['id', 'period'], inplace=True)
    with open("skillmodels/tests/estimation/test_model2.json") as j:
        model_dict = json.load(j)

    mod = SkillModel(
        model_dict=model_dict, dataset=df, estimator="chs", model_name="test_model"
    )
    # uinfo = mod.update_info
    # with open('/home/janos/Dropbox/Projects/skillmodels/test_uinfo.p', 'wb') as p:
    #     pickle.dump(uinfo, p)

    args = mod.likelihood_arguments_dict(params_type="short")

    params_df = pd.read_csv("skillmodels/tests/estimation/like_reg_params_new.csv")
    params_df["name2"].fillna("", inplace=True)
    params_df["name1"].replace("0", 0, inplace=True)
    params_df.set_index(["category", "period", "name1", "name2"], inplace=True)
    mod.start_params = params_df

    full_params_df = mod.generate_full_start_params()
    params = full_params_df["value"]

    res = log_likelihood_per_individual(params, **args)

    in_path = "skillmodels/tests/estimation/regression_test_fixture.pickle"
    with open(in_path, "rb") as p:
        last_result = pickle.load(p)

    aaae(res, last_result)

    # update the regression test
    # do not uncomment this if you are not absolutely sure that you want to
    # update the regression test!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # with open(in_path, 'wb') as p:
    #     pickle.dump(res, p)
