import pickle
import json
import pandas as pd
from skillmodels import SkillModel
from skillmodels.estimation.likelihood_function import \
    log_likelihood_per_individual

from numpy.testing import assert_array_almost_equal as aaae


def test_likelihood_value():
    df = pd.read_stata('skillmodels/tests/estimation/chs_test_ex2.dta')
    with open('skillmodels/tests/estimation/test_model2.json') as j:
        model_dict = json.load(j)

    mod = SkillModel(model_dict=model_dict, dataset=df, estimator='chs',
                     model_name='test_model')

    args = mod.likelihood_arguments_dict(params_type='short')

    params = [1,
              1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.1,
              1.095, 1.085, 1.075, 1.065, 1.055, 1.045, 1.035, 1.025, 1.015,
              1.005, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99,
              0.995, 0.985, 0.975, 0.965, 0.955, 0.945, 0.935, 0.925, 0.915,
              0.905, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.1,
              1.095, 1.085, 1.075, 1.065, 1.055, 1.045, 1.035, 1.025, 1.015,
              1.005, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99,
              0.995, 0.985, 0.975, 0.965, 0.955, 0.945, 0.935, 0.925, 0.915,
              0.905, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.1,
              1.095, 1.085, 1.075, 1.065, 1.055, 1.045, 1.035, 1.025, 1.015,
              1.005, 1, 1, 1, 1.2, 1.4, 0.8, 0.6, 1.2, 0.8, 1.2, 1.4, 0.8, 0.6,
              1.2, 1.4, 0.8, 0.6, 1.2, 1.4, 0.8, 0.6, 1.2, 1.4, 0.8, 0.6, 1.2,
              1.4, 0.8, 0.6, 1.2, 1.4, 0.8, 0.6, 1.2, 1.4, 0.8, 0.6, 1, 0.5,
              0.51, 0.52, 0.53, 0.54, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.58,
              0.57, 0.56, 0.55, 0.54, 0.53, 0.52, 0.51, 0.5, 0.51, 0.52, 0.53,
              0.54, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.58, 0.57, 0.56, 0.55,
              0.54, 0.53, 0.53, 0.52, 0.52, 0.51, 0.51, 0.5, 0.5, 0.5, 0.5,
              0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.1, 0.1, 0.447, 0, 0, 0.447,
              0, 0.447, 3, 3, -0.5, 0.6]

    res = log_likelihood_per_individual(params, **args)

    in_path = 'skillmodels/tests/estimation/regression_test_fixture.pickle'
    with open(in_path, 'rb') as p:
        last_result = pickle.load(p)
    aaae(res, last_result)

    # update the regression test
    # do not uncomment this if you are not absolutely sure that you want to
    # update the regression test!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # with open(in_path, 'wb') as p:
    #     pickle.dump(res, p)











