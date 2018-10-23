"""Collection of functions to generate tables.

The functions are used in several parts of skillmodels.

"""
import pandas as pd
from skillmodels.estimation.wa_functions import prepend_index_level


def tex_from_df():
    pass


def rst_from_df():
    pass


def statsmodels_results_to_df(res_list, decimals=3):
    sr_list = []
    periods = sorted(list(set([res.period for res in res_list])))

    for res in res_list:
        sr = res.params
        sr.name = res.name
        sr = prepend_index_level(sr, res.period)
        sr = sr.round(decimals)
        sr_list.append(sr)

    period_dfs = []
    for period in periods:
        to_concat = [sr for sr in sr_list if sr.index.levels[0][0] == period]
        df = pd.concat(to_concat, axis=1)
        period_dfs.append(df)

    df = pd.concat(period_dfs, axis=0)
    df.fillna('', inplace=True)
    return df


def skillmodels_to_df():
    # do I need an extra function here?
    pass
