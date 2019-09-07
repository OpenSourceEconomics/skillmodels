"""Collection of functions to generate tables.

The functions are used in several parts of skillmodels.

"""
import pandas as pd

from skillmodels.pre_processing.data_processor import prepend_index_level


def df_to_tex_table(df, title):
    start = r"\begin{table}[h!]\centering" + "\n"
    caption = r"\caption{{{}}}" + "\n"
    old_max_width = pd.get_option("display.max_colwidth")
    pd.set_option("display.max_colwidth", -1)
    end = r"\end{table}" + "\n"
    table = start + caption.format(title) + df.to_latex(escape=False) + end
    pd.set_option("display.max_colwidth", old_max_width)
    return table


def statsmodels_result_to_string_series(res, decimals=2, report_se=True):
    if hasattr(res, "name"):
        res_col = res.name
    else:
        res_col = "params"

    params = res.params
    params.name = "params"
    df = params.to_frame()
    df["p"] = res.pvalues
    white_star = r"\textcolor{white}{*}"
    df["stars"] = pd.cut(
        df["p"],
        bins=[-1, 0.01, 0.05, 0.1, 2],
        labels=["***", "**" + white_star, "*" + 2 * white_star, 3 * white_star],
    )
    fmt_str = "{:,." + str(decimals) + "f}"
    df[res_col] = params.map(fmt_str.format)
    df["phantom"] = r"\textcolor{white}{-}"
    df[res_col] = df[res_col].where(params < 0, df["phantom"] + df[res_col])
    df[res_col] += df["stars"].astype(str)

    if report_se is True:
        se_col = res.bse.map(fmt_str.format)
        se_col = " (" + se_col + ")"
        df[res_col] += se_col

    # this has a bug. it does not ensure that r_squared and nobs is moved
    # to the end and it breaks my tex build

    # df.loc['number of obs.', res_col] = str(int(res.nobs))
    # try:
    #     df.loc['adj. $R^2$', res_col] = fmt_str.format(res.rsquared_adj)
    # except AttributeError:
    #     pass
    return df[res_col]


def statsmodels_results_to_df(
    res_list, decimals=2, period_name="Period", report_se=True
):
    sr_list = []
    periods = sorted(list({res.period for res in res_list}))

    for res in res_list:
        sr = statsmodels_result_to_string_series(res, decimals=decimals)
        sr.name = res.name
        if len(periods) > 1:
            sr.index = (r"\hspace{0.5cm} " + sr.index).str.replace("_", r"\_")
            sr = prepend_index_level(sr, res.period)
        sr_list.append(sr)

    if len(periods) == 1:
        result = pd.concat(sr_list, axis=1, sort=True)
    else:
        period_dfs = []
        for period in periods:
            to_concat = [s for s in sr_list if s.index.levels[0][0] == period]
            df = pd.concat(to_concat, axis=1, sort=True)
            bold_period = r"\textbf{{{}}}".format(period_name + " " + str(period))
            ind = pd.MultiIndex.from_tuples([(period, bold_period)])
            first_row = pd.DataFrame(index=ind, columns=df.columns)
            df = pd.concat([first_row, df], axis=0)
            df = df.loc[period]
            period_dfs.append(df)

        result = pd.concat(period_dfs, axis=0, sort=True)

    result.fillna("", inplace=True)
    return result


def skillmodels_to_df():
    # do I need an extra function here?
    pass
