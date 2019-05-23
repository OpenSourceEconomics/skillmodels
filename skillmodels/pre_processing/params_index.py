import skillmodels.model_functions.transition_functions as tf
import pandas as pd


def params_index(
    update_info, controls, factors, nemf, transition_names, included_factors
):

    periods = list(range(len(controls)))

    ind_tups = _deltas_index_tuples(controls, update_info)
    ind_tups += _h_index_tuples(factors, update_info)
    ind_tups += _r_index_tuples(update_info)
    ind_tups += _q_index_tuples(periods, factors)
    ind_tups += _x_index_tuples(periods, factors)
    ind_tups += _w_index_tuples(nemf)
    ind_tups += _p_index_tuples(nemf, factors)
    ind_tups += _trans_coeffs_index_tuples(
        factors, periods, transition_names, included_factors
    )

    index = pd.MultiIndex.from_tuples(
        ind_tups, names=["category", "period", "name1", "name2"]
    )
    return index


def _deltas_index_tuples(controls, update_info):
    ind_tups = []
    for period, meas in update_info.index:
        for cont in ["constant"] + controls[period]:
            ind_tups.append(("delta", period, meas, cont))
    return ind_tups


def _h_index_tuples(factors, update_info):
    ind_tups = []
    for period, meas in update_info.index:
        for factor in factors:
            ind_tups.append(("h", period, meas, factor))
    return ind_tups


def _r_index_tuples(update_info):
    ind_tups = []
    for period, meas in update_info.index:
        ind_tups.append(("r", period, meas, ""))
    return ind_tups


def _q_index_tuples(periods, factors):
    ind_tups = []
    for period in periods[:-1]:
        for factor in factors:
            ind_tups.append(("q", period, factor, ""))
    return ind_tups


def _x_index_tuples(nemf, factors):
    ind_tups = []
    for emf in range(nemf):
        for factor in factors:
            ind_tups.append(("x", 0, emf, factor))
    return ind_tups


def _w_index_tuples(nemf):
    ind_tups = []
    for emf in range(nemf):
        ind_tups.append("w", 0, emf, "")
    return ind_tups


def _p_index_tuples(nemf, factors):
    ind_tups = []
    for emf in range(nemf):
        for row, factor1 in enumerate(factors):
            for col, factor2 in enumerate(factors):
                if row <= col:
                    ind_tups.append(("p", 0, emf, "{}-{}".format(factor1, factor2)))
    return ind_tups


def _trans_coeffs_index_tuples(factors, periods, transition_names, included_factors):
    ind_tups = []
    for period in periods[:-1]:
        for f, factor in enumerate(factors):
            func = getattr(tf, "index_tuples_{}".format(transition_names[f]))
            ind_tups += func(factor, included_factors[f], period)
    return ind_tups
