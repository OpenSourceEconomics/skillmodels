"""Construct a dictionary of model specifications."""
import warnings
from itertools import product

import numpy as np
import pandas as pd
from pandas import DataFrame

from skillmodels.pre_processing.constraints import constraints
from skillmodels.pre_processing.data_processor import pre_process_data
from skillmodels.pre_processing.params_index import params_index


def process_model(
    model_dict, dataset, model_name="some_model", dataset_name="some_dataset"
):
    model_specs = {}
    model_specs["model_dict"] = model_dict
    model_specs["data"] = pre_process_data(dataset)
    model_specs["dataset_name"] = dataset_name
    model_specs["_timeinf"] = model_dict.get("time_specific", {})
    model_specs["_facinf"] = model_dict["factor_specific"]
    model_specs["factors"] = tuple(sorted(model_specs["_facinf"].keys()))
    model_specs["nfac"] = len(model_specs["factors"])
    model_specs["nsigma"] = 2 * model_specs["nfac"] + 1
    general_settings = {
        "n_mixture_components": 1,
        "sigma_points_scale": 2,
        "bounds_distance": 1e-6,
        "time_invariant_measurement_system": False,
        "base_color": "#035096",
    }

    general_settings.update(model_dict.get("general", {}))
    model_specs.update({"nmixtures": general_settings.pop("n_mixture_components")})
    model_specs.update(general_settings)
    model_specs.update(_set_time_specific_attributes(model_specs))
    model_specs.update(_transition_equation_names(model_specs))
    model_specs.update(_transition_equation_included_factors(model_specs))
    model_specs.update(_set_anchoring_attributes(model_specs))
    model_specs.update(_check_measurements(model_specs))
    model_specs.update(_clean_controls_specification(model_specs))
    model_specs.update(
        {"nobs": int(len(model_specs["data"]) / model_specs["nperiods"])}
    )
    model_specs.update(_check_and_fill_normalization_specification(model_specs))
    model_specs.update({"nupdates": len(update_info(model_specs))})
    model_specs.update(_set_params_index(model_specs))
    model_specs.update(_set_constraints(model_specs))
    return model_specs


def _set_time_specific_attributes(model_specs):
    """Set model specs related to periods and stages as attributes."""
    time_spec_dict = {}
    time_spec_dict["nperiods"] = len(
        model_specs["_facinf"][model_specs["factors"][0]]["measurements"]
    )
    if "stagemap" in model_specs["_timeinf"]:
        time_spec_dict["stagemap"] = np.array(model_specs["_timeinf"]["stagemap"])
    else:
        sm = np.arange(time_spec_dict["nperiods"])
        sm[-1] = sm[-2]
        time_spec_dict["stagemap"] = sm

    time_spec_dict["periods"] = tuple(range(time_spec_dict["nperiods"]))
    time_spec_dict["stages"] = tuple(sorted(set(time_spec_dict["stagemap"])))
    time_spec_dict["nstages"] = len(time_spec_dict["stages"])
    time_spec_dict["stage_length_list"] = tuple(
        list(time_spec_dict["stagemap"][:-1]).count(s) for s in time_spec_dict["stages"]
    )

    assert len(time_spec_dict["stagemap"]) == time_spec_dict["nperiods"], (
        "You have to specify a list of length nperiods " "as stagemap. Check model {}"
    ).format(model_specs["model_name"])

    assert time_spec_dict["stagemap"][-1] == time_spec_dict["stagemap"][-2], (
        "If you specify a stagemap of length nperiods the last two "
        "elements have to coincide because no transition equation can be "
        "estimated in the last period. Check model {}"
    ).format(model_specs["model_name"])

    assert np.array_equal(time_spec_dict["stages"], range(time_spec_dict["nstages"])), (
        "The stages have to be numbered beginning with 0 and increase in "
        "steps of 1. Your stagemap in mode {} is invalid"
    ).format(model_specs["model_name"])

    for factor in model_specs["factors"]:
        length = len(model_specs["_facinf"][factor]["measurements"])
        assert length == time_spec_dict["nperiods"], (
            "The lists of lists with the measurements must have the "
            "same length for each factor in the model. In the model {} "
            "you have one list with length {} and another with length "
            "{}."
        ).format(model_specs["model_name"], time_spec_dict["nperiods"], length)
    return time_spec_dict


def _transition_equation_names(model_specs):
    """Construct a list with the transition equation name for each factor.

    The result is set as class attribute ``transition_names``.

    """
    trans_eq_dict = {}
    trans_eq_dict["transition_names"] = tuple(
        model_specs["_facinf"][f]["trans_eq"]["name"] for f in model_specs["factors"]
    )
    return trans_eq_dict


def _transition_equation_included_factors(model_specs):
    """Included factors and their position for each transition equation.

    Construct a list with included factors for each transition equation
    and set the results as class attribute ``included_factors``.

    Construct a list with the positions of included factors in the
    alphabetically ordered factor list and set the result as class
    attribute ``included_positions``.

    """
    trans_eq_incl_fac = {}
    included_factors = []
    included_positions = []

    for factor in model_specs["factors"]:
        trans_inf = model_specs["_facinf"][factor]["trans_eq"]
        args_f = sorted(trans_inf["included_factors"])
        pos_f = list(
            np.arange(model_specs["nfac"])[np.in1d(model_specs["factors"], args_f)]
        )
        included_factors.append(tuple(args_f))
        included_positions.append(np.array(pos_f, dtype=int))
        assert len(included_factors) >= 1, (
            "Each latent factor needs at least one included factor. This is "
            "violated for {}".format(factor)
        )

    trans_eq_incl_fac["included_factors"] = tuple(included_factors)
    trans_eq_incl_fac["included_positions"] = tuple(included_positions)
    return trans_eq_incl_fac


def _set_anchoring_attributes(model_specs):
    """Set attributes related to anchoring and make some checks."""
    anchoring_dict = {}
    if "anchoring" in model_specs["model_dict"]:
        anch_info = model_specs["model_dict"]["anchoring"]
        anchoring_dict["anchoring"] = True
        anchoring_dict["anch_outcome"] = anch_info["outcome"]
        anchoring_dict["anchored_factors"] = sorted(anch_info["factors"])
        anchoring_dict["centered_anchoring"] = anch_info.get("center", False)
        anchoring_dict["anch_positions"] = np.array(
            [
                model_specs["factors"].index(fac)
                for fac in anchoring_dict["anchored_factors"]
            ]
        )
        anchoring_dict["use_anchoring_controls"] = anch_info.get("use_controls", False)
        anchoring_dict["use_anchoring_constant"] = anch_info.get("use_constant", False)
        anchoring_dict["free_anchoring_loadings"] = anch_info.get(
            "free_loadings", False
        )

        assert isinstance(anchoring_dict["anchoring"], bool)
        assert isinstance(anchoring_dict["anch_outcome"], (str, int, tuple))
        assert isinstance(anchoring_dict["anchored_factors"], list)
        assert isinstance(anchoring_dict["centered_anchoring"], bool)
        assert isinstance(anchoring_dict["use_anchoring_controls"], bool)
        assert isinstance(anchoring_dict["use_anchoring_constant"], bool)
        assert isinstance(anchoring_dict["free_anchoring_loadings"], bool)
    else:
        anchoring_dict["anchoring"] = False
        anchoring_dict["anchored_factors"] = []
        anchoring_dict["use_anchoring_controls"] = False
        anchoring_dict["use_anchoring_constant"] = False
        anchoring_dict["free_anchoring_loadings"] = False
        anchoring_dict["anch_outcome"] = None
        anchoring_dict["centered_anchoring"] = False
    return anchoring_dict


def _check_measurements(model_specs):
    """Set a dictionary with the cleaned measurement specifications as attribute."""
    measurements = {}
    measurements_dict = {}
    for factor in model_specs["factors"]:
        measurements[factor] = model_specs["_facinf"][factor]["measurements"]

    for f, factor in enumerate(model_specs["factors"]):
        if model_specs["transition_names"][f] == "constant":
            for t in model_specs["periods"][1:]:
                assert len(measurements[factor][t]) == 0, (
                    "In model {} factor {} has a constant transition "
                    "equation. Therefore it can only have measurements "
                    "in the initial period. However, you specified measure"
                    "ments in period {}.".format(model_specs["model_name"], factor, t)
                )
    measurements_dict["measurements"] = measurements
    return measurements_dict


def _clean_controls_specification(model_specs):
    controls = model_specs["_timeinf"].get("controls", [[]] * model_specs["nperiods"])
    clean_cont_dict = {}
    missings_list = []
    for t in model_specs["periods"]:
        df = model_specs["data"].query(f"__period__ == {t}")
        missings_list.append(df[controls[t]].isnull().any(axis=1))
    clean_cont_dict["missing_controls"] = tuple(missings_list)
    clean_cont_dict["controls"] = tuple(tuple(con) for con in controls)
    return clean_cont_dict


def _check_and_clean_normalizations_list(model_specs, factor, norm_list, norm_type):
    """Check and clean a list with normalization specifications.

    Raise an error if invalid normalizations were specified.

    Transform the normalization list to the new standard specification
    (list of dicts) if the old standard (list of lists) was used.

    For the correct specification of a normalizations list refer to
    :ref:`model_specs`

    3 forms of invalid specification are checked and custom error
    messages are raised in each case:
    * Invalid length of the specification list
    * Invalid length of the entries in the specification list
    * Normalized variables that were not specified as measurement variables
      in the period where they were used

    """
    was_not_specified_message = (
        "Normalized measurements must be included in the measurement list "
        "of the factor they normalize in the period where they are used. "
        "In model {} you use the variable {} to normalize factor {} in "
        "period {} but it is not included as measurement."
    )

    assert len(norm_list) == model_specs["nperiods"], (
        "Normalizations lists must have one entry per period. In model {} "
        "you specify a normalizations list of length {} for factor {} "
        "but the model has {} periods"
    ).format(model_specs["model_name"], len(norm_list), factor, model_specs["nperiods"])

    for t, norminfo in enumerate(norm_list):
        if type(norminfo) != dict:
            assert len(norminfo) in [0, 2], (
                "The sublists in the normalizations must be empty or have "
                "length 2. In model {} in period {} you specify a "
                "list with len {} for factor {}"
            ).format(model_specs["model_name"], t, len(norminfo), factor)

    cleaned = []

    for norminfo in norm_list:
        if type(norminfo) == dict:
            cleaned.append(norminfo)
        else:
            cleaned.append({norminfo[0]: norminfo[1]})

    if norm_list != cleaned:
        warnings.warn(
            "Using lists of lists instead of lists of dicts for the "
            "normalization specification is deprecated.",
            DeprecationWarning,
        )

    norm_list = cleaned

    # check presence of variables
    for t, norminfo in enumerate(norm_list):
        normed_measurements = list(norminfo.keys())
        for normed_meas in normed_measurements:
            if normed_meas not in model_specs["measurements"][factor][t]:
                raise KeyError(
                    was_not_specified_message.format(
                        model_specs["model_name"], normed_meas, factor, t
                    )
                )

    # check validity of values
    for norminfo in norm_list:  #
        for n_val in norminfo.values():
            if norm_type == "variances":
                assert n_val > 0, "Variances can only be normalized to a value > 0."
            if norm_type == "loadings":
                assert n_val != 0, "Loadings cannot be normalized to 0."

    return norm_list


def _check_and_fill_normalization_specification(model_specs):
    """Check normalization specs or generate empty ones for each factor.

    The result is set as class attribute ``normalizations``.

    """
    norm_dict = {}
    norm = {}
    norm_types = ["loadings", "intercepts"]

    for factor in model_specs["factors"]:
        norm[factor] = {}

        for norm_type in norm_types:
            if "normalizations" in model_specs["_facinf"][factor]:
                norminfo = model_specs["_facinf"][factor]["normalizations"]
                if not set(norminfo.keys()).issubset(set(norm_types)):
                    raise ValueError(
                        "Normalization can be provided only "
                        "for loadings  and intercepts"
                    )
                if norm_type in norminfo:
                    norm_list = norminfo[norm_type]
                    norm[factor][norm_type] = _check_and_clean_normalizations_list(
                        model_specs, factor, norm_list, norm_type
                    )
                else:
                    norm[factor][norm_type] = [{}] * model_specs["nperiods"]
            else:
                norm[factor] = {nt: [{}] * model_specs["nperiods"] for nt in norm_types}

    norm_dict["normalizations"] = norm
    return norm_dict


def update_info(model_specs):
    """A DataFrame with all relevant information on Kalman updates.

    Construct a DataFrame that contains all relevant information on the
    numbers of updates, variables used and factors involved. Moreover it
    combines the model_specs of measurements and anchoring as both are
    incorporated into the likelihood via Kalman updates.

    In the model specs the measurement variables are specified in the way
    I found to be most human readable and least error prone. Here they are
    transformed into a pandas DataFrame that is more convenient for the
    construction of inputs for the likelihood function.

    Each row in the DataFrame corresponds to one Kalman update. Therefore,
    the length of the DataFrame is the total number of updates (nupdates).

    The DataFrame has a MultiIndex. The first level is the period in which,
    the update is made. The second level the name of measurement/anchoring
    outcome used in the update.

    The DataFrame has the following columns:

    * A column for each factor: df.loc[(t, meas), fac1] is 1 if meas is a
        measurement for fac1 in period t, else it is 0.
    * purpose: takes one of the values in ['measurement', 'anchoring']

    Returns:
        DataFrame

    """
    to_concat = [
        _factor_update_info(model_specs),
        _purpose_update_info(model_specs),
        _invariance_update_info(model_specs),
    ]

    df = pd.concat(to_concat, axis=1)
    return df


def _factor_update_info(model_specs):
    index = pd.MultiIndex(levels=[[], []], codes=[[], []], names=["period", "name"])
    df = DataFrame(data=None, index=index)

    # append rows for each update that has to be performed
    for t in model_specs["periods"]:
        for factor in model_specs["factors"]:
            for meas in model_specs["measurements"][factor][t]:
                if t in df.index and meas in df.loc[t].index:
                    # change corresponding row of the DataFrame
                    df.loc[(t, meas), factor] = 1
                else:
                    # add a new row to the DataFrame
                    ind = pd.MultiIndex.from_tuples(
                        [(t, meas)], names=["period", "variable"]
                    )
                    df2 = DataFrame(data=0, columns=model_specs["factors"], index=ind)
                    df2[factor] = 1
                    df = df.append(df2)

        if model_specs["anchoring"]:
            for factor in model_specs["anchored_factors"]:
                name = f"{model_specs['anch_outcome']}_{factor}"
                ind = pd.MultiIndex.from_tuples(
                    [(t, name)], names=["period", "variable"]
                )
                df2 = DataFrame(data=0, columns=model_specs["factors"], index=ind)
                df2[factor] = 1
                df = df.append(df2)
    return df


def _purpose_update_info(model_specs):
    factor_uinfo = _factor_update_info(model_specs)
    sr = pd.Series(index=factor_uinfo.index, name="purpose", data="measurement")

    if model_specs["anchoring"] is True:
        for t, factor in product(
            model_specs["periods"], model_specs["anchored_factors"]
        ):
            sr.loc[t, f"{model_specs['anch_outcome']}_{factor}"] = "anchoring"
    return sr


def _invariance_update_info(model_specs):
    """Update information relevant for time invariant measurement systems.

    Measurement equations are uniquely identified by their period and the
    name of their measurement.

    Two measurement equations count as equal if and only if:
    * their measurements have the same name
    * the same latent factors are measured
    * they occur in periods that use the same control variables.
    """
    factor_uinfo = _factor_update_info(model_specs)
    ind = factor_uinfo.index
    df = pd.DataFrame(
        columns=["is_repeated", "first_occurence"],
        index=ind,
        data=[[False, np.nan]] * len(ind),
    )

    purpose_uinfo = _purpose_update_info(model_specs)

    for t, meas in ind:
        if purpose_uinfo[t, meas] == "measurement":
            # find first occurrence
            for t2, meas2 in ind:
                if meas == meas2 and t2 <= t:
                    if model_specs["controls"][t] == model_specs["controls"][t2]:
                        info1 = factor_uinfo.loc[(t, meas)].to_numpy()
                        info2 = factor_uinfo.loc[(t2, meas2)].to_numpy()
                        if (info1 == info2).all():
                            first = t2
                            break

            if t != first:
                df.loc[(t, meas), "is_repeated"] = True
                df.loc[(t, meas), "first_occurence"] = first
    return df


def _set_params_index(model_specs):
    params_ind = {}
    params_ind["params_index"] = params_index(
        update_info(model_specs),
        model_specs["controls"],
        model_specs["factors"],
        model_specs["nmixtures"],
        model_specs["transition_names"],
        model_specs["included_factors"],
    )
    return params_ind


def _set_constraints(model_specs):
    dict_const = {}
    dict_const["constraints"] = constraints(
        update_info(model_specs),
        model_specs["controls"],
        model_specs["factors"],
        model_specs["normalizations"],
        model_specs["measurements"],
        model_specs["nmixtures"],
        model_specs["stagemap"],
        model_specs["transition_names"],
        model_specs["included_factors"],
        model_specs["time_invariant_measurement_system"],
        model_specs["anchored_factors"],
        model_specs["anch_outcome"],
        model_specs["bounds_distance"],
        model_specs["use_anchoring_controls"],
        model_specs["use_anchoring_constant"],
        model_specs["free_anchoring_loadings"],
    )
    return dict_const


def public_attribute_dict(model_specs):
    public_attributes = {
        key: val for key, val in model_specs.items() if not key.startswith("_")
    }
    public_attributes["update_info"] = update_info(model_specs)
    return public_attributes
