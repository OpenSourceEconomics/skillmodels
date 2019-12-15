import warnings
from itertools import product

import numpy as np
import pandas as pd
from pandas import DataFrame

from skillmodels.pre_processing.constraints import constraints
from skillmodels.pre_processing.data_processor import pre_process_data
from skillmodels.pre_processing.params_index import params_index


class ModelSpecProcessor:
    """Check, clean, extend and transform the model specs.

    Check the completeness, consistency and validity of the general and model
    specific specifications.

    Clean the model specs by handling variables that were specified but are
    not in the dataset or have no variance. Raise errors if specified.

    Transform the cleaned model specs into forms that are more practical for
    the construction of the quantities that are needed in the likelihood
    function.

    """

    def __init__(
        self, model_dict, dataset, model_name="some_model", dataset_name="some_dataset"
    ):
        self.model_dict = model_dict
        self.data = pre_process_data(dataset)
        self.model_name = model_name
        self.dataset_name = dataset_name
        self._timeinf = model_dict.get("time_specific", {})
        self._facinf = model_dict["factor_specific"]
        self.factors = tuple(sorted(self._facinf.keys()))
        self.nfac = len(self.factors)
        self.nsigma = 2 * self.nfac + 1

        # set the general model specifications
        general_settings = {
            "n_mixture_components": 1,
            "sigma_points_scale": 2,
            "bounds_distance": 1e-6,
            "time_invariant_measurement_system": False,
            "base_color": "#035096",
        }

        general_settings.update(model_dict.get("general", {}))

        self.nmixtures = general_settings.pop("n_mixture_components")
        self.__dict__.update(general_settings)
        self._set_time_specific_attributes()
        self._transition_equation_names()
        self._transition_equation_included_factors()
        self._set_anchoring_attributes()
        self._check_measurements()
        self._clean_controls_specification()
        self.nobs = int(len(self.data) / self.nperiods)
        self._check_and_fill_normalization_specification()
        self.nupdates = len(self.update_info())
        self._set_params_index()
        self._set_constraints()

    def _set_time_specific_attributes(self):
        """Set model specs related to periods and stages as attributes."""
        self.nperiods = len(self._facinf[self.factors[0]]["measurements"])
        if "stagemap" in self._timeinf:
            self.stagemap = np.array(self._timeinf["stagemap"])
        else:
            sm = np.arange(self.nperiods)
            sm[-1] = sm[-2]
            self.stagemap = sm

        self.periods = tuple(range(self.nperiods))
        self.stages = tuple(sorted(set(self.stagemap)))
        self.nstages = len(self.stages)
        self.stage_length_list = tuple(
            list(self.stagemap[:-1]).count(s) for s in self.stages
        )

        assert len(self.stagemap) == self.nperiods, (
            "You have to specify a list of length nperiods "
            "as stagemap. Check model {}"
        ).format(self.model_name)

        assert self.stagemap[-1] == self.stagemap[-2], (
            "If you specify a stagemap of length nperiods the last two "
            "elements have to coincide because no transition equation can be "
            "estimated in the last period. Check model {}"
        ).format(self.model_name)

        assert np.array_equal(self.stages, range(self.nstages)), (
            "The stages have to be numbered beginning with 0 and increase in "
            "steps of 1. Your stagemap in mode {} is invalid"
        ).format(self.model_name)

        for factor in self.factors:
            length = len(self._facinf[factor]["measurements"])
            assert length == self.nperiods, (
                "The lists of lists with the measurements must have the "
                "same length for each factor in the model. In the model {} "
                "you have one list with length {} and another with length "
                "{}."
            ).format(self.model_name, self.nperiods, length)

    def _transition_equation_names(self):
        """Construct a list with the transition equation name for each factor.

        The result is set as class attribute ``transition_names``.

        """
        self.transition_names = tuple(
            self._facinf[f]["trans_eq"]["name"] for f in self.factors
        )

    def _transition_equation_included_factors(self):
        """Included factors and their position for each transition equation.

        Construct a list with included factors for each transition equation
        and set the results as class attribute ``included_factors``.

        Construct a list with the positions of included factors in the
        alphabetically ordered factor list and set the result as class
        attribute ``included_positions``.

        """
        included_factors = []
        included_positions = []

        for factor in self.factors:
            trans_inf = self._facinf[factor]["trans_eq"]
            args_f = sorted(trans_inf["included_factors"])
            pos_f = list(np.arange(self.nfac)[np.in1d(self.factors, args_f)])
            included_factors.append(tuple(args_f))
            included_positions.append(np.array(pos_f, dtype=int))
            assert len(included_factors) >= 1, (
                "Each latent factor needs at least one included factor. This is "
                "violated for {}".format(factor)
            )

        self.included_factors = tuple(included_factors)
        self.included_positions = tuple(included_positions)

    def _set_anchoring_attributes(self):
        """Set attributes related to anchoring and make some checks."""
        if "anchoring" in self.model_dict:
            anch_info = self.model_dict["anchoring"]
            self.anchoring = True
            self.anch_outcome = anch_info["outcome"]
            self.anchored_factors = sorted(anch_info["factors"])
            self.centered_anchoring = anch_info.get("center", False)
            self.anch_positions = np.array(
                [self.factors.index(fac) for fac in self.anchored_factors]
            )
            self.use_anchoring_controls = anch_info.get("use_controls", False)
            self.use_anchoring_constant = anch_info.get("use_constant", False)
            self.free_anchoring_loadings = anch_info.get("free_loadings", False)

            assert isinstance(self.anchoring, bool)
            assert isinstance(self.anch_outcome, (str, int, tuple))
            assert isinstance(self.anchored_factors, list)
            assert isinstance(self.centered_anchoring, bool)
            assert isinstance(self.use_anchoring_controls, bool)
            assert isinstance(self.use_anchoring_constant, bool)
            assert isinstance(self.free_anchoring_loadings, bool)
        else:
            self.anchoring = False
            self.anchored_factors = []
            self.use_anchoring_controls = False
            self.use_anchoring_constant = False
            self.free_anchoring_loadings = False
            self.anch_outcome = None
            self.centered_anchoring = False

    def _check_measurements(self):
        """Set a dictionary with the cleaned measurement specifications as attribute."""
        measurements = {}
        for factor in self.factors:
            measurements[factor] = self._facinf[factor]["measurements"]

        for f, factor in enumerate(self.factors):
            if self.transition_names[f] == "constant":
                for t in self.periods[1:]:
                    assert len(measurements[factor][t]) == 0, (
                        "In model {} factor {} has a constant transition "
                        "equation. Therefore it can only have measurements "
                        "in the initial period. However, you specified measure"
                        "ments in period {}.".format(self.model_name, factor, t)
                    )

        self.measurements = measurements

    def _clean_controls_specification(self):
        controls = self._timeinf.get("controls", [[]] * self.nperiods)

        missings_list = []
        for t in self.periods:
            df = self.data.query(f"__period__ == {t}")
            missings_list.append(df[controls[t]].isnull().any(axis=1))
        self.missing_controls = tuple(missings_list)
        self.controls = tuple(tuple(con) for con in controls)

    def _check_and_clean_normalizations_list(self, factor, norm_list, norm_type):
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
        if norm_type == "variances":
            raise ValueError("Normalization for variances cannot be provided")
        assert len(norm_list) == self.nperiods, (
            "Normalizations lists must have one entry per period. In model {} "
            "you specify a normalizations list of length {} for factor {} "
            "but the model has {} periods"
        ).format(self.model_name, len(norm_list), factor, self.nperiods)

        for t, norminfo in enumerate(norm_list):
            if type(norminfo) != dict:
                assert len(norminfo) in [0, 2], (
                    "The sublists in the normalizations must be empty or have "
                    "length 2. In model {} in period {} you specify a "
                    "list with len {} for factor {}"
                ).format(self.model_name, t, len(norminfo), factor)

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
                if normed_meas not in self.measurements[factor][t]:
                    raise KeyError(
                        was_not_specified_message.format(
                            self.model_name, normed_meas, factor, t
                        )
                    )

        # check validity of values
        for norminfo in norm_list:  #
            for n_val in norminfo.values():
                if norm_type == "loadings":
                    assert n_val != 0, "Loadings cannot be normalized to 0."

        return norm_list

    def _check_and_fill_normalization_specification(self):
        """Check normalization specs or generate empty ones for each factor.

        The result is set as class attribute ``normalizations``.

        """
        norm = {}
        norm_types = ["loadings", "intercepts"]

        for factor in self.factors:
            norm[factor] = {}

            for norm_type in norm_types:
                if "normalizations" in self._facinf[factor]:
                    norminfo = self._facinf[factor]["normalizations"]
                    if norm_type in norminfo:
                        norm_list = norminfo[norm_type]
                        norm[factor][
                            norm_type
                        ] = self._check_and_clean_normalizations_list(
                            factor, norm_list, norm_type
                        )
                    else:
                        norm[factor][norm_type] = [{}] * self.nperiods
                else:
                    norm[factor] = {nt: [{}] * self.nperiods for nt in norm_types}

        self.normalizations = norm

    def update_info(self):
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
            self._factor_update_info(),
            self._purpose_update_info(),
            self._invariance_update_info(),
        ]

        df = pd.concat(to_concat, axis=1)

        return df

    def _factor_update_info(self):
        # create an empty DataFrame with and empty MultiIndex
        index = pd.MultiIndex(levels=[[], []], codes=[[], []], names=["period", "name"])
        df = DataFrame(data=None, index=index)

        # append rows for each update that has to be performed
        for t in self.periods:
            for factor in self.factors:
                for meas in self.measurements[factor][t]:
                    if t in df.index and meas in df.loc[t].index:
                        # change corresponding row of the DataFrame
                        df.loc[(t, meas), factor] = 1
                    else:
                        # add a new row to the DataFrame
                        ind = pd.MultiIndex.from_tuples(
                            [(t, meas)], names=["period", "variable"]
                        )
                        df2 = DataFrame(data=0, columns=self.factors, index=ind)
                        df2[factor] = 1
                        df = df.append(df2)

            if self.anchoring:
                for factor in self.anchored_factors:
                    name = f"{self.anch_outcome}_{factor}"
                    ind = pd.MultiIndex.from_tuples(
                        [(t, name)], names=["period", "variable"]
                    )
                    df2 = DataFrame(data=0, columns=self.factors, index=ind)
                    df2[factor] = 1
                    df = df.append(df2)
        return df

    def _purpose_update_info(self):
        factor_uinfo = self._factor_update_info()
        sr = pd.Series(index=factor_uinfo.index, name="purpose", data="measurement")

        if self.anchoring is True:
            for t, factor in product(self.periods, self.anchored_factors):
                sr.loc[t, f"{self.anch_outcome}_{factor}"] = "anchoring"
        return sr

    def _invariance_update_info(self):
        """Update information relevant for time invariant measurement systems.

        Measurement equations are uniquely identified by their period and the
        name of their measurement.

        Two measurement equations count as equal if and only if:
        * their measurements have the same name
        * the same latent factors are measured
        * they occur in periods that use the same control variables.
        """
        factor_uinfo = self._factor_update_info()
        ind = factor_uinfo.index
        df = pd.DataFrame(
            columns=["is_repeated", "first_occurence"],
            index=ind,
            data=[[False, np.nan]] * len(ind),
        )

        purpose_uinfo = self._purpose_update_info()

        for t, meas in ind:
            if purpose_uinfo[t, meas] == "measurement":
                # find first occurrence
                for t2, meas2 in ind:
                    if meas == meas2 and t2 <= t:
                        if self.controls[t] == self.controls[t2]:
                            info1 = factor_uinfo.loc[(t, meas)].to_numpy()
                            info2 = factor_uinfo.loc[(t2, meas2)].to_numpy()
                            if (info1 == info2).all():
                                first = t2
                                break

                if t != first:
                    df.loc[(t, meas), "is_repeated"] = True
                    df.loc[(t, meas), "first_occurence"] = first
        return df

    def _set_params_index(self):
        self.params_index = params_index(
            self.update_info(),
            self.controls,
            self.factors,
            self.nmixtures,
            self.transition_names,
            self.included_factors,
        )

    def _set_constraints(self):
        self.constraints = constraints(
            self.update_info(),
            self.controls,
            self.factors,
            self.normalizations,
            self.measurements,
            self.nmixtures,
            self.stagemap,
            self.transition_names,
            self.included_factors,
            self.time_invariant_measurement_system,
            self.anchored_factors,
            self.anch_outcome,
            self.bounds_distance,
            self.use_anchoring_controls,
            self.use_anchoring_constant,
            self.free_anchoring_loadings,
        )

    def public_attribute_dict(self):
        all_attributes = self.__dict__
        public_attributes = {
            key: val for key, val in all_attributes.items() if not key.startswith("_")
        }
        public_attributes["update_info"] = self.update_info()
        return public_attributes
