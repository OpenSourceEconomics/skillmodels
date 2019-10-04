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
        if "time_specific" in model_dict:
            self._timeinf = model_dict["time_specific"]
        else:
            self._timeinf = {}
        self._facinf = model_dict["factor_specific"]
        self.factors = tuple(sorted(list(self._facinf.keys())))
        self.nfac = len(self.factors)
        self.nsigma = 2 * self.nfac + 1

        # set the general model specifications
        general_settings = {
            "nemf": 1,
            "kappa": 2,
            "square_root_filters": True,
            "missing_variables": "raise_error",
            "controls_with_missings": "raise_error",
            "variables_without_variance": "raise_error",
            "bounds_distance": 1e-20,
            "estimate_X_zeros": False,
            "restrict_W_zeros": True,
            "restrict_P_zeros": True,
            "ignore_intercept_in_linear_anchoring": True,
            "anchoring_mode": "only_estimate_anchoring_equation",
            "time_invariant_measurement_system": False,
            "base_color": "#035096",
        }

        if "general" in model_dict:
            general_settings.update(model_dict["general"])
        self.__dict__.update(general_settings)
        self._set_time_specific_attributes()
        self._check_general_specifications()
        self._transition_equation_names()
        self._transition_equation_included_factors()
        self._set_anchoring_attributes()
        self._clean_measurement_specifications()
        self._clean_controls_specification()
        self.nobs = int(len(self.data) / self.nperiods)
        self._check_and_fill_normalization_specification()
        self._check_anchoring_specification()
        self.nupdates = len(self.update_info())
        self._nmeas_list()
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

    def _check_general_specifications(self):
        """Check consistency of the "general" model specifications."""
        if self.estimate_X_zeros is False:
            assert self.nemf == 1, (
                "If start states (X_zero) are not estimated it is not "
                "possible to have more than one element in the mixture "
                "distribution of the latent factors. Check model {}"
            ).format(self.model_name)

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
            included_positions.append(tuple(pos_f))
            assert len(included_factors) >= 1, (
                "Each latent factor needs at least one included factor. This is "
                "violated for {}".format(factor)
            )

        self.included_factors = tuple(included_factors)
        self.included_positions = tuple(included_positions)

    def _set_anchoring_attributes(self):
        """Set attributes related to anchoring and make some checks."""
        if "anchoring" in self.model_dict:
            assert len(self.model_dict["anchoring"]) <= 1, (
                "At most one anchoring equation can be estimated. You "
                "specify {} in model {}"
            ).format(len(self.model_dict["anchoring"]), self.model_name)
            (self.anch_outcome, self.anchored_factors), = self.model_dict[
                "anchoring"
            ].items()
            self.anchoring = True
            self.anch_positions = tuple(
                f for f in range(self.nfac) if self.factors[f] in self.anchored_factors
            )
            if self.anchoring_mode == "truly_anchor_latent_factors":
                self.anchor_in_predict = True
            else:
                self.anchor_in_predict = False
        else:
            self.anchoring = False
            self.anchored_factors = ()
            self.anchor_in_predict = False
            self.anch_outcome = None

    def _present(self, variable, period):
        """Check if **variable** is present in **period**.

        **variable** is considered present if it is in self.data and not all
        observations in **period** are NaN.

        args:
            variable (str): name of the variable whose presence is checked.
            period (int): period in which the presence of variable is checked.

        Returns:
            bool: True if **variable** is present and False otherwise.

        Raises:
            KeyError: if **variable** is not present and self.missing_variables
                is equal to 'raise_error'

        """
        message = (
            "In model {} you use variable {} which is not in dataset {}. "
            "in period {}. You can either delete the variable in the "
            'model_specs or set "missing_variables" to "drop_variable" '
            "to automatically drop missing variables."
        ).format(self.model_name, variable, self.dataset_name, period)

        columns = set(self.data.columns)
        df = self.data.query(f"__period__ == {period}")
        if variable in columns and df[variable].notnull().any():
            return True
        elif self.missing_variables == "raise_error":
            raise KeyError(message)
        else:
            return False

    def _has_variance(self, variable, period):
        """Check if **variable** has variance in **period**.

        **variable** is considered to have variance if it takes at least 2
        different values in **period**.

        args:
            variable (str): name of the variable being checked.
            period (int): period in which the variable is checked
        Returns:
            bool: True if **variable** has variance in **period**, else False

        Raises:
            ValueError: if **variable** is not present and
                ``self.variables_without_variance == 'raise_error'``

        """
        message = (
            "Variables have to take at least two different values as variables"
            " without variance cannot help to identify the model. In model {} "
            "you use variable {} which only takes the value {} in dataset {} "
            "in period {}. You can eiter drop the variable in the model_specs "
            'or set the "variables_without_variance" key in general settings '
            'to "drop_variable".'
        )

        series = self.data.query(f"__period__ == {period}")[variable]
        unique_non_missing_values = list(series[pd.notnull(series)].unique())
        nr_unique = len(unique_non_missing_values)

        if nr_unique <= 1:
            if self.variables_without_variance == "raise_error":
                raise ValueError(
                    message.format(
                        self.model_name,
                        variable,
                        unique_non_missing_values,
                        self.dataset_name,
                        period,
                    )
                )
            else:
                return False
        else:
            return True

    def _clean_measurement_specifications(self):
        """Drop missing measurement variables or raise errors.

        Check for each measurement variable of the model if it is present and
        has variance in all periods where it is used. If not drop it or
        raise an error according to ``self.missing_variables`` and
        ``self.variables_without_variance``.

        Set a dictionnary with the cleaned measurement specifications as class
        attribute ``measurements``.

        """
        measurements = {factor: [] for factor in self.factors}
        for factor, t in product(self.factors, self.periods):
            possible_meas = self._facinf[factor]["measurements"][t]
            present = [
                m
                for m in possible_meas
                if (self._present(m, t) and self._has_variance(m, t))
            ]
            measurements[factor].append(present)

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
        message = (
            "In model {} you use variable {} which has missing observations "
            "in period {} as control variable. You can either delete the "
            'variable in the model_specs or set the "controls_with_missings" '
            'key to "drop_variable" or "drop_observations" to automatically '
            "drop the variable (in period {}) or the missing observations "
            "(in all periods!), respectively"
        )

        raw_controls = self._timeinf.get("controls", [[]] * self.nperiods)
        controls = []
        bad_missings_list = []
        for t in self.periods:
            all_measurements = []
            for factor in self.factors:
                all_measurements += self.measurements[factor][t]
            df = self.data.query(f"__period__ == {t}")
            bad_missings = pd.Series(data=False, index=df.index)
            meas_df = df[all_measurements]
            controls_t = []
            for control in raw_controls[t]:
                new_bad_missings = df[control].isnull() & ~meas_df.isnull().all(axis=1)

                if self._present(control, t) and self._has_variance(control, t):
                    if not new_bad_missings.any():
                        controls_t.append(control)
                    elif self.controls_with_missings == "raise_error":
                        raise ValueError(message.format(self.model_name, control, t, t))
                    elif self.controls_with_missings == "drop_variable":
                        pass
                    elif self.controls_with_missings == "drop_observations":
                        controls_t.append(control)
                    else:
                        raise ValueError(
                            "controls_with_missings has to be raise_error, "
                            "drop_variable or drop_observation."
                        )

                bad_missings = bad_missings | new_bad_missings

            bad_missings_list.append(bad_missings)
            controls.append(tuple(controls_t))

        self.bad_missings = tuple(bad_missings_list)
        self.controls = tuple(controls)

    def _check_anchoring_specification(self):
        """Consistency checks for the model specs related to anchoring."""
        if hasattr(self, "anch_outcome"):
            for factor in self.factors:
                last_measurements = self.measurements[factor][self.nperiods - 1]
                assert self.anch_outcome not in last_measurements, (
                    "The anchoring outcome cannot be used as measurement "
                    "in the last period. In model {} you use the anchoring "
                    "outcome {} as measurement for factor {}"
                ).format(self.model_name, self.anch_outcome, factor)

    def _check_and_clean_normalizations_list(self, factor, norm_list, norm_type):
        """Check and clean a list with normalization specifications.

        Raise an error if invalid normalizations were specified.

        Transform the normalization list to the new standard specification
        (list of dicts) if the old standard (list of lists) was used.

        For the correct specification of a normalizations list refer to
        :ref:`model_specs`

        Four forms of invalid specification are checked and custom error
        messages are raised in each case:
        * Invalid length of the specification list
        * Invalid length of the entries in the specification list
        * Normalized variables that were not specified as measurement variables
          in the period where they were used
        * Normalized variables that have been dropped because they were
          not present in the dataset in the period where they were used.

        Errors are raised even if ``self.missing_variables == 'drop_variable'``
        . This is because in some cases the correct normalization is extremely
        important for the interpretation of the results. Therefore, if
        normalizations are specified manually they are not changed.

        """
        has_been_dropped_message = (
            "Normalized measurements must be present. In model {} you have "
            "specified {} as normalized variable for factor {} but it was "
            "dropped because it is not present in dataset {} in period {}"
            'and missing_variables == "drop_variable"'
        )

        was_not_specified_message = (
            "Normalized measurements must be included in the measurement list "
            "of the factor they normalize in the period where they are used. "
            "In model {} you use the variable {} to normalize factor {} in "
            "period {} but it is not included as measurement."
        )

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
                    if normed_meas in self._facinf[factor]["measurements"][t]:
                        raise KeyError(
                            has_been_dropped_message.format(
                                self.model_name,
                                normed_meas,
                                factor,
                                self.dataset_name,
                                t,
                            )
                        )
                    else:
                        raise KeyError(
                            was_not_specified_message.format(
                                self.model_name, normed_meas, factor, t
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

    def _check_and_fill_normalization_specification(self):
        """Check normalization specs or generate empty ones for each factor.

        The result is set as class attribute ``normalizations``.

        """
        norm = {}
        norm_types = ["loadings", "intercepts", "variances"]

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

    def _nmeas_list(self):
        info = self.update_info()
        nmeas_list = []
        last_period = self.periods[-1]
        for t in self.periods:
            if t != last_period or self.anchoring is False:
                nmeas_list.append(len(info.loc[t]))
            else:
                nmeas_list.append(len(info.loc[t]) - 1)
        self.nmeas_list = tuple(nmeas_list)

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

        The row order within one period is arbitrary except for the last
        period, where the row that corresponds to the anchoring update comes
        last (if anchoring is used).

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
        for t, (_f, factor) in product(self.periods, enumerate(self.factors)):
            measurements = self.measurements[factor][t].copy()
            if self.anchoring is True:
                if t == self.nperiods - 1 and factor in self.anchored_factors:
                    measurements.append(self.anch_outcome)

            for _m, meas in enumerate(measurements):
                # if meas is not the first measurement in period t
                # and the measurement has already been used in period t
                if t in df.index and meas in df.loc[t].index:
                    # change corresponding row of the DataFrame
                    df.loc[(t, meas), factor] = 1

                else:
                    # add a new row to the DataFrame
                    ind = pd.MultiIndex.from_tuples(
                        [(t, meas)], names=["period", "variable"]
                    )
                    dat = np.zeros((1, self.nfac))
                    df2 = DataFrame(data=dat, columns=self.factors, index=ind)
                    df2[factor] = 1
                    df = df.append(df2)

        # move anchoring update to last position
        if self.anchoring is True:
            anch_index = (self.nperiods - 1, self.anch_outcome)
            anch_row = df.loc[anch_index]
            df.drop(anch_index, axis=0, inplace=True)
            df = df.append(anch_row)
        return df

    def _invariance_update_info(self):
        """Update information relevant for time invariant measurement systems.
        Measurement equations are uniquely identified by their period and the
        name of their measurement.
        Two measurement equations count as equal if and only if:
        * their measurements have the same name
        * the same latent factors are measured
        * they occur in a periods that use the same control variables.
        """
        factor_uinfo = self._factor_update_info()
        ind = factor_uinfo.index
        df = pd.DataFrame(
            columns=["is_repeated", "first_occurence"],
            index=ind,
            data=[[False, np.nan]] * len(ind),
        )

        for t, meas in ind:
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

    def _purpose_update_info(self):
        factor_uinfo = self._factor_update_info()
        sr = pd.Series(
            index=factor_uinfo.index,
            name="purpose",
            data=["measurement"] * len(factor_uinfo),
        )
        if self.anchoring is True:
            anch_index = (self.nperiods - 1, self.anch_outcome)
            sr[anch_index] = "anchoring"
        return sr

    def _set_params_index(self):
        self.params_index = params_index(
            self.update_info(),
            self.controls,
            self.factors,
            self.nemf,
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
            self.nemf,
            self.stagemap,
            self.transition_names,
            self.included_factors,
            self.time_invariant_measurement_system,
            self.anchored_factors,
            self.anch_outcome,
            self.bounds_distance,
            self.estimate_X_zeros,
        )

    def public_attribute_dict(self):
        all_attributes = self.__dict__
        public_attributes = {
            key: val for key, val in all_attributes.items() if not key.startswith("_")
        }
        public_attributes["update_info"] = self.update_info()
        return public_attributes
