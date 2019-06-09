import pandas as pd
from pandas import DataFrame
import numpy as np
from itertools import product
import os
import warnings
from skillmodels.pre_processing.data_processor import pre_process_data
from skillmodels.pre_processing.params_index import params_index
from skillmodels.pre_processing.constraints import constraints


class ModelSpecProcessor:
    """Check, clean, extend and transform the model specs.

    Check the completeness, consistency and validity of the general and model
    specific specifications.

    Clean the model specs by handling variables that were specified but are
    not in the dataset or have no variance. Raise errors if specified.

    Extend the model specs by inferring dimensions, generating automatic
    normalization specifications or merging stages if too few measurements are
    available for some factors.

    Transform the cleaned model specs into forms that are more practical for
    the construction of the quantities that are needed in the likelihood
    function.

    """

    def __init__(
        self,
        model_dict,
        dataset,
        estimator,
        model_name="some_model",
        dataset_name="some_dataset",
        save_path=None,
        bootstrap_samples=None,
    ):
        self.model_dict = model_dict
        self.data = pre_process_data(dataset)
        self.estimator = estimator
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.save_path = save_path
        self.bootstrap_samples = bootstrap_samples
        if "time_specific" in model_dict:
            self._timeinf = model_dict["time_specific"]
        else:
            self._timeinf = {}
        self._facinf = model_dict["factor_specific"]
        self.factors = sorted(list(self._facinf.keys()))
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
            "robust_bounds": False,
            "bounds_distance": 1e-200,
            "estimate_X_zeros": False,
            "order_X_zeros": 0,
            "restrict_W_zeros": True,
            "restrict_P_zeros": True,
            "cholesky_of_P_zero": False,
            "probit_measurements": False,
            "probanch_function": "odds_ratio",
            "ignore_intercept_in_linear_anchoring": True,
            "start_values_per_quantity": {
                "delta": 1.0,
                "h": 1.0,
                "r": 1.0,
                "q": 0.1,
                "p_diags": 0.4472135955,
                "p_off_diags": 0.0,
                "trans_coeffs": 1.0,
            },
            # "numba_target": "cpu",
            "wa_standard_error_method": "bootstrap",
            "chs_standard_error_method": "op_of_gradient",
            "save_intermediate_optimization_results": False,
            "save_params_before_calculating_standard_errors": False,
            "maxiter": 1000000,
            "maxfun": 1000000,
            "bootstrap_nreps": 300,
            "bootstrap_sample_size": None,
            "bootstrap_nprocesses": None,
            "anchoring_mode": "only_estimate_anchoring_equation",
            "time_invariant_measurement_system": False,
            "base_color": "#035096",
        }

        if "general" in model_dict:
            general_settings.update(model_dict["general"])
        self.__dict__.update(general_settings)
        self.standard_error_method = getattr(
            self, "{}_standard_error_method".format(self.estimator)
        )
        if self.estimator == "wa":
            self.nemf = 1
            self.cholesky_of_P_zero = False
            self.square_root_filters = False
        self._set_time_specific_attributes()
        self._check_general_specifications()
        self._generate_save_directories()
        self._transition_equation_names()
        self._transition_equation_included_factors()
        self._set_anchoring_attributes()
        self._clean_measurement_specifications()
        self._clean_controls_specification()
        self.nobs = self.obs_to_keep.sum()
        self._set_bootstrap_sample_size()
        self._check_and_fill_normalization_specification()
        self._check_anchoring_specification()
        self.nupdates = len(self.update_info())
        self._nmeas_list()
        self._set_params_index()
        self._set_constraints()
        if self.estimator == "wa":
            self._wa_period_weights()
            self._wa_storage_df()
            self._wa_identified_transition_function_restrictions()
            assert self.time_invariant_measurement_system is False, (
                "Time invariant measurement system is not yet supported "
                "with the wa estimator."
            )

    def _set_bootstrap_sample_size(self):
        if self.bootstrap_samples is not None:
            bs_n = len(self.bootstrap_samples[0])
            if self.bootstrap_sample_size is not None:
                if bs_n != self.bootstrap_sample_size:
                    message = (
                        "The bootsrap_sample_size you specified in the general"
                        " section of the model dict in model {} does not "
                        "coincide with the bootstrap_samples you provide "
                        "and will be ignored."
                    ).format(self.model_name)
                    warnings.warn(message)
            self.bootstrap_sample_size = bs_n
        elif self.bootstrap_sample_size is None:
            self.bootstrap_sample_size = self.nobs

    def _generate_save_directories(self):
        if self.save_intermediate_optimization_results is True:
            os.makedirs(self.save_path + "/opt_results", exist_ok=True)
        if self.save_params_before_calculating_standard_errors is True:
            os.makedirs(self.save_path + "/params", exist_ok=True)

    def _set_time_specific_attributes(self):
        """Set model specs related to periods and stages as attributes."""
        self.nperiods = len(self._facinf[self.factors[0]]["measurements"])
        if "stagemap" in self._timeinf:
            self.stagemap = np.array(self._timeinf["stagemap"])
        else:
            sm = np.arange(self.nperiods)
            sm[-1] = sm[-2]
            self.stagemap = sm

        self.periods = list(range(self.nperiods))
        self.stages = sorted(list(set(self.stagemap)))
        self.nstages = len(self.stages)
        self.stage_length_list = [
            list(self.stagemap[:-1]).count(s) for s in self.stages
        ]

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

        assert self.wa_standard_error_method == "bootstrap", (
            "Currently, the only standard error method supported with the wa "
            "estimator is bootstrap."
        )

        chs_admissible = ["bootstrap", "op_of_gradient", "hessian_inverse"]
        assert self.chs_standard_error_method in chs_admissible, (
            "Currently, the only standard error methods supported with the "
            "chs estimator are {}".format(chs_admissible)
        )

        something_ist_saved = (
            self.save_intermediate_optimization_results
            or self.save_params_before_calculating_standard_errors
        )
        if something_ist_saved is True:
            assert self.save_path is not None, (
                "If you specified to save intermediate optimization "
                "results or estimated parameters you have to provide "
                "a save_path."
            )

        if self.estimator == "wa":
            assert self.probit_measurements is False, (
                "It is not possible to estimate probit measurement equations "
                "with the wa estimator."
            )

    def _transition_equation_names(self):
        """Construct a list with the transition equation name for each factor.

        The result is set as class attribute ``transition_names``.

        """
        self.transition_names = [
            self._facinf[f]["trans_eq"]["name"] for f in self.factors
        ]

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
            included_factors.append(args_f)
            included_positions.append(pos_f)

        self.included_factors = included_factors
        self.included_positions = included_positions

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
            self.anch_positions = [
                f for f in range(self.nfac) if self.factors[f] in self.anchored_factors
            ]
            if self.anchoring_mode == "truly_anchor_latent_factors":
                self.anchor_in_predict = True
            else:
                self.anchor_in_predict = False
        else:
            self.anchoring = False
            self.anchored_factors = []
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
        df = self.data.query('__period__ == {}'.format(period))
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

        series = self.data.query('__period__ == {}'.format(period))[variable]
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

            elif self.estimator == "wa":
                for t in self.periods:
                    assert len(measurements[factor][t]) >= 2, (
                        "In model {} factor {} has a non-constant transition "
                        "equation. Therefore it must have at least two "
                        "measurements in every period. However, this is "
                        "not the case in period {}".format(self.model_name, factor, t)
                    )

        self.measurements = measurements

    def _clean_controls_specification(self):
        """Check if control variables have NaNs and handle them as specified.

        Check for each control variable if it is present. If not, drop it or
        raise an error according to ``self.missing_variables``.

        Then check if they have missing values for some observations and drop
        the variable, drop the observations or raise a ValueError according to
        ``self.controls_with_missings``.

        Set the cleaned list of controls as class attribute ``controls``.

        Set a boolean array of length nr_individuals_in_the_sample that is True
        for observations that should be kept and False for observations that
        have to be dropped because of missing values in control variables
        as class attribute ``obs_to_keep``.

        """
        message = (
            "In model {} you use variable {} which has missing observations "
            "in period {} as control variable. You can either delete the "
            'variable in the model_specs or set the "controls_with_missings" '
            'key to "drop_variable" or "drop_observation" to automatically '
            "drop the variable (in period {}) or the missing observations "
            "(in all periods!), respectively"
        )

        self.uses_controls = False
        if "controls" in self._timeinf:
            for t in self.periods:
                if len(self._timeinf["controls"][t]) > 0:
                    self.uses_controls = True

        if self.estimator == "wa":
            if self.uses_controls is True:
                print(
                    "The control variables you specified in model {} will "
                    "be ignored when the model is estimated with the wa "
                    "estimator.".format(self.model_name)
                )

        obs_to_keep = np.ones(len(self.data) // self.nperiods, dtype=bool)
        controls = [[] for t in self.periods]

        if self.uses_controls:
            present_controls = []
            for t in self.periods:
                present_controls.append(
                    [c for c in self._timeinf["controls"][t] if (self._present(c, t))]
                )

            for t in self.periods:
                df = self.data.query('__period__ == {}'.format(t))
                for c, control in enumerate(present_controls[t]):
                    if df[control].notnull().all():
                        controls[t].append(control)
                    elif self.controls_with_missings == "drop_observations":
                        controls[t].append(control)
                        obs_to_keep = np.logical_and(
                            obs_to_keep, df[control].notnull().to_numpy()
                        )
                    elif self.controls_with_missings == "raise_error":
                        raise ValueError(message.format(self.model_name, control, t, t))

        self.controls = controls
        self.obs_to_keep = obs_to_keep

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

        if self.anchoring is True and self.estimator == "wa":
            assert self.anchor_in_predict is False, (
                "For the wa estimator the only possible anchoring_mode is ",
                "only_estimate_anchoring_equation. Check the specs of ",
                "model {}".format(self.model_name),
            )

    def _check_and_clean_normalizations_list(self, factor, norm_list, norm_type):
        """Check and clean a list with normalization specifications.

        Raise an error if invalid normalizations were specified.

        Transform the normalization list to the new standard specification
        (list of dicts) if the old standard (list of lists) was used.

        For the correct specification of a normalizations list refer to
        :ref:`model_specs`

        Four forms of invalid specification are checked and custom error
        messages are raised in each case:
            #. Invalid length of the specification list
            #. Invalid length of the entries in the specification list
            #. Normalized variables that were not specified as measurement
               variables in the period where they were used
            #. Normalized variables that have been dropped because they were
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
            raise DeprecationWarning(
                "Using lists of lists instead of lists of dicts for the "
                "normalization specification is deprecated."
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
        for t, norminfo in enumerate(norm_list):
            for n_meas, n_val in norminfo.items():
                if norm_type == "variances":
                    assert n_val > 0, "Variances can only be normalized to a value > 0."
                if norm_type == "loadings":
                    assert n_val != 0, "Loadings cannot be normalized to 0."

        if self.estimator == "wa":
            for norminfo in norm_list:
                msg = (
                    "The wa estimator currently allows at most one "
                    "normalization of {} in each period. This is "
                    "violated for factor {}: {}"
                )
                assert len(norminfo) <= 1, msg.format(norm_type, factor, norminfo)

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

        if self.estimator == "wa":
            for factor in self.factors:
                assert (
                    norm[factor]["variances"] == [{}] * self.nperiods
                ), "Normalized variances and wa estimator are incompatible."

        self.normalizations = norm

    def _nmeas_list(self):
        info = self.update_info()
        self.nmeas_list = []
        last_period = self.periods[-1]
        for t in self.periods:
            if t != last_period or self.anchoring is False:
                self.nmeas_list.append(len(info.loc[t]))
            else:
                self.nmeas_list.append(len(info.loc[t]) - 1)

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
        * A column with the loading_norm_values for each factor:
            df.loc[(t, meas), fac1] is equal to the normalized value if meas is
            a measurement with normalized factor loading for fac1 in period t.
            else it is 0.
        * intercept_norm_value: the value the intercept is normalized to or NaN
        * stage: maps updates to stages
        * purpose: takes one of the values in ['measurement', 'anchoring']
        * has_normalized_loading: True if any loading is normalized
        * has_normalized_intercept: True if the intercept is normalized
        * is_repeated: True if the same measurement equation has appeared
          in a previous period
        * first_occurrence: Period in which the same measurement equation has
          appeared first or np.nan.

        The row order within one period is arbitrary except for the last
        period, where the row that corresponds to the anchoring update comes
        last (if anchoring is used).

        Returns:
            DataFrame

        """
        to_concat = [
            self._factor_update_info(),
            self._normalization_update_info(),
            self._stage_udpate_info(),
            self._purpose_update_info(),
            self._invariance_update_info(),
        ]

        df = pd.concat(to_concat, axis=1)
        if self.time_invariant_measurement_system is True:
            df = self._rewrite_normalizations_for_time_inv_meas_system(df)

        return df

    def _factor_update_info(self):
        # create an empty DataFrame with and empty MultiIndex
        index = pd.MultiIndex(
            levels=[[], []], codes=[[], []], names=["period", "name"]
        )
        df = DataFrame(data=None, index=index)

        # append rows for each update that has to be performed
        for t, (f, factor) in product(self.periods, enumerate(self.factors)):
            measurements = self.measurements[factor][t].copy()
            if self.anchoring is True:
                if t == self.nperiods - 1 and factor in self.anchored_factors:
                    measurements.append(self.anch_outcome)

            for m, meas in enumerate(measurements):
                # if meas is not the first measurement in period t
                # and the measurement has already been used in period t
                if (f > 0 or m > 0) and meas in df.loc[t].index:
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

    def _normalization_update_info(self):
        bdf = self._factor_update_info()
        load_cols = ["{}_loading_norm_value".format(f) for f in self.factors]
        # for some reason it affects the likelihood value of a test model
        # whether the loading norm values have dtype float or int
        # therefore I fill them with 0.0 to have them explicitly as float.
        df = pd.DataFrame(index=bdf.index, columns=load_cols).fillna(0.0)

        df["intercept_norm_value"] = np.nan
        df["variance_norm_value"] = np.nan

        for (t, meas), factor in product(df.index, self.factors):
            if bdf.loc[(t, meas), factor] == 1:
                load_norm_column = "{}_loading_norm_value".format(factor)
                load_norminfo = self.normalizations[factor]["loadings"][t]

                if meas in load_norminfo:
                    df.loc[(t, meas), load_norm_column] = load_norminfo[meas]

                msg = "Incompatible normalizations of {} for {} in period {}"
                for normtype in ["intercepts", "variances"]:
                    norminfo = self.normalizations[factor][normtype][t]
                    if meas in norminfo:
                        col = "{}_norm_value".format(normtype[:-1])
                        if df.loc[(t, meas), col] != norminfo[meas]:
                            assert np.isnan(df.loc[(t, meas), col]), msg.format(
                                normtype, meas, t
                            )
                        df.loc[(t, meas), col] = norminfo[meas]

        df["has_normalized_loading"] = df[load_cols].sum(axis=1).astype(bool)
        df["has_normalized_intercept"] = pd.notnull(df["intercept_norm_value"])
        df["has_normalized_variance"] = pd.notnull(df["variance_norm_value"])
        return df

    def _stage_udpate_info(self):
        replace_dict = {t: stage for t, stage in enumerate(self.stagemap)}
        df = self._factor_update_info()
        df["period"] = df.index.get_level_values("period")
        df["stage"] = df["period"].replace(replace_dict)
        return df["stage"]

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

    def _rewrite_normalizations_for_time_inv_meas_system(self, df):
        """Return a copy of df with rewritten normalization info.

        make sure that all normalizations that are done in any occurrence of a
        measurement equation are also present in all other occurrences.

        """
        assert (
            self.time_invariant_measurement_system is True
        ), "Must not be called if measurement system is not time invariant."
        df = df.copy(deep=True)

        loading_msg = (
            "Incompatible normalizations of factor loadings for time "
            "invariant measurement system. Check normalizations of "
            "{} in periods {} and {} for factor {}"
        )

        other_msg = (
            "Incompatible normalizations of {} for time invariant measurement "
            "system. Check normalizations of {} in periods {} and {}"
        )

        # check normalizations and write them into first occurrence
        for factor in self.factors:
            normcol = "{}_loading_norm_value".format(factor)
            norm_dummy = "has_normalized_loading"
            for t, meas in df.index:
                repeated = df.loc[(t, meas), "is_repeated"] == True
                normalized = df.loc[(t, meas), normcol] != 0
                if repeated and normalized:
                    first_occ = df.loc[(t, meas), "first_occurence"]
                    nval = df.loc[(t, meas), normcol]
                    nval_first = df.loc[(first_occ, meas), normcol]
                    assert nval_first in [0, nval], loading_msg.format(
                        meas, first_occ, t, factor
                    )
                    df.loc[(first_occ, meas), normcol] = nval
                    df.loc[(first_occ, meas), norm_dummy] = True

        for norm_type in ["intercepts", "variances"]:
            for t, meas in df.index:
                normcol = "{}_norm_value".format(norm_type[:-1])
                norm_dummy = "has_normalized_{}".format(norm_type[:-1])
                repeated = df.loc[(t, meas), "is_repeated"] == True
                normalized = df.loc[(t, meas), norm_dummy] == True
                if repeated and normalized:
                    first_occ = df.loc[(t, meas), "first_occurence"]
                    nval = df.loc[(t, meas), normcol]
                    nval_first = df.loc[(first_occ, meas), normcol]
                    normalized_first = df.loc[(first_occ, meas), norm_dummy]
                    if normalized_first == True:
                        assert nval_first == nval, other_msg.format(
                            norm_type, meas, first_occ, t
                        )
                    df.loc[(first_occ, meas), normcol] = nval
                    df.loc[(first_occ, meas), norm_dummy] = True

        # copy consolidated normalizations to all other occurrences
        all_normcols = [
            "{}_loading_norm_value".format(factor) for factor in self.factors
        ]
        all_normcols += [
            "intercept_norm_value",
            "variance_norm_value",
            "has_normalized_loading",
            "has_normalized_intercept",
            "has_normalized_variance",
        ]

        for t, meas in df.index:
            if df.loc[(t, meas), "is_repeated"] == True:
                first_occurence = df.loc[(t, meas), "first_occurence"]
                df.loc[(t, meas), all_normcols] = df.loc[
                    (first_occurence, meas), all_normcols
                ]

        return df

    def new_meas_coeffs(self):
        """DataFrame that indicates if new parameters from params are needed.

        The DataFrame has the same index as update_info. The columns are the
        union of:

        * the latent factors
        * all control variables ever used
        * intercept
        * variance

        The entry in column c of line (t, meas) is True if the measurement
        equation of meas in period t needs a new entry from params for
        the type of measurement parameter that is associated with column c.
        Else it is False.

        """
        uinfo = self.update_info()
        all_controls = self._all_controls_list()
        new_params = pd.DataFrame(index=uinfo.index)

        for param in self.factors:
            normcol = "{}_loading_norm_value".format(param)
            not_normalized = ~uinfo[normcol].astype(bool)
            not_repeated = ~uinfo["is_repeated"]
            applicable = uinfo[param].astype(bool)
            if self.time_invariant_measurement_system is True:
                new_params[param] = not_normalized & not_repeated & applicable
            else:
                new_params[param] = not_normalized & applicable

        for param in ["intercept", "variance"]:
            not_normalized = ~uinfo["has_normalized_{}".format(param)]
            not_repeated = ~uinfo["is_repeated"]
            if self.time_invariant_measurement_system is True:
                new_params[param] = not_normalized & not_repeated
            else:
                new_params[param] = not_normalized

        for param in all_controls:
            not_repeated = ~uinfo["is_repeated"]
            applicable = pd.Series(index=uinfo.index, data=True)
            for t in self.periods:
                if param not in self.controls[t]:
                    applicable[t] = False
            if self.time_invariant_measurement_system is True:
                new_params[param] = not_repeated & applicable
            else:
                new_params[param] = applicable

        return new_params

    def _all_controls_list(self):
        """Control variables without duplicates in order of occurrence."""
        all_controls = []
        for cont_list in self.controls:
            for cont in cont_list:
                if cont not in all_controls:
                    all_controls.append(cont)
        return all_controls

    def _wa_storage_df(self):
        df = self.update_info().copy(deep=True)
        df = df[df["purpose"] == "measurement"]
        assert (
            df[self.factors].to_numpy().sum(axis=1) == 1
        ).all(), "In the wa estimator each measurement can only measure 1 factor."
        norm_cols = ["{}_loading_norm_value".format(f) for f in self.factors]
        # storage column for factor loadings, initialized with zeros for un-
        # normalized loadings and the norm_value for normalized loadings
        df["loadings"] = df[norm_cols].sum(axis=1)
        # storage column for intercepts, initialized with zeros for un-
        # normalized intercepts and the norm_value for normalized intercepts.
        df["intercepts"] = df["intercept_norm_value"].fillna(0)
        relevant_columns = [
            "has_normalized_intercept",
            "has_normalized_loading",
            "loadings",
            "intercepts",
        ]
        storage_df = df[relevant_columns].copy(deep=True)
        storage_df["meas_error_variances"] = 0.0
        self.storage_df = storage_df

    def _wa_identified_transition_function_restrictions(self):
        restriction_dict = {}
        for rtype in ["coeff_sum_value", "trans_intercept_value"]:
            df = pd.DataFrame(
                data=[[None] * self.nfac] * self.nstages,
                columns=self.factors,
                index=self.stages,
            )
            restriction_dict[rtype] = df
        self.identified_restrictions = restriction_dict

    def new_trans_coeffs(self):
        """Array that indicates if new parameters from params are needed.

        The transition equation of a factor either uses new parameters in each
        stage, reuses the parameters from the previous stage or does not need
        parameters at all.

        * For an AR1 process only one parameter is taken from params in the
          first stage. Then it is reused in all other stages.
        * For a constant process no parameters are needed at all.

        Returns:
            array of [nstages, nfac]. The s_th element in the
            f_th row is 1 if new parameters from params are used in stage s
            for the transition equation of factor f. It is 0 in the case
            of reuse and -1 if the transition equation doesn't take
            parameters at all.

        """
        new_params = np.zeros((self.nstages, self.nfac))

        for f, factor in enumerate(self.factors):
            name = self.transition_names[f]

            if name == "constant":
                new_params[:, f] = -1
            elif name == "ar1":
                new_params[0, f] = 1
                new_params[1:, f] = 0
            else:
                new_params[:, f] = 1

        return new_params

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
        )

    def _wa_period_weights(self):
        """Dataframe of shape (nperiods - 1, nfac) with weights.

        The weights are used to combine the transition parameters of several
        periods if they belong to the same stage. In the case of an ar1
        transition equations the paramaters of all periods are combined.
        Currently, weights are the same for all periods in a stage and only
        depend on the length of the stage and type of transition equation.
        The format of the DataFrame was chosen to facilitate the
        implementation of a more efficient weighting scheme later.

        """
        # OPTIONAL: base thes on new_params instead of special treatment for
        # ar1 and constant.
        arr = np.ones((self.nperiods - 1, self.nfac))

        for t, f in product(self.periods[:-1], range(self.nfac)):
            s = self.stagemap[t]
            arr[t, f] /= self.stage_length_list[s]
        df = pd.DataFrame(data=arr, index=self.periods[:-1], columns=self.factors)

        for f, factor in enumerate(self.factors):
            if self.transition_names[f] == "ar1":
                df[factor] = 1 / (self.nperiods - 1)
            elif self.transition_names[f] == "constant":
                df[factor] = np.nan
        self.wa_period_weights = df

    def public_attribute_dict(self):
        all_attributes = self.__dict__
        public_attributes = {
            key: val for key, val in all_attributes.items() if not key.startswith("_")
        }
        public_attributes["update_info"] = self.update_info()
        public_attributes["new_meas_coeffs"] = self.new_meas_coeffs()
        public_attributes["new_trans_coeffs"] = self.new_trans_coeffs()
        return public_attributes
