import pandas as pd
from pandas import DataFrame
import numpy as np
from itertools import product
import skillmodels.model_functions.transition_functions as tf


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
            self, model_name, dataset_name, model_dict, dataset, estimator,
            quiet_mode=False):
        # TODO: check where I could use quiet mode
        self.model_name = model_name
        self.estimator = estimator
        self.dataset_name = dataset_name
        self._data = dataset
        self._model_dict = model_dict
        self.quiet_mode = quiet_mode
        if 'time_specific' in model_dict:
            self._timeinf = model_dict['time_specific']
        else:
            self._timeinf = {}
        self._facinf = model_dict['factor_specific']
        self.factors = sorted(list(self._facinf.keys()))
        self.nfac = len(self.factors)
        self.nsigma = 2 * self.nfac + 1

        # set the general model specifications
        general_settings = \
            {"nemf": 1,
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
             "start_values_per_quantity":
                {
                    "deltas": 1.0,
                    "H": 1.0,
                    "R": 1.0,
                    "Q": 0.1,
                    "P_zero_diags": 0.4472135955,
                    "P_zero_off_diags": 0.0,
                    "psi": 0.1,
                    "tau": 0.1,
                    "trans_coeffs": 1.0
                },
             "numba_target": "cpu"}
        # TODO: Müssen noch andere argumente in general settings aufgenommen werden?

        if 'general' in model_dict:
            general_settings.update(model_dict['general'])
        self.__dict__.update(general_settings)

        self._set_time_specific_attributes()
        self._check_general_specifications()
        self._transition_equation_names()
        self._transition_equation_included_factors()
        self._set_anchoring_attributes()
        self._set_endogeneity_correction_attributes()
        self._clean_measurement_specifications()
        self._clean_controls_specification()
        self.nobs = self.obs_to_keep.sum()
        self._check_or_generate_normalization_specification()
        self._check_anchoring_specification()
        self.nupdates = len(self.update_info())
        self._nmeas_list()
        if self.estimator == 'wa':
            self._wa_period_weights()
            self._wa_storage_df()

    def _set_time_specific_attributes(self):
        """Set model specs related to periods and stages as attributes."""
        self.nperiods = len(self._facinf[self.factors[0]]['measurements'])
        if 'stagemap' in self._timeinf:
            self.stagemap = np.array(self._timeinf['stagemap'])
        else:
            sm = np.arange(self.nperiods)
            sm[-1] = sm[-2]
            self.stagemap = sm

        self.periods = range(self.nperiods)
        self.stages = list(set(self.stagemap))
        self.nstages = len(self.stages)
        self.stage_length_list = [
            list(self.stagemap[:-1]).count(s) for s in self.stages]

        assert len(self.stagemap) == self.nperiods, (
            'You have to specify a list of length nperiods '
            'as stagemap. Check model {}').format(self.model_name)

        assert self.stagemap[-1] == self.stagemap[-2], (
            'If you specify a stagemap of length nperiods the last two '
            'elements have to coincide because no transition equation can be '
            'estimated in the last period. Check model {}').format(
                self.model_name)

        assert np.array_equal(self.stages, range(self.nstages)), (
            'The stages have to be numbered beginning with 0 and increase in '
            'steps of 1. Your stagemap in mode {} is invalid').format(
                self.model_name)

        # TODO: remove this assert statement after implementing stages in WA!!!!!!!!!!
        if self.estimator == 'wa':
            assert list(self.stagemap)[:-1] == list(self.periods)[:-1], (
                'For the wa estimator stages cannot span more than 1 period.')

        for factor in self.factors:
            length = len(self._facinf[factor]['measurements'])
            assert length == self.nperiods, (
                'The lists of lists with the measurements must have the '
                'same length for each factor in the model. In the model {} '
                'you have one list with length {} and another with length '
                '{}.').format(self.model_name, self.nperiods, length)

    def _check_general_specifications(self):
        """Check consistency of the "general" model specifications."""
        if self.estimate_X_zeros is False:
            assert self.nemf == 1, (
                'If start states (X_zero) are not estimated it is not '
                'possible to have more than one element in the mixture '
                'distribution of the latent factors. Check model {}').format(
                    self.model_name)

    def _transition_equation_names(self):
        # todo: change to check_and_set
        """Construct a list with the transition equation name for each factor.

        The result is set as class attribute ``transition_names``.

        """
        self.transition_names = \
            [self._facinf[f]['trans_eq']['name'] for f in self.factors]

    def _transition_equation_included_factors(self):
        """Included factors and their position for each transition equation.

        Construct a list with included factors for each transition equation
        and set the results as class attribute ``included_factors``.

        Construct a list with the positions of included factors in the
        alphabetically ordered factor list and set the result as class
        attribute ``included_positions.

        """
        included_factors = []
        included_positions = []

        for factor in self.factors:
            trans_inf = self._facinf[factor]['trans_eq']
            args_f = sorted(trans_inf['included_factors'])
            pos_f = list(np.arange(self.nfac)[np.in1d(self.factors, args_f)])
            included_factors.append(args_f)
            included_positions.append(pos_f)

        self.included_factors = included_factors
        self.included_positions = included_positions

    def _set_anchoring_attributes(self):
        """Set attributes related to anchoring and make some checks."""
        # TODO: reimplement the anchoring procedure; check if all of this is still needed
        if 'anchoring' in self._model_dict:
            assert len(self._model_dict['anchoring']) <= 1, (
                'At most one anchoring equation can be estimated. You '
                'specify {} in model {}').format(
                    len(self._model_dict['anchoring']), self.model_name)
            (self.anch_outcome, self.anchored_factors), = \
                self._model_dict['anchoring'].items()
            self.anchoring = True
            if self._is_dummy(self.anch_outcome, self.periods[-1]):
                self.anchoring_update_type = 'probit'
            else:
                self.anchoring_update_type = 'linear'
            self.anch_positions = [f for f in range(self.nfac) if
                                   self.factors[f] in self.anchored_factors]
        else:
            self.anchoring = False
            self.anchored_factors = []

    def _set_endogeneity_correction_attributes(self):
        # TODO: implement a comparable endogeneity correction option for both estimators
        if 'endog_correction' in self._model_dict:
            info_dict = self._model_dict['endog_correction']
            self.endog_factor = info_dict['endog_factor']
            self.endog_correction = True
            self.endog_function = info_dict['endog_function']
        else:
            self.endog_correction = False

    def _present(self, variable, period):
        """Check if **variable** is present in **period**.

        **variable** is considered present if it is in self._data and not all
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
            'In model {} you use variable {} which is not in dataset {}. '
            'in period {}. You can either delete the variable in the '
            'model_specs or set "missing_variables" to "drop_variable" '
            'to automatically drop missing variables.').format(
                self.model_name, variable, self.dataset_name, period)

        columns = set(self._data.columns)
        df = self._data[self._data['period'] == period]
        if variable in columns and df[variable].notnull().any():
            return True
        elif self.missing_variables == 'raise_error':
            raise KeyError(message)
        else:
            return False

    def _is_dummy(self, variable, period):
        """Check if **variable** is a dummy variable in **period**.

        **variable** is considered a dummy variable if it takes the values
        0, 1 and NaN in **period**. If it only takes a subset of these values
        it is not considered a dummy variable.

        args:
            variable (str): name of the variable being checked.
            period (int): period in which the variable is checked

        Returns:
            bool: True if **variable** is a dummy in **period**, else False

        """
        series = self._data[self._data['period'] == period][variable]
        unique_values = series[pd.notnull(series)].unique()
        if sorted(unique_values) == [0, 1]:
            return True
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
            'Variables have to take at least two different values as variables'
            ' without variance cannot help to identify the model. In model {} '
            'you use variable {} which only takes the value {} in dataset {} '
            'in period {}. You can eiter drop the variable in the model_specs '
            'or set the "variables_without_variance" key in general settings '
            'to "drop_variable".')

        series = self._data[self._data['period'] == period][variable]
        unique_non_missing_values = list(series[pd.notnull(series)].unique())
        nr_unique = len(unique_non_missing_values)

        if nr_unique <= 1:
            if self.variables_without_variance == 'raise_error':
                raise ValueError(message.format(
                    self.model_name, variable, unique_non_missing_values,
                    self.dataset_name, period))
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
            possible_meas = self._facinf[factor]['measurements'][t]
            present = [m for m in possible_meas if (self._present(m, t) and
                       self._has_variance(m, t))]
            measurements[factor].append(present)

        for f, factor in enumerate(self.factors):
            if self.transition_names[f] == 'constant':
                for t in self.periods[1:]:
                    assert len(measurements[factor][t]) == 0, (
                        'In model {} factor {} has a constant transition '
                        'equation. Therefore it can only have measurements '
                        'in the initial period. However, you specified measure'
                        'ments in period {}.'.format(
                            self.model_name, factor, t))

            elif self.estimator == 'wa':
                for t in self.periods:
                    assert len(measurements[factor][t]) >= 2, (
                        'In model {} factor {} has a non-constant transition '
                        'equation. Therefore it must have at least two '
                        'measurements in every period. However, this is '
                        'not the case in period {}'.format(
                            self.model_name, factor, t))

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
            'In model {} you use variable {} which has missing observations '
            'in period {} as control variable. You can either delete the '
            'variable in the model_specs or set the "controls_with_missings" '
            'key to "drop_variable" or "drop_observation" to automatically '
            'drop the variable (in period {}) or the missing observations '
            '(in all periods!), respectively')

        self.uses_controls = False
        if 'controls' in self._timeinf:
            for t in self.periods:
                if len(self._timeinf['controls'][t]) > 0:
                    self.uses_controls = True

        if self.estimator == 'wa' and self.quiet_mode is False:
            if self.uses_controls is True:
                print('The control variables you specified in model {} will '
                      'be ignored when the model is estimated with the wa '
                      'estimator.'.format(self.model_name))

        obs_to_keep = np.ones(len(self._data) // self.nperiods, dtype=bool)
        controls = [[] for t in self.periods]

        if self.uses_controls:
            present_controls = []
            for t in self.periods:
                present_controls.append(
                    [c for c in self._timeinf['controls'][t]
                     if (self._present(c, t))])

            for t in self.periods:
                df = self._data[self._data['period'] == t]
                for c, control in enumerate(present_controls[t]):
                    if df[control].notnull().all():
                        controls[t].append(control)
                    elif self.controls_with_missings == 'drop_observations':
                        controls[t].append(control)
                        obs_to_keep = np.logical_and(
                            obs_to_keep, df[control].notnull().values)
                    elif self.controls_with_missings == 'raise_error':
                        raise ValueError(message.format(
                            self.model_name, control, t, t))

        self.controls = controls
        self.obs_to_keep = obs_to_keep

    def _check_anchoring_specification(self):
        """Consistency checks for the model specs related to anchoring."""
        if hasattr(self, 'anch_outcome'):
            for factor in self.factors:
                last_measurements = \
                    self.measurements[factor][self.nperiods - 1]
                assert self.anch_outcome not in last_measurements, (
                    'The anchoring outcome cannot be used as measurement '
                    'in the last period. In model {} you use the anchoring '
                    'outcome {} as measurement for factor {}').format(
                        self.model_name, self.anch_outcome, factor)

        if self.nemf >= 2 and self._is_dummy(self.anch_outcome):
            raise NotImplementedError(
                'Probability anchoring is not yet implemented for nemf >= 2 '
                'but your anchoring outcome {} in model {} is a dummy '
                'variable.').format(self.anch_outcmoe, self.model_name)

    def _check_normalizations_list(self, factor, norm_list):
        """Raise an error if invalid normalizations were specified.

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
            'Normalized measurements must be present. In model {} you have '
            'specified {} as normalized variable for factor {} but it was '
            'dropped because it is not present in dataset {} in period {}'
            'and missing_variables == "drop_variable"')

        was_not_specified_message = (
            'Normalized measurements must be included in the measurement list '
            'of the factor they normalize in the period where they are used. '
            'In model {} you use the variable {} to normalize factor {} in '
            'period {} but it is not included as measurement.')

        assert len(norm_list) == self.nperiods, (
            'Normalizations lists must have one entry per period. In model {} '
            'you specify a normalizations list of length {} for factor {} '
            'but the model has {} periods').format(
                self.model_name, len(norm_list), factor, self.nperiods)

        for t, norminfo in enumerate(norm_list):
            assert len(norminfo) in [0, 2], (
                'The sublists in the lists of normalizations must be empty or '
                'have a length of 2. In model {} in period {} you specify a '
                'list with len {} for factor {}').format(
                    self.model_name, t, len(norminfo), factor)

            if len(norminfo) == 2:
                normed_meas = norminfo[0]
                if normed_meas not in self.measurements[factor][t]:
                    if normed_meas in self._facinf[factor]['measurements'][t]:
                        raise KeyError(has_been_dropped_message.format(
                            self.model_name, normed_meas, factor,
                            self.dataset_name, t))
                    else:
                        raise KeyError(was_not_specified_message.format(
                            self.model_name, normed_meas, factor, t))

        return norm_list

    def _stage_has_fixed_start_period_list(self, norm_type):
        """Map list of stages to list of boolean values.

        The s_th entry in the resulting list is True if the location
        (norm_type='intercept') or scale (norm_type='loadings') in the first
        period of stage s is fixed through normalizations in previous stages
        or normalization of initial values.
        """
        # TODO: reread this function after writing the new automatic normalization specification
        has_fixed = []
        for s in self.stages:
            if s == 0:
                if norm_type == 'intercepts' and self.estimate_X_zeros is False:    # noqa
                    has_fixed.append(True)
                else:
                    has_fixed.append(False)

            elif self.stage_length_list[s - 1] == 1:
                has_fixed.append(False)
            else:
                has_fixed.append(True)
        return has_fixed

    def _first_period_in_stage(self, period):
        """Return True if period is the first period in its stage."""
        if period == 0:
            return True
        elif self.stagemap[period] > self.stagemap[period - 1]:
            return True
        else:
            return False

    def needs_normalization(self, factor, norm_type):
        """Boolean list of length nperiods.

        The t_th entry is True if factor needs a normalization of norm_type
        in period period t. Else it is False.

        """
        transition_name = self.transition_names[self.factors.index(factor)]


        needs_normalization = []
        # first period entry
        if norm_type == 'loadings' or self.estimate_X_zeros is True:
            needs_normalization.append(True)
        else:
            needs_normalization.append(False)

        if norm_type == 'loadings':
            func_string = 'output_has_known_scale_{}'
        else:
            func_string = 'output_has_known_location_{}'

        no_norm_after_0 = getattr(tf, func_string.format(transition_name))()

        if no_norm_after_0:
            needs_normalization += [False] * (self.nperiods - 1)
        elif transition_name == 'ar1':
            needs_normalization.append(True)
            needs_normalization += [False] * (self.nperiods - 2)
        else:
            has_fixed = self._stage_has_fixed_start_period_list(norm_type)
            for t in self.periods[1:]:
                stage = self.stagemap[t]
                if self._first_period_in_stage(t):
                    if has_fixed[stage]:
                        needs_normalization.append(False)
                    else:
                        needs_normalization.append(True)
                # if it is the second period in stage s
                elif self._first_period_in_stage(t - 1):
                    needs_normalization.append(True)
                else:
                    needs_normalization.append(False)
        return needs_normalization

    def generate_normalizations_list(self, factor, norm_type):
        """Generate normalizations automatically.

        If factor needs a normalization of 'loadings' in period t the loading
        of its first measurement in period t is normalized to 1.

        If factor needs a normalization of 'intercepts' in period t the
        intercept of its first measurement in period t is normalized
        to 0.

        args:
            factor (str): name of factor for which normalizations are generated
            norm_type (str): specifices the type of normalization. Takes the
                values 'loadings' or 'intercepts'.

        Returns:
            list: a normalizations list that has the same form as manually
            specified counterparts (see description in :ref:`model_specs`)

        """
        normalizations = []
        needs_normalization = self.needs_normalization(factor, norm_type)
        val = 0.0 if norm_type == 'intercepts' else 1.0
        for t in self.periods:
            if needs_normalization[t] is True:
                meas = self.measurements[factor][t][0]
                normalizations.append([meas, val])
            else:
                normalizations.append([])
        return normalizations

    def _check_or_generate_normalization_specification(self):
        """Check the normalization specs or generate it for each factor.

        The result is set as class attribute ``normalizations``.

        """
        norm = {}

        for factor in self.factors:
            norm[factor] = {}
            for norm_type in ['loadings', 'intercepts']:
                if 'normalizations' in self._facinf[factor]:
                    norminfo = self._facinf[factor]['normalizations']
                    if norm_type in norminfo:
                        norm_list = norminfo[norm_type]
                        norm[factor][norm_type] = \
                            self._check_normalizations_list(factor, norm_list)
                    else:
                        norm[factor][norm_type] = \
                            self.generate_normalizations_list(
                                factor, norm_type)
                else:
                    norm[factor][norm_type] = \
                        self.generate_normalizations_list(factor, norm_type)
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
        tranformed into a pandas DataFrame that is more convenient for the
        construction of arrays for the likelihood function and identification
        checks.

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
        * update_type: takes the value 'probit' if the measurement or
          anchoring_outcome is a dummy variable and self.probit_measurements
          is True, else 'linear'
        * has_normalized_loading: True if any loading is normalized
        * has_normalized_intercept: True if the intercept is normalized

        The row order within one period is arbitrary except for the last
        period, where the row that corresponds to the anchoring update comes
        last (if anchoring is used).

        Returns:
            DataFrame

        """
        # create an empty DataFrame with and empty MultiIndex
        index = pd.MultiIndex(
            levels=[[], []], labels=[[], []], names=['period', 'name'])
        df = DataFrame(data=None, index=index)

        norm_cols = ['{}_loading_norm_value'.format(f) for f in self.factors]
        cols = self.factors + norm_cols

        # append rows for each update that has to be performed
        for t, (f, factor) in product(self.periods, enumerate(self.factors)):
            stage = self.stagemap[t]
            load_norm_column = '{}_loading_norm_value'.format(factor)
            measurements = self.measurements[factor][t]
            if t == self.nperiods - 1 and factor in self.anchored_factors:
                measurements.append(self.anch_outcome)

            load_norminfo = self.normalizations[factor]['loadings'][t]
            load_normed_meas, load_norm_value = \
                load_norminfo if len(load_norminfo) == 2 else [None, None]

            intercept_norminfo = self.normalizations[factor]['intercepts'][t]
            intercept_normed_meas, intercept_norm_value = \
                intercept_norminfo if len(intercept_norminfo) == 2 else [None, None]    # noqa

            for m, meas in enumerate(measurements):
                # if meas is not the first measurement in period t
                # and the measurement has already been used in period t
                if (f > 0 or m > 0) and meas in df.loc[t].index:
                    # change corresponding row of the DataFrame
                    df.loc[(t, meas), factor] = 1
                    if meas == load_normed_meas:
                        df.loc[(t, meas), load_norm_column] = load_norm_value
                    if meas == intercept_normed_meas:
                        df.loc[(t, meas), 'intercept_norm_value'] = \
                            intercept_norm_value

                else:
                    # add a new row to the DataFrame
                    ind = pd.MultiIndex.from_tuples(
                        [(t, meas)], names=['period', 'variable'])
                    dat = np.zeros((1, len(cols)))
                    df2 = DataFrame(data=dat, columns=cols, index=ind)
                    df2[factor] = 1
                    if meas == load_normed_meas:
                        df2[load_norm_column] = load_norm_value
                    if meas == intercept_normed_meas:
                        df2['intercept_norm_value'] = intercept_norm_value
                    else:
                        df2['intercept_norm_value'] = np.nan
                    df2['stage'] = stage

                    df = df.append(df2)

        # create the has_normalized_loading_column
        df['has_normalized_loading'] = df[norm_cols].sum(axis=1).astype(bool)

        # create the has_normalized_intercept_column
        df['has_normalized_intercept'] = pd.notnull(df['intercept_norm_value'])

        # add the purpose_column
        df['purpose'] = 'measurement'
        if self.anchoring is True:
            anch_index = (self.nperiods - 1, self.anch_outcome)
            df.loc[anch_index, 'purpose'] = 'anchoring'

        # move anchoring row to end of DataFrame
        if self.anchoring is True:
            anch_row = df.loc[anch_index]
            df.drop(anch_index, axis=0, inplace=True)
            df = df.append(anch_row)

        # TODO: test how probit measurements are implemented in the code
        # add the update_type column
        df['update_type'] = 'linear'
        for t, variable in list(df.index):
            if self._is_dummy(variable, t) and self.probit_measurements is True:    # noqa
                df.loc[(t, variable), 'update_type'] = 'probit'

        if self.estimator == 'wa':
            df = df[df['purpose'] == 'measurement']

        return df

    def _wa_storage_df(self):
        df = self.update_info().copy(deep=True)
        norm_cols = ['{}_loading_norm_value'.format(f) for f in self.factors]
        # storage column for factor loadings, initialized with zeros for un-
        # normalized loadings and the norm_value for normalized loadings
        df['loadings'] = df[norm_cols].sum(axis=1)
        # storage column for intercepts, initialized with zeros for un-
        # normalized intercepts and the norm_value for normalized intercepts.
        df['intercepts'] = df['intercept_norm_value'].fillna(0)
        relevant_columns = \
            ['has_normalized_intercept', 'has_normalized_loading',
             'loadings', 'intercepts']
        storage_df = df[relevant_columns].copy(deep=True)
        storage_df['meas_error_variances'] = 0.0
        self.storage_df = storage_df

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
        # TODO: use this in WA estimator. Currently ar1 case is not handled correctly.
        new_params = np.zeros((self.nstages, self.nfac))

        for f, factor in enumerate(self.factors):
            name = self.transition_names[f]

            if name == 'constant':
                new_params[:, f] = -1
            elif name == 'ar1':
                new_params[0, f] = 1
                new_params[1:, f] = 0
            else:
                new_params[:, f] = 1

        return new_params

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
        df = pd.DataFrame(data=arr, index=self.periods[:-1],
                          columns=self.factors)

        for f, factor in enumerate(self.factors):
            if self.transition_names[f] == 'ar1':
                df[factor] = 1 / (self.nperiods - 1)
            elif self.transition_names[f] == 'constant':
                df[factor] = np.nan
        self.wa_period_weights = df

    def public_attribute_dict(self):
        all_attributes = self.__dict__
        public_attributes = {key: val for key, val in all_attributes.items()
                             if not key.startswith('_')}
        return public_attributes
