import pandas as pd
from pandas import DataFrame
import numpy as np
from itertools import product


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

    def __init__(self, model_name, dataset_name, model_dict, dataset):

        self.model_name = model_name
        self.dataset_name = dataset_name
        self._data = dataset
        self._model_dict = model_dict
        self._timeinf = model_dict['time_specific']
        self._facinf = model_dict['factor_specific']
        self.factors = sorted(list(self._facinf.keys()))
        self.nfac = len(self.factors)
        self.nsigma = 2 * self.nfac + 1

        # set the general model specifications
        general_settings = \
            {"nemf": 1,
             "sigma_method": "julier",
             "kappa": 2,
             "alpha": 0.5,
             "beta": 2,
             "square_root_filters": True,
             "missing_variables": "raise_error",
             "controls_with_missings": "raise_error",
             "variables_without_variance": "raise_error",
             "check_enough_measurements": "no_check",
             "robust_bounds": False,
             "bounds_distance": 1e-200,
             "add_constant": False,
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
        # call functions that set the basic class attributes and call functions
        if self.check_enough_measurements == 'raise_error':
            self._check_enough_measurements()

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
        present_controls = []
        for t in self.periods:
            present_controls.append(
                [c for c in self._timeinf['controls'][t]
                 if (self._present(c, t))])

        message = (
            'In model {} you use variable {} which has missing observations '
            'in period {} as control variable. You can either delete the '
            'variable in the model_specs or set the "controls_with_missings" '
            'key to "drop_variable" or "drop_observation" to automatically '
            'drop the variable (in period {}) or the missing observations '
            '(in all periods!), respectively')

        controls = [[] for t in self.periods]
        obs_to_keep = np.ones(len(self._data) // self.nperiods, dtype=bool)

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

    def _check_normalizations_list(self, factor):
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

        normalizations = self._facinf[factor]['normalizations']
        assert len(normalizations) == self.nperiods, (
            'Normalizations lists must have one entry per period. In model {} '
            'you specify a normalizations list of length {} for factor {} '
            'but the model has {} periods').format(
                self.model_name, len(normalizations), factor, self.nperiods)

        for t, norminfo in enumerate(normalizations):
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

        return normalizations

    def generate_normalizations_list(self, factor):
        """Generate normalizations automatically.

        This method automatically decides when a normalization is needed and
        which factor loadings are normalized. The normalized value is always
        one.

        There will be two modes to decide when a normalization is needed:

            #. chs: one normalization per period per factor
            #. parsimonious: based on my intuition less normalizations are
               needed; Making all the normalizations that chs suggest is more
               like making assumptions on equal scales of factors across
               periods. However, chs version or manual normalizations should be
               used if these assumptions are justified.

        The method tries to reduce computations by normalizing loadings in
        anchoring equations to one. This can make it unnecessary
        to anchor sigma_points at all in the case of linear anchoring and
        reduces the number of computations required for the anchoring in
        the probability anchoring cases.

        .. todo:: Write this function

        args:
            factor (str): name of factor for which normalizations are generated

        Returns:
            list: a normalizations list that has the same form as manually
            specified counterparts (see description in :ref:`model_specs`)

        """
        raise NotImplementedError(
            'Automatic generation of the normalization specifications is not' +
            'yet implemented.')

    def _check_or_generate_normalization_specification(self):
        """Check the normalization specs or generate it for each factor.

        The result is set as class attribute ``normalizations``.

        """
        norm = {}
        for factor in self.factors:
            if 'normalizations' in self._facinf[factor]:
                norm[factor] = self._check_normalizations_list(factor)
            else:
                norm[factor] = self.generate_normalizations_list(factor)
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
        * A column with the norm_value for each factor: df.loc[(t, meas), fac1]
          is equal to the normalized value if meas is a measurement with
          normalized factor loading for fac1 in period t. else it is 0.
        * stage: maps updates to stages
        * purpose: takes one of the values in ['measurement', 'anchoring']
        * update_type: takes the value 'probit' if the measurement or
          anchoring_outcome is a dummy variable, else 'linear'

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

        cols = self.factors + ['{}_norm_value'.format(f) for f in self.factors]

        # append rows for each update that has to be performed
        for t, (f, factor) in product(self.periods, enumerate(self.factors)):
            stage = self.stagemap[t]
            norm_column = '{}_norm_value'.format(factor)
            measurements = self.measurements[factor][t]
            if t == self.nperiods - 1 and factor in self.anchored_factors:
                measurements.append(self.anch_outcome)

            norminfo = self.normalizations[factor][t]
            normed_meas, norm_value = \
                norminfo if len(norminfo) == 2 else [None, None]

            for m, meas in enumerate(measurements):
                # if meas is not the first measurement in period t
                # and the measurement has already been used in period t
                if (f > 0 or m > 0) and meas in df.loc[t].index:
                    # change corresponding row of the DataFrame
                    df.loc[(t, meas), factor] = 1
                    if meas == normed_meas:
                        df.loc[(t, meas), norm_column] = norm_value
                else:
                    # add a new row to the DataFrame
                    ind = pd.MultiIndex.from_tuples(
                        [(t, meas)], names=['period', 'variable'])
                    dat = np.zeros((1, len(cols)))
                    df2 = DataFrame(data=dat, columns=cols, index=ind)
                    df2[factor] = 1
                    if meas == normed_meas:
                        df2[norm_column] = norm_value
                    df2['stage'] = stage

                    df = df.append(df2)

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

        # add the update_type column
        df['update_type'] = 'linear'
        for t, variable in list(df.index):
            if self._is_dummy(variable, t):
                df.loc[(t, variable), 'update_type'] = 'probit'

        return df

    def enough_measurements_array(self):
        """Arr[s, f] is True if factor f has enough measurements in stage s.

        To check if factor f has enough measurements to identify
        its transition equation in stage s, a simple heuristic is used:

        The heuristic is based on the concept of unproblematic periods:
        period t is a unproblematic period for factor f if it has >= 2
        measurements for f of which at least one only measures f and t + 1
        fulfills the same condition. Last periods cannot be unproblematic.

        A general transition equation for factor f in stage s is identified if
        s has at least one unproblematic period for factor f.

        Some transition equations need less: an ar1 transition equation
        is identified if its factor has at least one unproblematic period
        in any stage. Constant transition equations have no estimated parameter
        and thus are always identified:

        As it is only a heuristic, it is possible that models pass this
        test but do not converge and the other way round. However, experience
        suggests that the heuristic works pretty well.

        """
        df = self.update_info().copy(deep=True)
        # drop what is not needed and discard info on normalized measurements
        df = df[df['purpose'] == 'measurement'][self.factors].copy(deep=True)

        # construct df that is True at row t and column f if factor f has at
        # least two measurements in period t.
        two_or_more = df.groupby(level='period').sum().replace(
            {1: 0}).astype(bool)

        # construct df that is True at row t and column f if factor f has at
        # least 1 dedicated measurement in period t.
        df[df.sum(axis=1) >= 2] = 0
        one_or_more_dedicated = df.groupby(level='period').sum().astype(bool)

        # a df where both previous conditions must be fulfilled
        unproblematic = two_or_more & one_or_more_dedicated
        # incorporate the condition on the next period
        unproblematic.loc[:self.periods[-2]] &= unproblematic.loc[1:].values
        unproblematic['stage'] = self.stagemap

        # calculate the array of identified stages
        enough_meas = unproblematic.groupby('stage').sum().values.astype(bool)

        # handle the particularities of 'constant' and 'ar1' functions
        for f, factor in enumerate(self.factors):
            if self.transition_names[f] == 'constant':
                enough_meas[:, f] = True
            elif self.transition_names[f] == 'ar1' and enough_meas[:, f].any():
                enough_meas[:, f] = True

        return enough_meas

    def _check_enough_measurements(self):
        """Raise error if some factors have not enough measurements."""
        assert_message = (
            'Model {} with dataset {} has probably not enough measurements '
            'to identify all transition equations. You have three options to '
            'solve this: 1) ignore it by setting check_enough_measurements in '
            'the model_specs general section to "no_check". 2) add '
            'measurements to make the model identified (You can refer to the '
            'documentation of the enough_measurements_array method '
            'of the ModelParser class to understand in which periods you '
            'should add measurements). 3) set check_enough_measurements to '
            '"merge_stages" to automatically merge unidentified stages with '
            'identified stages.')

        assert self.enough_measurements_array().all() is True, assert_message

    def new_trans_coeffs(self):
        """Array that indicates if new parameters from params are needed.

        The transition equation of a factor either uses new parameters in each
        stage, reuses the parameters from the previous stage or does not need
        parameters at all.

        * For an AR1 process only one parameter is taken from params in the
          first stage. Then it is reused in all other stages.
        * For a constant process no parameters are needed at all.
        * If too few measurements are available for some factors and
          *check_enough_measurements* is set to *merge_stages* the transition
          equation of stage s reuses the parameters of stage s - 1 in order to
          ensure identification.

        Returns:
            boolean array of [nstages, nfac]. The s_th element in the
            f_th row is 1 if new parameters from params are used in stage s
            for the transition equation of factor f. It is 0 in the case
            of reuse and -1 if the transition equation doesn't take
            parameters at all.

        """
        new_params = np.zeros((self.nstages, self.nfac))
        enough = self.enough_measurements_array

        if self.check_enough_measurements == 'merge_stages':
            merge = True
        else:
            merge = False

        for f, factor in enumerate(self.factors):
            name = self.transition_names[f]

            if merge is True:
                assert enough[:, f].any(), (
                    'Not enough measurements in model {} for factor {}.'
                    'Merging stages can only help to identify the transition '
                    'equation of a factor if this factor has an identified '
                    'transition equation in at least one stage.').format(
                        self.model_name, self.factor)

            if name == 'constant':
                new_params[:, f] = -1
            elif name == 'ar1':
                new_params[0, f] = 1
                new_params[1:, f] = 0
            elif merge is True:
                next_needs = True
                for s in self.stages:
                    if next_needs is True:
                        new_params[s, f] = 1
                    else:
                        new_params[s, f] = 0
                    if s < self.stages[-1]:
                        if enough[s, f] == True and enough[s + 1:, f].any():
                            next_needs = True
                        else:
                            next_needs = False
            else:
                new_params[:, f] = 1

        return new_params

    def public_attribute_dict(self):
        all_attributes = self.__dict__
        public_attributes = {key: val for key, val in all_attributes.items()
                             if not key.startswith('_')}
        return public_attributes
