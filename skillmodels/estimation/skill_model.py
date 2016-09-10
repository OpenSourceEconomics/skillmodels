from skillmodels.pre_processing.model_spec_processor import ModelSpecProcessor
from skillmodels.pre_processing.data_processor import DataProcessor
from skillmodels.estimation.likelihood_function import \
    log_likelihood_per_individual
from skillmodels.estimation.wa_functions import initial_cov_matrix, \
    residual_measurements, iv_reg, initial_meas_coeffs, prepend_index_level
from statsmodels.base.model import GenericLikelihoodModel
from statsmodels.base.model import LikelihoodModelResults
from skillmodels.estimation.skill_model_results import SkillModelResults
import numpy as np
import skillmodels.model_functions.transition_functions as tf
import skillmodels.estimation.parse_params as pp
from itertools import product
from scipy.optimize import minimize
from statsmodels.tools.numdiff import approx_hess, approx_fprime
from patsy import dmatrix
import pandas as pd


class SkillModel(GenericLikelihoodModel):
    """Estimate dynamic nonlinear latent factor models with maximum likelihood.

    SkillModel is a subclass of GenericLikelihoodModel from statsmodels and
    inherits many useful methods such as statistical tests and the calculation
    of standard errors from its parent class. Its usage is described in
    :ref:`basic_usage`.

    When initialized, all public attributes of ModelSpecProcessor and the
    arrays with c_data and y_data from DataProcessor are set as attributes.
    Moreover update_info(), enough_measurements_array() and new_trans_coeffs()
    from ModelSpecCleaner are set as attributes.

    In addition to the methods inherited from GenericLikelihoodModel,
    SkillModel contains methods to determine how the params vector has to be
    parsed and methods to construct argument dictionaries for the likelihood
    function.

    """

    def __init__(
            self, model_name, dataset_name, model_dict, dataset, estimator):
        """Initialize the CHSModel class and set attributes."""
        self.estimator = estimator
        specs = ModelSpecProcessor(
            model_name, dataset_name, model_dict, dataset, estimator)
        self.__dict__.update(specs.public_attribute_dict())

        data = DataProcessor(
            model_name, dataset_name, model_dict, dataset, estimator)

        self.c_data = data.c_data() if self.estimator == 'chs' else None
        self.y_data = data.y_data()

        self.update_info = specs.update_info()
        self.new_trans_coeffs = specs.new_trans_coeffs()

        # create a list of all quantities that depend from params vector
        self.params_quants = \
            ['deltas', 'H', 'R', 'Q', 'P_zero', 'trans_coeffs']
        if self.estimate_X_zeros is True:
            self.params_quants.append('X_zero')
        if self.endog_correction is True:
            self.params_quants += ['psi', 'tau']
        if self.restrict_W_zeros is False:
            self.params_quants.append('W_zero')

        self.df_model = self.len_params(params_type='short')
        self.df_resid = self.nobs - self.df_model

    def _general_params_slice(self, length):
        """Slice object for params taking the "next" *length* elements.

        The class attribute param_counter is used to determine which are the
        "next" elements. It keeps track of which entries from the parameter
        vector params already have been used to construct quantities that are
        needed in the likelihood function.

        The method has a side effect on self.param_counter and should never
        be called by the user.

        """
        assert hasattr(self, 'param_counter'), (
            'Users must not call any of the private _params_slice methods '
            'but only the public params_slices() method that returns a '
            'dictionary with params_slices for each params_quant.')
        res = slice(int(self.param_counter), int(self.param_counter + length))
        self.param_counter += length
        return res

    def _initial_deltas(self):
        """List of initial arrays for control variable params in each period.

        The arrays have the shape [nupdates, nr_of_control_variables]
        which is potentially different in each period. They are filled with
        zeros.

        """
        deltas = []
        for t in self.periods:
            length = len(self.update_info.loc[t])
            width = len(self.controls[t]) + 1
            init = np.zeros((length, width))
            init[:, 0] = \
                self.update_info.loc[t]['intercept_norm_value'].fillna(0)
            deltas.append(init)
        return deltas

    def _deltas_bool(self):
        """List of length nperiods with boolean arrays.

        The arrays have the same shape as the corresponding initial_delta and
        are True where the initial_delta has to be overwritten with
        entries from the params vector.

        """
        deltas_bool = []
        for t in self.periods:
            length = len(self.update_info.loc[t])
            width = len(self.controls[t]) + 1
            has_normalized = self.update_info.loc[t][
                'has_normalized_intercept'].astype(int)
            boo = np.ones((length, width))
            boo[:, 0] -= has_normalized
            deltas_bool.append(boo.astype(bool))
        return deltas_bool

    def _params_slice_for_deltas(self, params_type):
        """A slice object, selecting the part of params mapped to deltas.

        The method has a side effect on self.param_counter and should never
        be called by the user.

        """
        deltas_bool = self._deltas_bool()
        params_slices = []
        for t in self.periods:
            length = deltas_bool[t].sum()
            params_slices.append(self._general_params_slice(length))
        return params_slices

    def _deltas_names(self, params_type):
        """List with names for the params mapped to deltas."""
        deltas_bool = self._deltas_bool()
        deltas_names = []
        for t in self.periods:
            updates = list(self.update_info.loc[t].index)
            for u, update in enumerate(updates):
                if deltas_bool[t][u, 0] == True:
                    constant_list = ['constant']
                else:
                    constant_list = []
                controls_list = constant_list + self.controls[t]
                for control in controls_list:
                    deltas_names.append('delta__{}__{}__{}'.format(
                        t, update, control))
        return deltas_names

    def _initial_psi(self):
        """Initial psi vector filled with ones."""
        return np.ones(self.nfac)

    def _psi_bool(self):
        """Boolean array.

        It has the same shape as initial_psi and is True where initial_psi
        has to be overwritten entries from params.

        """
        return np.array(self.factors) != self.endog_factor

    def _params_slice_for_psi(self, params_type):
        """A slice object, selecting the part of params mapped to psi.

        The method has a side effect on self.param_counter and should never
        be called by the user.

        """
        length = self.nfac - 1
        return self._general_params_slice(length)

    def _psi_names(self, params_type):
        """List with names for the params mapped to psi."""
        psi_names = []
        for factor in self.factors:
            if factor != self.endog_factor:
                psi_names.append('psi__{}'.format(factor))
        return psi_names

    def _initial_tau(self):
        """Initial tau array, filled with zeros."""
        return np.zeros((self.nstages, self.nfac))

    def _tau_bool(self):
        """Boolean array.

        It has the same shape as initial_tau and is True where initial_tau
        has to be overwritten with entries from params.

        """
        boo = self._initial_tau().astype(bool)
        for f, factor in enumerate(self.factors):
            if factor != self.endog_factor:
                if self.endog_factor in self.included_factors[f]:
                    boo[:, f] = True
        return boo

    def _params_slice_for_tau(self, params_type):
        """A slice object, selecting the part of params mapped to tau.

        The method has a side effect on self.param_counter and should never
        be called by the user.

        """
        length = self._tau_bool().sum()
        return self._general_params_slice(length)

    def _tau_names(self, params_type):
        """List with names for the params mapped to tau."""
        tau_names = []
        boo = self._tau_bool()
        for s, (f, factor) in product(self.stages, enumerate(self.factors)):
            if boo[s, f] == True:
                tau_names.append('tau__{}__{}'.format(s, factor))
        return tau_names

    def _initial_H(self):
        """Initial H array filled with zeros and normalized factor loadings.

        The array has the form [nupdates, nfac]. Most entries
        are zero, but if the factor loading of factor f in update equation
        u is normalized to some value then arr[u, f] is equal to this value.

        """
        column_list = ['{}_loading_norm_value'.format(f) for f in self.factors]
        df = self.update_info[column_list]
        return df.values

    def _H_bool(self):
        """Boolean array.

        It has the same shape as initial_H and is True where initial_H
        has to be overwritten entries from params.

        """
        measured = self.update_info[self.factors].values.astype(bool)
        normalized = self._initial_H().astype(bool)
        return np.logical_and(measured, np.logical_not(normalized))

    def _helpers_for_H_transformation_with_psi(self):
        """A boolean array and two empty arrays to store intermediate results.

        To reduce the number of computations when using endogeneity correction,
        some vectors with factor loadings (i.e. some rows of H) are transformed
        using psi.

        Returns:
            boolean array that indicates which rows are transformed

            numpy array to store intermediate results

            numpy arrey to store intermediate results

        """
        assert self.endog_correction is True, (
            'The psi_bool_for_H method should only be called if '
            'endog_correction is True. You did otherwise in model {}').format(
                self.model_name)

        psi_bool = \
            self.update_info[self.endog_factor].values.flatten().astype(bool)
        arr1 = np.zeros((psi_bool.sum(), 1))
        arr2 = np.zeros((psi_bool.sum(), self.nfac))
        return psi_bool, arr1, arr2

    def _params_slice_for_H(self, params_type):
        """A slice object, selecting the part of params mapped to H.

        The method has a side effect on self.param_counter and should never
        be called by the user.

        """
        length = self._H_bool().sum()
        return self._general_params_slice(length)

    def _H_names(self, params_type):
        """List with names for the params mapped to H."""
        H_names = []
        H_bool = self._H_bool()
        for k, (period, measure) in enumerate(list(self.update_info.index)):
            for f, factor in enumerate(self.factors):
                if H_bool[k, f] == True:
                    H_names.append('H__{}__{}__{}'.format(
                        period, factor, measure))
        return H_names

    def _initial_R(self):
        """1d numpy array of length nupdates filled with zeros."""
        return np.zeros(self.nupdates)

    def _params_slice_for_R(self, params_type):
        """A slice object, selecting the part of params mapped to R.

        The method has a side effect on self.param_counter and should never
        be called by the user.

        """
        return self._general_params_slice(self.nupdates)

    def _R_names(self, params_type):
        """List with names for the params mapped to R."""
        R_names = ['R__{}__{}'.format(t, measure)
                   for t, measure in list(self.update_info.index)]
        return R_names

    def _set_bounds_for_R(self, params_slice):
        """Set lower bounds for params mapped to R."""
        self.lower_bound[params_slice] = \
            self.robust_bounds * self.bounds_distance

    def _initial_Q(self):
        """Initial Q array filled with zeros."""
        return np.zeros((self.nstages, self.nfac, self.nfac))

    def _Q_bool(self):
        """Boolean array.

        It has the same shape as initial_Q and is True where initial_Q
        has to be overwritten entries from params.

        """
        Q_bool = np.zeros((self.nstages, self.nfac, self.nfac), dtype=bool)
        for s in self.stages:
            Q_bool[s, :, :] = np.diag(self.new_trans_coeffs[s] == 1)
        return Q_bool

    def _params_slice_for_Q(self, params_type):
        """A slice object, selecting the part of params mapped to Q.

        The method has a side effect on self.param_counter and should never
        be called by the user.

        """
        length = self._Q_bool().sum()
        return self._general_params_slice(length)

    def _Q_names(self, params_type):
        """List with names for the params mapped to Q."""
        Q_names = []
        for s, (f, factor) in product(self.stages, enumerate(self.factors)):
            if self.new_trans_coeffs[s, f] == 1:
                Q_names.append('Q__{}__{}'.format(s, factor))
        return Q_names

    def _Q_replacements(self):
        """List of pairs of index tuples.

        The first tuple indicates where to put the Element of Q described
        by the second index tuple.

        """
        replacements = []
        for s, f in product(self.stages, range(self.nfac)):
            if self.new_trans_coeffs[s, f] == 0:
                replacements.append([(s, f, f), (s - 1, f, f)])
        return replacements

    def _set_bounds_for_Q(self, params_slice):
        """Set lower bounds for params mapped to Q."""
        self.lower_bound[params_slice] = \
            self.robust_bounds * self.bounds_distance

    def _initial_X_zero(self):
        """Initial X_zero array filled with zeros."""
        init = np.zeros((self.nobs, self.nemf, self.nfac))
        flat_init = init.reshape(self.nobs * self.nemf, self.nfac)
        return init, flat_init

    def _X_zero_filler(self):
        """Helper array used to fill X_zero with parameters."""
        return np.zeros((self.nemf, self.nfac))

    def _params_slice_for_X_zero(self, params_type):
        """A slice object, selecting the part of params mapped to X_zero.

        The method has a side effect on self.param_counter and should never
        be called by the user.

        """
        return self._general_params_slice(self.nemf * self.nfac)

    def _X_zero_replacements(self):
        """List of pairs of index tuples.

        The first tuple indicates where to add the Element of X_zero described
        by the second index tuple.

        """
        replacements = []
        if self.nemf > 1:
            for i in range(1, self.nemf):
                replacements.append(
                    [(i, self.order_X_zeros),
                     (i - 1, self.order_X_zeros)])
        return replacements

    def _set_bounds_for_X_zero(self, params_slice):
        """Set lower bounds for parameters mapped to diagonal of P_zero."""
        bounds = self.lower_bound[params_slice].reshape(self.nemf, self.nfac)
        if self.nemf > 1:
            bounds[1:, self.order_X_zeros] = 0
        self.lower_bound[params_slice] = bounds.flatten()

    def _X_zero_names(self, params_type):
        """List with names for the params mapped to X_zero."""
        X_zero_names = []
        for n, (f, fac) in product(range(self.nemf), enumerate(self.factors)):
            if params_type == 'long' or n == 0 or f != self.order_X_zeros:
                format_string = 'X_zero__{}__{}'
            else:
                format_string = 'diff_X_zero__{}__{}'
            X_zero_names.append(format_string.format(n, fac))
        return X_zero_names

    def _initial_W_zero(self):
        """Initial W_zero array filled with 1/nemf."""
        return np.ones((self.nobs, self.nemf)) / self.nemf

    def _params_slice_for_W_zero(self, params_type):
        """A slice object, selecting the part of params mapped to W_zero.

        The method has a side effect on self.param_counter and should never
        be called by the user.

        """
        return self._general_params_slice(self.nemf)

    def _W_zero_names(self, params_type):
        """List with names for the params mapped to W_zero."""
        return ['W_zero__{}'.format(n) for n in range(self.nemf)]

    def _initial_P_zero(self):
        """Initial P_zero array filled with zeros."""
        if self.square_root_filters is False:
            init = np.zeros((self.nobs, self.nemf, self.nfac, self.nfac))
            flat_init = init.reshape(
                self.nobs * self.nemf, self.nfac, self.nfac)
        else:
            init = np.zeros(
                (self.nobs, self.nemf, self.nfac + 1, self.nfac + 1))
            flat_init = init.reshape(
                self.nobs * self.nemf, self.nfac + 1, self.nfac + 1)
        return init, flat_init

    def _P_zero_filler(self):
        """Helper array used to fill X_zero with parameters."""
        nr_matrices = 1 if self.restrict_P_zeros is True else self.nemf
        return np.zeros((nr_matrices, self.nfac, self.nfac))

    def _P_zero_bool(self):
        """Boolean array.

        It has the same shape as P_zero_filler and is True where P_zero_filler
        has to be overwritten entries from params.

        """
        helper = np.zeros((self.nfac, self.nfac), dtype=bool)
        helper[np.triu_indices(self.nfac)] = True
        filler_bool = np.zeros_like(self._P_zero_filler(), dtype=bool)
        filler_bool[:] = helper
        return filler_bool

    def _params_slice_for_P_zero(self, params_type):
        """A slice object, selecting the part of params mapped to P_zero.

        The method has a side effect on self.param_counter and should never
        be called by the user.

        """
        nr_lower_triangular_elements = 0.5 * self.nfac * (self.nfac + 1)
        if self.restrict_P_zeros is True:
            length = nr_lower_triangular_elements
        else:
            length = nr_lower_triangular_elements * self.nemf
        return self._general_params_slice(length)

    def _P_zero_names(self, params_type):
        """List with names for the params mapped to P_zero."""
        P_zero_names = []
        format_string = 'P_zero__{}__{}__{}'
        if self.cholesky_of_P_zero is True or params_type == 'short':
            format_string = 'cholesky_' + format_string

        nr_matrices = 1 if self.restrict_P_zeros is True else self.nemf

        for mat in range(nr_matrices):
            for row, factor1 in enumerate(self.factors):
                for column, factor2 in enumerate(self.factors):
                    if row <= column:
                        P_zero_names.append(format_string.format(
                            mat, factor1, factor2))
        return P_zero_names

    def _set_bounds_for_P_zero(self, params_slice):
        """Set lower bounds for parameters mapped to diagonal of P_zero."""
        param_indices = np.arange(10000)[params_slice]
        nr_matrices = 1 if self.restrict_P_zeros is True else self.nemf
        params_per_matrix = int(0.5 * self.nfac * (self.nfac + 1))

        assert len(param_indices) == nr_matrices * params_per_matrix, (
            'You specified an invalid params_slice in _set_bounds_for_P_zero',
            'in model {}. The number of elements it selects is not compatible '
            'with the number of factors and the restrict_P_zeros setting.')
        param_indices = param_indices.reshape(nr_matrices, params_per_matrix)
        diagonal_positions = np.cumsum([0] + list(range(self.nfac, 1, -1)))
        diagonal_indices = param_indices[:, diagonal_positions].flatten()
        self.lower_bound[diagonal_indices] = \
            self.robust_bounds * self.bounds_distance

    def _initial_trans_coeffs(self):
        """List of initial trans_coeffs arrays, each filled with zeros."""
        initial_params = []
        for f, factor in enumerate(self.factors):
            func = 'nr_coeffs_{}'.format(self.transition_names[f])
            width = getattr(tf, func)(
                included_factors=self.included_factors[f], params_type='long')
            initial_params.append(np.zeros((self.nstages, int(width))))
        return initial_params

    def _params_slice_for_trans_coeffs(self, params_type):
        """A slice object, selecting the part of params mapped to trans_coeffs.

        The method has a side effect on self.param_counter and should never
        be called by the user.

        """
        slices = [[] for factor in self.factors]
        for f, s in product(range(self.nfac), self.stages):
            func = 'nr_coeffs_{}'.format(self.transition_names[f])
            if self.new_trans_coeffs[s, f] in [1, -1]:
                length = getattr(tf, func)(
                    included_factors=self.included_factors[f],
                    params_type=params_type)
                slices[f].append(self._general_params_slice(length))
            else:
                slices[f].append(slices[f][s - 1])
        return slices

    def _set_bounds_for_trans_coeffs(self, params_slice):
        """Set lower and upper bounds for trans_coeffs.

        Check if src.model_code.transition_functions defines bounds functions
        for some types of transition function and call them.

        """
        new_info = self.new_trans_coeffs
        for f, s in product(range(self.nfac), self.stages):
            sl = params_slice[f][s]
            func = 'bounds_{}'.format(self.transition_names[f])
            if hasattr(tf, func) and new_info[s, f] in [-1, 1]:
                self.lower_bound[sl], self.upper_bound[sl] = \
                    getattr(tf, func)(
                        included_factors=self.included_factors[f])

    def _trans_coeffs_names(self, params_type):
        """List with names for the params mapped to trans_coeffs."""
        param_names = []
        for (f, factor), s in product(enumerate(self.factors), self.stages):
            name_func = 'coeff_names_{}'.format(self.transition_names[f])
            len_func = 'nr_coeffs_{}'.format(self.transition_names[f])
            args = {}
            args['included_factors'] = self.included_factors[f]
            args['params_type'] = params_type
            if self.new_trans_coeffs[s, f] in [-1, 1]:
                if hasattr(tf, name_func):
                    args['factor'] = factor
                    args['stage'] = s
                    param_names += getattr(tf, name_func)(**args)
                else:
                    fs = 'trans_coeff__{}__{}__{}'
                    length = getattr(tf, len_func)(**args)
                    param_names += [
                        fs.format(s, factor, i) for i in range(length)]
        return param_names

    def _transform_trans_coeffs_funcs(self):
        """List of length nfac.

        Holds the name of the function used to expand the parameters of the
        transition function for each factor or None if no such function exists.

        """
        funcs = []
        for f, factor in enumerate(self.factors):
            func = 'transform_coeffs_{}'.format(self.transition_names[f])
            if hasattr(tf, func):
                funcs.append(func)
            else:
                funcs.append(None)
        return funcs

    def params_slices(self, params_type):
        """A dictionary with slice objects for each params_quant.

        args:
            params_type (str): Takes the values 'short' and 'long'. See
                :ref:`params_type`.

        Returns:
            dict: The keys are the params_quants. The values are slice objects
            that map params to the corresponding quantity.

        """
        self.param_counter = 0
        slices = {}
        for quantity in self.params_quants:
            func = '_params_slice_for_{}'.format(quantity)
            slices[quantity] = getattr(self, func)(params_type)
        # safety measure

        del self.param_counter
        return slices

    def len_params(self, params_type):
        """Return the length of params, dependig on the :ref:`params_type`."""
        slices = self.params_slices(params_type)
        last_slice = slices[self.params_quants[-1]]
        while type(last_slice) == list:
            last_slice = last_slice[-1]
        len_params = last_slice.stop
        return len_params

    def bounds_list(self):
        """List with bounds tuples to pass to the optimizer."""
        self.lower_bound = np.empty(self.len_params(params_type='short'),
                                    dtype=object)
        self.lower_bound[:] = None
        self.upper_bound = self.lower_bound.copy()
        slices = self.params_slices(params_type='short')
        for quantity in self.params_quants:
            params_slice = slices[quantity]
            func = '_set_bounds_for_{}'.format(quantity)
            if hasattr(self, func):
                getattr(self, func)(params_slice=params_slice)
        bounds_list = list(zip(self.lower_bound, self.upper_bound))

        assert len(bounds_list) == self.len_params(params_type='short'), (
            'The bounds list has to have the same length as the short params')
        return bounds_list

    def param_names(self, params_type):
        """Parameter names, depending on the :ref:`params_type`."""
        param_names = []
        for quantity in self.params_quants:
            func = '_{}_names'.format(quantity, params_type)
            param_names += getattr(self, func)(params_type)
        assert len(param_names) == self.len_params(params_type=params_type)
        return param_names

    def _flatten_slice_list(self, slice_list):
        first_slice = slice_list
        while type(first_slice) == list:
            first_slice = first_slice[0]

        last_slice = slice_list
        while type(last_slice) == list:
            last_slice = last_slice[-1]

        return slice(first_slice.start, last_slice.stop)

    # will replace method with same name in GenericLikelihoodModel
    def expandparams(self, params):
        short_slices = self.params_slices(params_type='short')
        short_len = self.len_params(params_type='short')
        long_slices = self.params_slices(params_type='long')
        long_len = self.len_params(params_type='long')

        for quant in ['deltas', 'trans_coeffs']:
            long_slices[quant] = self._flatten_slice_list(
                long_slices[quant])

        short_slices['deltas'] = self._flatten_slice_list(
            short_slices['deltas'])

        to_transform = ['X_zero', 'P_zero', 'trans_coeffs']
        to_transform = [quant for quant in to_transform
                        if quant in self.params_quants]

        args = {quant: {} for quant in to_transform}

        args['P_zero']['params_for_P_zero'] = \
            params[short_slices['P_zero']]
        args['P_zero']['filler'] = self._P_zero_filler()
        args['P_zero']['boo'] = self._P_zero_bool()
        args['P_zero']['estimate_cholesky_of_P_zero'] = self.cholesky_of_P_zero

        if 'X_zero' in self.params_quants:
            args['X_zero']['params_for_X_zero'] = \
                params[short_slices['X_zero']]
            args['X_zero']['filler'] = self._X_zero_filler()
            args['X_zero']['replacements'] = self._X_zero_replacements()

        args['trans_coeffs']['params'] = params
        args['trans_coeffs']['initial'] = self._initial_trans_coeffs()
        args['trans_coeffs']['params_slice'] = \
            short_slices['trans_coeffs']

        args['trans_coeffs']['transform_funcs'] = \
            self._transform_trans_coeffs_funcs()
        args['trans_coeffs']['included_factors'] = self.included_factors

        assert len(params) == short_len, (
            'You use a params vector with invalid length in the expandparams '
            'function in model {}'.format(self.model_name))

        long_params = np.zeros(long_len)

        for quant in self.params_quants:
            if quant not in to_transform:
                long_params[long_slices[quant]] = params[short_slices[quant]]
            else:
                func = 'transform_params_for_{}'.format(quant)
                long_params[long_slices[quant]] = getattr(pp, func)(
                    **args[quant])

        return long_params

    def reduceparams(self, params):
        raise NotImplementedError(
            'A reduceparams method is not implemented in CHSModel')

    def generate_start_params(self):
        """Vector with start values for the optimization."""
        if hasattr(self, 'start_params'):
            assert len(self.start_params) == self.len_params('short'), (
                'In model {} with dataset {} your start_params have not the '
                'correct length. Your start params have length {}, the '
                'correct length is {}').format(
                self.model_name, self.dataset_name, len(self.start_params),
                self.len_params('short'))
            return self.start_params
        else:
            slices = self.params_slices(params_type='short')
            start = np.zeros(self.len_params(params_type='short'))
            vals = self.start_values_per_quantity
            for quant in ['deltas', 'H', 'R', 'Q', 'psi', 'tau', 'X_zero']:
                if quant in self.params_quants and quant in vals:
                    if type(slices[quant]) != list:
                        start[slices[quant]] = vals[quant]
                    else:
                        for sl in slices[quant]:
                            start[sl] = vals[quant]

            nr_matrices = 1 if self.restrict_P_zeros is True else self.nemf
            params_per_matrix = int(0.5 * self.nfac * (self.nfac + 1))
            p = start[slices['P_zero']].reshape(nr_matrices, params_per_matrix)

            p[:] = vals['P_zero_off_diags']
            diagonal_positions = np.cumsum([0] + list(range(self.nfac, 1, -1)))
            p[:, diagonal_positions] = vals['P_zero_diags']
            start[slices['P_zero']] = p.flatten()

            for (f, fac), s in product(enumerate(self.factors), self.stages):
                func = 'start_values_{}'.format(self.transition_names[f])
                sl = slices['trans_coeffs'][f][s]
                if hasattr(tf, func):
                    start[sl] = getattr(tf, func)(
                        factor=fac, included_factors=self.included_factors[f])
                else:
                    start[sl] = vals['trans_coeffs']

            if 'W_zero' in self.params_quants:
                if 'W_zero' in vals:
                    start[slices['W_zero']] = vals['W_zero']
                else:
                    start[slices['W_zero']] = np.ones(self.nemf) / self.nemf

            return start

    def sigma_weights(self):
        nsigma = 2 * self.nfac + 1
        if self.sigma_method == 'julier':
            s_weights_m = np.ones(nsigma) / (2 * (self.nfac + self.kappa))
            s_weights_m[0] = self.kappa / (self.nfac + self.kappa)
            s_weights_c = s_weights_m
        elif self.sigma_method == 'van_merwe':
            lambda_ = self.alpha ** 2 * (self.nfac + self.kappa) - self.nfac
            s_weights_m = np.ones(nsigma) / (2 * (self.nfac + lambda_))
            s_weights_m[0] = lambda_ / (self.nfac + lambda_)
            s_weights_c = np.copy(s_weights_m)
            s_weights_c[0] += (1 - self.alpha ** 2 + self.beta)
        return s_weights_m, s_weights_c

    def sigma_scaling_factor(self):
        scaling_factor = np.sqrt(self.kappa + self.nfac)
        if self.sigma_method == 'van_merwe':
            scaling_factor *= self.alpha
        return scaling_factor

    def _initial_quantities_dict(self):
        init_dict = {}
        needed_quantities = self.params_quants.copy()

        if 'X_zero' not in needed_quantities:
            needed_quantities.append('X_zero')
        if 'W_zero' not in needed_quantities:
            needed_quantities.append('W_zero')

        for quant in needed_quantities:
            if quant not in ['X_zero', 'P_zero']:
                init_dict[quant] = getattr(self, '_initial_{}'.format(quant))()
            else:
                normal, flat = getattr(self, '_initial_{}'.format(quant))()
                init_dict[quant] = normal
                init_dict['flat_{}'.format(quant)] = flat

        sp = np.zeros((self.nemf * self.nobs, self.nsigma, self.nfac))
        init_dict['sigma_points'] = sp
        init_dict['flat_sigma_points'] = sp.reshape(
            self.nemf * self.nobs * self.nsigma, self.nfac)

        return init_dict

    def _parse_params_args_dict(self, initial_quantities, params_type):
        pp = {}
        slices = self.params_slices(params_type=params_type)

        # when adding initial quantities it's very important not to make copies
        for quant in self.params_quants:
            entry = '{}_args'.format(quant)
            pp[entry] = {'initial': initial_quantities[quant]}
            pp[entry]['params_slice'] = slices[quant]
            if hasattr(self, '_{}_bool'.format(quant)):
                pp[entry]['boo'] = getattr(self, '_{}_bool'.format(quant))()
            if hasattr(self, '_{}_filler'.format(quant)):
                pp[entry]['filler'] = getattr(
                    self, '_{}_filler'.format(quant))()
            if hasattr(self, '_{}_replacements'.format(quant)):
                rep = getattr(self, '_{}_replacements'.format(quant))()
                if len(rep) > 0:
                    pp[entry]['replacements'] = rep

        if self.endog_correction is True:
            helpers = self._helpers_for_H_transformation_with_psi()
            pp['H_args']['psi'] = initial_quantities['psi']
            pp['H_args']['psi_bool_for_H'] = helpers[0]
            pp['H_args']['arr1'] = helpers[1]
            pp['H_args']['arr2'] = helpers[2]
            pp['H_args']['endog_position'] = self.endog_position
            pp['H_args']['initial_copy'] = initial_quantities['H'].copy()

        pp['P_zero_args']['cholesky_of_P_zero'] = self.cholesky_of_P_zero
        pp['P_zero_args']['square_root_filters'] = self.square_root_filters
        pp['P_zero_args']['params_type'] = params_type

        pp['R_args']['square_root_filters'] = self.square_root_filters

        if params_type == 'short':
            pp['trans_coeffs_args']['transform_funcs'] = \
                self._transform_trans_coeffs_funcs()
            pp['trans_coeffs_args']['included_factors'] = self.included_factors
        return pp

    def _restore_unestimated_quantities_args_dict(self, initial_quantities):
        r_args = {}
        if 'X_zero' not in self.params_quants:
            r_args['X_zero'] = initial_quantities['X_zero']
            # this could be vectors
            r_args['X_zero_value'] = 0.0
        if 'W_zero' not in self.params_quants:
            r_args['W_zero'] = initial_quantities['W_zero']
            r_args['W_zero_value'] = 1 / self.nemf
        return r_args

    def _update_args_dict(self, initial_quantities, like_vec):
        position_helper = self.update_info[self.factors].values.astype(bool)

        u_args_list = []
        k = 0
        for t in self.periods:
            nmeas = self.nmeas_list[t]
            if t == self.periods[-1] and self.anchoring is True:
                nmeas += 1
            for j in range(nmeas):
                u_args = [
                    initial_quantities['X_zero'],
                    initial_quantities['P_zero'],
                    like_vec,
                    self.y_data[k],
                    self.c_data[t],
                    initial_quantities['deltas'][t][j],
                    initial_quantities['H'][k],
                    initial_quantities['R'][k: k + 1],
                    np.arange(self.nfac)[position_helper[k]],
                    initial_quantities['W_zero']]
                if self.square_root_filters is False:
                    u_args.append(np.zeros((self.nobs, self.nfac)))
                u_args_list.append(u_args)
                k += 1
        return u_args_list

    def _transition_equation_args_dicts(self, initial_quantities):
        dict_list = [[{} for f in self.factors] for s in self.stages]

        for s, f in product(self.stages, range(self.nfac)):
            dict_list[s][f]['coeffs'] = \
                initial_quantities['trans_coeffs'][f][s, :]
            dict_list[s][f]['included_positions'] = self.included_positions[f]
        return dict_list

    def _transform_sigma_points_args_dict(self, initial_quantities):
        tsp_args = {}
        tsp_args['transition_function_names'] = self.transition_names
        # TODO: change this to if anchoring is really needed
        if self.anchoring is True:
            tsp_args['anchoring_type'] = self.anchoring_update_type
            tsp_args['anchoring_positions'] = self.anch_positions
            tsp_args['anch_params'] = initial_quantities['H'][-1, :]
            if self.ignore_intercept_in_linear_anchoring is False:
                tsp_args['intercept'] = \
                    initial_quantities['deltas'][-1][-1, 0:1]

        if self.endog_correction is True:
            tsp_args['psi'] = initial_quantities['psi']
            tsp_args['tau'] = initial_quantities['tau']
            tsp_args['endog_position'] = self.endog_position
            tsp_args['correction_func'] = self.endog_function
        tsp_args['transition_argument_dicts'] = \
            self._transition_equation_args_dicts(initial_quantities)
        return tsp_args

    def _predict_args_dict(self, initial_quantities):
        p_args = {}
        p_args['sigma_points'] = initial_quantities['sigma_points']
        p_args['flat_sigma_points'] = initial_quantities['flat_sigma_points']
        p_args['s_weights_m'], p_args['s_weights_c'] = self.sigma_weights()
        p_args['Q'] = initial_quantities['Q']
        p_args['transform_sigma_points_args'] = \
            self._transform_sigma_points_args_dict(initial_quantities)
        p_args['out_flat_states'] = initial_quantities['flat_X_zero']
        p_args['out_flat_covs'] = initial_quantities['flat_P_zero']
        return p_args

    def _calculate_sigma_points_args_dict(self, initial_quantities):
        sp_args = {}
        sp_args['states'] = initial_quantities['X_zero']
        sp_args['flat_covs'] = initial_quantities['flat_P_zero']
        sp_args['out'] = initial_quantities['sigma_points']
        sp_args['square_root_filters'] = self.square_root_filters
        sp_args['scaling_factor'] = self.sigma_scaling_factor()
        return sp_args

    def likelihood_arguments_dict(self, params_type):
        """Construct a dict with arguments for the likelihood function."""
        initial_quantities = self._initial_quantities_dict()

        args = {}
        args['like_vec'] = np.ones(self.nobs)
        args['parse_params_args'] = self._parse_params_args_dict(
            initial_quantities, params_type=params_type)
        args['stagemap'] = self.stagemap
        args['nmeas_list'] = self.nmeas_list
        args['anchoring'] = self.anchoring
        args['square_root_filters'] = self.square_root_filters
        args['update_types'] = list(self.update_info['update_type'])
        args['update_args'] = self._update_args_dict(
            initial_quantities, args['like_vec'])
        args['predict_args'] = self._predict_args_dict(initial_quantities)
        args['calculate_sigma_points_args'] = \
            self._calculate_sigma_points_args_dict(initial_quantities)
        args['restore_args'] = self._restore_unestimated_quantities_args_dict(
            initial_quantities)
        return args

    def nloglikeobs(self, params, args):
        return - log_likelihood_per_individual(params, **args)

    def nloglike(self, params, args):
        return - log_likelihood_per_individual(params, **args).sum()

    def loglikeobs(self, params, args):
        return log_likelihood_per_individual(params, **args)

    def loglike(self, params, args):
        return log_likelihood_per_individual(params, **args).sum()

    def fit(self, start_params=None, maxiter=1000000, maxfun=1000000,
            print_result=True, standard_error_method='op_of_gradient'):

        if start_params is None:
            start_params = self.generate_start_params()
        bounds = self.bounds_list()
        args = self.likelihood_arguments_dict(params_type='short')

        res = minimize(self.nloglike, start_params, args=(args),
                       method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': maxiter, 'maxfun': maxfun})

        optimize_dict = {}
        optimize_dict['success'] = res.success
        optimize_dict['nfev'] = res.nfev
        optimize_dict['log_lh_value'] = -res.fun
        optimize_dict['xopt'] = res.x.tolist()

        params = self.expandparams(res.x)
        args = self.likelihood_arguments_dict(params_type='long')

        if standard_error_method == 'hessian_inverse':
            hessian = self.hessian(params)
            try:
                cov = np.linalg.inv(-hessian)
            except:
                warning = 'The hessian could not be inverted.'
                optimize_dict['hessian_warning'] = warning
                standard_error_method = 'op_of_gradient'
                print('Model {} with dataset {}:'.format(
                    self.model_name, self.dataset_name))
                print(warning)
                print(hessian)

        if standard_error_method == 'op_of_gradient':
            gradient = self.score_obs(params, args)
            try:
                # what follows is equivalent to:
                # cov = np.linalg.inv(np.dot(gradient.T, gradient))
                # but it ensures that the resulting covariance matrix is
                # positive semi-definite
                u = np.linalg.qr(gradient)[1]
                u_inv = np.linalg.inv(u)
                cov = np.dot(u_inv, u_inv.T)
            except:
                warning = (
                    'The outer product of gradients could not be inverted. '
                    'No standard errors could be calculated.')

                optimize_dict['gradient_warning'] = warning
                print('Model {} with dataset {}:'.format(
                    self.model_name, self.dataset_name))
                print(warning)
                print(gradient)
                cov = np.diag(params * np.nan)

            mlefit = LikelihoodModelResults(self, params, cov)

        chsmlefit = SkillModelResults(self, mlefit, optimize_dict)
        return chsmlefit

    def meas_list_for_iv_equations(self, period, factor, suffix=''):
        """List of lists with names of measurements of included factors.

        Args:
            period (int): the period for which the list is generated
            factor (str): the factor for which the list is generated
            suffix (str): a suffix that is appended to all measurement names in
                the list. It is separated from the actual name by '_'.

        Returns:
            meas_list (list): List of lists with one sublist for each factor
                that appears on the right hand side of the transition equation
                of *factor*. Each sublist contains the names of all
                measurements in *period* of the corresponding right-hand-side
                factor.

        """
        suffix = '_' + suffix if suffix != '' else suffix
        f = self.factors.index(factor)
        inc_facs = self.included_factors[f]
        meas_list = []
        for inc in inc_facs:
            trans_name = self.transition_names[self.factors.index(inc)]
            if trans_name == 'constant' and period > 0:
                form_string = '{}_copied' + suffix
                sublist = [
                    form_string.format(m) for m in self.measurements[inc][0]]
            else:
                form_string = '{}' + suffix
                sublist = [form_string.format(m)
                           for m in self.measurements[inc][period]]
            meas_list.append(sublist)
        return meas_list

    def iv_equation_variable_lists(self, period, factor):
        """Nested lists with combination of measurement names for iv equations.

        In the WA estimator, the transition equations are rewritten in an
        errors-in-variables specification in which latent factors are
        approximated by residual measurements and then instrumented by other
        measurements.

        Args:
            period (int): the period for which the lists are generated
            factor (str): the factor for which the lists are generated

        Returns:
            x_list (list): contains one sublist for each iv equation that has
                to be estimated for *factor* in *period*. Each sublist contains
                the names of the measurements that are used to form the
                residual measurements for that iv equation. The length of each
                sublist is the number of factors that are included in the right
                hand side of the transition equation of *factor*.
            z_list (list): has the same length as x_list and contains the
                instruments for each transition equation. The instruments are
                lists of lists with one sublist for each included factor.

        """
        meas_lists_for_x = self.meas_list_for_iv_equations(
            period, factor, suffix='resid')
        x_list = list(map(list, product(*meas_lists_for_x)))

        z_list = []
        for x in x_list:
            z = []
            for sublist in meas_lists_for_x:
                z.append([m[:-6] for m in sublist if m not in x])
            z_list.append(z)

        return x_list, z_list

    def number_of_iv_parameters(self, factor):
        """Number of parameters in the IV equation of a parameter."""
        f = self.factors.index(factor)
        trans_name = self.transition_names[f]
        x_list, z_list = self.iv_equation_variable_lists(0, factor)
        example_x, example_z = x_list[0], z_list[0]
        x_formula, _ = getattr(tf, 'iv_formula_{}'.format(trans_name))(
            example_x, example_z)
        return x_formula.count('+') + 1

    def extended_meas_coeffs(self, coeff_type, period):
        """Series of coefficients for construction of residual measurements.

        Args:
            coeff_type (str): takes values 'loadings' and 'intercepts'
            period (int): period identifier

        Returns:
            coeffs (Series): Series of measurement coefficients in period
                extendend with period zero coefficients of constant factors.

        """
        coeffs = self.storage_df.loc[period, coeff_type]
        if period > 0 and 'constant' in self.transition_names:
            initial_meas = []
            for f, factor in enumerate(self.factors):
                if self.transition_names[f] == 'constant':
                    initial_meas += self.measurements[factor][0]
            constant_factor_coeffs =  \
                self.storage_df.loc[0, coeff_type].loc[initial_meas]
            constant_factor_coeffs.index = [
                '{}_copied'.format(m) for m in constant_factor_coeffs.index]
            coeffs = coeffs.append(constant_factor_coeffs)
        return coeffs

    def iv_data(self, period):
        """Construct DataFrames with data for all IV equations in period.

        .. Note:: These DataFrames contain everything that will be needed for
            IV equations but do not yet select the data for one particular
            equation.

        Args:
            period (int): period identifier

        Returns:
            y_data (DataFrame): measurements from the next period (period + 1)
            x_data (DataFrame): residual measurements from period, extended
                with residual measurements of constant factors from initial
                period.
            z_data (DataFrame): measurements from period, extended  with
                measurements of constant factors from initial period.
        """
        y_data = self.y_data[period + 1]

        loadings = self.extended_meas_coeffs('loadings', period)
        intercepts = self.extended_meas_coeffs('intercepts', period)

        x_data = residual_measurements(
            self.y_data[period], loadings=loadings, intercepts=intercepts)
        x_data['constant'] = 1

        z_data = self.y_data[period].copy()
        z_data['constant'] = 1

        return y_data, x_data, z_data

    def get_iv_coeffs(self, period, factor, y_data, x_data, z_data):
        """Estimate parameters of all IV equations of factor in period.

        Args:
            period (int): period identifier
            factor (str): name of a latent factor
            y_data, x_data, z_data (DataFrames): see iv_data

        Returns:
            iv_coeffs (DataFrame): pandas DataFrame with the estimated
                coefficients of all IV equations of factor in period. The
                DataFrame has a Multiindex. The first level are the names of
                the dependent variables (next period measurements) of the
                estimated equations. The second level consists of inegers that
                identify the different combinations of independent variables.

        """
        x_list, z_list = self.iv_equation_variable_lists(period, factor)
        f = self.factors.index(factor)
        trans_name = self.transition_names[f]
        depvars = self.measurements[factor][period + 1]
        nr_deltas = self.number_of_iv_parameters(factor)
        columns = ['delta_{}'.format(i) for i in range(nr_deltas)]

        ind_tuples = [(y, i) for y, i in product(depvars, range(len(x_list)))]
        index = pd.MultiIndex.from_tuples(ind_tuples)

        iv_coeffs = pd.DataFrame(data=0.0, index=index, columns=columns)

        for y in depvars:
            y_vector = y_data[y].values
            # all deltas on this level have the same mü and lambda
            # mü and/or lambda might be normalized
            for x, z in zip(x_list, z_list):
                iv_formula_func = getattr(tf, 'iv_formula_' + trans_name)
                x_formula, z_formula = iv_formula_func(x, z)
                x_matrix = dmatrix(x_formula, data=x_data,
                                   return_type='dataframe').values
                z_matrix = dmatrix(z_formula, data=z_data,
                                   return_type='dataframe').values
                deltas = iv_reg(y_vector, x_matrix, z_matrix)
                iv_coeffs.loc[(y, x_list.index(x))] = deltas

        return iv_coeffs

    def fit_wa(self):
        t = 0
        delta_df_list = []
        meas_coeffs, X_zero = initial_meas_coeffs(
            y_data=self.y_data[t],
            measurements=self.measurements, normalizations=self.normalizations)

        # if self.model_name == 'no_squares_translog_model':
        #     self.storage_df.update(prepend_index_level(meas_coeffs.round(decimals=3), t))
        # else:
        self.storage_df.update(prepend_index_level(meas_coeffs, t))

        P_zero = initial_cov_matrix(
            data=self.y_data[t], storage_df=self.storage_df,
            measurements=self.measurements)

        trans_coeff_storage = self._initial_trans_coeffs()
        per_period_iv_data = []

        for t in self.periods[:-1]:
            delta_df_list.append([])
            stage = self.stagemap[t]

            y_data, x_data, z_data = self.iv_data(t)
            per_period_iv_data.append({'y': y_data, 'x': x_data, 'z': z_data})

            for f, factor in enumerate(self.factors):
                trans_name = self.transition_names[f]
                if trans_name != 'constant':
                    iv_coeffs = self.get_iv_coeffs(
                        t, factor, y_data, x_data, z_data)

                    model_coeffs_from_iv_coeffs_func = getattr(
                        tf, 'model_coeffs_from_iv_coeffs_' + trans_name)
                    loading_norminfo = \
                        self.normalizations[factor]['loadings'][t + 1]
                    intercept_norminfo = \
                        self.normalizations[factor]['intercepts'][t + 1]

                    meas_coeffs, gammas = model_coeffs_from_iv_coeffs_func(
                        iv_coeffs=iv_coeffs,
                        loading_norminfo=loading_norminfo,
                        intercept_norminfo=intercept_norminfo)
                    meas_coeffs = prepend_index_level(meas_coeffs, t + 1)
                    # if self.model_name == 'no_squares_translog_model':
                    #     self.storage_df.update(meas_coeffs.round(decimals=3)) # ============================
                    # else:
                    self.storage_df.update(meas_coeffs)
                    trans_coeff_storage[f][stage] = gammas
                delta_df_list[t].append(iv_coeffs)

        return self.storage_df, X_zero, P_zero, trans_coeff_storage

    def score(self, params, args):
        return approx_fprime(
            params, self.loglike, args=(args, ), centered=True).ravel()

    def score_obs(self, params, args):
        return approx_fprime(params, self.loglikeobs, args=(args, ),
                             centered=True)

    def hessian(self, params, args):
        return approx_hess(params, self.loglike, args=(args, ))
