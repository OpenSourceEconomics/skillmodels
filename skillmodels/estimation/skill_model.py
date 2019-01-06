from skillmodels.pre_processing.model_spec_processor import ModelSpecProcessor
from skillmodels.pre_processing.data_processor import DataProcessor
from skillmodels.estimation.likelihood_function import log_likelihood_per_individual
from skillmodels.estimation.wa_functions import (
    initial_meas_coeffs,
    prepend_index_level,
    factor_covs_and_measurement_error_variances,
    iv_reg_array_dict,
    iv_reg,
    large_df_for_iv_equations,
    transition_error_variance_from_u_covs,
    anchoring_error_variance_from_u_vars,
    variable_permutations_for_iv_equations)
from skillmodels.visualization.table_functions import (
    statsmodels_results_to_df,
    df_to_tex_table,
)
from skillmodels.visualization.text_functions import (
    title_text,
    write_figure_tex_snippet,
    get_preamble,
)
from statsmodels.base.model import GenericLikelihoodModel
from statsmodels.base.model import LikelihoodModelResults
import statsmodels.formula.api as smf

from skillmodels.estimation.skill_model_results import (
    SkillModelResults,
    NotApplicableError,
)
from skillmodels.fast_routines.transform_sigma_points import transform_sigma_points
import numpy as np
import skillmodels.model_functions.transition_functions as tf
from skillmodels.estimation.parse_params import parse_params
import skillmodels.estimation.parse_params as pp
import skillmodels.model_functions.anchoring_functions as anch

from itertools import product
from scipy.optimize import minimize
from statsmodels.tools.numdiff import approx_hess, approx_fprime
import pandas as pd
import json
from multiprocessing import Pool
import warnings

import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join


class SkillModel(GenericLikelihoodModel):
    """Estimate dynamic nonlinear latent factor models.

    SkillModel is a subclass of GenericLikelihoodModel from statsmodels and
    inherits many useful methods such as statistical tests and the calculation
    of standard errors from its parent class. Its usage is described in
    :ref:`basic_usage`.

    When initialized, all public attributes of ModelSpecProcessor and the
    arrays with c_data and y_data from DataProcessor are set as attributes.

    In addition to the methods inherited from GenericLikelihoodModel,
    SkillModel contains methods to:

    * determine how the params vector has to be parsed
    * calculate wa estimates
    * construct arguments for the likelihood function of the chs estimator
    * calculate covariance matrices of the estimated parameters

    Args:
        model_dict (dict): see :ref:`basic_usage`.
        dataset (DataFrame): datset in long format. see :ref:`basic_usage`.
        estimator (str): takes the values 'wa' and 'chs'
        model_name (str): optional. Used to make error messages readable.
        dataset_name (str): same as model_name
        save_path (str): specifies where intermediate results are saved.
        bootstrap_samples (list): optional, see docs of bootstrap functions.

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
        self.estimator = estimator
        specs = ModelSpecProcessor(
            model_dict=model_dict,
            dataset=dataset,
            estimator=estimator,
            model_name=model_name,
            dataset_name=dataset_name,
            save_path=save_path,
            bootstrap_samples=bootstrap_samples,
        )
        specs_dict = specs.public_attribute_dict()
        data_proc = DataProcessor(specs_dict)
        self.data_proc = data_proc
        self.c_data = data_proc.c_data() if self.estimator == "chs" else None
        self.y_data = data_proc.y_data()
        self.__dict__.update(specs_dict)

        if bootstrap_samples is not None:
            self.bootstrap_samples = bootstrap_samples
            self._check_bs_samples()
        else:
            self.bootstrap_samples = self._generate_bs_samples()

        # create a list of all quantities that depend from params vector
        self.params_quants = ["deltas", "H", "R", "Q", "P_zero", "trans_coeffs"]
        if self.estimate_X_zeros is True:
            self.params_quants.append("X_zero")
        # Todo: maybe this does not make sense for the wa estimator
        if self.endog_correction is True:
            self.params_quants.append("psi")
        if self.restrict_W_zeros is False and self.estimator == "chs":
            self.params_quants.append("W_zero")

        self.df_model = self.len_params(params_type="short")
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
        assert hasattr(self, "param_counter"), (
            "Users must not call any of the private _params_slice methods "
            "but only the public params_slices() method that returns a "
            "dictionary with params_slices for each params_quant."
        )
        res = slice(int(self.param_counter), int(self.param_counter + length))
        self.param_counter += length
        return res

    def _initial_deltas(self):
        """List of initial arrays for control variable params in each period.

        The arrays have the shape [nupdates, nr_of_control_variables + 1]
        which is potentially different in each period. They are filled with
        zeros or the value of normalized intercepts.

        """
        deltas = []
        for t in self.periods:
            length = len(self.update_info.loc[t])
            width = len(self.controls[t]) + 1
            init = np.zeros((length, width))
            init[:, 0] = self.update_info.loc[t]["intercept_norm_value"].fillna(0)
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
            relevant = ["intercept"] + self.controls[t]
            boo = self.new_meas_coeffs.loc[t, relevant].astype(bool).values
            deltas_bool.append(boo)
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
                all_vars = ["constant"] + self.controls[t]
                free_vars = [
                    var
                    for v, var in enumerate(all_vars)
                    if deltas_bool[t][u, v] == True
                ]
                for control in free_vars:
                    deltas_names.append("delta__{}__{}__{}".format(t, update, control))
        return deltas_names

    def _deltas_replacements(self):
        """List of pairs of index tuples.

        The first tuple indicates where to put the row of deltas described
        by the second index tuple.

        """
        uinfo = self.update_info
        replacements = []
        for t in self.periods:
            for m, meas in enumerate(uinfo.loc[t].index):
                if uinfo.loc[(t, meas), "is_repeated"] == True:
                    first_occ_period = int(uinfo.loc[t, meas]["first_occurence"])
                    first_occ_position = int(
                        uinfo.loc[first_occ_period].index.get_loc(meas)
                    )
                    replacements.append(
                        [(t, m), (first_occ_period, first_occ_position)]
                    )
        if self.time_invariant_measurement_system is True:
            return replacements
        else:
            return []

    def _initial_psi(self):
        """Initial psi vector filled with ones."""
        # TODO: Check if psi is needed for endog correction in wa case
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
                psi_names.append("psi__{}".format(factor))
        return psi_names

    def _initial_H(self):
        """Initial H array filled with zeros and normalized factor loadings.

        The array has the form [nupdates, nfac]. Most entries
        are zero, but if the factor loading of factor f in update equation
        u is normalized to some value then arr[u, f] is equal to this value.

        """
        column_list = ["{}_loading_norm_value".format(f) for f in self.factors]
        df = self.update_info[column_list]
        return df.values

    def _H_bool(self):
        """Boolean array.

        It has the same shape as initial_H and is True where initial_H
        has to be overwritten entries from params.

        """
        return self.new_meas_coeffs[self.factors].values

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
        # TODO: remove this. It didn't bring the time improvement I hoped for
        assert self.endog_correction is True, (
            "The psi_bool_for_H method should only be called if "
            "endog_correction is True. You did otherwise in model {}"
        ).format(self.model_name)

        psi_bool = self.update_info[self.endog_factor].values.flatten().astype(bool)
        arr1 = np.zeros((psi_bool.sum(), 1))
        return psi_bool, arr1

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
                    H_names.append("H__{}__{}__{}".format(period, factor, measure))
        return H_names

    def _H_replacements(self):
        """List of tuples with index positions.

        The first entry in each tuple indicates where to put the row of H
        described by the second entry.

        """
        uinfo = self.update_info
        replacements = []
        for put_position, (t, meas) in enumerate(uinfo.index):
            if uinfo.loc[(t, meas), "is_repeated"] == True:
                first_occurence = int(uinfo.loc[(t, meas), "first_occurence"])
                take_position = uinfo.index.get_loc((first_occurence, meas))
                replacements.append((put_position, take_position))

        if self.time_invariant_measurement_system is True:
            return replacements
        else:
            return []

    def _initial_R(self):
        """1d numpy array of length nupdates filled with zeros."""
        return self.update_info["variance_norm_value"].fillna(0).values

    def _R_bool(self):
        return self.new_meas_coeffs["variance"].values

    def _params_slice_for_R(self, params_type):
        """A slice object, selecting the part of params mapped to R.

        The method has a side effect on self.param_counter and should never
        be called by the user.

        """
        return self._general_params_slice(self._R_bool().sum())

    def _R_names(self, params_type):
        """List with names for the params mapped to R."""
        boo = self._R_bool()
        R_names = []
        for b, (t, measure) in zip(boo, self.update_info.index):
            # == instead of is because of numpy bool problem
            if b == True:
                R_names.append("R__{}__{}".format(t, measure))
        return R_names

    def _R_replacements(self):
        """List of tuples with index positions.

        The first entry in each tuple indicates where to put the element of R
        described by the second entry.

        """
        return self._H_replacements()

    def _set_bounds_for_R(self, params_slice):
        """Set lower bounds for params mapped to R."""
        self.lower_bound[params_slice] = self.robust_bounds * self.bounds_distance

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
                Q_names.append("Q__{}__{}".format(s, factor))
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
        self.lower_bound[params_slice] = self.robust_bounds * self.bounds_distance

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
                    [(i, self.order_X_zeros), (i - 1, self.order_X_zeros)]
                )
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
            if params_type == "long" or n == 0 or f != self.order_X_zeros:
                format_string = "X_zero__{}__{}"
            else:
                format_string = "diff_X_zero__{}__{}"
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
        return ["W_zero__{}".format(n) for n in range(self.nemf)]

    def _initial_P_zero(self):
        """Initial P_zero array filled with zeros."""
        if self.square_root_filters is False:
            init = np.zeros((self.nobs, self.nemf, self.nfac, self.nfac))
            flat_init = init.reshape(self.nobs * self.nemf, self.nfac, self.nfac)
        else:
            init = np.zeros((self.nobs, self.nemf, self.nfac + 1, self.nfac + 1))
            flat_init = init.reshape(
                self.nobs * self.nemf, self.nfac + 1, self.nfac + 1
            )
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
        nr_upper_triangular_elements = 0.5 * self.nfac * (self.nfac + 1)
        if self.restrict_P_zeros is True or self.estimator == "wa":
            length = nr_upper_triangular_elements
        else:
            length = nr_upper_triangular_elements * self.nemf
        return self._general_params_slice(length)

    def _P_zero_names(self, params_type):
        """List with names for the params mapped to P_zero."""
        P_zero_names = []
        format_string = "P_zero__{}__{}__{}"
        if self.estimator == "chs":
            if self.cholesky_of_P_zero is True or params_type == "short":
                format_string = "cholesky_" + format_string

        if self.estimator == "chs" and self.restrict_P_zeros is True:
            nr_matrices = 1
        else:
            nr_matrices = self.nemf

        for mat in range(nr_matrices):
            for row, factor1 in enumerate(self.factors):
                for column, factor2 in enumerate(self.factors):
                    if row <= column:
                        P_zero_names.append(format_string.format(mat, factor1, factor2))
        return P_zero_names

    def _set_bounds_for_P_zero(self, params_slice):
        """Set lower bounds for parameters mapped to diagonal of P_zero."""
        param_indices = np.arange(10000)[params_slice]
        nr_matrices = 1 if self.restrict_P_zeros is True else self.nemf
        params_per_matrix = int(0.5 * self.nfac * (self.nfac + 1))

        assert len(param_indices) == nr_matrices * params_per_matrix, (
            "You specified an invalid params_slice in _set_bounds_for_P_zero",
            "in model {}. The number of elements it selects is not compatible "
            "with the number of factors and the restrict_P_zeros setting.",
        )
        param_indices = param_indices.reshape(nr_matrices, params_per_matrix)
        diagonal_positions = np.cumsum([0] + list(range(self.nfac, 1, -1)))
        diagonal_indices = param_indices[:, diagonal_positions].flatten()
        self.lower_bound[diagonal_indices] = self.robust_bounds * self.bounds_distance

    def _initial_trans_coeffs(self, short_version=False):
        """List of initial trans_coeffs arrays, each filled with zeros."""
        # Note: the short version is only needed for reduceparams. it does
        # not depend on the normal params_type argument.
        partype = "short" if short_version is True else "long"
        initial_params = []
        for f, factor in enumerate(self.factors):
            func = "nr_coeffs_{}".format(self.transition_names[f])
            width = getattr(tf, func)(
                included_factors=self.included_factors[f], params_type=partype
            )
            initial_params.append(np.zeros((self.nstages, int(width))))
        return initial_params

    def _params_slice_for_trans_coeffs(self, params_type):
        """A slice object, selecting the part of params mapped to trans_coeffs.

        The method has a side effect on self.param_counter and should never
        be called by the user.

        """
        slices = [[] for factor in self.factors]
        for f, s in product(range(self.nfac), self.stages):
            func = "nr_coeffs_{}".format(self.transition_names[f])
            if self.new_trans_coeffs[s, f] in [1, -1]:
                length = getattr(tf, func)(
                    included_factors=self.included_factors[f], params_type=params_type
                )
                slices[f].append(self._general_params_slice(length))
            else:
                slices[f].append(slices[f][s - 1])
        return slices

    def _set_bounds_for_trans_coeffs(self, params_slice):
        """Set lower and upper bounds for trans_coeffs.

        Check if the transition_functions module defines bounds functions
        for some types of transition function and call them.

        """
        new_info = self.new_trans_coeffs
        for f, s in product(range(self.nfac), self.stages):
            sl = params_slice[f][s]
            func = "bounds_{}".format(self.transition_names[f])
            if hasattr(tf, func) and new_info[s, f] in [-1, 1]:
                self.lower_bound[sl], self.upper_bound[sl] = getattr(tf, func)(
                    included_factors=self.included_factors[f]
                )

    def _trans_coeffs_names(self, params_type):
        """List with names for the params mapped to trans_coeffs."""
        param_names = []
        for (f, factor), s in product(enumerate(self.factors), self.stages):
            name_func = "coeff_names_{}".format(self.transition_names[f])
            len_func = "nr_coeffs_{}".format(self.transition_names[f])
            args = {}
            args["included_factors"] = self.included_factors[f]
            args["params_type"] = params_type
            if self.new_trans_coeffs[s, f] in [-1, 1]:
                if hasattr(tf, name_func):
                    args["factor"] = factor
                    args["stage"] = s
                    param_names += getattr(tf, name_func)(**args)
                else:
                    fs = "trans_coeff__{}__{}__{}"
                    length = getattr(tf, len_func)(**args)
                    param_names += [fs.format(s, factor, i) for i in range(length)]
        return param_names

    def _transform_trans_coeffs_funcs(self):
        """List of length nfac.

        Holds the name of the function used to expand or reduce the parameters
        of the transition function for each factor or None if no such function
        exists.

        """
        funcs = []
        for f, factor in enumerate(self.factors):
            func = "transform_coeffs_{}".format(self.transition_names[f])
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
            func = "_params_slice_for_{}".format(quantity)
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
        self.lower_bound = np.empty(self.len_params(params_type="short"), dtype=object)
        self.lower_bound[:] = None
        self.upper_bound = self.lower_bound.copy()
        slices = self.params_slices(params_type="short")
        for quantity in self.params_quants:
            params_slice = slices[quantity]
            func = "_set_bounds_for_{}".format(quantity)
            if hasattr(self, func):
                getattr(self, func)(params_slice=params_slice)
        bounds_list = list(zip(self.lower_bound, self.upper_bound))

        assert len(bounds_list) == self.len_params(
            params_type="short"
        ), "The bounds list has to have the same length as the short params"
        return bounds_list

    def param_names(self, params_type):
        """Parameter names, depending on the :ref:`params_type`."""
        param_names = []
        for quantity in self.params_quants:
            func = "_{}_names".format(quantity, params_type)
            param_names += getattr(self, func)(params_type)
        assert len(param_names) == self.len_params(params_type=params_type)
        return param_names

    def _flatten_slice_list(self, slice_list):
        """Make one slice object from a list with consecutive slice objects."""
        first_slice = slice_list
        while type(first_slice) == list:
            first_slice = first_slice[0]

        last_slice = slice_list
        while type(last_slice) == list:
            last_slice = last_slice[-1]

        return slice(first_slice.start, last_slice.stop)

    def _transform_params(self, params, direction):
        have_type = "short" if direction.startswith("short") else "long"
        want_type = "short" if direction.endswith("short") else "long"

        have_slices = self.params_slices(params_type=have_type)
        have_len = self.len_params(params_type=have_type)

        want_slices = self.params_slices(params_type=want_type)
        want_len = self.len_params(params_type=want_type)

        for quant in ["deltas", "trans_coeffs"]:
            want_slices[quant] = self._flatten_slice_list(want_slices[quant])

        have_slices["deltas"] = self._flatten_slice_list(have_slices["deltas"])

        to_transform = ["X_zero", "P_zero", "trans_coeffs"]
        to_transform = [quant for quant in to_transform if quant in self.params_quants]

        args = {quant: {"direction": direction} for quant in to_transform}

        args["P_zero"]["params_for_P_zero"] = params[have_slices["P_zero"]]
        args["P_zero"]["filler"] = self._P_zero_filler()
        args["P_zero"]["boo"] = self._P_zero_bool()
        args["P_zero"]["estimate_cholesky_of_P_zero"] = self.cholesky_of_P_zero

        if "X_zero" in self.params_quants:
            args["X_zero"]["params_for_X_zero"] = params[have_slices["X_zero"]]
            args["X_zero"]["filler"] = self._X_zero_filler()
            args["X_zero"]["replacements"] = self._X_zero_replacements()

        args["trans_coeffs"]["params"] = params

        short_version = True if direction == "long_to_short" else False
        args["trans_coeffs"]["initial"] = self._initial_trans_coeffs(short_version)

        args["trans_coeffs"]["params_slice"] = have_slices["trans_coeffs"]

        args["trans_coeffs"]["transform_funcs"] = self._transform_trans_coeffs_funcs()
        args["trans_coeffs"]["included_factors"] = self.included_factors

        assert len(params) == have_len, (
            "You use a params vector with invalid length in the expandparams "
            "function in model {}".format(self.model_name)
        )

        want_params = np.zeros(want_len)

        for quant in self.params_quants:
            if quant not in to_transform:
                want_params[want_slices[quant]] = params[have_slices[quant]]
            else:
                func = "transform_params_for_{}".format(quant)
                want_params[want_slices[quant]] = getattr(pp, func)(**args[quant])

        return want_params

    def expandparams(self, params):
        """Convert a params vector of type short to type long."""
        return self._transform_params(params, direction="short_to_long")

    def reduceparams(self, params):
        """Convert a params vector of type long to type short."""
        return self._transform_params(params, direction="long_to_short")

    def _wa_params_can_be_used_for_start_params(self, raise_warning=True):
        """Check if results of WA can be used as start params for CHS."""
        reasons = []
        if self.nemf != 1:
            reasons.append("A mixture distribution is estimated.")
        if self.uses_controls is True:
            reasons.append("Control variables are used.")
        update_types = list(self.update_info["update_type"])
        if "probit" in update_types or "logit" in update_types:
            reasons.append("Probit or logit updates are used.")
        df = self.update_info.copy(deep=True)
        df = df[df["purpose"] == "measurement"]
        if not (df[self.factors].values.sum(axis=1) == 1).all():
            reasons.append("Some measurements measure more than 1 factor.")
        if self.anchoring_mode == "truly_anchor_latent_factors":
            reasons.append("The anchoring mode is not supported in wa.")
        if "log_ces" in self.transition_names:
            reasons.append("The log_ces cannot be used in wa.")
        if self.update_info["has_normalized_variance"].any():
            reasons.append("Normalized variances are incompatible with wa")

        can_be_used = False if len(reasons) > 0 else True

        warn_message = (
            "In model {} with dataset {} it is not possible to use "
            "estimates from the wa estimator as start values for the "
            "chs estimator because of the following reasons:"
        ).format(self.model_name, self.dataset_name)
        for r in reasons:
            warn_message += "\n" + r

        if raise_warning is True:
            warnings.warn(warn_message)

        return can_be_used

    def _correct_len_of_start_params(self, raise_warning=True):
        """Check if self.start_params has the correct length."""
        if len(self.start_params) == self.len_params("short"):
            return True
        else:
            warn_message = (
                "The start_params vector you provided in model {} with "
                "dataset {} will be ignored because it does not have the "
                "correct length. Your start params have length {}, the "
                "correct length is {}"
            ).format(
                self.model_name,
                self.dataset_name,
                len(self.start_params),
                self.len_params("short"),
            )
            if raise_warning is True:
                warnings.warn(warn_message)
            return False

    def _generate_naive_start_params(self):
        """Generate start_params based on self.start_values_per_quantity."""
        slices = self.params_slices(params_type="short")
        start = np.zeros(self.len_params(params_type="short"))
        vals = self.start_values_per_quantity
        for quant in ["deltas", "H", "R", "Q", "psi", "X_zero"]:
            if quant in self.params_quants and quant in vals:
                if type(slices[quant]) != list:
                    start[slices[quant]] = vals[quant]
                else:
                    for sl in slices[quant]:
                        start[sl] = vals[quant]

        nr_matrices = 1 if self.restrict_P_zeros is True else self.nemf
        params_per_matrix = int(0.5 * self.nfac * (self.nfac + 1))
        p = start[slices["P_zero"]].reshape(nr_matrices, params_per_matrix)

        p[:] = vals["P_zero_off_diags"]
        diagonal_positions = np.cumsum([0] + list(range(self.nfac, 1, -1)))
        p[:, diagonal_positions] = vals["P_zero_diags"]
        start[slices["P_zero"]] = p.flatten()

        for (f, fac), s in product(enumerate(self.factors), self.stages):
            func = "start_values_{}".format(self.transition_names[f])
            sl = slices["trans_coeffs"][f][s]
            if hasattr(tf, func):
                start[sl] = getattr(tf, func)(
                    factor=fac, included_factors=self.included_factors[f]
                )
            else:
                start[sl] = vals["trans_coeffs"]

        if "W_zero" in self.params_quants:
            if "W_zero" in vals:
                start[slices["W_zero"]] = vals["W_zero"]
            else:
                start[slices["W_zero"]] = np.ones(self.nemf) / self.nemf

            return start

    def _generate_wa_based_start_params(self):
        """Use wa estimates to construct a start_params vector for chs.

        Fit the model with the wa estimator and use reduceparams to get a
        params vector of type short. Then replace all estimated variances that
        are below 0.05 with 0.05 for robustness reasons.

        """
        long_params = self.estimate_params_wa
        short_params = self.reduceparams(long_params)

        # for robustness, replace very small estimated variances
        # this is necessary because these variances are estimated quite
        # imprecisely with the wa estimator and might even be negative which
        # would totally crash the chs estimator.
        slices = self.params_slices("short")
        for quant in ["R", "Q"]:
            sl = slices[quant]
            short_params[sl][short_params[sl] <= 0.05] = 0.05

        return short_params

    def generate_start_params(self):
        """Vector with start values for the optimization.

        If valid start_params are provided in the model dictionary, these will
        be used. Else, if the model is compatible with the wa estimator, the
        wa estimates will be used. Else, naive start_params are generated.

        """
        len_correct = self._correct_len_of_start_params()
        if hasattr(self, "start_params") and len_correct is True:
            start = self.start_params
        elif self._wa_params_can_be_used_for_start_params() is True:
            try:
                start = self._generate_wa_based_start_params()
            except:
                warn_message = (
                    "Fitting model {} with the wa estimator in order to get "
                    "start values for the chs estimator failed. Instead "
                    "naive start params will be used.".format(self.model_name)
                )
                warnings.warn(warn_message)
                start = self._generate_naive_start_params()
        else:
            start = self._generate_naive_start_params()
        return start

    def sigma_weights(self):
        """Calculate the sigma weight according to the julier algorithm."""
        nsigma = 2 * self.nfac + 1
        s_weights_m = np.ones(nsigma) / (2 * (self.nfac + self.kappa))
        s_weights_m[0] = self.kappa / (self.nfac + self.kappa)
        s_weights_c = s_weights_m
        return s_weights_m, s_weights_c

    def sigma_scaling_factor(self):
        """Calculate invariant part of sigma points according to the julier."""
        scaling_factor = np.sqrt(self.kappa + self.nfac)
        return scaling_factor

    def _initial_quantities_dict(self):
        init_dict = {}
        needed_quantities = self.params_quants.copy()

        if "X_zero" not in needed_quantities:
            needed_quantities.append("X_zero")
        if "W_zero" not in needed_quantities:
            needed_quantities.append("W_zero")

        for quant in needed_quantities:
            if quant not in ["X_zero", "P_zero"]:
                init_dict[quant] = getattr(self, "_initial_{}".format(quant))()
            else:
                normal, flat = getattr(self, "_initial_{}".format(quant))()
                init_dict[quant] = normal
                init_dict["flat_{}".format(quant)] = flat

        sp = np.zeros((self.nemf * self.nobs, self.nsigma, self.nfac))
        init_dict["sigma_points"] = sp
        init_dict["flat_sigma_points"] = sp.reshape(
            self.nemf * self.nobs * self.nsigma, self.nfac
        )

        return init_dict

    def _parse_params_args_dict(self, initial_quantities, params_type):
        pp = {}
        slices = self.params_slices(params_type=params_type)

        # when adding initial quantities it's very important not to make copies
        for quant in self.params_quants:
            entry = "{}_args".format(quant)
            pp[entry] = {"initial": initial_quantities[quant]}
            pp[entry]["params_slice"] = slices[quant]
            if hasattr(self, "_{}_bool".format(quant)):
                pp[entry]["boo"] = getattr(self, "_{}_bool".format(quant))()
            if hasattr(self, "_{}_filler".format(quant)):
                pp[entry]["filler"] = getattr(self, "_{}_filler".format(quant))()
            if hasattr(self, "_{}_replacements".format(quant)):
                rep = getattr(self, "_{}_replacements".format(quant))()
                if len(rep) > 0:
                    pp[entry]["replacements"] = rep

        if self.endog_correction is True:
            helpers = self._helpers_for_H_transformation_with_psi()
            pp["H_args"]["psi"] = initial_quantities["psi"]
            pp["H_args"]["psi_bool_for_H"] = helpers[0]
            pp["H_args"]["arr1"] = helpers[1]
            pp["H_args"]["endog_position"] = self.endog_position
            pp["H_args"]["initial_copy"] = initial_quantities["H"].copy()

        pp["P_zero_args"]["cholesky_of_P_zero"] = self.cholesky_of_P_zero
        pp["P_zero_args"]["square_root_filters"] = self.square_root_filters
        pp["P_zero_args"]["params_type"] = params_type

        pp["R_args"]["square_root_filters"] = self.square_root_filters

        if params_type == "short":
            pp["trans_coeffs_args"][
                "transform_funcs"
            ] = self._transform_trans_coeffs_funcs()
            pp["trans_coeffs_args"]["included_factors"] = self.included_factors
        return pp

    def _restore_unestimated_quantities_args_dict(self, initial_quantities):
        r_args = {}
        if "X_zero" not in self.params_quants:
            r_args["X_zero"] = initial_quantities["X_zero"]
            # this could be vectors
            r_args["X_zero_value"] = 0.0
        if "W_zero" not in self.params_quants:
            r_args["W_zero"] = initial_quantities["W_zero"]
            r_args["W_zero_value"] = 1 / self.nemf
        return r_args

    def _update_args_dict(self, initial_quantities, like_vec):
        position_helper = self.update_info[self.factors].values.astype(bool)

        u_args_list = []
        # this is necessary to make some parts of the likelihood arguments
        # dictionaries usable to calculate marginal effecs with all estimators.
        if self.estimator == "chs":
            k = 0
            for t in self.periods:
                nmeas = self.nmeas_list[t]
                if t == self.periods[-1] and self.anchoring is True:
                    nmeas += 1
                for j in range(nmeas):
                    u_args = [
                        initial_quantities["X_zero"],
                        initial_quantities["P_zero"],
                        like_vec,
                        self.y_data[k],
                        self.c_data[t],
                        initial_quantities["deltas"][t][j],
                        initial_quantities["H"][k],
                        initial_quantities["R"][k : k + 1],
                        np.arange(self.nfac)[position_helper[k]],
                        initial_quantities["W_zero"],
                    ]
                    if self.square_root_filters is False:
                        u_args.append(np.zeros((self.nobs, self.nfac)))
                    u_args_list.append(u_args)
                    k += 1
        return u_args_list

    def _transition_equation_args_dicts(self, initial_quantities):
        dict_list = [[{} for f in self.factors] for s in self.stages]

        for s, f in product(self.stages, range(self.nfac)):
            dict_list[s][f]["coeffs"] = initial_quantities["trans_coeffs"][f][s, :]
            dict_list[s][f]["included_positions"] = self.included_positions[f]
        return dict_list

    def _transform_sigma_points_args_dict(self, initial_quantities):
        tsp_args = {}
        tsp_args["transition_function_names"] = self.transition_names
        if self.anchor_in_predict is True:
            tsp_args["anchoring_type"] = self.anchoring_update_type
            tsp_args["anchoring_positions"] = self.anch_positions
            tsp_args["anch_params"] = initial_quantities["H"][-1, :]
            if self.ignore_intercept_in_linear_anchoring is False:
                tsp_args["intercept"] = initial_quantities["deltas"][-1][-1, 0:1]

        if self.endog_correction is True:
            tsp_args["psi"] = initial_quantities["psi"]
            tsp_args["endog_position"] = self.endog_position
            tsp_args["correction_func"] = self.endog_function
        tsp_args["transition_argument_dicts"] = self._transition_equation_args_dicts(
            initial_quantities
        )
        return tsp_args

    def _predict_args_dict(self, initial_quantities):
        p_args = {}
        p_args["sigma_points"] = initial_quantities["sigma_points"]
        p_args["flat_sigma_points"] = initial_quantities["flat_sigma_points"]
        p_args["s_weights_m"], p_args["s_weights_c"] = self.sigma_weights()
        p_args["Q"] = initial_quantities["Q"]
        p_args["transform_sigma_points_args"] = self._transform_sigma_points_args_dict(
            initial_quantities
        )
        p_args["out_flat_states"] = initial_quantities["flat_X_zero"]
        p_args["out_flat_covs"] = initial_quantities["flat_P_zero"]
        return p_args

    def _calculate_sigma_points_args_dict(self, initial_quantities):
        sp_args = {}
        sp_args["states"] = initial_quantities["X_zero"]
        sp_args["flat_covs"] = initial_quantities["flat_P_zero"]
        sp_args["out"] = initial_quantities["sigma_points"]
        sp_args["square_root_filters"] = self.square_root_filters
        sp_args["scaling_factor"] = self.sigma_scaling_factor()
        return sp_args

    def likelihood_arguments_dict(self, params_type):
        """Construct a dict with arguments for the likelihood function."""
        initial_quantities = self._initial_quantities_dict()

        args = {}
        args["like_vec"] = np.ones(self.nobs)
        args["parse_params_args"] = self._parse_params_args_dict(
            initial_quantities, params_type=params_type
        )
        args["stagemap"] = self.stagemap
        args["nmeas_list"] = self.nmeas_list
        args["anchoring"] = self.anchoring
        args["square_root_filters"] = self.square_root_filters
        args["update_types"] = list(self.update_info["update_type"])
        args["update_args"] = self._update_args_dict(
            initial_quantities, args["like_vec"]
        )
        args["predict_args"] = self._predict_args_dict(initial_quantities)
        args["calculate_sigma_points_args"] = self._calculate_sigma_points_args_dict(
            initial_quantities
        )
        args["restore_args"] = self._restore_unestimated_quantities_args_dict(
            initial_quantities
        )
        return args

    def nloglikeobs(self, params, args):
        """Negative log likelihood function per individual.

        This is the function used to calculate the standard errors based on
        the outer product of gradients.

        """
        return -log_likelihood_per_individual(params, **args)

    def nloglike(self, params, args):
        """Negative log likelihood function.

        This is the function used to fit the model as numeric optimization
        methods are implemented as minimizers.

        """
        if self.save_intermediate_optimization_results is True:
            path = self.save_path + "/opt_results/iteration{}.json"
            with open(path.format(self.optimize_iteration_counter), "w") as j:
                json.dump(params.tolist(), j)
            self.optimize_iteration_counter += 1
        return -log_likelihood_per_individual(params, **args).sum()

    def loglikeobs(self, params, args):
        """Log likelihood per individual."""
        return log_likelihood_per_individual(params, **args)

    def loglike(self, params, args):
        """Log likelihood."""
        return log_likelihood_per_individual(params, **args).sum()

    def estimate_params_chs(
        self, start_params=None, params_type="short", return_optimize_dict=True
    ):
        """Estimate the params vector with the chs estimator.

        Args:
            start_params (np.ndarray): start values that take precedence over
                all other ways of specifying start values.
            params_type (str): specify which type of params is returned.
                the default is short.
            return_optimize_dict (bool): if True, in addition to the params
                vector a dictionary with information from the numerical
                optimization is returned.

        """
        if start_params is None:
            start_params = self.generate_start_params()
        bounds = self.bounds_list()
        args = self.likelihood_arguments_dict(params_type="short")
        if self.save_intermediate_optimization_results is True:
            self.optimize_iteration_counter = 0
        res = minimize(
            self.nloglike,
            start_params,
            args=(args),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.maxiter, "maxfun": self.maxfun},
        )

        optimize_dict = {}
        optimize_dict["success"] = res.success
        optimize_dict["nfev"] = res.nfev
        optimize_dict["log_lh_value"] = -res.fun
        optimize_dict["xopt"] = res.x.tolist()

        params = self.expandparams(res.x) if params_type == "long" else res.x

        if optimize_dict["success"] is False:
            warnings.warn(
                "The model {} in dataset {} terminated unsuccessfully. Its "
                "parameters should not be used.".format(
                    self.model_name, self.dataset_name
                )
            )

        if self.save_params_before_calculating_standard_errors is True:
            path = self.save_path + "/params/params.json"
            with open(path, "w") as j:
                json.dump(params.tolist(), j)

        if return_optimize_dict is True:
            return params, optimize_dict
        else:
            return params

    def number_of_iv_parameters(self, factor=None, anch_equation=False):
        """Number of parameters in the IV equation of a factor."""
        if anch_equation is False:
            assert factor is not None, ""
            f = self.factors.index(factor)
            trans_name = self.transition_names[f]
            x_list, z_list = variable_permutations_for_iv_equations(self.factors, self.included_factors,
                                                                    self.transition_names, self.measurements,
                                                                    self.anchored_factors, 0, factor)
        else:
            trans_name = "linear"
            x_list, z_list = variable_permutations_for_iv_equations(self.factors, self.included_factors,
                                                                    self.transition_names, self.measurements,
                                                                    self.anchored_factors,
                                                                    self.periods[-1], None
                                                                    )

        example_x, example_z = x_list[0], z_list[0]
        x_formula, _ = getattr(tf, "iv_formula_{}".format(trans_name))(
            example_x, example_z
        )
        return x_formula.count("+") + 1

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
        if period > 0 and "constant" in self.transition_names:
            initial_meas = []
            for f, factor in enumerate(self.factors):
                if self.transition_names[f] == "constant":
                    initial_meas += self.measurements[factor][0]
            constant_factor_coeffs = self.storage_df.loc[0, coeff_type].loc[
                initial_meas
            ]
            constant_factor_coeffs.index = [
                "{}_copied".format(m) for m in constant_factor_coeffs.index
            ]
            coeffs = coeffs.append(constant_factor_coeffs)
        return coeffs

    def residual_measurements(self, period):
        """Residual measurements for the wa estimator in one period.

        Args:
            period (int): period identifier

        Returns
            res_meas (DataFrame): residual measurements from period, extended
            with residual measurements of constant factors from initial
            period.

        """
        loadings = self.extended_meas_coeffs("loadings", period)
        intercepts = self.extended_meas_coeffs("intercepts", period)

        res_meas = (self.y_data[period] - intercepts) / loadings
        res_meas.columns = [col + "_resid" for col in res_meas.columns]
        return res_meas

    def all_iv_estimates(self, period, data, factor=None):
        """Coeffs and residual covs for all IV equations of factor in period.

        Args:
            period (int): period identifier
            factor (str): name of a latent factor
            data (DataFrames): see large_df_for_iv_equations in wa_functions

        Returns:
            iv_coeffs (DataFrame): pandas DataFrame with the estimated
            coefficients of all IV equations of factor in period. The
            DataFrame has a Multiindex. The first level are the names of
            the dependent variables (next period measurements) of the
            estimated equations. The second level consists of integers that
            identify the different combinations of independent variables.

        Returns:
            u_cov_df (DataFrame): pandas DataFrame with covariances of iv
            residuals u with alternative dependent variables. The index is
            the same Multiindex as in iv_coeffs. The columns are all
            possible dependent variables of *factor* in *period*.

        """
        last_period = self.periods[-1]
        if period != last_period:
            assert factor is not None, ""

        indep_permutations, instr_permutations = variable_permutations_for_iv_equations(self.factors,
                                                                                        self.included_factors,
                                                                                        self.transition_names,
                                                                                        self.measurements,
                                                                                        self.anchored_factors,
                                                                                        period, factor
                                                                                        )

        if period != last_period:
            nr_deltas = self.number_of_iv_parameters(factor)
            trans_name = self.transition_names[self.factors.index(factor)]
            depvars = self.measurements[factor][period + 1]
        else:
            nr_deltas = self.number_of_iv_parameters(anch_equation=True)
            trans_name = "linear"
            depvars = [self.anch_outcome]

        iv_columns = ["delta_{}".format(i) for i in range(nr_deltas)]
        u_cov_columns = ["cov_u_{}".format(y) for y in depvars]

        ind_tuples = [
            (dep_name, indep_loc)
            for dep_name, indep_loc in product(depvars, range(len(indep_permutations)))
        ]
        index = pd.MultiIndex.from_tuples(ind_tuples)

        iv_coeffs = pd.DataFrame(data=0.0, index=index, columns=iv_columns)
        if period != last_period:
            u_cov_df = pd.DataFrame(data=0.0, index=index, columns=u_cov_columns)
        else:
            u_var_sr = pd.Series(
                data=0.0, index=range(len(indep_permutations)), name="u_var"
            )

        for dep in depvars:
            counter = 0
            for indep, instr in zip(indep_permutations, instr_permutations):
                iv_arrs = iv_reg_array_dict(dep, indep, instr, trans_name, data)
                non_missing_index = iv_arrs.pop("non_missing_index")
                deltas = iv_reg(**iv_arrs)
                iv_coeffs.loc[(dep, counter)] = deltas
                y, x = iv_arrs["depvar_arr"], iv_arrs["indepvars_arr"]
                u = pd.Series(data=y - np.dot(x, deltas), index=non_missing_index)

                if period != last_period:
                    for dep2 in depvars:
                        if dep2 != dep:
                            u_cov_df.loc[(dep, counter), dep2] = u.cov(
                                data[("y", dep2)]
                            )
                        else:
                            u_cov_df.loc[(dep, counter), dep2] = np.nan
                else:
                    u_var_sr[counter] = u.var()
                counter += 1
        if period != last_period:
            return iv_coeffs, u_cov_df
        else:
            return iv_coeffs, u_var_sr

    def model_coeffs_from_iv_coeffs_args_dict(self, period, factor):
        """Dictionary with optional arguments of model_coeffs_from_iv_coeffs.

        The arguments contain the normalizations and identified restrictions
        that are needed to identify the model coefficients of interest.

        """
        args = {}

        load_norm = self.normalizations[factor]["loadings"][period]
        if len(load_norm) == 0:
            load_norm = None
        args["loading_norminfo"] = load_norm

        inter_norm = self.normalizations[factor]["intercepts"][period]
        if len(inter_norm) == 0:
            inter_norm = None
        args["intercept_norminfo"] = inter_norm

        stage = self.stagemap[period]
        for rtype in ["coeff_sum_value", "trans_intercept_value"]:
            restriction = self.identified_restrictions[rtype].loc[stage, factor]
            if restriction is not None:
                args[rtype] = restriction

        return args

    def update_identified_restrictions(self, stage, factor, coeff_sum, intercept):
        """Update self.identified_restrictions if necessary.

        Identified restrictions are sums of coefficients of transition
        equations or intercepts of transition equations. They are used in the
        wa estimator to calculate model coefficients from the IV regression
        coefficients.

        Using restrictions that were identified in earlier periods of a stage
        to calculate the model parameters of later periods in a stage makes
        it possible to use development stages without over-normalization.

        """
        if coeff_sum is not None:
            self.identified_restrictions["coeff_sum_value"].loc[
                stage, factor
            ] = coeff_sum
        if intercept is not None:
            self.identified_restrictions["trans_intercept_value"].loc[
                stage, factor
            ] = intercept

    def _calculate_wa_quantities(self):
        """Helper function.

        In this function the wa estimates are calculated, but not yet written
        into a single params vector that can be used for later processing.

        """
        self.identified_restrictions["coeff_sum_value"][:] = None
        self.identified_restrictions["trans_intercept_value"][:] = None
        t = 0
        # identify measurement system and factor means in initial period
        meas_coeffs, X_zero = initial_meas_coeffs(
            y_data=self.y_data[t],
            measurements=self.measurements,
            normalizations=self.normalizations,
        )
        self.storage_df.update(prepend_index_level(meas_coeffs, t))

        # generate variables to store trans_coeffs and transition variances
        trans_coeff_storage = self._initial_trans_coeffs()
        trans_var_cols = [
            fac
            for f, fac in enumerate(self.factors)
            if self.transition_names[f] != "constant"
        ]
        trans_var_df = pd.DataFrame(data=0.0, columns=trans_var_cols, index=self.stages)

        # apply the WA IV approach in all period for all factors and calculate
        # all model parameters of interest from the iv parameters
        for t, stage in zip(self.periods[:-1], self.stagemap[:-1]):
            # generate the large IV DataFrame for period t
            resid_meas = self.residual_measurements(period=t)
            iv_data = large_df_for_iv_equations(
                depvar_data=self.y_data[t + 1],
                indepvars_data=resid_meas,
                instruments_data=self.y_data[t],
            )

            for f, factor in enumerate(self.factors):
                trans_name = self.transition_names[f]
                if trans_name != "constant":
                    # get iv estimates (parameters and residual covariances)
                    iv_coeffs, u_cov_df = self.all_iv_estimates(t, iv_data, factor)

                    # get model parameters from iv parameters
                    model_coeffs_func = getattr(
                        tf, "model_coeffs_from_iv_coeffs_" + trans_name
                    )

                    optional_args = self.model_coeffs_from_iv_coeffs_args_dict(
                        period=t + 1, factor=factor
                    )

                    meas_coeffs, gammas, n_i_coeff_sum, n_i_intercept = model_coeffs_func(
                        iv_coeffs=iv_coeffs, **optional_args
                    )

                    self.storage_df.update(prepend_index_level(meas_coeffs, t + 1))
                    weight = self.wa_period_weights.loc[t, factor]
                    trans_coeff_storage[f][stage] += weight * gammas

                    self.update_identified_restrictions(
                        stage, factor, n_i_coeff_sum, n_i_intercept
                    )

                    # get transition error variance from residual covariances
                    trans_var_df.loc[stage, factor] += (
                        weight
                        * transition_error_variance_from_u_covs(
                            u_cov_df, meas_coeffs["loadings"]
                        )
                    )

        if self.anchoring is True:
            t = self.periods[-1]
            resid_meas = self.residual_measurements(period=t)
            iv_data = large_df_for_iv_equations(
                depvar_data=self.y_data[t],
                indepvars_data=resid_meas,
                instruments_data=self.y_data[t],
            )
            iv_coeffs, u_var_sr = self.all_iv_estimates(t, iv_data)
            deltas = iv_coeffs.mean().values
            anch_intercept = deltas[-1]
            anch_loadings = deltas[:-1]
            indep_permutations, instr_permutations = variable_permutations_for_iv_equations(self.factors,
                                                                                            self.included_factors,
                                                                                            self.transition_names,
                                                                                            self.measurements,
                                                                                            self.anchored_factors,
                                                                                            period=t, factor=None
                                                                                            )
            anch_variance = anchoring_error_variance_from_u_vars(
                u_vars=u_var_sr,
                indepvars_permutations=indep_permutations,
                anch_loadings=anch_loadings,
                meas_loadings=self.storage_df.loc[t, "loadings"],
                anchored_factors=self.anchored_factors,
            )
        else:
            anch_intercept = None
            anch_loadings = None
            anch_variance = None

        # calculate measurement error variances and factor covariance matrices
        factor_cov_list = []
        for t in self.periods:
            loadings = self.extended_meas_coeffs(period=t, coeff_type="loadings")
            all_meas = list(loadings.index)
            meas_cov = self.y_data[t][all_meas].cov()
            # loadings = self.storage_df.loc[t, 'loadings']
            meas_per_f = self._measurement_per_factor_dict(period=t)

            p, meas_error_variances = factor_covs_and_measurement_error_variances(
                meas_cov=meas_cov, loadings=loadings, meas_per_factor=meas_per_f
            )
            factor_cov_list.append(p)
            self.storage_df.update(prepend_index_level(meas_error_variances, t))
        P_zero = factor_cov_list[0]

        # print('calculated_cov_matrices')
        # for cov_mat in factor_cov_list:
        #     print(cov_mat)

        return (
            self.storage_df,
            X_zero,
            P_zero,
            trans_coeff_storage,
            trans_var_df,
            anch_intercept,
            anch_loadings,
            anch_variance,
        )

    def _measurement_per_factor_dict(self, period, include_copied=True):
        d = {}
        for f, factor in enumerate(self.factors):
            if self.transition_names[f] != "constant" or period == 0:
                d[factor] = self.measurements[factor][period]
            elif include_copied is True:
                initial_meas = self.measurements[factor][0]
                d[factor] = ["{}_copied".format(m) for m in initial_meas]
        return d

    def estimate_params_wa(self):
        """Estimate the params vector with wa."""
        storage_df, X_zero, P_zero, trans_coeffs, trans_var_df, anch_intercept, anch_loadings, anch_variance = (
            self._calculate_wa_quantities()
        )

        params = np.zeros(self.len_params(params_type="long"))
        slices = self.params_slices(params_type="long")

        # write intercepts in params
        delta_start_index = slices["deltas"][0].start
        delta_stop_index = slices["deltas"][-1].stop
        all_intercepts = list(
            storage_df[storage_df["has_normalized_intercept"] == False][
                "intercepts"
            ].values
        )
        if anch_intercept is not None:
            all_intercepts.append(anch_intercept)
        params[delta_start_index:delta_stop_index] = all_intercepts
        # write loadings in params
        all_loadings = list(
            storage_df[storage_df["has_normalized_loading"] == False]["loadings"].values
        )
        if anch_loadings is not None:
            all_loadings += list(anch_loadings)
        params[slices["H"]] = all_loadings

        # write measurement variances in params
        all_meas_variances = list(storage_df["meas_error_variances"].values)
        if anch_variance is not None:
            all_meas_variances.append(anch_variance)
        params[slices["R"]] = all_meas_variances

        # write transition variances (Q) in params.

        # first take the mean of those trans_var estimates that have to be
        # averaged between stages. This is for example necessary if the
        # transition function is ar1. Averaging reduces to a addition as the
        # estimates are weighted already.
        for f, factor in enumerate(self.factors):
            if self.transition_names[f] != "constant":
                for s in reversed(self.stages):
                    if self.new_trans_coeffs[s, f] == 0:
                        trans_var_df.loc[s - 1, factor] += trans_var_df.loc[s, factor]
                        trans_var_df.loc[s, factor] = np.nan

        # then write them in a flat list in the correct order
        trans_var_list = []
        for s in self.stages:
            for f, factor in enumerate(self.factors):
                if self.new_trans_coeffs[s, f] == 1:
                    trans_var_list.append(trans_var_df.loc[s, factor])

        # finally write them in params
        params[slices["Q"]] = trans_var_list

        # write P_zero in params
        params[slices["P_zero"]] = P_zero

        # write trans_coeffs in params
        for f in range(self.nfac):
            for coeffs, sl in zip(trans_coeffs[f], slices["trans_coeffs"][f]):
                params[sl] += coeffs

        # write X_zero in params
        if self.estimate_X_zeros is True:
            params[slices["X_zero"]] = X_zero

        if self.save_params_before_calculating_standard_errors is True:
            path = self.save_path + "/params/params.json"
            with open(path, "w") as j:
                json.dump(params.tolist(), j)

        return params

    def score(self, params):
        """Gradient of loglike with respect to each parameter.

        To calculate the gradient, simple numerical derivatives are used.
        """
        if self.estimator == "wa":
            raise NotApplicableError(
                "score only works for likelihood based estimators."
            )
        elif not hasattr(self, "stored_score"):
            args = self.likelihood_arguments_dict("long")
            self.stored_score = approx_fprime(
                params, self.loglike, args=(args,), centered=True
            ).ravel()

        return self.stored_score

    def score_obs(self, params):
        """Gradient of loglikeobs with respect to each parameter.

        To calculate the gradient, simple numerical derivatives are used.

        """
        if self.estimator == "wa":
            raise NotApplicableError(
                "score_obs only works for likelihood based estimators."
            )
        elif not hasattr(self, "stored_score_obs"):
            args = self.likelihood_arguments_dict("long")
            self.stored_score_obs = approx_fprime(
                params, self.loglikeobs, args=(args,), centered=True
            )
        return self.stored_score_obs

    def hessian(self, params):
        """Hessian matrix of loglike.

        To calculate the hessian, simple numerical derivatives are used.

        """
        if self.estimator == "wa":
            raise NotApplicableError(
                "hessian only works for likelihood based estimators."
            )
        elif not hasattr(self, "stored_hessian"):
            args = self.likelihood_arguments_dict("long")
            self.stored_hessian = approx_hess(params, self.loglike, args=(args,))
        return self.stored_hessian

    def op_of_gradient_cov_matrix(self, params):
        """Covariance matrix of params based on outer product of gradients."""
        assert len(params) == self.len_params("long"), (
            "Standard errors can only be calculated for params vectors of the "
            "long type. Your params vector has incorrect length in model {} "
            "with dataset {}"
        ).format(self.model_name, self.dataset_name)

        gradient = self.score_obs(params)
        # what follows is equivalent to:
        # cov = np.linalg.inv(np.dot(gradient.T, gradient))
        # but it ensures that the resulting covariance matrix is
        # positive semi-definite
        u = np.linalg.qr(gradient)[1]
        u_inv = np.linalg.inv(u)
        cov = np.dot(u_inv, u_inv.T)
        return cov

    def hessian_inverse_cov_matrix(self, params):
        """Covariance matrix of params based on inverse of hessian."""
        assert len(params) == self.len_params("long"), (
            "Standard errors can only be calculated for params vectors of the "
            "long type. Your params vector has incorrect length in model {} "
            "with dataset {}"
        ).format(self.model_name, self.dataset_name)

        hessian = self.hessian(params)
        cov = np.linalg.inv(-hessian)
        return cov

    def _check_bs_samples(self):
        """Check validity of provided bootstrap samples."""
        assert hasattr(self.bootstrap_samples, "__iter__"), (
            "The bootstrap_samples you provided are not iterable. "
            "The bootstrap_samples must be iterable with each item providing "
            "the person_identifiers chosen to be in that bootstrap sample. "
            "See the documentation on model_specs for more information. "
            "This error occured in model {} and dataset {}".format(
                self.model_name, self.dataset_name
            )
        )

        identifiers = self.data[self.person_identifier].unique()
        for item in self.bootstrap_samples:
            assert set(item).issubset(identifiers), (
                "The bootstrap_samples you provided contain person_identifiers"
                " which are not in the dataset {}. These missing identifiers "
                "are {}.\nYou specified {} as person_identifier. This error "
                "occurred in model {}".format(
                    self.dataset_name,
                    [i for i in item if i not in identifiers],
                    self.person_identifier,
                    self.model_name,
                )
            )

        assert len(self.bootstrap_samples) >= self.bootstrap_nreps, (
            "You only provided {} bootstrap samples but specified {} "
            "replications. You must either reduce the number of replications "
            "or provide more samples in model {} and dataset {}".format(
                len(self.bootstrap_samples),
                self.bootstrap_nreps,
                self.model_name,
                self.dataset_name,
            )
        )

    def _generate_bs_samples(self):
        """List of lists.

        Each sublist contains the 'person_identifiers' of the individuals that
        were sampled from the dataset with replacement.

        """
        individuals = np.array(self.data[self.person_identifier].unique())
        selected_indices = np.random.randint(
            low=0,
            high=len(individuals),
            size=(self.bootstrap_nreps, self.bootstrap_sample_size),
        )
        bootstrap_samples = individuals[selected_indices].tolist()
        return bootstrap_samples

    def _select_bootstrap_data(self, rep):
        """Return the data of the resampled individuals."""
        data = self.data.set_index(
            [self.person_identifier, self.period_identifier], drop=False
        )
        current_sample = self.bootstrap_samples[rep]
        bs_index = pd.MultiIndex.from_product(
            [current_sample, self.periods],
            names=[self.person_identifier, self.period_identifier],
        )
        bs_data = data.loc[bs_index].reset_index(drop=True)
        return bs_data

    def _bs_fit(self, rep, params):
        """Check the bootstrap data and re-fit the model with it."""
        bootstrap_data = self._select_bootstrap_data(rep)

        if self.save_path is not None:
            bs_save_path = self.save_path + "/bootstrap/{}".format(rep)
        else:
            bs_save_path = None

        new_mod = SkillModel(
            model_dict=self.model_dict,
            dataset=bootstrap_data,
            estimator=self.estimator,
            model_name=self.model_name + "_{}".format(rep),
            dataset_name=self.dataset_name + "_{}".format(rep),
            save_path=bs_save_path,
        )

        if new_mod.len_params("long") != len(params):
            warn_message = (
                "No bootstrap results were calculated for replication {} "
                "because the resulting parameter vectors would not have "
                "had the right length. This can be caused if variables "
                "were dropped automatically because they had no variance. "
                "in the bootstrap data. It happened in model {} for "
                "dataset {}."
            ).format(rep, self.model_name, self.dataset_name)
            warnings.warn(warn_message)
            bs_params = [np.nan] * len(params)
            if self.estimator == "chs":
                optimize_dict = {
                    "success": "Not started because the resulting "
                    + "parameter vector would have had the wrong length."
                }

        elif self.estimator == "chs":
            start_params = new_mod.reduceparams(params)
            bs_params, optimize_dict = new_mod.estimate_params_chs(
                start_params=start_params, return_optimize_dict=True, params_type="long"
            )

        elif self.estimator == "wa":
            bs_params = new_mod.estimate_params_wa()

        if self.estimator == "chs" and optimize_dict["success"] is False:
            bs_params = [np.nan] * len(bs_params)
            warnings.warn(
                "The optimization for bootstrap replication {} has failed. "
                "It is therefore not included in the calculation of standard "
                "errors. This occured for model {} in dataset {}".format(
                    rep, self.model_name, self.dataset_name
                )
            )
            bs_params = [np.nan] * len(params)
        return bs_params

    def all_bootstrap_params(self, params):
        """Return a DataFrame of all bootstrap parameters.

        Create the resampled datasets from lists of person identifiers and fit
        re-fit the model.

        The boostrap replications are estimated in parallel, using the
        Multiprocessing module from the Python Standard Library.

        """
        assert len(params) == self.len_params("long"), (
            "Standard errors can only be calculated for params vectors of the "
            "long type. Your params vector has incorrect length in model {} "
            "with dataset {}"
        ).format(self.model_name, self.dataset_name)

        if not hasattr(self, "stored_bootstrap_params"):
            bs_fit_args = [(r, params) for r in range(self.bootstrap_nreps)]
            with Pool(self.bootstrap_nprocesses) as p:
                bootstrap_params = p.starmap(self._bs_fit, bs_fit_args)
            ind = ["rep_{}".format(rep) for rep in range(self.bootstrap_nreps)]
            cols = self.param_names("long")
            bootstrap_params_df = pd.DataFrame(
                data=bootstrap_params, index=ind, columns=cols
            )
            bootstrap_params_df.dropna(inplace=True)
            self.stored_bootstrap_params = bootstrap_params_df
        return self.stored_bootstrap_params

    def bootstrap_conf_int(self, params, alpha=0.05):
        """Parameter confidence intervals from bootstrap parametres.

        args:
            bootstrap_params (df): pandas DataFrame where each column gives the
                parameters from one bootstrap replication.
            alpha (float): the significance level of the confidence interval.

        Returns:
            conf_int_df (df): pandas DataFrame with two columns that give the
            lower and upper bound of the confidence interval based on the
            distribution of the bootstrap parameters.

        """
        bs_params = self.all_bootstrap_params(params)

        lower = bs_params.quantile(0.5 * alpha, axis=0).rename("lower")
        upper = bs_params.quantile(1 - 0.5 * alpha, axis=0).rename("upper")
        conf_int_df = pd.concat([lower, upper], axis=1)
        return conf_int_df

    def bootstrap_cov_matrix(self, params):
        """Calculate the paramater covariance matrix using bootstrap."""
        bs_params = self.all_bootstrap_params(params)
        return bs_params.cov()

    def bootstrap_mean(self, params):
        bs_params = self.all_bootstrap_params(params)
        return bs_params.mean()

    def bootstrap_pvalues(self, params):
        bs_params = self.all_bootstrap_params(params)
        # safety measure
        params = np.array(params)
        nan_where_less_extreme = bs_params[abs(bs_params) >= abs(params)]
        numerator = nan_where_less_extreme.count() + 1
        p_values = numerator / (len(bs_params) + 1)
        return p_values

    def fit(self, start_params=None, params=None):
        """Fit the model and return an instance of SkillModelResults."""
        if self.estimator == "chs":
            params, optimize_dict = self.estimate_params_chs(
                start_params, return_optimize_dict=True, params_type="long"
            )

        elif self.estimator == "wa":
            params = self.estimate_params_wa()
            optimize_dict = None

        cov_func = getattr(self, "{}_cov_matrix".format(self.standard_error_method))
        cov = cov_func(params)

        like_res = LikelihoodModelResults(self, params, cov)

        skillmodel_res = SkillModelResults(self, like_res, optimize_dict)
        return skillmodel_res

    def _generate_start_factors(self):
        """Start factors for simulations or marginal effects.

        Returns:
            start_factors (np.ndarray): array of size (self.nobs, self.nfac)
                with a simulated sample of initial latent factors.

        """
        assert self.nemf == 1, (
            "Start factors for simulation can currently only be generated "
            "if nemf == 1."
        )

        # get start means
        slices = self.params_slices("long")
        if "X_zero" in self.params_quants:
            X_zero = self.me_params[slices["X_zero"]]
        else:
            X_zero = np.zeros(self.nfac)

        # get start covariance matrix
        p_params = self.me_params[slices["P_zero"]]
        p_helper = np.zeros((self.nfac, self.nfac))
        p_helper[np.triu_indices(self.nfac)] = p_params

        if self.estimator == "chs" and self.cholesky_of_P_zero is True:
            P_zero = np.dot(p_helper.T, p_helper)
        else:
            p_helper_2 = (p_helper - np.diag(np.diagonal(p_helper))).T
            P_zero = p_helper + p_helper_2
        # generate the sample
        start_factors = np.random.multivariate_normal(
            mean=X_zero, cov=P_zero, size=self.nobs
        )

        return start_factors

    def _machine_accuracy(self):
        return np.MachAr().eps

    def _get_epsilon(self, centered):
        change = np.zeros(self.nperiods - 1)
        intermediate_factors = self._predict_final_factors(
            change, return_intermediate=True
        )

        pos = self.factors.index(self.me_of)
        epsilon = np.zeros(self.nperiods - 1)
        for t in self.periods[:-1]:
            epsilon[t] = np.abs(intermediate_factors[t][:, pos]).max()

        epsilon[epsilon <= 0.1] = 0.1

        accuracy = self._machine_accuracy()
        exponent = 1 / 2 if centered is False else 1 / 3

        epsilon *= accuracy ** exponent
        return epsilon

    def _predict_final_factors(self, change, return_intermediate=False):

        assert self.endog_correction is False, (
            "Currently, final factors can only be predicted if no "
            "endogeneity correction is used."
        )

        assert (
            len(change) == self.nperiods - 1
        ), "factor_change must have len nperiods - 1."

        if return_intermediate is True:
            intermediate_factors = []

        changed_pos = self.factors.index(self.me_of)
        args = self.likelihood_arguments_dict("long")
        tsp_args = args["predict_args"]["transform_sigma_points_args"]
        pp_args = args["parse_params_args"]
        parse_params(self.me_params, **pp_args)

        factors = self.me_at.copy()

        for t, stage in enumerate(self.stagemap[:-1]):
            if return_intermediate is True:
                intermediate_factors.append(factors.copy())
            factors[:, changed_pos] += change[t]
            transform_sigma_points(stage, factors, **tsp_args)

        if return_intermediate is False:
            return factors
        else:
            return intermediate_factors

    def _select_final_factor(self, final_factors):
        pos = self.factors.index(self.me_on)
        return final_factors[:, pos]

    def _anchor_final_factors(self, final_factors, anch_loadings):

        if self.anchoring is True:
            assert self.anchoring_update_type == "linear", (
                "Currently, marginal effects only work for linearly anchored "
                "factors."
            )
            anch_func = "anchor_flat_sigma_points_{}".format(self.anchoring_update_type)
            getattr(anch, anch_func)(
                final_factors, self.anch_positions, anch_loadings, intercept=None
            )

        return final_factors

    def _anchoring_outcome_from_final_factors(
        self, final_factors, anch_loadings, anch_intercept
    ):

        assert self.anchoring is True, (
            "Marginal effects on a anchoring outcome can only be calculated "
            "if anchoring equations were estimated."
        )

        assert self.anchoring_update_type == "linear", (
            "Currently, marginal effects on an anchoring outcome can only "
            "be calculated for linear anchoring."
        )

        anchored = self._anchor_final_factors(final_factors, anch_loadings)
        selector = anch_loadings.astype(bool).astype(int)
        anch_outcome = np.dot(anchored, selector) + anch_intercept
        return anch_outcome

    def _marginal_effect_outcome(self, change):

        final_factors = self._predict_final_factors(change)

        if self.anchoring is True:
            # construct_anch_loadings and anch_intercept
            slices = self.params_slices("long")
            relevant_params = self.me_params[slices["H"]][-len(self.anchored_factors) :]

            anch_loadings = np.zeros(self.nfac)
            for p, pos in enumerate(self.anch_positions):
                anch_loadings[pos] = relevant_params[p]

            # the last entry of the last delta slice
            anch_intercept = self.me_params[slices["deltas"][-1]][-1]

            if self.me_on == "anch_outcome":
                return self._anchoring_outcome_from_final_factors(
                    final_factors, anch_loadings, anch_intercept
                )
            elif self.me_anchor_on is True:
                anchored_final_factors = self._anchor_final_factors(
                    final_factors, anch_loadings
                )
                return self._select_final_factor(anchored_final_factors)

        else:
            return self._select_final_factor(final_factors)

    def _basic_heatmap_args(self):
        args = {
            "cmap": "coolwarm",
            "center": 0.0,
            "vmax": 0.5,
            "vmin": -0.5,
            "annot": True,
        }
        return args

    def _basic_pairplot_args(self):
        plot_kws = self._basic_regplot_args()
        del plot_kws["order"]
        args = {"plot_kws": plot_kws, "kind": "reg", "diag_kind": "kde"}
        return args

    def _basic_regplot_args(self):
        args = {
            "scatter_kws": {"alpha": 0.2},
            "fit_reg": True,
            "order": 5,
            "color": self.base_color,
            "truncate": True,
        }
        return args

    def measurement_heatmap(
        self,
        periods="all",
        factors="all",
        figsize=None,
        heatmap_kws={},
        save_path=None,
        dpi=200,
        write_tex=False,
        width=None,
        height=None,
    ):
        """Heatmap of the correlation matrix of measurements.

        Args:
            periods: periods to include. Can be the name of one period, a list
                like object with periods or 'all'.
            factors: factors to include. Can be the name of one factor, a list
                like object with factors or 'all'.
            figsize (tuple): size of the matplotlib figure. If None provided,
                the figure automatically scales with the size of the
                correlation matrix.
            heatmap_kws (dict): dictionary with arguments for sns.heatmap()
            save_path (str): path where the plot will be saved. Needs a valid
                file extension (.png, .jpg, .eps, ...); Other documents are
                saved in the same directory.
            dpi (int): resolution of the plot
            write_tex (bool): if True, a tex file with the plot is written.

         """
        if write_tex is True:
            assert (
                save_path is not None
            ), "To write a tex file, please provide a save_path"
        df = self.data_proc.measurements_df(periods=periods, factors=factors)
        corr = df.corr()

        if figsize is None:
            figsize = (len(corr), 0.8 * len(corr))

        if width is None and height is None:
            if len(corr) <= 5:
                width = 0.5
            elif len(corr) <= 9:
                width = 0.8
            else:
                width = 1

        kwargs = self._basic_heatmap_args()
        kwargs.update(heatmap_kws)
        fig, ax = plt.subplots(figsize=figsize)
        ax = sns.heatmap(data=corr, ax=ax, **kwargs)

        base_title = "Correlations of Measurements"
        title = title_text(basic_name=base_title, periods=periods, factors=factors)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

        if write_tex is True:
            write_figure_tex_snippet(save_path, title, width=width, height=height)
        return fig, ax

    def score_heatmap(
        self,
        periods="all",
        factors="all",
        agg_method="means",
        figsize=None,
        heatmap_kws={},
        save_path=None,
        dpi=200,
        write_tex=False,
        width=None,
        height=None,
    ):

        if write_tex is True:
            assert (
                save_path is not None
            ), "To write a tex file, please provide a save_path"

        df = self.data_proc.score_df(
            periods=periods, factors=factors, order="by_factor", agg_method=agg_method
        )
        corr = df.corr()

        if width is None and height is None:
            if len(corr) <= 5:
                width = 0.5
            elif len(corr) <= 9:
                width = 0.8
            else:
                width = 1

        if figsize is None:
            figsize = (len(corr), 0.8 * len(corr))

        kwargs = self._basic_heatmap_args()
        kwargs.update(heatmap_kws)
        fig, ax = plt.subplots(figsize=figsize)
        ax = sns.heatmap(data=corr, ax=ax, **kwargs)

        base_title = "Correlations of Factor Scores"
        title = title_text(basic_name=base_title, periods=periods, factors=factors)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

        if write_tex is True:
            write_figure_tex_snippet(save_path, title, width=width, height=height)
        return fig, ax

    def measurement_pairplot(
        self,
        periods="all",
        factors="all",
        group=None,
        pair_kws={},
        save_path=None,
        dpi=200,
        write_tex=False,
        width=None,
        height=None,
    ):

        if write_tex is True:
            assert (
                save_path is not None
            ), "To write a tex file, please provide a save_path"

        if group is None:
            other_vars = []
        else:
            other_vars = [group]

        if width is None and height is None:
            width = 1

        df = self.data_proc.measurements_df(
            periods=periods, factors=factors, other_vars=other_vars
        )

        kwargs = self._basic_pairplot_args()
        kwargs.update(pair_kws)

        vars = [col for col in df.columns if col != group]

        grid = sns.pairplot(data=df, vars=vars, hue=group, **kwargs)

        base_title = "Joint Distribution of Measurements"
        title = title_text(base_title, periods=periods, factors=factors)

        if write_tex is True:
            write_figure_tex_snippet(save_path, title, width=width, height=height)

        if save_path is not None:
            grid.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return grid

    def score_pairplot(
        self,
        periods="all",
        factors="all",
        agg_method="means",
        group=None,
        pair_kws={},
        save_path=None,
        dpi=200,
        write_tex=False,
        width=None,
        height=None,
    ):

        if write_tex is True:
            assert (
                save_path is not None
            ), "To write a tex file, please provide a save_path"

        if group is None:
            other_vars = []
        else:
            other_vars = [group]

        if width is None and height is None:
            width = 1

        df = self.data_proc.score_df(
            periods=periods,
            factors=factors,
            other_vars=other_vars,
            agg_method=agg_method,
        )

        kwargs = self._basic_pairplot_args()
        kwargs.update(pair_kws)

        vars = [col for col in df.columns if col != group]

        grid = sns.pairplot(data=df, hue=group, vars=vars, **kwargs)

        base_title = "Joint Distribution of Factor Scores"
        title = title_text(base_title, periods=periods, factors=factors)

        if write_tex is True:
            write_figure_tex_snippet(save_path, title, width=width, height=height)

        if save_path is not None:
            grid.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return grid

    def autoregression_plot(
        self,
        period,
        factor,
        agg_method="mean",
        reg_kws={},
        figsize=(10, 5),
        save_path=None,
        dpi=200,
        write_tex=False,
        width=None,
        height=None,
    ):

        if write_tex is True:
            assert (
                save_path is not None
            ), "To write a tex file, please provide a save_path"
        if width is None and height is None:
            width = 0.8

        kwargs = self._basic_regplot_args()
        kwargs.update(reg_kws)

        df = self.data_proc.score_df(
            factors=factor, periods=[period, period + 1], agg_method=agg_method
        )

        fig, ax = plt.subplots(figsize=figsize)

        x = "{}_{}".format(factor, period)
        y = "{}_{}".format(factor, period + 1)

        sns.regplot(x=x, y=y, data=df, ax=ax, **kwargs)

        base_title = "Autoregression Plot"
        title = title_text(base_title, periods=period, factors=factor)

        if write_tex is True:
            write_figure_tex_snippet(save_path, title, width=width, height=height)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

        return fig, ax

    def score_regression_residual_plot(
        self,
        factor,
        period=None,
        stage=None,
        controls=[],
        other_vars=[],
        agg_method="mean",
        reg_kws={},
        save_path=None,
        write_tex=False,
        width=None,
        height=None,
        dpi=200,
    ):

        if write_tex is True:
            assert (
                save_path is not None
            ), "To write a tex file, please provide a save_path"
        if width is None and height is None:
            height = 0.9

        mod = self._score_regression_model(
            period=period,
            stage=stage,
            factor=factor,
            controls=controls,
            agg_method=agg_method,
        )

        res = mod.fit()

        data = self.data_proc.reg_df(
            factor=factor,
            period=period,
            stage=stage,
            controls=controls + other_vars,
            agg_method=agg_method,
        )

        data["residuals"] = res.resid
        data["fitted"] = res.fittedvalues

        y_name = "{}_t_plusone".format(factor)
        to_plot = [col for col in data.columns if col not in ["residuals", y_name]]
        figsize = (10, len(to_plot) * 5)
        fig, axes = plt.subplots(nrows=len(to_plot), figsize=figsize)

        kwargs = self._basic_regplot_args()
        kwargs.update(reg_kws)
        for ax, var in zip(axes, to_plot):
            sns.regplot(y="residuals", x=var, ax=ax, data=data, **kwargs)

        base_title = "Residual Plot"
        title = title_text(base_title, periods=period, factors=factor, stages=stage)

        if write_tex is True:
            write_figure_tex_snippet(save_path, title, width=width, height=height)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

        return fig

    def factor_score_dynamics_plot(
        self,
        factor,
        agg_method="mean",
        group=None,
        figsize=None,
        write_tex=False,
        dpi=200,
        save_path=None,
        width=None,
        height=None,
    ):

        if figsize is None:
            figsize = (12, 8)
        to_concat = []
        for period in self.periods:
            df = self.data_proc.score_df(
                periods=period,
                factors=factor,
                other_vars=[group, self.period_identifier],
                agg_method=agg_method,
            )
            to_concat.append(df)

        data = pd.concat(to_concat, axis=0, sort=True)

        fig, ax = plt.subplots(figsize=figsize)
        sns.pointplot(
            x=self.period_identifier,
            y=factor,
            hue=group,
            data=data,
            ax=ax,
            kind="bar",
            dodge=0.15,
            join=True,
            capsize=0.05,
        )
        sns.despine(fig=fig, ax=ax)

        base_title = "Factor Score Dynamics"
        title = title_text(base_title, periods="all", factors=factor)

        if write_tex is True:
            write_figure_tex_snippet(save_path, title, width=width, height=height)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

        return fig, ax

    def anchoring_plot(
        self,
        period,
        anchor_var,
        agg_method="mean",
        reg_kws={},
        figsize=None,
        write_tex=False,
        dpi=200,
        save_path=None,
        width=None,
        height=None,
    ):

        if figsize is None:
            figsize = (12, 7)

        kwargs = self._basic_regplot_args()
        kwargs["order"] = 1
        kwargs.update(reg_kws)

        if width is None:
            width = 1

        if height is None:
            height = 1

        nonconstant_factors = []
        for factor, trans_name in zip(self.factors, self.transition_names):
            if trans_name != "constant":
                nonconstant_factors.append(factor)

        data = self.data_proc.score_df(
            periods=period,
            factors=nonconstant_factors,
            other_vars=[anchor_var],
            agg_method=agg_method,
        )

        fig, axes = plt.subplots(
            figsize=figsize, ncols=len(nonconstant_factors), sharey=True
        )

        for ax, factor in zip(axes, nonconstant_factors):
            sns.regplot(data=data, y=anchor_var, x=factor, ax=ax, **kwargs)
            sns.despine(fig=fig, ax=ax)

        title = "Relationship of Factor Scores and {} in Period {}".format(
            anchor_var, period
        )

        if write_tex is True:
            write_figure_tex_snippet(save_path, title, width=width, height=height)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

        return fig, ax

    def score_regression_table(
        self,
        periods=None,
        stages=None,
        controls=[],
        agg_method="mean",
        write_tex=False,
        save_path=None,
    ):

        assert (
            periods is None or stages is None
        ), "You cannot specify periods and stages for a score regression."

        assert not (
            periods is None and stages is None
        ), "You have to specify periods or stages for a score regression"

        if write_tex is True:
            assert save_path is not None

        if periods == "all":
            periods = self.periods[:-1]

        if stages == "all":
            stages = self.stages

        if periods is not None:
            time = periods

        if isinstance(periods, int) or isinstance(periods, float):
            periods = [periods]

        if isinstance(stages, int) or isinstance(stages, float):
            stages = [stages]

        if periods is not None:
            time = periods
            time_name = "period"
        else:
            time = stages
            time_name = "stage"

        results = []
        for t in time:
            for factor in self.factors:
                ind = self.factors.index(factor)
                trans_name = self.transition_names[ind]
                if trans_name != "constant":
                    mod_kwargs = {
                        "factor": factor,
                        "controls": controls,
                        "agg_method": agg_method,
                        time_name: t,
                    }

                    mod = self._score_regression_model(**mod_kwargs)
                    res = mod.fit()
                    res.name = factor
                    res.period = t
                    results.append(res)

        df = statsmodels_results_to_df(
            res_list=results, decimals=2, period_name=time_name.capitalize()
        )

        base_title = "OLS Estimation of Transition Equations"
        title = title_text(base_title, periods=periods, stages=stages, factors="all")

        if write_tex is True:
            with open(save_path, "w") as t:
                t.write(df_to_tex_table(df, title))

        # base_name =
        # if output == 'tex':

        return df

    def _score_regression_model(
        self, factor, period=None, stage=None, controls=[], agg_method="mean"
    ):

        df = self.data_proc.reg_df(
            factor=factor,
            period=period,
            stage=stage,
            controls=controls,
            agg_method=agg_method,
        )

        ind = self.factors.index(factor)
        included = self.included_factors[ind]

        y_var = "{}_{}".format(factor, "t_plusone")
        x_vars = ["{}_{}".format(var, "t") for var in included + controls]

        formula = y_var + " ~ " + " + ".join(x_vars)

        mod = smf.ols(formula=formula, data=df)
        return mod

    def visualize_model(self, save_path, anchor_var=None):
        """Visualize a SkillModel.

        Generate plots and tables that illustrate how well the measurements
        fit together, how stable factors are over time and how strong the
        transition equation deviate from linearity.

        Args:
            save_path (str): path to a directory in which plots and tex output
                is saved.

        """
        tex_lines = []
        tex_input = "\input{{{}}}"
        cp = "\n" + r"\clearpage" + "\n"
        float_barrier = r"\FloatBarrier" + "\n"
        section = cp + float_barrier + r"\section{{{}}}" + "\n"
        subsection = float_barrier + r"\subsection{{{}}}" + "\n"
        maketitle = "\maketitle\n"
        toc = r"\tableofcontents" + "\n"
        title = r"\title{{{}}}" + "\n"
        title = title.format(
            "Visualization of {}".format(self.model_name.replace("_", " "))
        )

        tex_lines.append(section.format("Visualization of the Measurement System"))
        tex_lines.append(subsection.format("Correlations by Period"))

        for period in self.periods:
            base_name = "meas_heat_in_period_{}".format(period)
            path = join(save_path, "{}.png".format(base_name))
            self.measurement_heatmap(periods=period, save_path=path, write_tex=True)
            # plt.show()
            plt.close()
            tex_lines.append(tex_input.format(base_name + ".tex"))
            for factor in self.factors:
                if len(self.measurements[factor][period]) >= 2:
                    base_name = "meas_pair_in_period_{}_for_factor_{}".format(
                        period, factor
                    )
                    path = join(save_path, "{}.png".format(base_name))
                    self.measurement_pairplot(
                        periods=period, factors=factor, save_path=path, write_tex=True
                    )
                    plt.close()
                    tex_lines.append(tex_input.format(base_name + ".tex"))
        tex_lines.append(subsection.format("Correlations by Factor"))

        for factor in self.factors:
            base_name = "meas_heat_for_factor_{}".format(factor)
            path = join(save_path, "{}.png".format(base_name))
            self.measurement_heatmap(factors=factor, save_path=path, write_tex=True)
            # plt.show()
            plt.close()
            tex_lines.append(tex_input.format(base_name + ".tex"))

        tex_lines.append(section.format("Visualization of Factor Scores"))

        tex_lines.append(subsection.format("Correlations by Factor"))
        for factor in self.factors:
            ind = self.factors.index(factor)
            trans_name = self.transition_names[ind]
            if trans_name != "constant":
                base_name = "score_heat_for_factor_{}".format(factor)
                path = join(save_path, "{}.png".format(base_name))
                self.score_heatmap(factors=factor, save_path=path, write_tex=True)
                plt.close()
                tex_lines.append(tex_input.format(base_name + ".tex"))

        tex_lines.append(subsection.format("Joint Distribution by Period"))
        for period in self.periods:
            base_name = "score_pair_in_period_{}".format(period)
            path = join(save_path, base_name + ".png")
            self.score_pairplot(periods=period, save_path=path, write_tex=True)
            plt.close()
            tex_lines.append(tex_input.format(base_name + ".tex"))

        tex_lines.append(section.format("Persistence of Factor Scores"))

        for factor in self.factors:
            ind = self.factors.index(factor)
            trans_name = self.transition_names[ind]
            if trans_name != "constant":
                for period in self.periods[:-1]:
                    base_name = "autoreg_for_factor_{}_in_period_{}".format(
                        factor, period
                    )
                    path = join(save_path, "{}.png".format(base_name))
                    self.autoregression_plot(
                        period=period, factor=factor, save_path=path, write_tex=True
                    )
                    plt.close()
                    tex_lines.append(tex_input.format(base_name + ".tex"))

        if anchor_var is not None:
            tex_lines.append(section.format("Anchoring"))
            for period in self.periods:
                base_name = "anchorplot_in_period_{}".format(period)
                path = join(save_path, "{}.png".format(base_name))
                self.anchoring_plot(
                    period=period, anchor_var=anchor_var, write_tex=True, save_path=path
                )

                plt.close()
                tex_lines.append(tex_input.format(base_name + ".tex"))

        tex_lines.append(section.format("OLS Estimates of Transition Equations"))

        tex_lines.append(subsection.format("Parameter Estimates"))

        base_name = "reg_table_by_{}.tex"
        path = join(save_path, base_name.format("periods"))
        self.score_regression_table(periods="all", write_tex=True, save_path=path)
        tex_lines.append(tex_input.format(base_name.format("periods")))

        path = join(save_path, base_name.format("stages"))
        self.score_regression_table(stages="all", write_tex=True, save_path=path)
        tex_lines.append(tex_input.format(base_name.format("stages")))

        tex_lines.append(cp + float_barrier)

        tex_lines.append(subsection.format("Residual Plots"))
        for factor in self.factors:
            ind = self.factors.index(factor)
            trans_name = self.transition_names[ind]
            if trans_name != "constant":
                for stage in self.stages:
                    base_name = "resid_for_factor_{}_in_stage_{}".format(factor, stage)
                    path = join(save_path, "{}.png".format(base_name))
                    self.score_regression_residual_plot(
                        factor=factor, stage=stage, save_path=path, write_tex=True
                    )
                    plt.close()
                    tex_lines.append(tex_input.format(base_name + ".tex"))

        preamble = get_preamble()

        base_name = "visualization_of_{}.tex".format(self.model_name)

        with open(join(save_path, base_name), "w") as t:
            t.write(preamble + "\n\n\n")
            t.write(title)
            t.write(maketitle)
            t.write(toc)
            for line in tex_lines:
                t.write(line + "\n")

            t.write("\n\n\n\end{document}\n")
