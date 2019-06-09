from skillmodels.pre_processing.model_spec_processor import ModelSpecProcessor
from skillmodels.pre_processing.data_processor import DataProcessor
from skillmodels.estimation.likelihood_function import log_likelihood_per_individual
from skillmodels.estimation.wa_functions import calculate_wa_estimated_quantities
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
from skillmodels.simulation.simulate_data import simulate_datasets
from estimagic.optimization.start_helpers import make_start_params_helpers
from estimagic.optimization.start_helpers import get_start_params_from_helpers


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
        self.params_quants = ["delta", "h", "r", "q", "p", "trans_coeffs"]
        if self.estimate_X_zeros is True:
            self.params_quants.append("x")
        if self.restrict_W_zeros is False and self.estimator == "chs":
            self.params_quants.append("w")

    def _initial_delta(self):
        """List of initial arrays for control variable params in each period.

        The arrays have the shape [nupdates, nr_of_control_variables + 1]
        which is potentially different in each period. They are filled with zeros.

        """
        delta = []
        for t in self.periods:
            length = len(self.update_info.loc[t])
            width = len(self.controls[t]) + 1
            init = np.zeros((length, width))
            delta.append(init)
        return delta

    def _initial_h(self):
        """Initial h array filled with zeros.

        The array has the form [nupdates, nfac]. Most entries
        are zero, but if the factor loading of factor f in update equation
        u is normalized to some value then arr[u, f] is equal to this value.

        """
        return np.zeros((self.nupdates, self.nfac))

    def _initial_r(self):
        """1d numpy array of length nupdates filled with zeros."""
        return np.zeros(self.nupdates)

    def _initial_q(self):
        """Initial Q array filled with zeros."""
        return np.zeros((self.nperiods - 1, self.nfac, self.nfac))

    def _initial_x(self):
        """Initial X_zero array filled with zeros."""
        init = np.zeros((self.nobs, self.nemf, self.nfac))
        flat_init = init.reshape(self.nobs * self.nemf, self.nfac)
        return init, flat_init

    def _initial_w(self):
        """Initial W_zero array filled with 1/nemf."""
        return np.ones((self.nobs, self.nemf)) / self.nemf

    def _initial_p(self):
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

    def _initial_trans_coeffs(self):
        """List of initial trans_coeffs arrays, each filled with zeros."""
        initial = []
        for f, factor in enumerate(self.factors):
            func = getattr(tf, "index_tuples_{}".format(self.transition_names[f]))
            nparams = len(func(
                factor=factor, included_factors=self.included_factors[f], period=0))
            initial.append(np.zeros((self.nperiods - 1, nparams)))
        return initial

    def len_params(self, params_type):
        """Return the length of params, dependig on the :ref:`params_type`."""
        slices = self.params_slices(params_type)
        last_slice = slices[self.params_quants[-1]]
        while type(last_slice) == list:
            last_slice = last_slice[-1]
        len_params = last_slice.stop
        return len_params

    def expandparams(self, params):
        """Convert a params vector of type short to type long."""
        raise NotImplementedError

    def reduceparams(self, params):
        """Convert a params vector of type long to type short."""
        raise NotImplementedError

    def _wa_params_can_be_used_for_start_params(self, raise_warning=True):
        """Check if results of WA can be used as start params for CHS."""
        reasons = []
        if self.nemf != 1:
            reasons.append("A mixture distribution is estimated.")
        if self.uses_controls is True:
            reasons.append("Control variables are used.")
        df = self.update_info.copy(deep=True)
        df = df[df["purpose"] == "measurement"]
        if not (df[self.factors].to_numpy().sum(axis=1) == 1).all():
            reasons.append("Some measurements measure more than 1 factor.")
        if self.anchoring_mode == "truly_anchor_latent_factors":
            reasons.append("The anchoring mode is not supported in wa.")
        if "log_ces" in self.transition_names:
            reasons.append("The log_ces cannot be used in wa.")
        # if self.update_info["has_normalized_variance"].any():
        #     reasons.append("Normalized variances are incompatible with wa")

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
        raise NotImplementedError

    def start_params_helpers(self):
        """DataFrames with the free and fixed parameters of the model."""
        free, fixed = make_start_params_helpers(
            self.params_index, self.constraints)
        return free, fixed

    def generate_full_start_params(self):
        """Vector with start values for the optimization.

        If valid start_params are provided in the model dictionary, these will
        be used. Else, naive start_params are generated.

        """
        free, fixed = self.start_params_helpers()
        if hasattr(self, 'start_params'):
            sp = self.start_params
            assert isinstance(sp, pd.DataFrame)
            if len(sp.index.intersection(self.params_index)) == len(self.params_index):
                full_sp = sp

            elif len(sp.index.intersection(free.index)) == len(free.index):
                full_sp = get_start_params_from_helpers(
                    sp, fixed, self.constraints, self.params_index)
            else:
                raise ValueError(
                    "Index of start parameters has to be the same as the index of ",
                    "free parameters from start_params_helpers.")
        else:
            free['value'] = 0.0
            free.loc['h', 'value'] = 1.0
            free.loc['r', 'value'] = 1.0
            free.loc['q', 'value'] = 1.0
            free.loc['trans_coeffs'] = 1 / self.nfac
            free.loc['w'] = 1 / self.nemf
            for emf in range(self.nemf):
                p_diags = [("p", 0, emf, fac) for fac in self.factors]
                free.loc[p_diags, 'value'] = 1

            full_sp = get_start_params_from_helpers(
                free, fixed, self.constraints, self.params_index)
        return full_sp

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

        if "x" not in needed_quantities:
            needed_quantities.append("x")
        if "w" not in needed_quantities:
            needed_quantities.append("w")

        for quant in needed_quantities:
            if quant not in ["x", "p"]:
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

    def _parse_params_args_dict(self, initial_quantities):
        pp = {
            'initial_quantities': initial_quantities,
            'factors': self.factors,
            'square_root_filters': self.square_root_filters
        }

        return pp

    def _restore_unestimated_quantities_args_dict(self, initial_quantities):
        r_args = {}
        if "x" not in self.params_quants:
            r_args["x"] = initial_quantities["x"]
            # this could be vectors
            r_args["x_value"] = 0.0
        if "w" not in self.params_quants:
            r_args["w"] = initial_quantities["w"]
            r_args["w_value"] = 1 / self.nemf
        return r_args

    def _update_args_dict(self, initial_quantities, like_vec):
        position_helper = self.update_info[self.factors].to_numpy().astype(bool)

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
                        initial_quantities["x"],
                        initial_quantities["p"],
                        like_vec,
                        self.y_data[k],
                        self.c_data[t],
                        initial_quantities["delta"][t][j],
                        initial_quantities["h"][k],
                        initial_quantities["r"][k: k + 1],
                        np.arange(self.nfac)[position_helper[k]],
                        initial_quantities["w"],
                    ]
                    if self.square_root_filters is False:
                        u_args.append(np.zeros((self.nobs, self.nfac)))
                    u_args_list.append(u_args)
                    k += 1
        return u_args_list

    def _transition_equation_args_dicts(self, initial_quantities):
        dict_list = [[{} for f in self.factors] for t in self.periods[:-1]]

        for t, f in product(self.periods[:-1], range(self.nfac)):
            dict_list[t][f]["coeffs"] = initial_quantities["trans_coeffs"][f][t, :]
            dict_list[t][f]["included_positions"] = self.included_positions[f]
        return dict_list

    def _transform_sigma_points_args_dict(self, initial_quantities):
        tsp_args = {}
        tsp_args["transition_function_names"] = self.transition_names
        if self.anchor_in_predict is True:
            tsp_args["anchoring_positions"] = self.anch_positions
            tsp_args["anch_params"] = initial_quantities["h"][-1, :]
            if self.ignore_intercept_in_linear_anchoring is False:
                tsp_args["intercept"] = initial_quantities["delta"][-1][-1, 0:1]

        tsp_args["transition_argument_dicts"] = self._transition_equation_args_dicts(
            initial_quantities
        )
        return tsp_args

    def _predict_args_dict(self, initial_quantities):
        p_args = {}
        p_args["sigma_points"] = initial_quantities["sigma_points"]
        p_args["flat_sigma_points"] = initial_quantities["flat_sigma_points"]
        p_args["s_weights_m"], p_args["s_weights_c"] = self.sigma_weights()
        p_args["q"] = initial_quantities["q"]
        p_args["transform_sigma_points_args"] = self._transform_sigma_points_args_dict(
            initial_quantities
        )
        p_args["out_flat_states"] = initial_quantities["flat_x"]
        p_args["out_flat_covs"] = initial_quantities["flat_p"]
        return p_args

    def _calculate_sigma_points_args_dict(self, initial_quantities):
        sp_args = {}
        sp_args["states"] = initial_quantities["x"]
        sp_args["flat_covs"] = initial_quantities["flat_p"]
        sp_args["out"] = initial_quantities["sigma_points"]
        sp_args["square_root_filters"] = self.square_root_filters
        sp_args["scaling_factor"] = self.sigma_scaling_factor()
        return sp_args

    def likelihood_arguments_dict(self, params_type):
        """Construct a dict with arguments for the likelihood function."""
        initial_quantities = self._initial_quantities_dict()

        args = {}
        args["like_vec"] = np.ones(self.nobs)
        args["parse_params_args"] = self._parse_params_args_dict(initial_quantities)
        args["stagemap"] = self.stagemap
        args["nmeas_list"] = self.nmeas_list
        args["anchoring"] = self.anchoring
        args["square_root_filters"] = self.square_root_filters
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

    def simulate(self, nobs, params, params_type):
        """Simulate a dataset generated by the model at *params*.

        Args:
            nobs (int): number of observations to simulate
            params (np.array): parameters
            params_type (str): 'short' or 'long'

        Returns:
            observed_data (pd.DataFrame)
            latent_data (pd.DataFrame)

        """
        initial_quantities = self._initial_quantities_dict()
        pp_args = self._parse_params_args_dict(initial_quantities)

        parse_params(params, **pp_args)

        factor_names = self.factors

        for period in self.periods:
            assert self.controls[period] == self.controls[0], (
                'simulate only works if the same controls are used in each period.')

        control_names = self.controls[0]

        loadings_df = pd.DataFrame(
            data=initial_quantities['H'],
            columns=self.factors,
            index=self.update_info.index
        )
        deltas = initial_quantities['deltas']
        transition_names = self.transition_names
        base_trans_args = self._transition_equation_args_dicts(
            initial_quantities)
        transition_argument_dicts = []
        for t, s in enumerate(self.stagemap):
            transition_argument_dicts.append(base_trans_args[s])

        # todo: do I have standard deviations instead of variances in square-root case?
        base_shock_variances = np.diagonal(initial_quantities['Q'], axis1=1, axis2=2)

        shock_variances = []
        for t, s in enumerate(self.stagemap):
            shock_variances.append(base_shock_variances[s])

        meas_variances = pd.Series(
            data=initial_quantities['R'], index=self.update_info.index)
        if self.square_root_filters is True:
            meas_variances **= 2

        dist_name = 'multivariate_normal'

        X_zero = initial_quantities['X_zero']
        P_zero = initial_quantities['P_zero']

        dist_arg_dict = []
        for n in range(self.nemf):
            factor_mean = X_zero[0, n]
            control_mean = np.zeros(len(control_names))

            factor_cov = P_zero[0, n]
            if self.square_root_filters is True:
                factor_cov = factor_cov[1:, 1:]
                factor_cov = np.dot(factor_cov.T, factor_cov)
            nfac = len(factor_cov)
            dim = nfac + len(control_names)
            full_cov = np.eye(dim) * 0.9 + np.ones((dim, dim)) * 0.1
            full_cov[:nfac, :nfac] = factor_cov
            d = {
                "mean": np.hstack([factor_mean, control_mean]),
                "cov": full_cov
            }
            dist_arg_dict.append(d)
        weights = np.ones(self.nemf)

        observed_data, latent_data = simulate_datasets(
            factor_names=factor_names,
            control_names=control_names,
            nobs=nobs,
            nper=self.nperiods,
            transition_names=transition_names,
            transition_argument_dicts=transition_argument_dicts,
            shock_variances=shock_variances,
            loadings_df=loadings_df,
            deltas=deltas,
            meas_variances=meas_variances,
            dist_name=dist_name,
            dist_arg_dict=dist_arg_dict,
            weights=weights
        )

        return observed_data, latent_data

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

    def estimate_params_wa(self):
        """Estimate the params vector with wa."""
        storage_df, X_zero, P_zero, trans_coeffs, trans_var_df, anch_intercept, anch_loadings, anch_variance = calculate_wa_estimated_quantities(
            self.identified_restrictions,
            self.y_data,
            self.measurements,
            self.normalizations,
            self.storage_df,
            self.factors,
            self.transition_names,
            self.included_factors,
            self.nstages,
            self.stages,
            self.periods,
            self.stagemap,
            self.anchored_factors,
            self.anch_outcome,
            self.wa_period_weights,
            self.anchoring,
        )

        params = np.zeros(self.len_params(params_type="long"))
        slices = self.params_slices(params_type="long")

        # write intercepts in params
        delta_start_index = slices["deltas"][0].start
        delta_stop_index = slices["deltas"][-1].stop
        all_intercepts = list(
            storage_df[storage_df["has_normalized_intercept"] == False][
                "intercepts"
            ].to_numpy()
        )
        if anch_intercept is not None:
            all_intercepts.append(anch_intercept)
        params[delta_start_index:delta_stop_index] = all_intercepts
        # write loadings in params
        all_loadings = list(
            storage_df[storage_df[
                "has_normalized_loading"] == False]["loadings"].to_numpy()
        )
        if anch_loadings is not None:
            all_loadings += list(anch_loadings)
        params[slices["H"]] = all_loadings

        # write measurement variances in params
        all_meas_variances = list(storage_df["meas_error_variances"].to_numpy())
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

        identifiers = self.data['__id__'].unique()
        for item in self.bootstrap_samples:
            assert set(item).issubset(identifiers), (
                "The bootstrap_samples you provided contain person_identifiers"
                " which are not in the dataset {}. These missing identifiers "
                "are {}.\nYou specified {} as person_identifier. This error "
                "occurred in model {}".format(
                    self.dataset_name,
                    [i for i in item if i not in identifiers],
                    '__id__',
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
        individuals = np.array(self.data['__id__'].unique())
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
            ['__id__', '__period__'], drop=False
        )
        current_sample = self.bootstrap_samples[rep]
        bs_index = pd.MultiIndex.from_product(
            [current_sample, self.periods],
            names=['__id__', '__period__'],
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
                    "success": "Not started because the resulting " +
                    "parameter vector would have had the wrong length."
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
            anch_func = "anchor_flat_sigma_points_linear"
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

        anchored = self._anchor_final_factors(final_factors, anch_loadings)
        selector = anch_loadings.astype(bool).astype(int)
        anch_outcome = np.dot(anchored, selector) + anch_intercept
        return anch_outcome

    def _marginal_effect_outcome(self, change):

        final_factors = self._predict_final_factors(change)

        if self.anchoring is True:
            # construct_anch_loadings and anch_intercept
            slices = self.params_slices("long")
            relevant_params = self.me_params[slices["H"]][-len(self.anchored_factors):]

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
                other_vars=[group, '__period__'],
                agg_method=agg_method,
            )
            to_concat.append(df)

        data = pd.concat(to_concat, axis=0, sort=True)

        fig, ax = plt.subplots(figsize=figsize)
        sns.pointplot(
            x='__period__',
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
