from skillmodels.pre_processing.model_spec_processor import ModelSpecProcessor
from skillmodels.pre_processing.data_processor import DataProcessor
from skillmodels.estimation.likelihood_function import log_likelihood_per_individual
from skillmodels.visualization.table_functions import (
    statsmodels_results_to_df,
    df_to_tex_table,
)
from skillmodels.visualization.text_functions import (
    title_text,
    write_figure_tex_snippet,
    get_preamble,
)
import statsmodels.formula.api as smf

import numpy as np
import skillmodels.model_functions.transition_functions as tf
from skillmodels.estimation.parse_params import parse_params

from itertools import product
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join
from skillmodels.simulation.simulate_data import simulate_datasets
from estimagic.optimization.start_helpers import make_start_params_helpers
from estimagic.optimization.start_helpers import get_start_params_from_free_params
from estimagic.optimization.optimize import maximize
from skillmodels.pre_processing.constraints import add_bounds


class SkillModel:
    """Estimate dynamic nonlinear latent factor models.

    Its usage is described in :ref:`basic_usage`.
    When initialized, all public attributes of ModelSpecProcessor and the
    arrays with c_data and y_data from DataProcessor are set as attributes.

    Args:
        model_dict (dict): see :ref:`basic_usage`.
        dataset (DataFrame): datset in long format. see :ref:`basic_usage`.
        model_name (str): optional. Used to make error messages readable.
        dataset_name (str): same as model_name
        save_path (str): specifies where intermediate results are saved.

    """

    def __init__(
        self,
        model_dict,
        dataset,
        model_name="some_model",
        dataset_name="some_dataset",
    ):
        specs = ModelSpecProcessor(
            model_dict=model_dict,
            dataset=dataset,
            model_name=model_name,
            dataset_name=dataset_name,
        )
        specs_dict = specs.public_attribute_dict()
        data_proc = DataProcessor(specs_dict)
        self.data_proc = data_proc
        self.c_data = data_proc.c_data()
        self.y_data = data_proc.y_data()
        self.__dict__.update(specs_dict)

        # create a list of all quantities that depend from params vector
        self.params_quants = ["delta", "h", "r", "q", "p", "trans_coeffs"]
        if self.estimate_X_zeros is True:
            self.params_quants.append("x")
        if self.restrict_W_zeros is False:
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

    def start_params_helpers(self):
        """DataFrames with the free and fixed parameters of the model."""
        free, fixed = make_start_params_helpers(
            self.params_index, self.constraints)
        return free, fixed

    def generate_full_start_params(self, start_params):
        """Vector with start values for the optimization.

        If valid start_params are provided in the model dictionary, these will
        be used. Else, naive start_params are generated.

        """
        free, fixed = self.start_params_helpers()
        if start_params is not None:
            sp = start_params
            assert isinstance(sp, pd.DataFrame)
            if len(sp.index.intersection(self.params_index)) == len(self.params_index):
                full_sp = sp

            elif len(sp.index.intersection(free.index)) == len(free.index):
                full_sp = get_start_params_from_free_params(
                    sp, self.constraints, self.params_index)
            else:
                raise ValueError(
                    "Index of start parameters has to be either self.params_index or "
                    "the index of free parameters from start_params_helpers.")
        else:
            free['value'] = 0.0
            free.loc['h', 'value'] = 1.0
            free.loc['r', 'value'] = 1.0
            free.loc['q', 'value'] = 1.0
            free.loc['trans', 'value'] = 1 / self.nfac
            free.loc['w', 'value'] = 1 / self.nemf
            for emf in range(self.nemf):
                p_diags = [("p", 0, emf, f"{fac}-{fac}") for fac in self.factors]
                free.loc[p_diags, 'value'] = 1

            full_sp = get_start_params_from_free_params(
                free, self.constraints, self.params_index)

        full_sp["group"] = None

        full_sp = add_bounds(full_sp, bounds_distance=self.bounds_distance)
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

    def likelihood_arguments_dict(self):
        """Construct a dict with arguments for the likelihood function."""
        initial_quantities = self._initial_quantities_dict()

        args = {}
        args["like_vec"] = np.ones(self.nobs)
        args["parse_params_args"] = self._parse_params_args_dict(initial_quantities)
        args["periods"] = self.periods
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

    def simulate(self, nobs, params):
        """Simulate a dataset generated by the model at *params*.

        Args:
            nobs (int): number of observations to simulate
            params (np.array): parameters

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
            data=initial_quantities['h'],
            columns=self.factors,
            index=self.update_info.index
        )
        deltas = initial_quantities['delta']
        transition_names = self.transition_names
        transition_argument_dicts = self._transition_equation_args_dicts(
            initial_quantities)

        shock_variances = np.diagonal(initial_quantities['q'], axis1=1, axis2=2)

        meas_variances = pd.Series(
            data=initial_quantities['r'], index=self.update_info.index)
        if self.square_root_filters is True:
            meas_variances **= 2

        dist_name = 'multivariate_normal'

        X_zero = initial_quantities['x']
        P_zero = initial_quantities['p']

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

    def fit(self, start_params=None, params=None):
        """Fit the model and return an instance of SkillModelResults."""
        args = self.likelihood_arguments_dict()
        start_params = self.generate_full_start_params(start_params)

        def criterion(params, args):
            like_vec = log_likelihood_per_individual(params, **args)
            return like_vec.sum()

        db_options = {"rollover": 1000}

        res = maximize(
            criterion,
            start_params,
            constraints=self.constraints,
            algorithm='scipy_L-BFGS-B',
            criterion_args=(args, ),
            dashboard=False,
        )
        return res

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
