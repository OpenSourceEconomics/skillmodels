import warnings

import numpy as np
import pandas as pd


def prepend_index_level(df, to_prepend):
    df = df.copy()
    df.index = pd.MultiIndex.from_tuples([(to_prepend, x) for x in df.index])
    return df


def check_value_errors(varlist, data):
    report_df = pd.DataFrame(columns=["period", "variable", "issue"])
    t = data["__period__"].unique()[0]
    for ind, var in enumerate(varlist):
        len_val = len(data[var].value_counts())
        if len_val == 0:
            report_df.loc[ind] = [t, var, "All values are missing"]
        elif len_val == 1:
            report_df.loc[ind] = [
                t,
                var,
                "All non missing values are equal to {}".format(
                    data[var].unique()[~np.isnan(data[var].unique())][0]
                ),
            ]
    return report_df


def pre_process_data(df):
    """Transform an unbalanced and unsorted dataset.

    Args:
        df (DataFrame): panel dataset in long format. It has a MultiIndex
            where the first level indicates the period and the second
            the individual.

    Returns:
        balanced (DataFrame): balanced panel. It has a MultiIndex. The first
            level is called __id__ and enumerates individuals. The second
            level is called __period__ and counts periods, starting at 0.

    """
    assert (
        "__id__" not in df.columns
    ), "The variable name __id__ is used internally and must not occur."

    assert (
        "__period__" not in df.columns
    ), "The variable name __period__ is used internally and must not occur."

    df = df.sort_index()
    all_ids, all_periods = list(df.index.levels[0]), list(df.index.levels[1])
    nobs = len(all_ids)
    nperiods = len(all_periods)

    new_period = np.tile(np.arange(nperiods), nobs)
    old_period = all_periods * nobs
    old_id = np.repeat(all_ids, nperiods)
    new_id = np.arange(nobs).repeat(nperiods)

    balanced = pd.DataFrame(
        data=np.column_stack([old_id, old_period, new_id, new_period]),
        columns=df.index.names + ["__id__", "__period__"],
    )

    balanced.set_index(df.index.names, inplace=True)
    balanced = pd.concat([balanced, df], axis=1)
    balanced.set_index(["__id__", "__period__"], inplace=True, drop=False)

    return balanced


class DataProcessor:
    """Transform a pandas DataFrame in long format into numpy arrays."""

    def __init__(self, specs_processor_attribute_dict):
        self.__dict__.update(specs_processor_attribute_dict)
        self._check_observable_data()

    def c_data(self):
        """A List of 2d arrays with control variables for each period.

        The arrays are of the shape[nind, number of control variables in t].

        """
        c_data = []
        self.data["constant"] = 1.0
        const_list = ["constant"]

        for t in self.periods:
            df = self.data[self.data["__period__"] == t]
            arr = df[const_list + list(self.controls[t])].to_numpy()
            c_data.append(arr)
        return c_data

    def y_data(self):
        """A 2d numpy array that holds the measurement variables.

        The array is of shape [nupdates, nind].

        """
        dims = (self.nupdates, self.nobs)
        y_data = np.zeros(dims)

        counter = 0
        for t in self.periods:
            measurements = list(self.update_info.loc[t].index)
            df = self.data[self.data["__period__"] == t][measurements].copy(deep=True)
            missing_index = self.missing_controls[t][self.missing_controls[t]].index
            num_missing = len(missing_index)

            if num_missing > 0:
                warnings.warn(
                    f"In model {self.model_name}, period {t} the measurements of "
                    f"{num_missing} observations have been set to np.nan because a "
                    "control variable is missing and controls_with_missings is set to "
                    "drop_observations.\n"
                )
                df.loc[missing_index] = np.nan

            y_data[counter : counter + len(measurements), :] = df.to_numpy().T
            counter += len(measurements)
        return y_data

    def measurements_df(self, periods="all", factors="all", other_vars=None):
        other_vars = [] if other_vars is None else other_vars
        if periods == "all":
            periods = self.periods

        if factors == "all":
            factors = self.factors

        if isinstance(periods, int) or isinstance(periods, float):
            periods = [periods]

        if isinstance(factors, str):
            factors = [factors]

        if len(other_vars) == 0 or isinstance(other_vars[0], str):
            other_vars = [other_vars] * len(periods)

        other_vars_dict = {}
        for i, period in enumerate(periods):
            other_vars_dict[period] = other_vars[i]

        period_dfs = []
        for period in periods:
            relevant_variables = ["__id__"]
            for factor in factors:
                for meas in self.measurements[factor][period]:
                    if meas not in relevant_variables:
                        relevant_variables.append(meas)
            relevant_variables += other_vars_dict[period]
            df = self.data[self.data["__period__"] == period]
            df = df[relevant_variables].set_index("__id__")
            period_dfs.append(df)

        if len(period_dfs) == 1:
            measurements_df = period_dfs[0]
        else:
            for period, df in enumerate(period_dfs):
                rename_dict = {col: col + f"_{period}" for col in df.columns}
                df.rename(columns=rename_dict, inplace=True)

            measurements_df = pd.concat(period_dfs, axis=1)

        return measurements_df

    def score_df(
        self,
        periods="all",
        factors="all",
        other_vars=None,
        agg_method="mean",
        order="by_factor",
    ):
        other_vars = [] if other_vars is None else other_vars
        if periods == "all":
            periods = self.periods

        if factors == "all":
            factors = self.factors

        if isinstance(periods, int) or isinstance(periods, float):
            periods = [periods]

        if isinstance(factors, str):
            factors = [factors]

        if len(other_vars) == 0 or isinstance(other_vars[0], str):
            other_vars = [other_vars] * len(periods)

        other_vars_dict = {}
        for i, period in enumerate(periods):
            other_vars_dict[period] = other_vars[i]

        relevant_uinfo = self.update_info.loc[periods, factors]
        assert (
            relevant_uinfo.sum(axis=1) <= 1
        ).all(), "score_df only works with dedicated measurement systems."

        trans_name_dict = {}
        for f, factor in enumerate(self.factors):
            trans_name_dict[factor] = self.transition_names[f]

        period_dfs = {}
        for period in periods:
            to_concat = []
            for factor in factors:

                if trans_name_dict[factor] == "constant":
                    meas_df = self.measurements_df(factors=factor)
                else:
                    meas_df = self.measurements_df(periods=period, factors=factor)
                if agg_method == "mean":
                    score_sr = meas_df.mean(axis=1)
                else:
                    scaled = (meas_df - meas_df.mean()) / meas_df.std()
                    score_sr = scaled.mean(axis=1)
                    if agg_method == "norm_scaled":
                        raise NotImplementedError

                score_sr.name = factor
                to_concat.append(score_sr)
            to_concat.append(
                self.measurements_df(periods=period, factors=[], other_vars=other_vars)
            )

            period_dfs[period] = pd.concat(to_concat, axis=1)

        to_concat = []
        for period in periods:
            df = period_dfs[period]
            if len(periods) > 1:
                rename_dict = {col: col + f"_{period}" for col in df.columns}
                df.rename(columns=rename_dict, inplace=True)
            to_concat.append(df)
        score_df = pd.concat(to_concat, axis=1)

        if isinstance(score_df, pd.Series):
            score_df = score_df.to_frame()

        if order == "by_factor" and len(to_concat) > 1:
            ordered_columns = []
            for factor in factors:
                for period in periods:
                    ordered_columns.append(f"{factor}_{period}")
            for col in score_df.columns:
                if col not in ordered_columns:
                    ordered_columns.append(col)
            score_df = score_df[ordered_columns]

        return score_df

    def reg_df(self, factor, period=None, stage=None, controls=None, agg_method="mean"):
        controls = [] if controls is None else controls
        assert (
            period is None or stage is None
        ), "You cannot specify a period and a stage for a score regression."

        assert not (
            period is None and stage is None
        ), "You have to specify a period or a stage for a score regression"

        if period is not None:
            assert (
                period in self.periods[:-1]
            ), "The score regression is not possible in the last period."

        if stage is not None:
            periods = [p for p in self.periods[:-1] if self.stagemap[p] == stage]
        else:
            periods = [period]

        ind = self.factors.index(factor)
        included = self.included_factors[ind]

        period_dfs = []
        for p in periods:
            df_old = self.score_df(
                periods=p, factors=included, other_vars=controls, agg_method=agg_method
            )
            df_old.rename(mapper=lambda x: x + "_t", axis=1, inplace=True)

            df_new = self.score_df(periods=p + 1, factors=factor, agg_method=agg_method)
            df_new.rename(mapper=lambda x: x + "_t_plusone", axis=1, inplace=True)
            df = pd.concat([df_new, df_old], axis=1)

            df = prepend_index_level(df, p)

            period_dfs.append(df)

        reg_df = pd.concat(period_dfs, axis=0)
        return reg_df

    def _check_observable_data(self):
        report_issues = pd.DataFrame()
        for t in self.periods:
            df = self.data.query(f"__period__ == {t}")
            # create time relevant measurements list
            meas_t = []
            for fac in self.factors:
                meas_t.append(self.measurements[fac][t])
            # flatten
            meas_t = {item for sublist in meas_t for item in sublist}
            varlist = list(self.controls[t]) + list(meas_t)
            to_concat = check_value_errors(varlist, df)
            report_issues = pd.concat([report_issues, to_concat])
        report_issues=report_issues.reset_index(drop=True)
        if len(report_issues) != 0:
            raise ValueError(f"Invalid dataset:\n{report_issues}")
