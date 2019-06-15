"""Functions to simulate a dataset generated by a latent factor model."""
import pandas as pd
import numpy as np
from numpy.random import multivariate_normal, choice, binomial

import skillmodels.model_functions.transition_functions as tf
import skillmodels.simulation._elliptical_functions as ef


def add_missings(data, meas_names, p_b, p_r):
    """Add np.nans to data.

    nans are only added to measurements, not to control variables or factors.

    The function does not modify data in place. (create a new one)

    Args:
        data (pd.DataFrame): contains the observable part of a simulated dataset
        meas_names (list): list of strings of names of each measurement variable
        p_b (float): probability of a measurement to become missing
        p_r (float): probability of a measurement to remain missing in the next period
    Returns:
        data_with_missings (pd.DataFrame): Dataset with a share of measurements
        replaced by np.nan values

    Notes:
        - Time_periods should be sorted for each individual
        - p is NOT the marginal probability of a measurement being missing.
          The marginal probability is given by: p_m = p/(1-serial_corr), where
          serial_corr = (p_r-p_b) in general != 0, since p_r != p_b. This means that in
          average the share of missing values (in the entire dataset) will be larger
          than p. Thus, p and q should be set accordingly given the desired share
          of missing values.
        - I would still suggest to draw period_0 bernoulli with the marginal
          probaility. Having run the function in a loop with 100 iteration, for
          100 pairs of p_r and p_b , the average share of missing measurements in all
          measurements is very close to p_b/(1-(p_r-p_b)) (the average percentage
          deviation is less than 2 %). Sounds like a nice result and a simple
          formula for deciding on p_b and p_r.

    """
    nmeas = len(meas_names)
    data_with_missings = data.copy(deep=True)
    for i in set(data_with_missings.index):
        ind_data = data_with_missings.loc[i][meas_names].to_numpy()
        s_0 = binomial(1, p_b, nmeas)
        ind_data[0, np.where(s_0 == 1)] = np.nan
        for t in range(1, len(ind_data)):
            indc_nan = np.isnan(ind_data[t - 1])
            prob = p_r * indc_nan + p_b * (1 - indc_nan)
            s_m = binomial(1, prob)
            ind_data[t, np.where(s_m == 1)] = np.nan
        data_with_missings.loc[i, meas_names] = ind_data

    return data_with_missings


def simulate_datasets(
    factor_names,
    control_names,
    nobs,
    nper,
    transition_names,
    transition_argument_dicts,
    shock_variances,
    loadings_df,
    deltas,
    meas_variances,
    dist_name,
    dist_arg_dict,
    weights,
    policies=None
):
    """Simulate datasets generated by a latent factor model.

    This function calls the remaining functions in this module.

    Args:
        nper (int): number of time periods the dataset contains
        nobs (int): number of observations
        factor_names (list): list of strings of names of each factor
        control_names (list): list of strings of names of each control variable
        loadings_df (pd.DataFrame): The factor loadings. It has a multi
            index where the first level indicates the period and the second one
            the variable. The columns are the names of the factors.
        deltas (list): list of numpy array of size (nmeas, ncontrols). The list has
            length nper.
        transition_names (list): list of strings with the names of the transition
           function of each factor.
        transition_argument_dicts (list): list lists of dictionaries. Each sublsit has
            length nfac with and contanis the arguments for the transition function of
            each factor. There is one sublist for each period.
        shock_variances (np.ndarra): numpy array of size (nper, nfac) with the shock
            variances in each period.
        meas_variances (pd.Series): The index is the same as in loadings_df. The data
            are the variances of the measurements errors.
        dist_name (string): the elliptical distribution to use in the mixture
        dist_arg_dict (list or dict): list of length nemf of dictionaries with the
            relevant arguments of the mixture distributions. Arguments with default
            values should NOT be included in the dictionaries. Lengths of arrays in the
            arguments should be in accordance with nfac + ncont
        weights (np.ndarray): size (nemf). The weight of each mixture element.
        policies (list): list of dictionaries. Each dictionary specifies a
            a stochastic shock to a latent factor AT THE END of "period" for "factor"
            with mean "effect_size" and "standard deviation"

    Returns:
        observed_data (pd.DataFrame): Dataset with measurements and control variables
            in long format
        latent_data (pd.DataFrame): Dataset with lantent factors in long format
    Notes:
        - the key names of dist_arg_dict can be looked up in the module
          _elliptical_functions. For multivariate_normal it's [mean, cov].
    """
    ncont = len(control_names)
    nfac = len(factor_names)
    fac = np.zeros((nper, nobs, nfac))
    if policies is None:
        fac[0], cont = generate_start_factors_and_control_variables_elliptical(
            nobs, nfac, ncont, dist_name, dist_arg_dict, weights
        )

        cont = pd.DataFrame(data=cont, columns=['constant'] + control_names)

        for t in range(nper - 1):
            fac[t + 1] = next_period_factors(
                fac[t], transition_names, transition_argument_dicts[t], shock_variances[t]
            )
    else:
        def add_effect(fac, effect, sd, nobs):
            """function to add stochastic effects to a
            vertical factor vector of lenght nobs

            """
            if sd == 0:
                fac += effect
            elif sd > 0:
                rand_effects = np.random.normal(effect, sd, nobs)
                fac += rand_effects 
            else:
                raise ValueError('negative sd')
            return fac

        fac[0], cont = generate_start_factors_and_control_variables_elliptical(
            nobs, nfac, ncont, dist_name, dist_arg_dict, weights
        )
        for d in policies:
            period = d['period']
            if period == 0:
                factor = d['factor']
                for i in range (0,nfac):
                    if factor == factor_names[i]:
                        effect = d['effect_size']
                        sd = d['standard_deviation']
                        fac[0,:,i] = add_effect(fac[0,:,i], effect, sd, nobs)
                    else:
                        pass
            else:
                pass

        cont = pd.DataFrame(data=cont, columns=['constant'] + control_names)

        for t in range(nper - 1):
            fac[t + 1] = next_period_factors(
                fac[t], transition_names, transition_argument_dicts[t], shock_variances[t]
            )
            for d in policies:
                period = d['period']
                if period == t+1:
                    factor = d['factor']
                    for i in range(nfac):
                        if factor == factor_names[i]:
                            effect = d['effect_size']
                            sd = d['standard_deviation']
                            fac[t+1,:,i] = add_effect(fac[t+1,:,i], effect, sd, nobs)
                        else:
                            pass
                else:
                    pass 
    observed_data_by_period = []
    for t in range(nper):
        meas = pd.DataFrame(
            data=measurements_from_factors(
                fac[t],
                cont.to_numpy(),
                loadings_df.loc[t].to_numpy(),
                deltas[t],
                meas_variances.loc[t].to_numpy(),
            ),
            columns=loadings_df.loc[t].index,
        )
        meas['period'] = t
        observed_data_by_period.append(pd.concat([meas, cont], axis=1))

    observed_data = pd.concat(observed_data_by_period, axis=0, sort=True)
    observed_data['id'] = observed_data.index
    observed_data.sort_values(['id', 'period'], inplace=True)
    observed_data.set_index(['id', 'period'], inplace=True)

    latent_data_by_period = []
    for t in range(nper):
        lat = pd.DataFrame(data=fac[t], columns=factor_names)
        lat['period'] = t
        latent_data_by_period.append(lat)

    latent_data = pd.concat(latent_data_by_period, axis=0, sort=True)
    latent_data['id'] = latent_data.index
    latent_data.sort_values(['id', 'period'], inplace=True)
    latent_data.set_index(['id', 'period'], inplace=True)

    return observed_data, latent_data


def generate_start_factors_and_control_variables_elliptical(
    nobs, nfac, ncont, dist_name, dist_arg_dict, weights=1
):
    """Draw initial states and control variables from a (mixture of) normals.

    Args:
        nobs (int): number of observations
        nfac (int): number of factor (latent) variables
        ncont (int): number of control variables
        dist_name (string): the elliptical distribution to use in the mixture
        dist_arg_dict (list or dict): list of length nemf of dictionaries with the
          relevant arguments of the mixture distributions. Arguments with default
          values should NOT be included in the dictionaries. Lengths of arrays in the
          arguments should be in accordance with nfac + ncont
        weights (np.ndarray): size (nemf). The weight of each mixture element.
                              Default value is equal to 1.

    Returns:
        start_factors (np.ndarray): shape (nobs, nfac),
        controls (np.ndarray): shape (nobs, ncontrols),

    """
    if np.size(weights) == 1:
        out = getattr(ef, dist_name)(size=nobs, **dist_arg_dict[0])
    else:
        helper_array = choice(np.arange(len(weights)), p=weights, size=nobs)
        out = np.zeros((nobs, nfac + ncont))
        for i in range(nobs):
            out[i] = getattr(ef, dist_name)(**dist_arg_dict[helper_array[i]])
    start_factors = out[:, 0:nfac]
    controls = out[:, nfac:]
    controls = np.hstack([np.ones((nobs, 1)), controls])

    return start_factors, controls


def next_period_factors(
    factors, transition_names, transition_argument_dicts, shock_variances
):
    """Apply transition function to factors and add shocks.

    Args:
        factors (np.ndarray): shape (nobs, nfac)
        transition_names (list): list of strings with the names of the transition
            function of each factor.
        transition_argument_dicts (list): list of dictionaries of length nfac with
            the arguments for the transition function of each factor. A detailed
            description of the arguments of transition functions can be found in the
            module docstring of skillmodels.model_functions.transition_functions.
        shock_variances (np.ndarray): numpy array of length nfac.

    Returns:
        next_factors (np.ndarray): shape(nobs,nfac)

    """
    nobs, nfac = factors.shape
    # sigma_points = factors
    factors_tp1 = np.zeros((nobs, nfac))
    for i in range(nfac):
        factors_tp1[:, i] = getattr(tf, transition_names[i])(
            factors, **transition_argument_dicts[i]
        )
    # Assumption: In general err_{Obs_j,Fac_i}!=err{Obs_k,Fac_i}, where j!=k
    errors = multivariate_normal([0] * nfac, np.diag(shock_variances), nobs).reshape(
        nobs, nfac
    )
    next_factors = factors_tp1 + errors

    return next_factors


def measurements_from_factors(factors, controls, loadings, deltas, variances):
    """Generate the variables that would be observed in practice.

    This generates the data for only one period. Let nmeas be the number
    of measurements in that period.

    Args:
        factors (pd.DataFrame or np.ndarray): DataFrame of shape (nobs, nfac)
        controls (pd.DataFrame or np.ndarray): DataFrame of shape (nobs, ncontrols)
        loadings (np.ndarray): numpy array of size (nmeas, nfac)
        deltas (np.ndarray): numpy array of size (nmeas, ncontrols)
        variances (np.ndarray): numpy array of size (nmeas) with the variances of the
            measurements. Measurement error is assumed to be independent across
            measurements

    Returns:
        measurements (np.ndarray): array of shape (nobs, nmeas) with measurements.
    """
    nmeas = loadings.shape[0]
    nobs, nfac = factors.shape
    # Assumption: In general eps_{Obs_j,Meas_i}!=eps_{Obs_k,Meas_i}  where j!=k
    epsilon = multivariate_normal([0] * nmeas, np.diag(variances), nobs)
    states = factors
    conts = controls
    states_part = np.dot(states, loadings.T)
    control_part = np.dot(conts, deltas.T)
    meas = states_part + control_part + epsilon
    return meas
