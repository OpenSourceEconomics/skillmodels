""" Contains functions for simulating random vectors of arbytrary size from:
    - multivariate student's t
    - multivariate symmetric stable (based on Nolan (2018) and Nolan (2013))
    - calls multivariate normal from np.random to be able to use with getattr() 
      in simulate_data

"""
import numpy as np
from numpy.random import multivariate_normal


def _mv_student_t(mean, cov, d_f, size=1):
    """Generate random sample from d-dimensional t_distribution
    Args:
        mean (np.ndarray): vector of mean of size d
        cov (np.ndarray): covariance matrix of shape (d,d)
        d_f (float): degree of freedom
        size (float): the sample size
    Returns:
        mv_t (np.ndarray): shape (size, d)
    Notes:
       - Cov is the variance-covariance of the normal component.
       - So the cov matrix of the resulting t will be cov*d_f/(d_f-2)
       - Ref: bit.ly/2NDhbWM

    """
    d_dim = len(cov)
    x_chi = np.sqrt(np.random.chisquare(d_f, size) / d_f).reshape(size, 1, 1)
    y_norm = multivariate_normal(np.zeros(d_dim), cov, size).reshape(size, 1, d_dim)
    mv_t = mean + (y_norm / x_chi).reshape(size, d_dim)
    return mv_t


def _uv_elip_stable(alpha, gamma, delta=0, beta=1, size=1):
    """An algorithm to for simulating random variables from stable 
    
    distribution.
    Args:
        alpha (float): measure of concentration
        gamma (float): scale parameter      
        delta (float): location parameter
        beta (float): measure of skewness.
        size (float): sample size
    Returns:
        stable_u (np.ndarray): S_1(alpha, beta, gamma, delta) random vector of length size
    Notes:
       
       - ref: [1] Chambers et al., 1976 , [2] Nolan, 2018 [3] Weron, 1995
       - to be used in _mv_elip_stable
       - This is the general case. For the purpose of generating from
         a multivariate elliptically contoured (symmetric) stabel rv would suffice 
         to set beta = 1  and restrict alpha < 1 (strictly).
       - the extreme skewness of the univariate_stable component creates the heave tails of 
         the multivariate in the multivariate distribution.
    """
    theta = np.random.uniform(-np.pi / 2, np.pi / 2, size)
    w_exp = np.random.exponential(1, size)
    if alpha == 1:
        comp_1 = np.tan(theta) * (0.5 * np.pi + beta * theta)
        comp_2 = -beta * np.log(
            0.5 * w_exp * np.cos(theta) / (0.5 * np.pi + beta * theta)
        )
        zeta = np.power(0.5 * np.pi, -1) * (comp_1 + comp_2)
        stable_u = gamma * zeta + (
            delta + beta * np.power(0.5 * np.pi, -1) * gamma * np.log(gamma)
        )
    else:
        theta_0 = (
            np.arctan(beta * np.tan(0.5 * np.pi * alpha)) / alpha
        )  #!!! Don't know if alpha is inside arctan or artcan is divided by alpha
        theta_0 = np.ones(size) * theta_0
        z_1 = np.sin(alpha * (theta_0 + theta)) / np.power(
            (np.cos(theta) * np.cos(alpha * theta_0)), (1 / alpha)
        )
        z_2 = np.power(
            (np.cos((alpha - 1) * theta + alpha * theta_0) / w_exp), (1 / alpha - 1)
        )
        zeta = z_1 * z_2
        stable_u = gamma * zeta + delta

    return stable_u


def _mv_elip_stable(alpha, sigma_mat, delta, size=1):
    """An algorithm to generate d-dimensional multivariate elliptically contoured stable rv
     Args:
        alpha (float): measure of concentration strictly between 0 and 2
        sigma_mat (np.ndarray): positive definite matrix of shape (d,d)
        delta (np.ndarray): shift vector of size d
     Returns:
        stable_m (np.ndarray): rv of shape (size, d)
     Notes:
       - This is a special symmetric case of mv stable ([1]) 
       - ref: [1] bit.ly/2XyEOUX, [2] Nolan, 2013, [3] Teimouri et al, 2018
     """

    a_stab = _uv_elip_stable(
        0.5 * alpha, 2 * np.power(np.cos(np.pi * alpha / 4), (2 / alpha)), size=size
    ).reshape(size, 1)
    g_norm = multivariate_normal(np.zeros(len(sigma_mat)), sigma_mat, size)
    stable_m = np.sqrt(a_stab) * g_norm + delta
    return stable_m
