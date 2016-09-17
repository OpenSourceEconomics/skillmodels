.. _model_specs:

********************
Model specifications
********************

Models are specified as `Python dictionaries <http://introtopython.org/dictionaries.html>`_. To improve reuse of the model specifications these dictionaries should be stored in `JSON <http://www.json.org/>`_ files. Users who are unfamiliar with `JSON <http://www.json.org/>`_ and `Python dictionaries <http://introtopython.org/dictionaries.html>`_ should read up on these topics first.

Example 2 from the CHS replication files
****************************************

Below the model-specification is illustrated using Example 2 from the CHS `replication files`_. If you want you can read it in section 4.1 of their readme file but I briefly reproduce it here for convenience.

Suppose that there are three factors fac1, fac2 and fac3 and that there are 8 periods that all belong to the same development stage. fac1 evolves according to a log_ces production function and depends on its past values as well as the past values of all other factors. Moreover, it is linearly anchored with anchoring outcome Q1. This results in the following transition equation:

.. math::

    fac1_{t + 1} = \frac{1}{\phi \lambda_1} ln\big(\gamma_{1,t}e^{\phi \lambda_1 fac1_t} + \gamma_{2,t}e^{\phi \lambda_2 fac2_t} + \gamma_{3,t}e^{\phi \lambda_3 fac3_t}\big) + \eta_{1, t}

where the lambdas are anchoring parameters from a linear anchoring equation. fac1 is measured by measurements y1, y2 and y3 in all periods.  To sum up: fac1 has the same properties as cognitive and non-cognitive skills in the CHS paper.

The evolution of fac2 is described by a linear function and fac2 only depends on its own past values, not on other factors, i.e. has the following transition equation:

.. math::

    fac2_{t + 1} = lincoeff \cdot fac2_t + \eta_{2, t}

It is measured by y4, y5 and y6 in all periods. Thus fac2 has the same properties as parental investments in the CHS paper.

fac3 is constant over time. It is measured by y7, y8 and y9 in the first period and has no measurements in other periods. This makes it similar to parental skills in the CHS paper.

In all periods and for all measurement equations the control variables x1 and x2 are used, where x2 is a constant.

What has to be specified?
*************************

Before thinking about how to translate the above example into a model specification it is helpful to recall what information is needed to define a general latent factor model:

    #. What are the latent factors of the model and how are they related over time? (transition equations)
    #. What are the measurement variables of each factor in each period and how are measurements and factors related? (measurement equations)
    #. What are the normalizations of scale (normalized factor loadings) and location (normalized intercepts or means)?
    #. What are the control variables in each period?
    #. If development stages are used: Which periods belong to which stage?
    #. If anchoring is used: Which factors are anchored and what is the anchoring outcome?
    #. If endogeneity correction is used: Which factor is endogenous and what is the correction function?

Translate the example to a model dictionary
*******************************************

The first three points have to be specified for each latent factor of the model. This is done by adding a subdictionary for each latent factor to the ``factor_specific`` key of the model dictionary. In the example model from the CHS `replication files`_ the dictionary for the first factor takes the following form:

.. literalinclude:: test_model2.json
    :lines: 2-28

The value that corresponds to the ``measurements`` key is a list of lists. It has one sublist for each period of the model. Each sublist consists of the measurement variables of ``fac1`` in the corresponding period. Note that in the example the measurement variables are the same in each period. However, this is extremely rare in actual models. If a measurement measures more than one latent factor, it simply is included in the ``measurements`` list of all factors it measures. As in the Fortran code by CHS it is currently only possible to estimate models with linear measurement system. I plan to add the possibility to incorporate binary variables in a way similar to a probit model but have currently no plans to support arbitrary nonlinear measurement systems.

.. Note:: If the WA estimator is used, factors with constant transition equation can only have measurements in the initial period and all other factors need at least two measurements per period. If the CHS estimator is used these limitations do not hold as long as there are enough measurements to identify the model. If development stages span more than one period, models can be identified even if some factors have no measurements in some periods.

The value that corresponds to the ``normalizations`` key is a dictionary in which the normalizations for factor loadings and intercepts are specified. Its values (for each type of normalization) are lits of lists with one sublist per period. Each sublist has length 2. Its first entry is the name of the measurement whose factor loading is normalized. Its second entry is the value the loading is normalized to. For loadings it is typical to normalize to one but in theory any non-zero value is ok. Intercepts are typically normalized to zero. The example model has no normalizations on intercepts and this is ok due to the known location and scale of the CES production function.

Specifying normalizations is optional. If none are specified, they are generated automatically. The automatic generation takes into account the critique of `Wiswall and Agostinelli <https://dl.dropboxusercontent.com/u/33774399/wiswall_webpage/agostinelli_wiswall_renormalizations.pdf>`_, i.e. uses less normalizations for production functions with known scale and location. Moreover, it uses less normalizations if development stages span more than one period, because the location and scale of the transition function can be identified from the first period of such a long stage and no further normalization is needed.

.. Caution:: If you don't want to use normalizations you have to explicitly specify normalization lists with empty sublists (as in the example model). Simply not specifying normalizations triggers the automatic generation of normalization specifications.

.. Note:: I argue that the Wiswall Agostinelli critique is incomplete as transition function can also have known INPUT locations and/or scales. The log_CES function is an example for a function with known input scale. In this case even the initial normalizations might be unnecessary but I haven't yet tried to show that the restrictions the log_CES function poses on the input scale and location is sufficient for identification. Therefore the automatic generation of normalizations simply treats the log_CES function as a KLS function. I would currently recommend to rather use the translog function that (probably) doesn't have this problem.

.. Note:: The model shown below uses too many normalizations to make the results comparable with the
    parameters from the CHS replication files.

The value that corresponds to the ``trans_eq`` key is a dictionary. The ``name`` entry specifies the name of the transition equation. The ``included_factors`` entry specifies which factors enter the transition equation. The transition equations already implemented are:

    * ``linear``
    * ``log_ces`` (Known Location and Scale (KLS) version)
    * ``constant``
    * ``ar1`` (linear equation with only one included factor and the same coefficient in all stages)
    * ``translog`` (non KLS version; a log-linear-in-parameters function including squares and interaction terms. Not yet supported by wa estimator.).
    * ``no_squares_translog`` (as translog but without squares and therefore supported by wa estimator.)

To see how new types of transition equations can be added see :ref:`model_functions`.

The specification for fac2 is very similar and you can look it up in ``src.model_specs``. The specification for fac3 looks a bit different as this factor is only measured in the first period:

.. literalinclude:: test_model2.json
    :lines: 54-77

Here it is important to note that empty sublists have to be added to the ``measurements`` and ``normalizations`` key if no measurements are available in certain periods. Simply leaving the sublists out will result in an error.

Points 4 and 5 are only specified once for all factors by adding entries to the ``time_specific`` key of the model dictionary:

.. literalinclude:: test_model2.json
    :lines: 79-93

The value that corresponds to ``controls`` is a list of lists analogous to the specification of measurements. Note that the constat variable ``x2`` is not specified here as constants are added automatically in each measurement equation without normalized intercept.

The value that corresponds to ``stagemap`` is a list of length nperiods. The t_th entry indicates to which stage period t belongs. In the example it is just a list of zeros. (stages have to be numbered starting with zeros and incrementing in steps of one.)

The anchoring equation is specified as follows:

.. literalinclude:: test_model2.json
    :lines: 94

Q1 is the anchoring outcome and the list contains all anchored factors. In the example this is just fac1.

The example does not use endogeneity correction but adding it would be very easy. If for example fac2 was the endogenous factor one only had to add the following subdictionary to the model dictionary:

.. code::

    "endog_correction": {"endog_factor": "fac2", "endog_function": "linear"}

.. Note:: Endogeneity correction is already implemented but not completely tested yet.

The "general" section of the model dictionary:
**********************************************

Usually a research project comprises the estimation of more than one model and there are some specifications that are likely not to vary across these models. The default values for these specifications are hardcoded. If some or all of these values are redefined in the "general" section of the model dictionary the ones from the model dictionary have precedence. The specifications are:

    * ``nemf``: number of elements in the mixture of normals distribution of the latent factors. Usually set to 1 which corresponds to the assumption that the factors are normally distributed.
    * ``kappa``: scaling parameter for the sigma_points. Usually set to 2.
    * ``square_root_filters``: takes the values true and false and specifies if square-root implementations of the kalman filters are used. I strongly recommend always using square-root filters. As mentioned in section 3.2.2 of CHS' readme file the standard filters often crash unless very good start values for the maximization are available. Using the square-root filters completely avoids this problem.
    * ``missing_variables``: Takes the values "raise_error" or "drop_variable" and specifies what happens if a variable is not in the dataset or has only missing values. Automatically dropping these variables is handy when the same model is estimated with several similar but not exactly equal datasets.
    * ``controls_with_missings``: Takes the values "raise_error", "drop_variable" or "drop_observations". Recall that measurement variables can have missing observations as long as they are missing at random and at least some observations are not missing. For control variables this is not the case and it is necessary to drop the missing observations or the contol variable.
    * ``variables_without_variance``: takes the same values as ``missing_variables`` and specifies what happens if a measurement or anchoring variable has no variance. Control variables without variance are not dropped as this would drop constants.
    * ``robust_bounds``: takes the values true or false and refers to the bounds on some parameters during the maximization of the likelihood function. If true the lower bound for estimated variances is not set to zero but to ``bounds_distance``. This improves the stability of the estimator but is usually unnecessary if square-root filters are used.
    * ``bounds_distance``: a small number
    * ``estimate_X_zeros``: takes the values true or false. If true the start mean of the factor distribution is estimated, else it is normalized to zero. This is an alternative normalization of location in the initial period. If set to False you have to specify less normalizations of intercepts that otherwise. The automatic generation of normalizations correctly handles this case. If nemf > 1 you have to set estimate_X_zeros to True.
    * ``order_X_zeros``: Takes an integer value between 0 and nfac - 1.  If ``estimate_X_zeros`` is true and nemf > 1 the model would not be identified without imposing an order on the start means. The value of order_X_zeros determines which factor (in the alphabetically ordered factor list) is used to impose this order.
    * ``restrict_W_zeros``: takes the values true or false. If true the start weights of the mixture distribution is not estimated but set to 1 / nemf for each factor.
    * ``restrict_P_zeros``: takes the values true or false. If true the covariance matrices of all elements in the mixture distribution of the factors is required to be the same. CHS use this because their models with nemf > 1 do not converge otherwise.
    * ``cholesky_of_P_zero``: takes the values true or false. If true both the "long" and "short" parameter vector contain the cholesky factor of the covariance matrix of the factor distribution, which increases robustness. Else the "short" vector contains the cholesky factor and the "long" version the entries of the normal covariance matrix. See :ref:`params_type` for an explanation.
    * ``probit_measurements``: takes the values true and false. If true measurements that take only the values 0 and 1 are not incorporated with a linear measurement equation but similar to a probit model.

    .. Note:: This is not yet ready and will raise a NotImplementedError.

    * ``probanch_function``: takes the values "probability" and "odds_ratio". See Appendix 7.2 of the CHS paper for an explanation.

    .. Note:: Probability anchoring is not yet ready and will raise a NotImplementedError.

    * ``ignore_intercept_in_linear_anchoring``: takes the values true and false. Often the results remain interpretable if the intercept of the anchoring equation is ignored in the anchoring process. CHS do so in the example model (see equation above).
    * ``start_params``: a start vector for the maximization
    * ``start_values_per_quantity``: a dictionary with values that are used to construct the start vector for the maximization if the start vector is not provided directly.
    * ``wa_standard_error_method``: a string that indicates which method is used to calculate standard_errors if the WA estimator is used. Curently "bootstrap" is the only option.
    * ``chs_standard_error_method``:  a string that indicates which method is used to calculate standard_errors if the CHS estimator is used. Currently the options "op_of_gradient" (outer product of gradient), "hessian_inverse" and "bootstrap" are supported with the CHS estimator.
    * ``save_intermediate_optimization_results``: boolean variable. If true, the the optional arguments of SkillModel (save_path, model_name, dataset_name) have to be specified in order to generate a directory where the intermediate results are stored. The default value is False.
    * ``save_params_before_calculating_standard_errors``: boolean variable. If true, the the optional arguments of SkillModel (save_path, model_name, dataset_name) have to be specified in order to generate a directory where the intermediate results are stored. The default value is False.
    *``maxiter`` and ``maxfun`` are the maximal number of iterations or function evaluations for estimators that use numerical optimization techniques. The default for both is one million which probably won't be reached in practice.

.. _replication files:
    https://www.econometricsociety.org/content/supplement-estimating-technology-cognitive-and-noncognitive-skill-formation-0

.. _numba documentation:
    http://numba.pydata.org/numba-doc/latest/user/vectorize.html








