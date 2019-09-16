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

    #. What are the latent factors of the model and how are they related over time?
       (transition equations)
    #. What are the measurement variables of each factor in each period and how are
       measurements and factors related? (measurement equations)
    #. What are the normalizations of scale (normalized factor loadings or variances)
       and location (normalized intercepts or means)?
    #. What are the control variables in each period?
    #. If development stages are used: Which periods belong to which stage?
    #. If anchoring is used: Which factors are anchored and what is the anchoring
       outcome?

Translate the example to a model dictionary
*******************************************

The first three points have to be specified for each latent factor of the model. This is done by adding a subdictionary for each latent factor to the ``factor_specific`` key of the model dictionary. In the example model from the CHS `replication files`_ the dictionary for the first factor takes the following form:

.. literalinclude:: test_model2.json
    :lines: 2-28

The value that corresponds to the ``measurements`` key is a list of lists. It has one sublist for each period of the model. Each sublist consists of the measurement variables of ``fac1`` in the corresponding period. Note that in the example the measurement variables are the same in each period. However, this is extremely rare in actual models. If a measurement measures more than one latent factor, it simply is included in the ``measurements`` list of all factors it measures. As in the Fortran code by CHS it is currently only possible to estimate models with linear measurement system. I plan to add the possibility to incorporate binary variables in a way similar to a probit model but have currently no plans to support arbitrary nonlinear measurement systems.

The value that corresponds to the ``normalizations`` key is a dictionary in which the normalizations for factor loadings, intercepts and measurement variances are specified. Its values (for each type of normalization) are lists of dictionaries. There is one dictionary for each period. Its keys are the names of the measurements whose factor loading are normalized. The values are the number the loading is normalized to. For loadings it is typical to normalize to one, but any non-zero value is ok. Intercepts are typically normalized to zero, but any value is ok. Variances have to be normalized to a positive number.

The example model has no normalizations on intercepts and this is ok due to the known location and scale of the CES production function. The same result would have obtained if one had simply omitted 'intercepts' in the normalization dictionary (as was the case for normalized variances).

.. Note:: A previous version of skillmodels supported automatic generation of normalization specifications. This was dropped because I don't need it and it would have been hard to maintain while implementing time invariant measurement systems.

.. Note:: The model shown below deliberately uses too many normalizations in order to make the results comparable with the parameters from the CHS replication files.

The value that corresponds to the ``trans_eq`` key is a dictionary. The ``name`` entry specifies the name of the transition equation. The ``included_factors`` entry specifies which factors enter the transition equation. The transition equations already implemented are:

    * ``linear``
    * ``log_ces`` (Known Location and Scale (KLS) version. See :ref:`log_ces_problem`.)
    * ``constant``
    * ``ar1`` (linear equation with only one included factor and the same coefficient in
      all stages)
    * ``translog`` (non KLS version; a log-linear-in-parameters function including
      squares and interaction terms.
    * ``no_squares_translog`` (as translog but without squares. Used by Wiswall and
      Agostinelli.


To see how new types of transition equations can be added see :ref:`model_functions`.

The specification for fac2 is very similar and not reproduced here. The specification for fac3 looks a bit different as this factor is only measured in the first period:

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

Q1 is the anchoring outcome and the list contains all anchored factors. In the example this is just fac1. If you don't want to anchor the latent factors, simply omit the anchoring part in the model dictionary.

The "general" section of the model dictionary:
**********************************************

Usually a research project comprises the estimation of more than one model and there are some specifications that are likely not to vary across these models. The default values for these specifications are hardcoded. If some or all of these values are redefined in the "general" section of the model dictionary the ones from the model dictionary have precedence. The specifications are:

    * ``nemf``: number of elements in the mixture of normals distribution of the latent
      factors. Usually set to 1 which corresponds to the assumption that the factors
      are normally distributed. Only used in CHS estimator.
    * ``kappa``: scaling parameter for the sigma_points. Usually set to 2.
      Only used in CHS estimator.
    * ``square_root_filters``: takes the values true (default) and false and
      specifies if square-root implementations of the kalman filters are used.
      I strongly recommend always using square-root filters. As mentioned in
      section 3.2.2 of CHS' readme file the standard filters often crash unless
      very good start values for the maximization are available. Using the
      square-root filters completely avoids this problem. Only used in CHS
      estimator.
    * ``missing_variables``: Takes the values "raise_error" or "drop_variable" and
      specifies what happens if a variable is not in the dataset or has only missing
      values. Automatically dropping these variables is handy when the same model is
      estimated with several similar but not exactly equal datasets.
    * ``controls_with_missings``: Takes the values "raise_error", "drop_variable" or
      "drop_observations". Recall that measurement variables can have missing
      observations as long as they are missing at random and at least some observations
      are not missing. For control variables this is not the case and it is necessary
      to drop the missing observations or the contol variable.
    * ``variables_without_variance``: takes the same values as ``missing_variables`` and
      specifies what happens if a measurement or anchoring variable has no variance.
      Control variables without variance are not dropped as this would drop constants.
    * ``robust_bounds``: takes the values true or false and refers to the bounds on
      some parameters during the maximization of the likelihood function. If true the
      lower bound for estimated variances is not set to zero but to ``bounds_distance``.
      This improves the stability of the estimator but is usually unnecessary if
      square-root filters are used. Only used in CHS estimator.
    * ``bounds_distance``: a small number. Only used in CHS estimator.
    * ``estimate_X_zeros``: takes the values true or false. If true the start mean of
      the factor distribution is estimated, else it is normalized to zero. This is an
      alternative normalization of location in the initial period. If set to False you
      have to specify less normalizations of intercepts that otherwise. The automatic
      generation of normalizations correctly handles this case. If nemf > 1 you have to
      set estimate_X_zeros to True.
    * ``order_X_zeros``: Takes an integer value between 0 and nfac - 1.  If
      ``estimate_X_zeros`` is true and nemf > 1 the model would not be identified
      without imposing an order on the start means. The value of order_X_zeros
      determines which factor (in the alphabetically ordered factor list) is used
      to impose this order. Only used in CHS estimator.
    * ``restrict_W_zeros``: takes the values true or false. If true the start weights
      of the mixture distribution is not estimated but set to 1 / nemf for each factor.
      Only used in CHS estimator.
    * ``restrict_P_zeros``: takes the values true or false. If true the covariance
      matrices of all elements in the mixture distribution of the factors is required
      to be the same. CHS use this because their models with nemf > 1 do not converge
      otherwise. Only used in CHS estimator.

    .. Note:: This is not yet ready and will raise a NotImplementedError.

    * ``probanch_function``: takes the values "probability" and "odds_ratio". See Appendix 7.2 of the CHS paper for an explanation.

    .. Note:: Probability anchoring is not yet ready and will raise a NotImplementedError.

    * ``ignore_intercept_in_linear_anchoring``: takes the values true and false. Often the results remain interpretable if the intercept of the anchoring equation is ignored in the anchoring process. CHS do so in the example model (see equation above). Only used if anchoring_mode equals 'truly_anchor_latent_factors'
    * ``anchoring_mode``: Takes the values 'only_estimate_anchoring_equation' and 'truly_anchor_latent_factors'. The default is 'only_estimate_anchoring_equation'. It means that an anchoring equation is estimated that can be used for the calculation of interpretable marginal effects. This option does, however, not make the estimated transition parameters interpretable. The other other option requires more computer power and can make the transition parameters interpretable if enough age invariant measures are available and used for normalizations.

    * ``time_invariant_measurement_system``: Takes the values True or False (default). If True, measurement equations that occur in several periods are restricted to have the same parameters across those periods. To determine whether a measurement equation occurs more than once, the notion of equality has to be defined for measurement equations. Measurement equations are equal if and only if: 1) they use the same measurement variable. 2) The same latent factors are measured. 3) They occur in periods that use the same control variables. Currently, time invariant measurement system can only be used with the CHS-estimator.

    .. Note:: If ``time_invariant_measurement_system`` is True, the normalizations across all occurrences of a measurement equation have to be compatible. It is considered compatible if a parameter is normalized in some periods but not in others. It is not compatible if a parameter is normalized to different values. In this case, an error will be raised.



A note on endogeneity correction methods:
*****************************************

In the empirical part of their paper, CHS use two methods for endogeneity correction. Both require very strong assumptions on the scale of factors. Below I give an overview of the proposed endogeneity correction methods that can serve as a starting point for someone who wants to extend skillmodels in that direction:

In secton 4.2.4 CHS extend their basic model with a time invariant individual specific heterogeneity component, i.e. a fixed effect. The time invariance assumption can only be valid if the scale of all factors remains the same throughout the model. This is highly unlikely, unless age invariant measurements (as defined by Wiswall and Agostinelli) are available and used for normalization in all periods for all factors. With KLS transition functions the assumption of the factor scales remaining constant in all periods is highly unlikely (see: :ref:`KLS_not_constant`). Moreover, this approach requires 3 adult outcomes. If you have a dataset with enough time invariant measurements and enough adult outcomes, this method is suitable for you and you could use the Fortran code by CHS as a starting point.

In 4.2.5 they make a endogeneity correction with time varying heterogeneity. However, this heterogeneity follows the same AR1 process in each period and relies on an estimated time invariant investment equation, so it also requires the factor scales to be constant. This might not be a good assumption in many applications. Moreover, this correction method relies on a exclusion restriction (Income is an argument of the investment function but not of the transition functions of other latent factors) or suitable functional form assumptions for identification.

To use this correction method in models where not enough age invariant measurements are available to ensure constant factor scales, one would have to replace the AR1 process by a linear transition function with different estimated parameters in each period and also estimate a different investment function in each period. I don't know if this model is identified.

I don't know if these methods could be used in the WA estimator.

Wiswall and Agostinelli use a simpler model of endegeneity of investments that could be used with both estimators. See section 6.1.2 of their `paper`_.

.. _paper:
    https://dl.dropboxusercontent.com/u/45673846/agostinelli_wiswall_estimation.pdf


    .. _replication files:
        https://www.econometricsociety.org/content/supplement-estimating-technology-cognitive-and-noncognitive-skill-formation-0

  
