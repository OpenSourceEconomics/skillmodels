.. _introduction:


************
Introduction
************

Welcome to skillmodels, a Python implementation of estimators for skill formation models. The econometrics of skill formation models is a very active field and several estimators were proposed. None of them is implemented in standard econometrics packages.

The aims of this package are as follows:
    - Implement all proposed estimators for nonlinear skill formation models
    - Use the same model specification for all
    - Use the same result processing for all

I will start implementing two estimators: the Kalman filter based maximum likelihood estimator proposed by Cunha, Heckman and Schennach (CHS),  `Econometrica (2010)`_ and the Method of Moments estimator proposed by Wiswall and Agostinelli (WA), `Working Paper (2016)`_.

.. _Econometrica (2010):
    http://onlinelibrary.wiley.com/doi/10.3982/ECTA6551/abstract

.. _Working Paper (2016):
    https://dl.dropboxusercontent.com/u/45673846/agostinelli_wiswall_estimation.pdf

Both were developed for skill formation models but are by no means limited to this particular application. They can be applied to any dynamic nonlinear latent factor model.

The CHS estimator implemented here differs in two points from the one implemented in their `replication files`_: 1) It uses different normalizations that take into account the `critique`_ of Wiswall and Agostinelli. 2) It can optionally use more robust square-root implementations of the Kalman filters.

The WA estimator differs in three points: 1) In order to make the wa estimates usable as start values for the chs estimator, I extended it to also estimate measurement error variances and anchoring equation variances. 2) Development stages (i.e. the same technology of skill formation in several periods) can be used. 3) It is possible to use non-KLS transition functions as long as enough normalizations are provided.

Most of the code is unit tested. Furthermore, the results of the CHS estimator have been compared to the Fortran code by CHS for two basic models with hypothetical data from their `replication files`_. For the WA estimator, I wrote a comprehensive integration test with simulated data.

The following documentation is ordered from high-level to low-level. To get started it is sufficient to read about Model specifications and Basic usage. Reading more is only necessary if you want to extend the code or understand the implementation details.


**Citation**

It took countless hours to write skillmodels. I make it available under a very permissive license in the hope that it helps other people to do great research that advances our knowledge about the formation of cognitive and noncognitive siklls. If you find skillmodels helpful, please don't forget to cite it. You can find a suggested citation in the README file on `GitHub`_.


**Feedback**

If you find skillmodels helpful for research or teaching, please let me know. If you encounter any problems with the installation or while using skillmodels, please complain or open an issue at `GitHub`_.





.. _critique:
    https://dl.dropboxusercontent.com/u/33774399/wiswall_webpage/agostinelli_wiswall_renormalizations.pdf

.. _replication files:
    https://www.econometricsociety.org/content/supplement-estimating-technology-cognitive-and-noncognitive-skill-formation-0

.. _GitHub:
    https://github.com/janosg/skillmodels

















