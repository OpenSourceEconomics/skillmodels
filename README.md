skillmodels
===========

Introduction
------------


Welcome to skillmodels, a Python implementation of estimators for skill formation models. The econometrics of skill formation models is a very active field and several estimators were proposed. None of them is implemented in standard econometrics packages.

The aims of this package are as follows:

* Implement all proposed estimators for nonlinear skill formation models
* Use the same model specification for all
* Use the same results processing for all

I will start implementing two estimators: the Kalman filter based maximum likelihood estimator proposed by Cunha, Heckman and Schennach (CHS),  [Econometrica (2010)](http://onlinelibrary.wiley.com/doi/10.3982/ECTA6551/abstract) and the Method of Moments estimator proposed by Wiswall and Agostinelli (WA), [Working Paper (2016)](https://dl.dropboxusercontent.com/u/45673846/agostinelli_wiswall_estimation.pdf).

The estimators were developed for the estimation of skill formation models but are by no means limited to this particular application. It can be applied to any dynamic nonlinear latent factor model.

The CHS estimator implemented here differs in two points from the estimator implemented in their [replication files](https://www.econometricsociety.org/content/supplement-estimating-technology-cognitive-and-noncognitive-skill-formation-0): 1) It uses different normalizations that take into account the [critique](https://dl.dropboxusercontent.com/u/33774399/wiswall_webpage/agostinelli_wiswall_renormalizations.pdf) of Wiswall and Agostinelli. 2) It can optionally use more robust square-root implementations of the Kalman filters.

The WA estimator differs in three points: 1) In order to make the wa estimates usable as start values for the chs estimator, I extended it to also estimate measurement error variances and anchoring equation variances. 2) Development stages (i.e. the same technology of skill formation in several periods) can be used. 3) It is possible to use non-KLS transition functions as long as enough normalizations are provided.

Most of the code is unit tested. Furthermore, the results of the CHS estimator have been compared to the Fortran code by CHS for two basic models with hypothetical data from their [replication files](https://www.econometricsociety.org/content/supplement-estimating-technology-cognitive-and-noncognitive-skill-formation-0). For the WA estimator, I wrote a comprehensive integration test with simulated data.


Installation
------------

The package can be installed via conda. To do so, type the following in a terminal:

conda install -c suri5471 skillmodels

This should work for all platforms, but only 64-bit linux is tested.


Documentation
-------------

After cloning the repository you can cd to the docs directory and type:

make html

to generate detailed documentation.





