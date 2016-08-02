skillmodels
===========

Welcome to skillmodels, a Python implementation of estimators for skill formation models. The econometrics of skill formation models is a very active field and several estimators were proposed. None of them is implemented in standard econometrics packages.

The aims of this package are as follows:
    - Implement all proposed estimators for nonlinear skill formation models
    - Use the same model specification for all
    - Use the same result processing for all

I will start implementing two estimators: the Kalman filter based maximum likelihood estimator proposed by Cunha, Heckman and Schennach (CHS),  [Econometrica (2010)](http://onlinelibrary.wiley.com/doi/10.3982/ECTA6551/abstract) and the Method of Moments estimator proposed by Wiswall and Agostinelli (WA), [Working Paper (2016)](https://dl.dropboxusercontent.com/u/45673846/agostinelli_wiswall_estimation.pdf).

The estimators were developed for the estimation of skill formation models but are by no means limited to this particular application. It can be applied to any dynamic nonlinear latent factor model.

The CHS estimator implemented here differs in two points from the estimator implemented in their [replication files](https://www.econometricsociety.org/content/supplement-estimating-technology-cognitive-and-noncognitive-skill-formation-0): 1) It uses different normalizations that take into account the [critique](https://dl.dropboxusercontent.com/u/33774399/wiswall_webpage/agostinelli_wiswall_renormalizations.pdf) of Wiswall and Agostinelli. 2) It can optionally use more robust square-root implementations of the Kalman filters.

