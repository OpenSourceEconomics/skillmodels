Welcome to the documentation of skillmodels!
============================================



Structure of the Documentation
==============================


.. raw:: html

    <div class="container" id="index-container">
        <div class="row">
            <div class="col-lg-6 col-md-6 col-sm-6 col-xs-12 d-flex">
                <a href="getting_started/index.html" id="index-link">
                    <div class="card text-center intro-card shadow">
                        <img src="_static/images/light-bulb.svg" class="card-img-top"
                             alt="getting_started-icon" height="52"
                        >
                        <div class="card-body flex-fill">
                            <h5 class="card-title">Getting Started</h5>
                            <p class="card-text">
                                New users of estimagic should read this first
                            </p>
                        </div>
                    </div>
                </a>
            </div>
            <div class="col-lg-6 col-md-6 col-sm-6 col-xs-12 d-flex">
                <a href="how_to_guides/index.html" id="index-link">
                    <div class="card text-center intro-card shadow">
                        <img src="_static/images/book.svg" class="card-img-top"
                             alt="how-to guides icon" height="52"
                        >
                        <div class="card-body flex-fill">
                            <h5 class="card-title">How-to Guides</h5>
                            <p class="card-text">
                                Detailed instructions for specific and advanced tasks.
                            </p>
                        </div>
                    </div>
                </a>
            </div>
            <div class="col-lg-6 col-md-6 col-sm-6 col-xs-12 d-flex">
                <a href="explanations/index.html" id="index-link">
                    <div class="card text-center intro-card shadow">
                        <img src="_static/images/books.svg" class="card-img-top"
                             alt="explanations icon" height="52"
                        >
                        <div class="card-body flex-fill">
                            <h5 class="card-title">Explanations</h5>
                            <p class="card-text">
                                Background information to key topics
                                underlying the package.
                            </p>
                        </div>
                    </div>
                </a>
            </div>
            <div class="col-lg-6 col-md-6 col-sm-6 col-xs-12 d-flex">
                <a href="reference_guides/index.html" id="index-link">
                    <div class="card text-center intro-card shadow">
                        <img src="_static/images/coding.svg" class="card-img-top"
                             alt="reference guides icon" height="52"
                        >
                        <div class="card-body flex-fill">
                            <h5 class="card-title">Reference Guides</h5>
                            <p class="card-text">
                                Overview of functions and modules as well as
                                implementation details
                            </p>
                        </div>
                    </div>
                 </a>
            </div>
        </div>
    </div>


Welcome to skillmodels, a Python implementation of estimators for skill
formation models. The econometrics of skill formation models is a very active
field and several estimators were proposed. None of them is implemented in
standard econometrics packages.


Skillmodels implements the Kalman filter based maximum likelihood estimator
proposed by Cunha, Heckman and Schennach (CHS), (`Econometrica 2010`_)


Skillmodels was developed for skill formation models but is by no means
limited to this particular application. It can be applied to any dynamic
nonlinear latent factor model.

The CHS estimator implemented here differs in two points from the one
implemented in their `replication files`_: 1) It uses different normalizations
that take into account the `critique`_ of Wiswall and Agostinelli. 2) It can
optionally use more robust square-root implementations of the Kalman filters.


Most of the code is unit tested. Furthermore, the results have been compared
to the Fortran code by CHS for two basic models with hypothetical data from
their `replication files`_.


**Citation**

It took countless hours to write skillmodels. I make it available under a very
permissive license in the hope that it helps other people to do great research
that advances our knowledge about the formation of cognitive and noncognitive
siklls. If you find skillmodels helpful, please don't forget to cite it. You
can find a suggested citation in the README file on `GitHub`_.


**Feedback**

If you find skillmodels helpful for research or teaching, please let me know.
If you encounter any problems with the installation or while using
skillmodels, please complain or open an issue at `GitHub`_.



.. _critique:
    https://tinyurl.com/y3wl43kz

.. _replication files:
    https://tinyurl.com/yyuq2sa4

.. _GitHub:
    https://github.com/janosg/skillmodels


.. _Econometrica 2010:
    http://onlinelibrary.wiley.com/doi/10.3982/ECTA6551/abstract


.. toctree::
    :maxdepth: 1

    getting_started/index
    how_to_guides/index
    explanations/index
    reference_guides/index
