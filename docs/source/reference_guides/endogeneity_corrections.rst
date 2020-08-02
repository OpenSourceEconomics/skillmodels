A note on endogeneity correction methods:
*****************************************

In the empirical part of their paper, CHS use two methods for endogeneity
correction. Both require very strong assumptions on the scale of factors.
Below I give an overview of the proposed endogeneity correction methods that
can serve as a starting point for someone who wants to extend skillmodels in
that direction:

In secton 4.2.4 CHS extend their basic model with a time invariant individual
specific heterogeneity component, i.e. a fixed effect. The time invariance
assumption can only be valid if the scale of all factors remains the same
throughout the model. This is highly unlikely, unless age invariant
measurements (as defined by Wiswall and Agostinelli) are available and used
for normalization in all periods for all factors. With KLS transition
functions the assumption of the factor scales remaining constant in all
periods is highly unlikely (see: :ref:`KLS_not_constant`). Moreover, this
approach requires 3 adult outcomes. If you have a dataset with enough time
invariant measurements and enough adult outcomes, this method is suitable for
you and you could use the Fortran code by CHS as a starting point.

In 4.2.5 they make a endogeneity correction with time varying heterogeneity.
However, this heterogeneity follows the same AR1 process in each period and
relies on an estimated time invariant investment equation, so it also requires
the factor scales to be constant. This might not be a good assumption in many
applications. Moreover, this correction method relies on a exclusion
restriction (Income is an argument of the investment function but not of the
transition functions of other latent factors) or suitable functional form
assumptions for identification.

To use this correction method in models where not enough age invariant
measurements are available to ensure constant factor scales, one would have to
replace the AR1 process by a linear transition function with different
estimated parameters in each period and also estimate a different investment
function in each period. I don't know if this model is identified.

I don't know if these methods could be used in the WA estimator.

Wiswall and Agostinelli use a simpler model of endegeneity of investments that
could be used with both estimators. See section 6.1.2 of their `paper`_.

.. _paper:
    https://tinyurl.com/y5ezloh2


.. _replication files:
    https://tinyurl.com/yyuq2sa4
