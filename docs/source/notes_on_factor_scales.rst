**********************************
Notes on Scales and Normalizations
**********************************

Here I collect Notes on different aspects of the discussion about factor scales and re-normalization. This discussion originates in the `critique`_ by Wiswall and Agostinelli but I argue below, that this critique is not yet complete.

Wiswall and Agostinelli define a class of transition functions with Known Location and Scale (KLS) that require less normalizations. You should read this definition in their paper.

The critique by Wiswall and Agostinelli invalidates all empirical estimates of CHS, but not their general estimation routine. To get estimates that don't suffer from renormalization you can either use less normalizations or non-KLS transition functions. As there is no natural scale of skills, none of the approaches is better or worse. Nevertheless, I prefer using flexible Non-KLS transition functions with one normalization per period and factor because it is compatible with using development stages.


.. _KLS_not_constant:

Why KLS functions don't keep the scales constant
************************************************

Skills have no natural scale, but after reading the critique paper by Wiswall and Agostinelli one could easily get the impression that using KLS transition functions and less normalizations is better, because it identifies some sort of natural scale. Moreover in their `estimation`_ paper (p. 7), they write: "We argue that our limited normalization is appropriate for the dynamic setting of child development we analyze.  With our normalization for the initial period only, latent skills  in all periods  share  a common  location  and scale  with  respect to  the one chosen normalizing measure."

The following example intuitively shows firstly that the scale identified with KLS functions is as arbitrary as a scale identified through normalizations and secondly that this scale is not constant over time in general.

Consider a simple model of financial investments with two latent factors: a stock variable wealth (w) and a flow variable investment (i). Suppose periods last one year and annual interest rate on wealth is 10 percent. New investments are deposited at the end of the year (get interests only in the next year).

The most intuitive scales to describe the system would be to measure all latent factors in all periods in the same currency, say Dollars. In this case the transition equation of wealth is given by:

.. math::

    w_{t + 1} = 1.1 w_t + i_t

However, it would also be possible to measure w in period t in Dollars, i in period t in 1000 Dollars and w in period t + 1 in Dollar cents. The transition equation -- that still describes the exactly same system -- is then:

.. math::

    w_{t + 1} = 110 w_t + 100000 i_t

The parameters now reflect the actual technology and scale changes between periods. They are much harder to interpret than before. In fact any linear function

.. math::

    f: \mathbb{R}^2 \rightarrow \mathbb{R}

could describe the example system -- just in different combinations of scales.

When latent factor models are estimated, the scales of each factor are usually set through normalizations in each period. The main point of the first paper is that a KLS transition function prohibits to make such normalizations except for the initial period. One could say that after that, the transition function chooses the scale.

The CES function has KLS and contains the subset of all linear functions without intercept and parameters summing to 1 as special cases. It can therefore be used to describe the example system. After setting the scale of both factors to Dollars in the initial period, the CES function would then choose the scales for all other periods.

The linear function that is a CES function and describes the system is:

.. math::
    w_{t + 1} = \frac{1}{2.1} (1.1 w_t + i_t) \approx 0.524 w_t + 0.476 i_t

The scale of w in period t + 1 chosen by this function is thus 1 / 2.1 or approximately 0.476 Dollars which means that wealth in period t + 1 is approximately measured in 100 Philippine Pesos.


.. _log_ces_problem:

Why the CES and log_CES functions are problematic
*************************************************

The definition of Known Location and Scale refers only to the scale of the (always one-dimensional) output of a transition function. After reading the Wiswall and Agostinelli critique, I wondered if the CES and log_CES functions also pose restrictions on the scales of their inputs, i.e. can describe a system only at a certain location or scale of inputs.

According to Wiswall and Agostinelli, when using a log_CES function (which belongs to the KLS class), one needs initial normalizations of location and scale for all factors in the model.
I made some pen-and-paper-calculations and estimated models with simulated data and the results suggest that less normalizations are needed with the log_CES function.

While one does need to make initial normalizations for the location of all factors, it is sufficient to normalize the scale of only one factor in the initial period and the model is still identified. However, these are only simulations and I do not have a formal result that shows that the restrictions the log_CES function poses on the scale of its inputs are always enough for identification.

I would therefore currently advise not to use the CES or log_CES function without thinking deeply about the normalizations you need. The automatic generation of normalizations treats the log_ces function simply as a KLS function.


.. _normalization_and_stages:

Normalizations and Development stages
*************************************

CHS use development stages, i.e. several periods of childhood in which the parameters of the technology of skill formation remain the same. Wiswall and Agostinelli do not use or analyze this case, but development stages do change the normalization requirements.

I always had the intuition that with development stages it is possible to identify a scale from the first period of the stage, such that no later normalizations are necessary until the next stage. When extending the WA estimator to be compatible with development stages, I could confirm this intuition as one nice feature of this estimator is that its identification strategy has to be very explicit.

If development stages are used, one only has to make normalizations in the first period of each stage, except for the initial stage where the first two periods have to be normalized. My recommendation is to use automatic normalizations if you use development stages because it is very easy to get confused.

This shows another type of over-normalization in the original CHS paper.

.. _critique:
    https://dl.dropboxusercontent.com/u/33774399/wiswall_webpage/agostinelli_wiswall_renormalizations.pdf

.. _estimation:
    https://dl.dropboxusercontent.com/u/45673846/agostinelli_wiswall_estimation.pdf
