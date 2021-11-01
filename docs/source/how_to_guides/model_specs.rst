
.. _model_specs:

********************
Model specifications
********************

Models are specified as
`Python dictionaries <http://introtopython.org/dictionaries.html>`_.
To improve reuse of the model specifications these dictionaries can be stored in json
or yaml files.

Example 2 from the CHS replication files
****************************************

Below the model-specification is illustrated using Example 2 from the CHS
`replication files`_. If you want you can read it in section 4.1 of their
readme file but I briefly reproduce it here for convenience.

There are three latent factors fac1, fac2 and fac3 and 8 periods that all belong to the
same development stage. fac1 evolves according to a log_ces production function and
depends on its past values as well as the past values of all other factors.
Moreover, it is linearly anchored with anchoring outcome Q1. This results in the
following transition equation:

.. math::

    fac1_{t + 1} = \frac{1}{\phi \lambda_1} ln\big(\gamma_{1,t}e^{\phi
    \lambda_1 fac1_t} + \gamma_{2,t}e^{\phi \lambda_2 fac2_t} +
    \gamma_{3,t}e^{\phi \lambda_3 fac3_t}\big) + \eta_{1, t}

where the lambdas are anchoring parameters from a linear anchoring equation.
fac1 is measured by measurements y1, y2 and y3 in all periods.  To sum up:
fac1 has the same properties as cognitive and non-cognitive skills in the CHS
paper.

The evolution of fac2 is described by a linear function and fac2 only depends
on its own past values, not on other factors, i.e. has the following
transition equation:

.. math::

    fac2_{t + 1} = lincoeff \cdot fac2_t + \eta_{2, t}

It is measured by y4, y5 and y6 in all periods. Thus fac2 has the same
properties as parental investments in the CHS paper.

fac3 is constant over time. It is measured by y7, y8 and y9 in the first
period and has no measurements in other periods. This makes it similar to
parental skills in the CHS paper.

In all periods and for all measurement equations the control variables x1 and
x2 are used, where x2 is a constant.

What has to be specified?
*************************

Before thinking about how to translate the above example into a model
specification it is helpful to recall what information is needed to define a
general latent factor model:

    #. What are the latent factors of the model and how are they related over time?
       (transition equations)
    #. What are the measurement variables of each factor in each period and how are
       measurements and factors related? (measurement equations)
    #. What are the normalizations of scale (normalized factor loadings)
       and location (normalized intercepts or means)?
    #. What are the control variables in each period?
    #. If development stages are used: Which periods belong to which stage?
    #. If anchoring is used: Which factors are anchored and what is the anchoring
       outcome?
    #. Are there any observed factors?

Translating the model to a dictionary
*************************************

before explaining how the model dictionary is written, here a full specification of the
example model as yaml file:


.. literalinclude:: ../../../skillmodels/tests/model2.yaml
  :language: yaml
  :linenos:

The model specification is a nested dictionary. The outer keys (which I call sections)
are ``"factors"``, ``"anchoring"``, ``"controls"``, ``"stagemap"`` and
``"estimation_options"``. All but the first are optional, but typically you will use at
least some of them.


``factors``
-----------

The factors are described as a dictionary. The keys are the names of the factors.
Any python string is possible as factor name. The values are dictionaries with three
entries:

- measurements: A nested list that is as long as the number of periods of the model.
  Each sublist contains the names of the measurements in that period. If A factor has
  no measurements in a period, it has to be an empty list. If a factor only has
  measurements up to a certain period you can leave out the empty lists at the end. In
  the example this is done for factor 3. Note that even in that case, the measurements
  have to be specified as nested list. If a factor only starts having measurements in
  some period, you still have to specify the empty lists for all periods before that
  period.

- transition_equation: A string with the name of a transition equation. For a list of
  all possible values see :ref:`transition_functions`. The list might seem a bit short
  but not that many other transition equations can be expressed with parameter
  constraints on the existing transition equations.

- normalizations: This entry is optional. It is a dictionary that can have the keys
  ``"loadings"`` and ``"intercepts"``. The values are lists of dictionaries. The list
  needs to contain one dictionary per period of the model. The keys of the dictionaries
  are names of measurements. The values are the value they are normalized to. Note that
  loadings cannot be normalized to zero.


Note that we could not express the constraint that ``"fac2"`` only depends on its own
past values in the model dictionary. This has to be expressed as an additional
constraint during optimization. Fortunately, this is very easy with estimagic.

``"anchoring"``
---------------

The specification for anchoring is a dictionary. It has the following entries:

- ``"outcomes"``: a dictionary that maps names of factors to variables that are used
  as anchoring outcome. Factors that are not anchored can simply be left out.
- ``"free_controls"``: Whether the control variables used in the measurement equations
  should also be used in the anchoring equations. Default False. This is mainly there
  to support the CHS example model and will probably not be set to True in any real
  application.
- ``"free_constant"``: Whether the anchoring equation should have a constant. Default
  False. This should be set to True if there are normalizations of location (i.e.
  normalized intercepts) in the measurement equations.
- ``"free_loadings"``: If true, the loadings are estimated, otherwise they are fixed to
  one. Default False. This should be set to True if there are normalizations of scale
  (i.e. normalized loadings) in the measurement equations.
- ``"ignore_constant_when_anchoring"``: If true, no constant is used when anchoring the
  latent factors, even if one was estimated. Default False. This is mainly there
  to support the CHS example model and will probably not be set to True in any real
  application.



``"controls"``
--------------

A list of variables that are used as controls in the measurement equations. You do not
have to specify as constant as control variable, because it is always included. If you
want to get rid of controls in some periods, you have to normalize their coefficients
to zero.

``"stagemap"``
--------------


A list that has one entry less than the number of periods of the model. It maps periods
to development stages. See :ref:`stages_vs_periods` for the meaning of development
stages.


``"observed_factors"``
----------------------

A list with variable names. Those variable names must be present in the dataset and
contain information about observed factors. An example of an observed factor could
be income, a treatment assignment or age.


Observed factors do not have transition equations, do not require multiple measurements
per period and are not part of the covariance matrix of the latent factors. As such,
adding an observed factor is computationally much less demanding than adding an
unobserved factor.


``"estimation_options"``
------------------------

Another dictionary. It has the following entries.

- ``"sigma_points_scale"``: The scaling factor of Julier sigma points. Default 2 which
  was shown to work well for the example models by Cunha, Heckman and Schennach.
- ``"robust_bounds"``: Bool. If true, bound constraints are made stricter. This avoids
  exploding likelihoods when the standard deviation of the measurement error is zero.
  Default True.
- ``"bounds_distance"``: By how much the bounds are made stricter. Only relevant when
  robust bounds are used. Default ``0.001``.
- ``"clipping_lower_bound": Strongly negative value at which the log likelihood is
  clipped a log likelihood of -infinity. The clipping is done using a soft maximum
  to avoid non-differentiable points in the likelihood. Default ``-1e-250``. Set to
  ``None`` to disable this completely.
- ``"clipping_upper_bound". Same as ``"clipping_lower_bound"`` but from above. Default
  None because typically the better way of avoiding upwards exploding likelihoods is to
  set bounds strictly above zero for the measurement error standard deviations.
- ``"clipping_lower_hardness"`` and ``"clipping_upper_hardness"``. How closely the soft
  maximum or minimum we use for clipping approximates its hard counterpart. Default 1
  which is an extremely close approximation of the hard maximum or minimum. If you want
  to make the likelihood function smoother you should set it to a much lower value.




.. _replication files:
    https://tinyurl.com/yyuq2sa4
