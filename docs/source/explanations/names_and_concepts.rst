.. _names_and_concepts:


==================
Names and concepts
==================

This section contains an overview of frequently used variable names and
concepts. It's not necessary to read this section if you are only interested in
using the code, but you might want to skim it if you are interested in what the
code actually does or plan to adapt it to your use case.

Most of those quantities are generated once during the :ref:`model_processing`
and appear as arguments of many other functions.

.. _dimensions:

``dimensions``
==============

Dimensions of the model quantities. All of them are integers.

- n_states: Number of latent factors or states in the model. Note that the terms
  state and factor are used interchangeably throughout the documentation.
- n_periods: Number of periods of the model. There is one more period than
  transition equations of the model.
- n_mixtures: Number of elements in the finite mixture of normals distribution.
- n_controls: Number of control variables in the measurement equations. This
  includes the intercept of the measurement equation. Thus n_controls is always
  1 or larger.


.. _labels:

``labels``
==========

Labels for the model quantities. All of them are lists.


- factors: Names of the latent factors.
- controls: Names of the control variables. The first entry is always "constant".
- periods: List of integers, starting at zero. The indices of the periods.
- stagemap: Maps periods to stages. Has one entry less than the number of periods.
- stages: The indices of the stages of the model.


.. _stages_vs_periods:


Development-Stages vs Periods
=============================

A development is a group of consecutive periods for which the technology of skill
formation remains the same. Thus the number of stages is always <= the number of
periods of a model.

Thus development stages are just equality constraints on the estimated parameter
vector. Because they are very frequently used, skillmodels can generate the
constraints automatically if you specify a stagemap in your model dictionary.


Example: If you have a model with 5 periods you can estimate at most 4 different
production functions (one for each transition between periods). If you want to
keep the parameters of the technology of skill formation constant between two
consecutive periods, you would specify the following stagemap: ``[0, 0, 1, 1]``


.. _anchoring:

``anchoring``
=============




.. _update_info:


``update_info``
===============



.. _normalizations:

``normalizations``
==================


.. _estimation_options:


``estimation_options``
======================
