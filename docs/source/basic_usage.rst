.. _basic_usage:

***********
Basic Usage
***********

Fitting models is very similar to fitting models in `Statsmodels`_

The main object a user interacts with is ``SkillModel`` (see :ref:`estimation`) which is a subclass of Statsmodels `GenericLikelihoodModel`_. To create an instance of this class just type:


.. code::

    from skillmodels import SkillModel
    mod = SkillModel(model_dict, dataset, estimator)

* model_dict is the actual model dictionary (usually loaded from a json file)
* dataset is a pandas dataframe in long format. It has to contain columns that identify the period and individual. The names of these columns are indicated as 'period_identifier' and 'person_identifier' in the general section of the model dictionary. The default values are 'period' and 'id'.
* estimator is a string that can take the values 'wa' (Wiswall Agostinelli Estimator) and 'chs' (Cunha, Heckman, Schennach estimator).

.. Note:: Currently only the CHS estimator is implemented.

Using the fit() method of ``SkillModel`` like so:

.. code::

    res = mod.fit()

will return an instance of ``SkillModelResults`` which is a subclass of ``GenericLikelihoodModelResults`` and has (or will have) most of the usual attributes and methods described `here`_. For example you can:

.. code::

    # access the estimated parameter vector (long form)
    estimated_parameters = res.params
    # access standard errors based on the method you specified in the general
    # section of the model dictionary.
    standard_errrors = res.bse
    # access the t-valuess of the parameters
    tvalues = res.tvalues
    # access the p-values of the parameters
    pvalues = res.pvalues
    # access the confidence intervals (5 %)
    confidence_intervals = res.conf_int

Some methods are not yet implemented but will be in the future:

    * save, load and remove_data
    * summary
    * predict
    * other ways to calculate the standard errors

It should already work to use the t-test, f-test and wald-test as described `here`_ but I haven't tested it yet.

.. Note:: As done in several places in Statsmodels, I will continue to use the structure of the
    LikelihoodResults and GenericLikelihoodModel classes also for other estimators. Of course, the ML specific ways to calculate standard errors and make tests are will raise an error if other estimators were used to fit the model.


.. _Statsmodels:
    http://statsmodels.sourceforge.net/stable/

.. _GenericLikelihoodModel:
    http://statsmodels.sourceforge.net/devel/examples/notebooks/generated/generic_mle.html

.. _here:
    http://nipy.bic.berkeley.edu/nightly/statsmodels/doc/html/dev/generated/statsmodels.base.model.GenericLikelihoodModelResults.html#statsmodels.base.model.GenericLikelihoodModelResults
