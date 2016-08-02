.. _basic_usage:

***********
Basic Usage
***********

Fitting models is very similar to fitting models in `Statsmodels`_

The main object a user interacts with is ``CHSModel`` (see :ref:`estimation`) which is a subclass of Statsmodels `GenericLikelihoodModel`_. To create an instance of this class just type:

.. Note:: This class will be renamed as it will support more estimators than CHS in the future.
    The estimator used will then be an argument of the class.

.. code::

    from skillmodels import CHSModel
    mod = CHSModel('model_name', 'dataset_name', model_dict, dataset)

* 'model_name' a string that is used in error messages to clearly identify where the errors occured and as names of directories for the storage of intermediate results.
* 'dataset_name' is used for the same purpose
* 'model_dict' is the actual model dictionary (usually loaded from a json file)
* 'dataset' is a pandas dataframe in long format. It has to contain a variable named 'period' that indicates the period starting with 0 and incrementing in steps of 1.

Using the fit() method of ``CHSModel`` like so:

.. code::

    res = mod.fit()

will return an instance of ``CHSModelResults`` which is a subclass of ``GenericLikelihoodModelResults`` and has (or will have) most of the usual attributes and methods described `here`_. For example you can:

.. code::

    # access the estimated parameter vector (long form)
    estimated_parameters = res.params
    # access standard errors based on the outer product of gradients
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
