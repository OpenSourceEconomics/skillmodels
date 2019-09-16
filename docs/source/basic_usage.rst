.. _basic_usage:

***********
Basic Usage
***********

Some Python Skills you will need
********************************

To use skillmodels you probably will have to create a pandas DataFrame from a file on your disk and a python dictionary from a json file. Both things are very easy:

To create the DataFrame:

.. code::

    # from a .dta file
    import pandas as pd
    dataset = pd.read_stata('some/path/to/data.dta')

.. code::

    # from a .csv file
    import pandas as pd
    dataset = pd.read_csv('some/path/to/data.csv')

For any other file formats or advanced options for loading .dta and .csv files, see the `pandas documentation`_.

To load the model dictionary:

.. code::

    import json

    with open('some/path/to/model_dict.json') as j:
        model_dict = json.load(j)

In case of any problem check the documentation of the `json module`_.


Usage of Skillmodels
********************

Fitting models is very similar to fitting models in `Statsmodels`_

The main object a user interacts with is ``SkillModel`` (see :ref:`estimation`). To create an instance of this class just type:

.. code::

    from skillmodels import SkillModel
    mod = SkillModel(model_dict, dataset, estimator)

The mandatory arguments of SkillModel are:

* model_dict: dictionary that defines the model (usually loaded from a json file).
  See :ref:`model_specs` for details.
* dataset: a pandas dataframe in long format. Needs to have a MultiIndex in which
  the first level indicates the individual and the second the period.

Optional arguments of SkillModel are:

* model_name: a string that gives a name to the model that will be used in error
  messages or warnings. If you estimate several models it will help you a lot to
  locate the problems.
* dataset_name: same as model_name

Using the fit() method of ``SkillModel`` like so:

.. code::

    res = mod.fit()

.. Note:: Some functions of the CHS estimator use numpy functions that call fast multithreaded
    routines from the Intel Math Kernel Library (MKL). This is usually what you want, but if you fit several models in parallel (e.g. if you have datasets from different countries or you calculate boostrap standard errors) you might get better results if reduce the number of threads used. To do so, see the `documentation`_ of Anaconda.

.. _here:
    http://nipy.bic.berkeley.edu/nightly/statsmodels/doc/html/dev/generated/statsmodels.base.model.GenericLikelihoodModelResults.html#statsmodels.base.model.GenericLikelihoodModelResults

.. _documentation:
    https://docs.continuum.io/mkl-service/

.. _pandas documentation:
    http://pandas.pydata.org/pandas-docs/stable/io.html

.. _json module:
    https://docs.python.org/3.4/library/json.html
.. _Statsmodels:
    https://pypi.org/project/statsmodels/
