skillmodels
===========

Introduction
------------


Welcome to skillmodels, a Python implementation of estimators for skill formation models. The econometrics of skill formation models is a very active field and several estimators were proposed. None of them is implemented in standard econometrics packages.


Installation
------------

The package can be installed via conda. To do so, type the following in a terminal:


conda config --add channels conda-forge
conda config --add channels janosg
conda install skillmodels


Documentation
-------------

[The documentation is hosted at readthedocs](https://skillmodels.readthedocs.io/en/latest/)

Alternatively, you can build it locally. After cloning the repository you can cd to the docs directory and type:

make html



Building the package with conda-build
-------------------------------------

If you want to make changes to the source code of skillmodels or build the package on your machine for any other reason, you can do so with conda-build.

* clone the repository and make your changes
* Install conda-build 2.0 or higher (earlier versions won't work!)
* Adjust the version number in setup.py and meta.yaml.
* Open a shell in the directory that contains skillmodels (i.e. one level higher than your git repository)
* type: conda build skillmodels

For more information see the [conda documentation](http://conda.pydata.org/docs/building/build.html)


Citation
--------

It took countless hours to write skillmodels. I make it available under a very permissive license in the hope that it helps other people to do great research that advances our knowledge about the formation of cognitive and noncognitive siklls. If you find skillmodels helpful, please don't forget to cite it. Below you can find the bibtex entry for a suggested citation. The suggested citation will be updated once the code becomes part of a published paper.

```
@Unpublished{Gabler2018,
  Title                    = {A Python Library to Estimate Nonlinear Dynamic Latent Factor Models},
  Author                   = {Janos Gabler},
  Year                     = {2018},
  Url                      = {https://github.com/janosg/skillmodels}
}
```


Feedback
--------

If you find skillmodels helpful for research or teaching, please let me know. If you encounter any problems with the installation or while using skillmodels, please complain or open an issue at [GitHub](https://github.com/janosg/skillmodels)
