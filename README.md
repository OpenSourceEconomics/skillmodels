# skillmodels

## Introduction

Welcome to skillmodels, a Python implementation of estimators for skill formation
models. The econometrics of skill formation models is a very active field and several
estimators were proposed. None of them is implemented in standard econometrics packages.

## Installation

> **Warning:** To run skillmodels you need to install jax and jaxlib. At the time of
> writing, in most use cases, it is faster on a CPU than on a GPU, so it should be
> sufficient to install the CPU version, which is available on all platforms. In any
> case, for installation of jax and jaxlib, please consult the jax
> [docs](https://jax.readthedocs.io/en/latest/installation.html#supported-platforms).

Skillmodels can be installed via PyPI or via GitHub. To do so, type the following in a
terminal:

```console
$ pip install skillmodels
```

or, for the latest development version, type:

```console
$ pip install git+https://github.com/OpenSourceEconomics/skillmodels.git
```

## Documentation

[The documentation is hosted at readthedocs](https://skillmodels.readthedocs.io/en/latest/)

## Developing

We use [pixi](https://pixi.sh/latest/) for our local development environment. If you
want to work with or extend the skillmodels code base you can run the tests using

```console
$ git clone https://github.com/OpenSourceEconomics/skillmodels.git
$ pixi run tests
```

This will install the development environment and run the tests. You can run
[mypy](https://mypy-lang.org/) using

```console
$ pixi run mypy
```

Before committing, install the pre-commit hooks using

```console
$ pre-commit install
```

#### Documentation

You can build the documentation locally. After cloning the repository you can cd to the
docs directory and type:

```console
$ make html
```

## Citation

It took countless hours to write skillmodels. I make it available under a very
permissive license in the hope that it helps other people to do great research that
advances our knowledge about the formation of cognitive and noncognitive siklls. If you
find skillmodels helpful, please don't forget to cite it. Below you can find the bibtex
entry for a suggested citation. The suggested citation will be updated once the code
becomes part of a published paper.

```
@Unpublished{Gabler2024,
  Title                    = {A Python Library to Estimate Nonlinear Dynamic Latent Factor Models},
  Author                   = {Janos Gabler},
  Year                     = {2024},
  Url                      = {https://github.com/OpenSourceEconomics/skillmodels}
}
```

## Feedback

If you find skillmodels helpful for research or teaching, please let me know. If you
encounter any problems with the installation or while using skillmodels, please complain
or open an issue at [GitHub](https://github.com/OpenSourceEconomics/skillmodels)
