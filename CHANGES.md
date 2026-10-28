# Changes

This is a record of all past skillmodels releases and what went into them in reverse
chronological order. We follow [semantic versioning](https://semver.org/) and all
releases are available on [PyPI](https://pypi.org/project/skillmodels/).

## Unreleased

- Support probit and Tobit measurements (`ModelSpec.measurement_models`) in the CHS
  estimator in every period, via an exact half-line update under the Gaussian
  predicted state with moment-matched filtered states.
- Support ordered-probit measurements (`OrderedProbitMeasurement`) in the CHS
  estimator, the shared measurement kernel and `simulate_dataset`, with per-period
  cutpoint parameters.
- The shared measurement kernel evaluates the log probability of an ordered-probit
  category with `skillmodels.common.normal_terms.log_normal_interval_mass`, which keeps
  the finite mass and derivatives of arbitrarily narrow categories.

## 0.1.1

- Drop the unused `jaxopt` dependency.
- Migrate the deprecated pixi `[system-requirements]` CUDA tables to named
  per-platform virtual packages (`linux-64-cuda12`/`linux-64-cuda13`), and refresh the
  CI toolchain (pixi 0.71.2, updated GitHub Actions and pre-commit hooks).

## 0.1

- Add the Antweiler–Freyberger (AF) sequential MLE and the Attanasio–Meghir–Nix (AMN)
  estimators alongside the existing CHS Kalman filter, with a shared control-function
  correction, a public measurement-family interface on `ModelSpec`, and the AF
  source/destination calendar adapter.
- Add a GitHub Action to build and publish the package to PyPI on tagged releases.
