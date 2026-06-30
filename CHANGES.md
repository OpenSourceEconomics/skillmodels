# Changes

This is a record of all past skillmodels releases and what went into them in reverse
chronological order. We follow [semantic versioning](https://semver.org/) and all
releases are available on [PyPI](https://pypi.org/project/skillmodels/).

## 0.1

- Add the Antweiler–Freyberger (AF) sequential MLE and the Attanasio–Meghir–Nix (AMN)
  estimators alongside the existing CHS Kalman filter, with a shared control-function
  correction, a public measurement-family interface on `ModelSpec`, and the AF
  source/destination calendar adapter.
- Add a GitHub Action to build and publish the package to PyPI on tagged releases.
