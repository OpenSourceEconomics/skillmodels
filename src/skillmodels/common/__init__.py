"""Estimator-agnostic infrastructure shared by CHS, AF, and AMN.

This subpackage holds everything that the three estimator subpackages
build on but do not own: the user-facing model specification
(`ModelSpec`, `FactorSpec`, `AnchoringSpec`), the data and parameter
processing pipeline (`process_model`, `process_data`, `params_index`,
`parse_params`), the constraint plumbing (`constraints`,
`decorators`), shared transition-function library, and the
visualisation helpers that operate on the common filtered-states
DataFrame format.

The dependency rule for this package: it imports from no estimator
subpackage. Conversely, `chs`, `af`, and `amn` import freely from here.
"""
