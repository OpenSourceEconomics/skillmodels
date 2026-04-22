"""Reproduction of Antweiler-Freyberger MATLAB skill-formation results.

Reference: `/home/hmg/sciebo/Skill estimation/` (local only; the data and
result artefacts are not committed). The test modules in this package load
`complete_7_9_11.xls` and the MATLAB `.mat` result files, translate MATLAB's
flat parameter vectors into skillmodels' 4-level MultiIndex, build a
`ModelSpec` that mirrors the MATLAB production function, and compare the
estimated parameters and likelihood against MATLAB's converged values.

Tests skip cleanly when the reference directory is not available.
"""
