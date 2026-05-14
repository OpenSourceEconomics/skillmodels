# cnlsy_7_9_11.csv

Long-format measurements derived from the CNLSY (Children of the National
Longitudinal Survey of Youth) public-use sample, ages 7 / 9 / 11. Produced
by `matlab_ces_repro/load_cnlsy.py` from the bundled
`complete_7_9_11.xls`, which itself is the input file used in the
Attanasio & Freyberger (2025) application.

The CSV is a tidy, period-indexed view of the same dataset:

- `(caseid, period)` MultiIndex (period ∈ {0, 1, 2}, mapping to ages 7 / 9 / 11).
- Skill measurements (`skill_math`, `skill_recog`, `skill_comp`) standardised
  within each period.
- Time-invariant cognitive (`mc_*`, 6 ASVAB sub-tests) and non-cognitive
  (`mn_neg`, `mn_pos`, `mn_rotter`) blocks standardised across the whole
  sample; written only in period 0, NaN elsewhere.
- Investment measurements (`inv_*`, parental involvement) standardised
  within each period and present in periods 0 and 1 only.
- `log_income_observed`: log family income (already in logs in the source);
  the period-2 value is held forward from period 1 to keep the CHS observed-
  factor column NaN-free (period 2 isn't used by the AF transition).

CNLSY is a U.S. Bureau of Labor Statistics public-use dataset. Redistribution
of a processed subset for documentation purposes is permitted under BLS terms.
