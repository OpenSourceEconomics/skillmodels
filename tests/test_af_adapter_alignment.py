"""The AF calendar adapter aligns the whole per-observation payload by individual ID.

The adapter sources mixed-calendar measurement columns on a sorted individual-ID
intersection, while controls, observed factors, the period-0 conditional distribution,
and the chain-link payloads are read positionally in input-row order. Unless every
per-observation array shares one canonical order, a measurement row for one person is
paired with another person's controls or latent payload (a silent point-estimate change
on a reordered or unbalanced panel).

`_align_adapter_panel` canonicalises the panel to one ID-sorted order on the adapter
path, and requires the panel to be balanced (AF aligns periods positionally), raising
otherwise.
"""

import numpy as np
import pandas as pd
import pytest

from skillmodels.af.estimate import _align_adapter_panel


def _panel(ids_per_period: dict[int, list[int]]) -> pd.DataFrame:
    rows = []
    for period, ids in ids_per_period.items():
        for i in ids:
            rows.append({"id": i, "period": period, "y": float(10 * period + i)})
    return pd.DataFrame(rows).set_index(["id", "period"])


def test_align_adapter_panel_canonicalises_shuffled_order() -> None:
    rng = np.random.default_rng(0)
    balanced = _panel({0: [1, 2, 3], 1: [1, 2, 3]})
    shuffled = balanced.sample(frac=1.0, random_state=np.random.RandomState(1))
    assert shuffled.index.tolist() != balanced.sort_index().index.tolist()

    aligned = _align_adapter_panel(shuffled, n_periods=2)

    # Canonical (id, period) order regardless of input row order.
    assert aligned.index.tolist() == balanced.sort_index().index.tolist()
    # Within each period the rows are id-ascending, so the period slices align.
    for period in (0, 1):
        slice_ids = aligned.xs(period, level="period").index.tolist()
        assert slice_ids == sorted(slice_ids)
    _ = rng


def test_align_adapter_panel_rejects_unbalanced() -> None:
    # Individual 3 is missing from period 1: AF cannot positionally align the periods.
    unbalanced = _panel({0: [1, 2, 3], 1: [1, 2]})
    with pytest.raises(ValueError, match="balanced"):
        _align_adapter_panel(unbalanced, n_periods=2)
