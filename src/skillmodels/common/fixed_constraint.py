"""`FixedConstraintWithValue`: leaf data type used across constraint code.

Lives in its own module so that low-level callers (`transition_functions`,
`af/params`, etc.) can import it without triggering the heavier
`skillmodels.common.constraints` module — `constraints.py` imports
`transition_functions`, which would otherwise force a circular import or
a `TYPE_CHECKING` guard that beartype.claw cannot resolve at decoration
time.
"""

import functools
from dataclasses import dataclass

import optimagic as om
import pandas as pd

from skillmodels.common.selector import select_by_loc


@dataclass(frozen=True)
class FixedConstraintWithValue(om.FixedConstraint):
    """Fixed constraint that carries the target value and parameter location.

    `om.FixedConstraint` fixes parameters at their start values but does not carry a
    target value. This wrapper adds `loc` (the parameter location in the params
    DataFrame) and `value` (the value to set before optimization).
    """

    loc: pd.MultiIndex | tuple | str | None = None
    """Parameter location in the params DataFrame."""
    value: float | None = None
    """Value to enforce on the parameter."""

    def __post_init__(self) -> None:
        """Validate that `loc` and `value` are not None and derive `selector`."""
        if self.loc is None:
            msg = "loc must not be None"
            raise TypeError(msg)
        if self.value is None:
            msg = "value must not be None"
            raise TypeError(msg)
        object.__setattr__(
            self,
            "selector",
            functools.partial(select_by_loc, loc=self.loc),
        )
