from __future__ import annotations

from enum import IntEnum
from typing import Union


class OdeSolverType(IntEnum):
    """Available ODE integration methods.

    The value of each member is the code passed to the Rust solver over the XLA
    custom call. A case-insensitive name string (e.g. ``"bdf"``) is accepted
    wherever an ``OdeSolverType`` is expected, via `coerce`.
    """

    BDF = 0
    TSIT45 = 1
    ESDIRK34 = 2
    TR_BDF2 = 3

    @classmethod
    def coerce(cls, value: "OdeSolverLike") -> "OdeSolverType":
        """Resolve an ``OdeSolverType`` or solver-name string to an ``OdeSolverType``.

        Args:
            value: an ``OdeSolverType`` member or a case-insensitive name such as
                ``"bdf"``, ``"tsit45"``, ``"esdirk34"``, ``"tr_bdf2"``.

        Returns:
            The matching ``OdeSolverType``.

        Raises:
            ValueError: if ``value`` is not a known solver.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls[value.upper()]
            except KeyError:
                pass
        available = [s.name.lower() for s in cls]
        raise ValueError(f"unknown solver {value!r}; available: {available}")


OdeSolverLike = Union[OdeSolverType, str]
