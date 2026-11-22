"""The types of the public signatures, so that the choices show up in them, and the error every
other module raises. This module imports nothing from the package: everything else may import
it."""

from typing import Literal

import numpy as np
from jax.typing import ArrayLike, DTypeLike
from jaxtyping import PyTree

Solver = Literal["bdf", "esdirk34", "tr_bdf2", "tsit45"]
LinearSolver = Literal["default", "lu", "klu"]
MatrixType = Literal["nalgebra_dense", "faer_dense", "faer_sparse"]
Ad = Literal["adjoint", "forward"]

# The two scalar types a solver can have, normalized: `solver_dtype` returns one of these, so
# they are what a dtype-keyed table is keyed on. A scalar type like `jnp.float64` compares equal
# to the matching `np.dtype` but does not hash to it, so it cannot be used as the key.
F32 = np.dtype(np.float32)
F64 = np.dtype(np.float64)
SOLVER_DTYPES = (F32, F64)


class DiffsolJaxError(RuntimeError):
    pass


def solver_dtype(spec: DTypeLike) -> np.dtype:
    """`spec` as the dtype of a solver, which must be `float32` or `float64`."""
    try:
        dtype = np.dtype(spec)
    except TypeError as e:
        raise DiffsolJaxError(f"dtype must be float32 or float64, got {spec!r}") from e
    if dtype not in SOLVER_DTYPES:
        raise DiffsolJaxError(f"dtype must be float32 or float64, got {dtype}")
    return dtype


__all__ = [
    "Ad",
    "ArrayLike",
    "DTypeLike",
    "DiffsolJaxError",
    "F32",
    "F64",
    "LinearSolver",
    "MatrixType",
    "PyTree",
    "Solver",
    "solver_dtype",
]
