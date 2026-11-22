"""The types of the public signatures, so that the choices show up in them, and the error every
other module raises. This module imports nothing from the package: everything else may import
it."""

from typing import Literal

import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike, DTypeLike
from jaxtyping import PyTree

Solver = Literal["bdf", "esdirk34", "tr_bdf2", "tsit45"]
LinearSolver = Literal["default", "lu", "klu"]
MatrixType = Literal["nalgebra_dense", "faer_dense", "faer_sparse"]
Ad = Literal["adjoint", "forward"]

# The times are f64 whatever the solver's dtype is: diffsol-c takes them as f64, and the
# precision of a time is not what `dtype=` is for.
F64 = jnp.float64

F32 = np.dtype(np.float32)
SOLVER_DTYPES = (F32, np.dtype(np.float64))


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
