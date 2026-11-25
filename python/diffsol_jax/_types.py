from typing import Literal

import jax
import numpy as np
from jax.typing import ArrayLike, DTypeLike
from jaxtyping import PyTree

Solver = Literal["bdf", "esdirk34", "tr_bdf2", "tsit45"]
LinearSolver = Literal["default", "lu", "klu"]
MatrixType = Literal["nalgebra_dense", "faer_dense", "faer_sparse"]
Ad = Literal["adjoint", "forward"]

F32 = np.dtype(np.float32)
F64 = np.dtype(np.float64)
SOLVER_DTYPES = (F32, F64)


class DiffsolJaxError(RuntimeError):
    pass


def solver_dtype(spec: DTypeLike | None) -> np.dtype:
    if spec is None:
        spec = np.float64 if jax.config.read("jax_enable_x64") else np.float32
    try:
        dtype = np.dtype(spec)
    except TypeError as e:
        raise DiffsolJaxError(f"dtype must be float32 or float64, got {spec!r}") from e
    if dtype not in SOLVER_DTYPES:
        raise DiffsolJaxError(f"dtype must be float32 or float64, got {dtype}")
    return dtype


__all__ = [
    "F32",
    "F64",
    "Ad",
    "ArrayLike",
    "DTypeLike",
    "DiffsolJaxError",
    "LinearSolver",
    "MatrixType",
    "PyTree",
    "Solver",
    "solver_dtype",
]
