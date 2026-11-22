"""The types of the public signatures, so that the choices show up in them, and the error every
other module raises. This module imports nothing from the package: everything else may import
it."""

from typing import Literal

import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import PyTree

Solver = Literal["bdf", "esdirk34", "tr_bdf2", "tsit45"]
LinearSolver = Literal["default", "lu", "klu"]
MatrixType = Literal["nalgebra_dense", "faer_dense", "faer_sparse"]
Ad = Literal["adjoint", "forward"]

F64 = jnp.float64


class DiffsolJaxError(RuntimeError):
    pass


__all__ = [
    "Ad",
    "ArrayLike",
    "DiffsolJaxError",
    "F64",
    "LinearSolver",
    "MatrixType",
    "PyTree",
    "Solver",
]
