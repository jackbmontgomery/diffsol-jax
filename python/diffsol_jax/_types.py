"""The types of the public signatures, so that the choices show up in them."""

from typing import Literal

from jax.typing import ArrayLike
from jaxtyping import PyTree

Solver = Literal["bdf", "esdirk34", "tr_bdf2", "tsit45"]
LinearSolver = Literal["default", "lu", "klu"]
MatrixType = Literal["nalgebra_dense", "faer_dense", "faer_sparse"]
Ad = Literal["adjoint", "forward"]

__all__ = ["Ad", "ArrayLike", "LinearSolver", "MatrixType", "PyTree", "Solver"]
