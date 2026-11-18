"""The types of the public signatures, so that the choices show up in them."""

from typing import Any, Literal

from jax.typing import ArrayLike

Solver = Literal["bdf", "esdirk34", "tr_bdf2", "tsit45"]
LinearSolver = Literal["default", "lu", "klu"]
MatrixType = Literal["nalgebra_dense", "faer_dense", "faer_sparse"]
Ad = Literal["adjoint", "forward"]
# a pytree: the state, the parameters, or the output of a problem
PyTree = Any

__all__ = ["Ad", "ArrayLike", "LinearSolver", "MatrixType", "PyTree", "Solver"]
