"""The types of the solve options, so that the choices show up in the signature."""

from typing import Any, Literal

from jax.typing import ArrayLike

Solver = Literal["bdf", "esdirk34", "tr_bdf2", "tsit45"]
LinearSolver = Literal["default", "lu", "klu"]
MatrixType = Literal["nalgebra_dense", "faer_dense", "faer_sparse"]
Ad = Literal["adjoint", "forward"]
# the parameters of the equations: one array [..., nparams], or any pytree, or None
Args = Any

__all__ = ["Ad", "Args", "ArrayLike", "LinearSolver", "MatrixType", "Solver"]
