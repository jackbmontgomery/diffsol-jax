"""The solve: one function over a problem."""

import jax
import jax.numpy as jnp

from . import _primitives
from ._lib import DiffsolJaxError
from ._params import F64
from ._problem import ODEProblem
from ._types import Ad, Args, ArrayLike, LinearSolver, MatrixType, Solver


def solve(
    problem: ODEProblem,
    y0: ArrayLike,
    t_eval: ArrayLike,
    args: Args = None,
    *,
    solver: Solver = "bdf",
    linear_solver: LinearSolver = "default",
    matrix_type: MatrixType = "nalgebra_dense",
    rtol: float = 1e-6,
    atol: float = 1e-6,
    ad: Ad = "adjoint",
    max_checkpoints: int = 8,
) -> jax.Array:
    """Solve `problem` from the initial state `y0` at the times `t_eval`.

    Differentiable with respect to both `y0` and `args`, and `jax.vmap` over either becomes more
    lanes of one batched solve rather than a loop of solves.

    Args:
        problem: an [`ODEProblem`][diffsol_jax.ODEProblem]: the equations.
        y0: the initial state `[..., nstates]`. `nstates` is its last axis.
        t_eval: the times to return the solution at, `[nt]`, increasing from `t0`.
        args: the parameters of the equations: one array `[..., nparams]`, or a pytree whose
            float-array leaves are the parameters, or `None` for no parameters. A leading axis of
            an array `args` is lanes, like a leading axis of `y0`; batch a pytree with
            `jax.vmap`.
        solver: the diffsol solver: `bdf`, `esdirk34`, `tr_bdf2` or `tsit45` (explicit, so not
            for a DAE).
        linear_solver: `default`, `lu` or `klu`.
        matrix_type: `nalgebra_dense`, `faer_dense` or `faer_sparse`.
        rtol: the relative tolerance of the solver.
        atol: the absolute tolerance of the solver.
        ad: how to take the reverse-mode derivative: `adjoint`, the diffsol adjoint with
            checkpointing, or `forward`, the transpose of the diffsol forward sensitivities.
        max_checkpoints: how many adjoint checkpoints to keep alive at once.

    Returns:
        `ys [..., nt, nout]`, the solution (or the output of `problem.out`) at `t_eval`.
    """
    if ad not in ("adjoint", "forward"):
        raise DiffsolJaxError("ad must be 'adjoint' or 'forward'")
    t_eval = jnp.asarray(t_eval, F64)
    if t_eval.ndim != 1:
        raise DiffsolJaxError("t_eval must have one axis")
    model = problem.compile(
        y0,
        args,
        solver=solver,
        linear_solver=linear_solver,
        matrix_type=matrix_type,
        max_checkpoints=max_checkpoints,
    )
    params = model.pack(y0, args)
    prim_cls = _primitives.SolveAdjoint if ad == "adjoint" else _primitives.SolveForward
    prim = prim_cls(
        jax.typeof(params),
        jax.typeof(t_eval),
        model=model,
        rtol=float(rtol),
        atol=float(atol),
    )
    return prim(params, t_eval)
