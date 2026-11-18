"""The solve: one function over a problem."""

import jax
import jax.numpy as jnp

from . import _primitives
from ._lib import DiffsolJaxError
from ._params import F64
from ._problem import ODEProblem
from ._types import Ad, Args, ArrayLike


def solve(
    problem: ODEProblem,
    y0: ArrayLike,
    t_eval: ArrayLike,
    args: Args = None,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    ad: Ad = "adjoint",
) -> jax.Array:
    """Solve `problem` from the initial state `y0` at the times `t_eval`.

    Differentiable with respect to both `y0` and `args`, and `jax.vmap` over either becomes more
    lanes of one batched solve rather than a loop of solves.

    Args:
        problem: an [`ODEProblem`][diffsol_jax.ODEProblem]: the equations, the sizes, and the
            diffsol solver. The solver options it was built with (`solver`, `linear_solver`,
            `matrix_type`, `max_checkpoints`) belong to it, not to the solve.
        y0: the initial state `[..., nstates]`.
        t_eval: the times to return the solution at, `[nt]`, increasing from `t0`.
        args: the parameter values, in the layout the problem declared: an array
            `[..., nparams]`, or the pytree matching its `params` template, or `None`.
        rtol: the relative tolerance of the solver.
        atol: the absolute tolerance of the solver.
        ad: how to take the reverse-mode derivative: `adjoint`, the diffsol adjoint with
            checkpointing, or `forward`, the transpose of the diffsol forward sensitivities.

    Returns:
        `ys [..., nt, nout]`, the solution (or the output of `problem.out`) at `t_eval`.
    """
    if ad not in ("adjoint", "forward"):
        raise DiffsolJaxError("ad must be 'adjoint' or 'forward'")
    t_eval = jnp.asarray(t_eval, F64)
    if t_eval.ndim != 1:
        raise DiffsolJaxError("t_eval must have one axis")
    params = problem._pack(y0, args)
    prim_cls = _primitives.SolveAdjoint if ad == "adjoint" else _primitives.SolveForward
    prim = prim_cls(
        jax.typeof(params),
        jax.typeof(t_eval),
        problem=problem,
        rtol=float(rtol),
        atol=float(atol),
    )
    return prim(params, t_eval)
