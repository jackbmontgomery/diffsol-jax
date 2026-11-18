"""The solve: one function over a problem."""

import jax
import jax.numpy as jnp

from . import _primitives
from ._lib import DiffsolJaxError
from ._params import F64
from ._problem import ODEProblem
from ._types import Ad, ArrayLike, PyTree


def solve(
    problem: ODEProblem,
    y0: PyTree,
    params: PyTree,
    t_eval: ArrayLike,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    ad: Ad = "adjoint",
) -> PyTree:
    """Solve `problem` from the initial state `y0` at the times `t_eval`.

    Differentiable with respect to both `y0` and `params`, and a `jax.vmap` over either becomes
    more lanes of one batched solve rather than a loop of solves. Batching is by `jax.vmap`:
    the leaves of `y0` and `params` have the shapes the problem declares, with no extra axes.

    Args:
        problem: an [`ODEProblem`][diffsol_jax.ODEProblem]: the equations, the layouts, and the
            diffsol solver. The options it was built with (`solver`, `linear_solver`,
            `matrix_type`, `max_checkpoints`) belong to it, not to the solve.
        y0: the initial state, in the problem's `state` structure.
        params: the parameter values, in the problem's `params` structure, or `None` for a
            problem with no parameters.
        t_eval: the times to return the solution at, `[nt]`, increasing from `t0`.
        rtol: the relative tolerance of the solver.
        atol: the absolute tolerance of the solver.
        ad: how to take the reverse-mode derivative: `adjoint`, the diffsol adjoint with
            checkpointing, or `forward`, the transpose of the diffsol forward sensitivities.

    Returns:
        `ys`, the solution at `t_eval` in the problem's output structure (its `state`, or what
        `out` returns): each leaf is `[nt, *leaf_shape]`. For a flat `state=n` and no `out`,
        that is one array `[nt, n]`.
    """
    if ad not in ("adjoint", "forward"):
        raise DiffsolJaxError("ad must be 'adjoint' or 'forward'")
    t_eval = jnp.asarray(t_eval, F64)
    if t_eval.ndim != 1:
        raise DiffsolJaxError("t_eval must have one axis")
    flat = problem._pack(y0, params)
    prim_cls = _primitives.SolveAdjoint if ad == "adjoint" else _primitives.SolveForward
    prim = prim_cls(
        jax.typeof(flat),
        jax.typeof(t_eval),
        problem=problem,
        rtol=float(rtol),
        atol=float(atol),
    )
    return problem._unpack(prim(flat, t_eval))
