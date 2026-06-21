from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp

from .ffi import _ffi_solve

if TYPE_CHECKING:
    from .problem import ODEProblem


def _as_vec(rhs: Callable) -> Callable:
    """Wrap ``rhs`` so it always returns a length-``n_state`` vector.

    The user RHS may return a tuple of scalar components or a single vector;
    ``jax.jvp`` needs a single array output.
    """

    def vec(t, y, p):
        out = rhs(t, y, p)
        if isinstance(out, (tuple, list)):
            return jnp.stack(jnp.broadcast_arrays(*out))
        return jnp.asarray(out)

    return vec


def _make_aug_rhs(rhs: Callable, n_state: int, n_param: int) -> Callable:
    """Build the augmented RHS for the state ``[y; S]`` (``S`` flattened by column).

    The augmented state has length ``n_state * (1 + n_param)``: the first
    ``n_state`` entries are ``y``; the remaining entries are the sensitivity
    columns ``S_:,0, S_:,1, ...`` each of length ``n_state``. Each column's
    derivative is a ``jax.jvp`` of the user RHS in the direction
    ``(S_:,j, e_j)``.
    """
    rhs_vec = _as_vec(rhs)

    def aug(t, y_aug, p):
        y = y_aug[:n_state]
        outs = [rhs_vec(t, y, p)]
        for j in range(n_param):
            col = y_aug[n_state + j * n_state : n_state + (j + 1) * n_state]
            e_j = jnp.array([1.0 if k == j else 0.0 for k in range(n_param)])
            _, dcol = jax.jvp(lambda yy, pp: rhs_vec(t, yy, pp), (y, p), (col, e_j))
            outs.append(dcol)
        return jnp.concatenate(outs)

    return aug


def _make_solver(problem: "ODEProblem", method: int) -> Callable:
    r"""Return a ``custom_jvp`` solve function for one ``(problem, method)`` pair.

    The primal is the Cranelift dense solve; the JVP solves the augmented system
    to obtain ``J = \partial ys / \partial p`` and contracts it with ``dp``. Reverse
    mode falls out of JAX's automatic transpose of that (linear) contraction
    [https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html].
    """
    n_state = problem.n_state
    n_param = problem.n_params
    n_times = problem.n_times
    n_aug = n_state * (1 + n_param)

    @jax.custom_jvp
    def solve(params, t_span):
        return _ffi_solve(problem._handle, params, t_span, n_times, n_state, method)

    @solve.defjvp
    def solve_jvp(primals, tangents):
        params, t_span = primals
        dp, _dt_span = tangents  # t_span tangents are not propagated

        ys, ts = solve(params, t_span)

        aug_handle = problem._ensure_aug_handle()
        ys_aug, _ = _ffi_solve(aug_handle, params, t_span, n_times, n_aug, method)
        # ys_aug[:, n_state:] is S flattened by column: (n_times, n_param, n_state).
        # Transpose to J[t, state, param].
        jac = ys_aug[:, n_state:].reshape(n_times, n_param, n_state).transpose(0, 2, 1)

        dys = jnp.tensordot(jac, dp, axes=([2], [0]))
        return (ys, ts), (dys, jnp.zeros_like(ts))

    return solve
