"""
Step 3 test: grad_y0 from adjoint matches finite differences.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from diffsol_jax import ODEProblem
from diffsol_jax import _solve_forward, _solve_adjoint

jax.config.update("jax_enable_x64", True)

PARAMS = jnp.array([1.5, 1.0, 0.75, 3.0])
Y0 = jnp.array([1.0, 0.5])
T_SPAN = jnp.array([0.0, 5.0])
N_TIMES = 50
ODE_SOLVERS = ["bdf", "tsit45", "esdirk34", "tr_bdf2"]


def lotka_volterra(t, y, p):
    x, yy = y[0], y[1]
    alpha, beta, delta, gamma = p[0], p[1], p[2], p[3]
    return (alpha * x - beta * x * yy, delta * x * yy - gamma * yy)


def make_problem(ode_solver="bdf"):
    return ODEProblem(
        lotka_volterra, Y0, PARAMS, n_times=N_TIMES, ode_solver=ode_solver
    )


@pytest.mark.parametrize("ode_solver", ODE_SOLVERS)
def test_grad_y0_matches_fd(ode_solver):
    """grad_y0 from adjoint matches finite-difference estimate."""
    prob = make_problem(ode_solver)
    handle = prob._handle
    n_state = len(Y0)
    method_code = {"bdf": 0, "tsit45": 1, "esdirk34": 2, "tr_bdf2": 3}[ode_solver]
    t0, t_final = float(T_SPAN[0]), float(T_SPAN[1])

    def loss_ys(y0):
        ys, _ = _solve_forward(handle, PARAMS, y0, t0, t_final, N_TIMES, n_state, method_code)
        return jnp.sum(ys ** 2)

    # cotangent g_ys = d(loss)/d(ys) — shape (N_TIMES, n_state)
    ys, _ = _solve_forward(handle, PARAMS, Y0, t0, t_final, N_TIMES, n_state, method_code)
    g_ys = 2.0 * ys  # gradient of sum(ys**2) w.r.t. ys

    _, grad_y0 = _solve_adjoint(
        handle, PARAMS, Y0, T_SPAN, g_ys, N_TIMES, n_state, method_code
    )

    # finite-difference reference
    eps = 1e-4
    fd = jnp.array([
        (loss_ys(Y0.at[i].set(Y0[i] + eps)) - loss_ys(Y0.at[i].set(Y0[i] - eps))) / (2 * eps)
        for i in range(n_state)
    ])

    rel = jnp.max(jnp.abs(grad_y0 - fd) / (jnp.abs(fd) + 1e-6))
    assert rel < 2e-2, f"{ode_solver}: grad_y0 rel err {rel:.2e}\n  adj={np.asarray(grad_y0)}\n  fd ={np.asarray(fd)}"


@pytest.mark.parametrize("ode_solver", ODE_SOLVERS)
def test_grad_y0_nonzero(ode_solver):
    """grad_y0 is not identically zero — adjoint is actually computing it."""
    prob = make_problem(ode_solver)
    handle = prob._handle
    n_state = len(Y0)
    method_code = {"bdf": 0, "tsit45": 1, "esdirk34": 2, "tr_bdf2": 3}[ode_solver]

    ys, _ = _solve_forward(handle, PARAMS, Y0, float(T_SPAN[0]), float(T_SPAN[1]),
                           N_TIMES, n_state, method_code)
    g_ys = jnp.ones_like(ys)

    _, grad_y0 = _solve_adjoint(
        handle, PARAMS, Y0, T_SPAN, g_ys, N_TIMES, n_state, method_code
    )
    assert jnp.max(jnp.abs(grad_y0)) > 1e-6, \
        f"{ode_solver}: grad_y0 suspiciously zero: {np.asarray(grad_y0)}"
