"""
Basic forward solve correctness test via _solve_forward.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from diffsol_jax import ODEProblem, _solve_forward
from scipy.integrate import solve_ivp

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


def scipy_solve(params=PARAMS):
    def f(t, y):
        a, b, d, g = params
        return [a * y[0] - b * y[0] * y[1], d * y[0] * y[1] - g * y[1]]

    ts = np.linspace(float(T_SPAN[0]), float(T_SPAN[1]), N_TIMES)
    sol = solve_ivp(
        f,
        (float(T_SPAN[0]), float(T_SPAN[1])),
        list(Y0),
        t_eval=ts,
        rtol=1e-8,
        atol=1e-10,
    )
    return sol.y.T  # shape (N_TIMES, n_state)


def make_problem(ode_solver="bdf"):
    return ODEProblem(lotka_volterra, Y0, PARAMS, n_times=N_TIMES)


@pytest.mark.parametrize("ode_solver", ODE_SOLVERS)
def test_forward_matches_scipy(ode_solver):
    """_solve_forward trajectory matches scipy reference."""
    prob = ODEProblem(lotka_volterra, Y0, PARAMS, n_times=N_TIMES)
    handle = prob._handle
    n_state = len(Y0)
    method_code = {"bdf": 0, "tsit45": 1, "esdirk34": 2, "tr_bdf2": 3}[ode_solver]

    ys, _ = _solve_forward(
        handle,
        PARAMS,
        float(T_SPAN[0]),
        float(T_SPAN[1]),
        N_TIMES,
        n_state,
        method_code,
    )

    ref = scipy_solve()
    diff = np.max(np.abs(np.asarray(ys) - ref))
    assert diff < 1e-3, f"{ode_solver}: max diff vs scipy {diff:.2e}"


def test_forward_matches_solve():
    """_solve_forward result matches ODEProblem.solve."""
    prob = make_problem()
    handle = prob._handle
    n_state = len(Y0)

    ys_ffi, _ = _solve_forward(
        handle,
        PARAMS,
        float(T_SPAN[0]),
        float(T_SPAN[1]),
        N_TIMES,
        n_state,
        0,  # bdf
    )

    _, ys_solve = prob.solve(PARAMS, T_SPAN)

    np.testing.assert_allclose(
        np.asarray(ys_ffi),
        np.asarray(ys_solve),
        rtol=1e-10,
        err_msg="_solve_forward and solve() should give identical trajectories",
    )
