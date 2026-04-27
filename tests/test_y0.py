"""
Step 1 & 2 tests: y0 plumbed through Rust FFI and C++ shim.

Verifies:
- Passing baked-in y0 explicitly produces identical trajectories.
- Passing a different y0 shifts the trajectory as expected (vs scipy).
- All four solver methods accept explicit y0.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from diffsol_jax import ODEProblem
from diffsol_jax import _solve_forward

jax.config.update("jax_enable_x64", True)

PARAMS = jnp.array([1.5, 1.0, 0.75, 3.0])
Y0_DEFAULT = jnp.array([1.0, 0.5])
T_SPAN = jnp.array([0.0, 5.0])
N_TIMES = 50
ODE_SOLVERS = ["bdf", "tsit45", "esdirk34", "tr_bdf2"]


def lotka_volterra(t, y, p):
    x, yy = y[0], y[1]
    alpha, beta, delta, gamma = p[0], p[1], p[2], p[3]
    return (alpha * x - beta * x * yy, delta * x * yy - gamma * yy)


def scipy_solve(y0, params=PARAMS):
    def f(t, y):
        a, b, d, g = params
        return [a * y[0] - b * y[0] * y[1], d * y[0] * y[1] - g * y[1]]

    ts = np.linspace(float(T_SPAN[0]), float(T_SPAN[1]), N_TIMES)
    sol = solve_ivp(f, (float(T_SPAN[0]), float(T_SPAN[1])), list(y0),
                    t_eval=ts, rtol=1e-8, atol=1e-10)
    return sol.y.T  # shape (N_TIMES, n_state)


def make_problem(ode_solver="bdf"):
    return ODEProblem(
        lotka_volterra, Y0_DEFAULT, PARAMS, n_times=N_TIMES, ode_solver=ode_solver
    )


def test_baked_y0_matches_explicit():
    """Passing baked-in y0 explicitly gives same result as the implicit path."""
    prob = make_problem()
    _, ys_implicit = prob.solve(Y0_DEFAULT, PARAMS, T_SPAN)

    handle = prob._handle
    n_state = len(Y0_DEFAULT)
    method_code = 0  # bdf

    ys_explicit, _ = _solve_forward(
        handle, PARAMS, Y0_DEFAULT,
        float(T_SPAN[0]), float(T_SPAN[1]),
        N_TIMES, n_state, method_code,
    )

    np.testing.assert_allclose(
        np.asarray(ys_explicit), np.asarray(ys_implicit), rtol=1e-10,
        err_msg="explicit y0 == baked y0 should give identical trajectories",
    )


@pytest.mark.parametrize("ode_solver", ODE_SOLVERS)
def test_explicit_y0_matches_scipy(ode_solver):
    """Trajectory with explicit y0 == [1.0, 0.5] matches scipy reference."""
    prob = ODEProblem(
        lotka_volterra, Y0_DEFAULT, PARAMS, n_times=N_TIMES, ode_solver=ode_solver
    )
    handle = prob._handle
    n_state = len(Y0_DEFAULT)
    method_code = {"bdf": 0, "tsit45": 1, "esdirk34": 2, "tr_bdf2": 3}[ode_solver]

    ys, _ = _solve_forward(
        handle, PARAMS, Y0_DEFAULT,
        float(T_SPAN[0]), float(T_SPAN[1]),
        N_TIMES, n_state, method_code,
    )

    ref = scipy_solve(Y0_DEFAULT)
    diff = np.max(np.abs(np.asarray(ys) - ref))
    assert diff < 1e-3, f"{ode_solver}: max diff vs scipy {diff:.2e}"


@pytest.mark.parametrize("ode_solver", ODE_SOLVERS)
def test_different_y0_shifts_trajectory(ode_solver):
    """Passing a different y0 produces a trajectory that matches scipy with that y0."""
    y0_alt = jnp.array([2.0, 1.0])

    prob = ODEProblem(
        lotka_volterra, Y0_DEFAULT, PARAMS, n_times=N_TIMES, ode_solver=ode_solver
    )
    handle = prob._handle
    n_state = len(Y0_DEFAULT)
    method_code = {"bdf": 0, "tsit45": 1, "esdirk34": 2, "tr_bdf2": 3}[ode_solver]

    ys_alt, _ = _solve_forward(
        handle, PARAMS, y0_alt,
        float(T_SPAN[0]), float(T_SPAN[1]),
        N_TIMES, n_state, method_code,
    )

    ref_alt = scipy_solve(y0_alt)
    diff = np.max(np.abs(np.asarray(ys_alt) - ref_alt))
    assert diff < 1e-3, f"{ode_solver}: y0=[2,1] max diff vs scipy {diff:.2e}"

    # Also verify it differs from the default-y0 trajectory
    ref_default = scipy_solve(Y0_DEFAULT)
    assert np.max(np.abs(ref_alt - ref_default)) > 0.1, \
        "alt y0 trajectory should differ from default"
