"""
Step 5 test: JVP (forward sensitivity) via diffsol_jvp_rust.

Verifies:
- dys ≈ (solve(p + eps*dp, y0 + eps*dy0) - solve(p, y0)) / eps  (finite-difference)
- Zero tangents produce zero dys.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from diffsol_jax import ODEProblem
from diffsol_jax import _solve_forward, _solve_jvp

jax.config.update("jax_enable_x64", True)

PARAMS = jnp.array([1.5, 1.0, 0.75, 3.0])
Y0 = jnp.array([1.0, 0.5])
T_SPAN = jnp.array([0.0, 5.0])
N_TIMES = 50
# JVP only implemented via BDF; esdirk34/tr_bdf2 fall back to BDF in adjoint too
ODE_SOLVERS = ["bdf"]


def lotka_volterra(t, y, p):
    x, yy = y[0], y[1]
    alpha, beta, delta, gamma = p[0], p[1], p[2], p[3]
    return (alpha * x - beta * x * yy, delta * x * yy - gamma * yy)


def make_problem(ode_solver="bdf"):
    return ODEProblem(
        lotka_volterra, Y0, PARAMS, n_times=N_TIMES, ode_solver=ode_solver
    )


@pytest.mark.parametrize("ode_solver", ODE_SOLVERS)
def test_jvp_matches_fd(ode_solver):
    """dys from JVP matches finite-difference directional derivative."""
    prob = make_problem(ode_solver)
    handle = prob._handle
    n_state = len(Y0)
    n_params = len(PARAMS)
    method_code = {"bdf": 0, "tsit45": 1, "esdirk34": 2, "tr_bdf2": 3}[ode_solver]
    t0, t_final = float(T_SPAN[0]), float(T_SPAN[1])

    rng = np.random.default_rng(42)
    dp = jnp.array(rng.standard_normal(n_params))
    dy0 = jnp.array(rng.standard_normal(n_state))

    dys = _solve_jvp(handle, PARAMS, Y0, t0, t_final, dp, dy0, N_TIMES, n_state, method_code)

    # Finite-difference reference (centered for O(eps^2) accuracy)
    eps = 1e-4
    ys_plus, _ = _solve_forward(
        handle, PARAMS + eps * dp, Y0 + eps * dy0, t0, t_final, N_TIMES, n_state, method_code
    )
    ys_minus, _ = _solve_forward(
        handle, PARAMS - eps * dp, Y0 - eps * dy0, t0, t_final, N_TIMES, n_state, method_code
    )
    fd = (ys_plus - ys_minus) / (2 * eps)

    rel = jnp.max(jnp.abs(dys - fd) / (jnp.abs(fd) + 1e-6))
    assert rel < 5e-2, (
        f"{ode_solver}: JVP rel err {rel:.2e}\n"
        f"  max|dys|={float(jnp.max(jnp.abs(dys))):.3e}\n"
        f"  max|fd| ={float(jnp.max(jnp.abs(fd))):.3e}"
    )


@pytest.mark.parametrize("ode_solver", ODE_SOLVERS)
def test_jvp_zero_tangents(ode_solver):
    """Zero tangents produce zero dys."""
    prob = make_problem(ode_solver)
    handle = prob._handle
    n_state = len(Y0)
    n_params = len(PARAMS)
    method_code = {"bdf": 0, "tsit45": 1, "esdirk34": 2, "tr_bdf2": 3}[ode_solver]
    t0, t_final = float(T_SPAN[0]), float(T_SPAN[1])

    dp = jnp.zeros(n_params)
    dy0 = jnp.zeros(n_state)

    dys = _solve_jvp(handle, PARAMS, Y0, t0, t_final, dp, dy0, N_TIMES, n_state, method_code)

    assert jnp.max(jnp.abs(dys)) < 1e-10, (
        f"{ode_solver}: zero tangents should give zero dys, got max={float(jnp.max(jnp.abs(dys))):.2e}"
    )


@pytest.mark.parametrize("ode_solver", ODE_SOLVERS)
def test_jvp_params_only(ode_solver):
    """With dy0=0, JVP matches FD on params alone."""
    prob = make_problem(ode_solver)
    handle = prob._handle
    n_state = len(Y0)
    n_params = len(PARAMS)
    method_code = {"bdf": 0, "tsit45": 1, "esdirk34": 2, "tr_bdf2": 3}[ode_solver]
    t0, t_final = float(T_SPAN[0]), float(T_SPAN[1])

    rng = np.random.default_rng(7)
    dp = jnp.array(rng.standard_normal(n_params))
    dy0 = jnp.zeros(n_state)

    dys = _solve_jvp(handle, PARAMS, Y0, t0, t_final, dp, dy0, N_TIMES, n_state, method_code)

    eps = 1e-4
    ys_plus, _ = _solve_forward(
        handle, PARAMS + eps * dp, Y0, t0, t_final, N_TIMES, n_state, method_code
    )
    ys_minus, _ = _solve_forward(
        handle, PARAMS - eps * dp, Y0, t0, t_final, N_TIMES, n_state, method_code
    )
    fd = (ys_plus - ys_minus) / (2 * eps)

    rel = jnp.max(jnp.abs(dys - fd) / (jnp.abs(fd) + 1e-6))
    assert rel < 5e-2, f"{ode_solver}: params-only JVP rel err {rel:.2e}"


@pytest.mark.parametrize("ode_solver", ODE_SOLVERS)
def test_jvp_y0_only(ode_solver):
    """With dp=0, JVP matches FD on y0 alone."""
    prob = make_problem(ode_solver)
    handle = prob._handle
    n_state = len(Y0)
    n_params = len(PARAMS)
    method_code = {"bdf": 0, "tsit45": 1, "esdirk34": 2, "tr_bdf2": 3}[ode_solver]
    t0, t_final = float(T_SPAN[0]), float(T_SPAN[1])

    rng = np.random.default_rng(13)
    dp = jnp.zeros(n_params)
    dy0 = jnp.array(rng.standard_normal(n_state))

    dys = _solve_jvp(handle, PARAMS, Y0, t0, t_final, dp, dy0, N_TIMES, n_state, method_code)

    eps = 1e-4
    ys_plus, _ = _solve_forward(
        handle, PARAMS, Y0 + eps * dy0, t0, t_final, N_TIMES, n_state, method_code
    )
    ys_minus, _ = _solve_forward(
        handle, PARAMS, Y0 - eps * dy0, t0, t_final, N_TIMES, n_state, method_code
    )
    fd = (ys_plus - ys_minus) / (2 * eps)

    rel = jnp.max(jnp.abs(dys - fd) / (jnp.abs(fd) + 1e-6))
    assert rel < 5e-2, f"{ode_solver}: y0-only JVP rel err {rel:.2e}"
