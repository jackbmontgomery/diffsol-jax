"""
JVP (forward sensitivity) via diffsol_jvp_rust.

Verifies:
- dys ≈ (solve(p + eps*dp) - solve(p)) / eps  (finite-difference, params only)
- Zero tangents produce zero dys.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from diffsol_jax import ODEProblem, _solve_forward, _solve_jvp

jax.config.update("jax_enable_x64", True)

PARAMS = jnp.array([1.5, 1.0, 0.75, 3.0])
Y0 = jnp.array([1.0, 0.5])
T_SPAN = jnp.array([0.0, 5.0])
N_TIMES = 50
ODE_SOLVERS = ["tsit45", "bdf"]


def lotka_volterra(t, y, p):
    x, yy = y[0], y[1]
    alpha, beta, delta, gamma = p[0], p[1], p[2], p[3]
    return (alpha * x - beta * x * yy, delta * x * yy - gamma * yy)


def make_problem(ode_solver="bdf"):
    return ODEProblem(lotka_volterra, Y0, PARAMS, n_times=N_TIMES)


@pytest.mark.parametrize("ode_solver", ODE_SOLVERS)
def test_jvp_matches_fd(ode_solver):
    prob = make_problem(ode_solver)
    handle = prob._handle
    n_state = len(Y0)
    n_params = len(PARAMS)
    method_code = {"bdf": 0, "tsit45": 1, "esdirk34": 2, "tr_bdf2": 3}[ode_solver]
    t0, t_final = float(T_SPAN[0]), float(T_SPAN[1])

    rng = np.random.default_rng(42)
    dp = jnp.array(rng.standard_normal(n_params))

    dys = _solve_jvp(handle, PARAMS, t0, t_final, dp, N_TIMES, n_state, method_code)

    eps = 1e-4
    ys_plus, _ = _solve_forward(
        handle, PARAMS + eps * dp, t0, t_final, N_TIMES, n_state, method_code
    )
    ys_minus, _ = _solve_forward(
        handle, PARAMS - eps * dp, t0, t_final, N_TIMES, n_state, method_code
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
    prob = make_problem(ode_solver)
    handle = prob._handle
    n_state = len(Y0)
    n_params = len(PARAMS)
    method_code = {"bdf": 0, "tsit45": 1, "esdirk34": 2, "tr_bdf2": 3}[ode_solver]
    t0, t_final = float(T_SPAN[0]), float(T_SPAN[1])

    dp = jnp.zeros(n_params)

    dys = _solve_jvp(handle, PARAMS, t0, t_final, dp, N_TIMES, n_state, method_code)

    assert jnp.max(jnp.abs(dys)) < 1e-10, (
        f"{ode_solver}: zero tangents should give zero dys, "
        f"got max={float(jnp.max(jnp.abs(dys))):.2e}"
    )
