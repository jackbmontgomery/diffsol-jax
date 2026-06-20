"""
JVP (forward sensitivity) through the public ``jax.jvp`` interface.

Verifies:
- dys ≈ (solve(p + eps*dp) - solve(p - eps*dp)) / (2 eps)  (finite-difference, params only)
- Zero tangents produce zero dys.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from diffsol_jax import ODEProblem

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


def make_problem():
    return ODEProblem(lotka_volterra, Y0, PARAMS, n_times=N_TIMES)


@pytest.mark.parametrize("ode_solver", ODE_SOLVERS)
def test_jvp_matches_fd(ode_solver):
    prob = make_problem()

    def ys_of(p):
        return prob.solve(p, T_SPAN, ode_solver=ode_solver)[1]

    rng = np.random.default_rng(42)
    dp = jnp.array(rng.standard_normal(len(PARAMS)))

    _, dys = jax.jvp(ys_of, (PARAMS,), (dp,))

    eps = 1e-4
    fd = (ys_of(PARAMS + eps * dp) - ys_of(PARAMS - eps * dp)) / (2 * eps)

    rel = jnp.max(jnp.abs(dys - fd) / (jnp.abs(fd) + 1e-6))
    assert rel < 5e-2, (
        f"{ode_solver}: JVP rel err {rel:.2e}\n"
        f"  max|dys|={float(jnp.max(jnp.abs(dys))):.3e}\n"
        f"  max|fd| ={float(jnp.max(jnp.abs(fd))):.3e}"
    )


@pytest.mark.parametrize("ode_solver", ODE_SOLVERS)
def test_jvp_zero_tangents(ode_solver):
    prob = make_problem()

    def ys_of(p):
        return prob.solve(p, T_SPAN, ode_solver=ode_solver)[1]

    dp = jnp.zeros(len(PARAMS))
    _, dys = jax.jvp(ys_of, (PARAMS,), (dp,))

    assert jnp.max(jnp.abs(dys)) < 1e-10, (
        f"{ode_solver}: zero tangents should give zero dys, "
        f"got max={float(jnp.max(jnp.abs(dys))):.2e}"
    )
