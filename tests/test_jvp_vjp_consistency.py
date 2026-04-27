"""
JVP/VJP consistency — dot-product / adjoint test.

For random dp and g:
    <g, J·dp>  ==  <grad_p, dp>

LHS uses _solve_jvp; RHS uses _solve_adjoint.
These are independent Rust paths, so agreement at near-machine-precision
confirms forward and reverse modes are mathematically consistent.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from diffsol_jax import ODEProblem
from diffsol_jax import _solve_jvp, _solve_adjoint

jax.config.update("jax_enable_x64", True)

PARAMS = jnp.array([1.5, 1.0, 0.75, 3.0])
Y0 = jnp.array([1.0, 0.5])
T_SPAN = jnp.array([0.0, 5.0])
N_TIMES = 50
# JVP only implemented with BDF internally; matches adjoint BDF path
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
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_jvp_vjp_dotproduct(ode_solver, seed):
    """<g, J·dp>  ==  <grad_p, dp> to solver tolerance."""
    prob = make_problem(ode_solver)
    handle = prob._handle
    n_state = len(Y0)
    n_params = len(PARAMS)
    method_code = {"bdf": 0, "tsit45": 1, "esdirk34": 2, "tr_bdf2": 3}[ode_solver]
    t0, t_final = float(T_SPAN[0]), float(T_SPAN[1])

    rng = np.random.default_rng(seed)
    dp = jnp.array(rng.standard_normal(n_params))
    g = jnp.array(rng.standard_normal((N_TIMES, n_state)))

    # LHS: <g, J·dp>
    dys = _solve_jvp(handle, PARAMS, t0, t_final, dp, N_TIMES, n_state, method_code)
    lhs = float(jnp.sum(g * dys))

    # RHS: <grad_p, dp>
    grad_p, _grad_y0 = _solve_adjoint(
        handle, PARAMS, T_SPAN, g, N_TIMES, n_state, method_code
    )
    rhs = float(jnp.dot(grad_p, dp))

    scale = max(abs(lhs), abs(rhs), 1e-12)
    rel = abs(lhs - rhs) / scale
    assert rel < 1e-4, (
        f"{ode_solver} seed={seed}: dot-product mismatch\n"
        f"  lhs={lhs:.10e}  rhs={rhs:.10e}  rel={rel:.2e}"
    )
