"""
JVP/VJP consistency — dot-product / adjoint test.

For random dp and g:
    <g, J·dp>  ==  <grad_p, dp>

LHS uses forward mode (``jax.jvp``); RHS uses reverse mode (``jax.vjp``). Both
go through the public ``solve``; reverse mode is JAX's automatic transpose of the
forward-sensitivity contraction, so agreement to solver tolerance confirms the
two modes are mathematically consistent.
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
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_jvp_vjp_dotproduct(ode_solver, seed):
    """<g, J·dp>  ==  <grad_p, dp> to solver tolerance."""
    prob = make_problem()

    def ys_of(p):
        return prob.solve(p, T_SPAN, ode_solver=ode_solver)[1]

    rng = np.random.default_rng(seed)
    dp = jnp.array(rng.standard_normal(len(PARAMS)))
    g = jnp.array(rng.standard_normal((N_TIMES, len(Y0))))

    # LHS: <g, J·dp> via forward mode
    _, dys = jax.jvp(ys_of, (PARAMS,), (dp,))
    lhs = float(jnp.sum(g * dys))

    # RHS: <grad_p, dp> via reverse mode
    _, vjp_fn = jax.vjp(ys_of, PARAMS)
    (grad_p,) = vjp_fn(g)
    rhs = float(jnp.dot(grad_p, dp))

    scale = max(abs(lhs), abs(rhs), 1e-12)
    rel = abs(lhs - rhs) / scale
    assert rel < 1e-3, (
        f"{ode_solver} seed={seed}: dot-product mismatch\n"
        f"  lhs={lhs:.10e}  rhs={rhs:.10e}  rel={rel:.2e}"
    )
