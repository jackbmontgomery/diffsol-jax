import jax
import jax.numpy as jnp
import numpy as np
import pytest
from diffsol_jax import make_diffsol_solver
from scipy.integrate import solve_ivp

jax.config.update("jax_enable_x64", True)

PARAMS = jnp.array([1.5, 1.0, 0.75, 3.0])
Y0 = jnp.array([1.0, 0.5])
T_SPAN = jnp.array([0.0, 10.0])
ODE_SOLVERS = ["bdf", "tsit45", "esdirk34", "tr_bdf2"]


def lotka_volterra(t, y, p):
    x, yy = y[0], y[1]
    alpha, beta, delta, gamma = p[0], p[1], p[2], p[3]
    return (alpha * x - beta * x * yy, delta * x * yy - gamma * yy)


def scipy_reference(ts):
    def f(t, y):
        a, b, d, g = PARAMS
        return [a * y[0] - b * y[0] * y[1], d * y[0] * y[1] - g * y[1]]

    return solve_ivp(
        f, (0.0, 10.0), [1.0, 0.5], t_eval=np.asarray(ts), rtol=1e-8, atol=1e-10
    )


@pytest.mark.parametrize("ode_solver", ODE_SOLVERS)
def test_lv_matches_scipy(ode_solver):
    solver, _ = make_diffsol_solver(
        lotka_volterra,
        y0=Y0,
        p_example=PARAMS,
        param_names=["alpha", "beta", "delta", "gamma"],
        state_names=["x", "y"],
        ode_solver=ode_solver,
        n_times=100,
    )
    ys, ts = solver(PARAMS, T_SPAN)
    ref = scipy_reference(ts)
    diff = np.max(np.abs(np.asarray(ys) - ref.y.T))
    assert diff < 1e-3, f"{ode_solver}: max diff {diff:.2e}"


@pytest.mark.parametrize("ode_solver", ODE_SOLVERS)
def test_lv_grad_matches_fd(ode_solver):
    solver, _ = make_diffsol_solver(
        lotka_volterra,
        y0=Y0,
        p_example=PARAMS,
        param_names=["alpha", "beta", "delta", "gamma"],
        state_names=["x", "y"],
        ode_solver=ode_solver,
        n_times=50,
    )

    def loss(p):
        ys, _ = solver(p, T_SPAN)
        return jnp.sum(ys**2)

    grad = jax.grad(loss)(PARAMS)

    eps = 1e-4
    fd = jnp.array(
        [
            (
                loss(PARAMS.at[i].set(PARAMS[i] + eps))
                - loss(PARAMS.at[i].set(PARAMS[i] - eps))
            )
            / (2 * eps)
            for i in range(len(PARAMS))
        ]
    )
    rel = jnp.max(jnp.abs(grad - fd) / (jnp.abs(fd) + 1e-6))
    assert rel < 1e-2, f"{ode_solver}: grad rel error {rel:.2e}"


def test_unknown_solver_raises():
    with pytest.raises(ValueError, match="unknown solver"):
        make_diffsol_solver(
            lotka_volterra, y0=Y0, p_example=PARAMS, ode_solver="rk4_custom"
        )
