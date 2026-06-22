import jax
import jax.numpy as jnp
import numpy as np
import pytest
from diffsol_jax import ODEProblem
from scipy.integrate import solve_ivp

jax.config.update("jax_enable_x64", True)

PARAMS = [1.5, 1.0, 0.75, 3.0]
Y0 = [1.0, 0.5]
T_SPAN = [0.0, 10.0]
ODE_SOLVERS = ["bdf", "tsit45", "esdirk34", "tr_bdf2"]
DTYPES = [jnp.float32, jnp.float64]

SETTINGS = {
    jnp.float32: dict(rtol=1e-5, atol=1e-6, fwd_tol=3e-2, grad_tol=1e-1, fd_eps=5e-3),
    jnp.float64: dict(rtol=1e-8, atol=1e-10, fwd_tol=1e-3, grad_tol=1e-2, fd_eps=1e-4),
}


def _inputs(dtype):
    return (
        jnp.array(PARAMS, dtype=dtype),
        jnp.array(Y0, dtype=dtype),
        jnp.array(T_SPAN, dtype=dtype),
    )


def lotka_volterra(t, y, p):
    x, yy = y[0], y[1]
    alpha, beta, delta, gamma = p[0], p[1], p[2], p[3]
    return (alpha * x - beta * x * yy, delta * x * yy - gamma * yy)


def scipy_reference(ts):
    def f(t, y):
        a, b, d, g = PARAMS
        return [a * y[0] - b * y[0] * y[1], d * y[0] * y[1] - g * y[1]]

    return solve_ivp(
        f,
        (0.0, 10.0),
        Y0,
        t_eval=np.asarray(ts, dtype=np.float64),
        rtol=1e-8,
        atol=1e-10,
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("ode_solver", ODE_SOLVERS)
def test_lv_matches_scipy(ode_solver, dtype):
    cfg = SETTINGS[dtype]
    params, y0, t_span = _inputs(dtype)

    ode_problem = ODEProblem(lotka_volterra, y0, params, n_times=100)
    ts, ys = ode_problem.solve(
        params, t_span, ode_solver=ode_solver, rtol=cfg["rtol"], atol=cfg["atol"]
    )

    assert ys.dtype == dtype
    assert ts.dtype == dtype

    ref = scipy_reference(ts)
    diff = np.max(np.abs(np.asarray(ys, dtype=np.float64) - ref.y.T))
    assert diff < cfg["fwd_tol"], f"{ode_solver}/{dtype}: max diff {diff:.2e}"


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("ode_solver", ODE_SOLVERS)
def test_lv_grad_matches_fd(ode_solver, dtype):
    cfg = SETTINGS[dtype]
    params, y0, t_span = _inputs(dtype)

    ode_problem = ODEProblem(lotka_volterra, y0, params, n_times=100)

    def loss(p):
        _, ys = ode_problem.solve(
            p, t_span, ode_solver=ode_solver, rtol=cfg["rtol"], atol=cfg["atol"]
        )
        return jnp.sum(ys**2)

    grad = jax.grad(loss)(params)
    assert grad.dtype == dtype

    eps = cfg["fd_eps"]
    fd = jnp.array(
        [
            (
                loss(params.at[i].set(params[i] + eps))
                - loss(params.at[i].set(params[i] - eps))
            )
            / (2 * eps)
            for i in range(len(params))
        ],
        dtype=dtype,
    )
    # Compare in float64 to avoid the ratio itself being dominated by f32 noise.
    grad = np.asarray(grad, dtype=np.float64)
    fd = np.asarray(fd, dtype=np.float64)
    rel = np.max(np.abs(grad - fd) / (np.abs(fd) + 1e-6))
    assert rel < cfg["grad_tol"], f"{ode_solver}/{dtype}: grad rel error {rel:.2e}"


def test_unknown_solver_raises():
    params, y0, t_span = _inputs(jnp.float64)
    prob = ODEProblem(lotka_volterra, y0=y0, params=params)
    with pytest.raises(ValueError, match="unknown solver"):
        prob.solve(params, t_span, ode_solver="rk4_custom")
