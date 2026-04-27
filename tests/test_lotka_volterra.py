import jax
import jax.numpy as jnp
import numpy as np
from diffsol_jax import ODEProblem
from scipy.integrate import solve_ivp

jax.config.update("jax_enable_x64", True)


def lotka_volterra(t, y, p):
    x, yy = y[0], y[1]
    alpha, beta, delta, gamma = p[0], p[1], p[2], p[3]
    return (alpha * x - beta * x * yy, delta * x * yy - gamma * yy)


def test_lv_matches_scipy():
    params = jnp.array([1.5, 1.0, 0.75, 3.0])
    y0 = jnp.array([1.0, 0.5])
    ode_problem = ODEProblem(lotka_volterra, y0, params, n_times=200)
    ts, ys = ode_problem.solve(y0, params, jnp.array([0.0, 10.0]))

    def f(t, y):
        a, b, d, g = params
        return [a * y[0] - b * y[0] * y[1], d * y[0] * y[1] - g * y[1]]

    ref = solve_ivp(
        f, (0.0, 10.0), [1.0, 0.5], t_eval=np.asarray(ts), rtol=1e-8, atol=1e-10
    )
    diff = np.max(np.abs(np.asarray(ys) - ref.y.T))
    assert diff < 1e-3, f"LV diff too large: {diff}"


def test_lv_under_jit():
    params = jnp.array([1.5, 1.0, 0.75, 3.0])
    y0 = jnp.array([1.0, 0.5])
    ode_problem = ODEProblem(lotka_volterra, y0, params, n_times=200)
    jit_solver = jax.jit(lambda p: ode_problem.solve(y0, p, jnp.array([0.0, 10.0])))
    _, ys1 = jit_solver(params)
    _, ys2 = jit_solver(params * 1.01)
    assert ys1.shape == (200, 2)
    assert not jnp.allclose(ys1, ys2)
