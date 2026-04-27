import jax
import jax.numpy as jnp
import numpy as np
from diffsol_jax import ODEProblem
from scipy.integrate import solve_ivp

jax.config.update("jax_enable_x64", True)


def lorenz(t, y, p):
    x, yy, z = y[0], y[1], y[2]
    sigma, rho, beta = p[0], p[1], p[2]
    return (sigma * (yy - x), x * (rho - z) - yy, x * yy - beta * z)


def test_lorenz_short_horizon():
    params = jnp.array([10.0, 28.0, 8.0 / 3.0])
    y0 = jnp.array([1.0, 0.0, 0.0])
    ode_problem = ODEProblem(lorenz, y0, params)
    ts, ys = ode_problem.solve(y0, params, jnp.array([0.0, 2.0]))

    def f(t, y):
        s, r, b = params
        return [s * (y[1] - y[0]), y[0] * (r - y[2]) - y[1], y[0] * y[1] - b * y[2]]

    ref = solve_ivp(
        f, (0.0, 2.0), [1.0, 0.0, 0.0], t_eval=np.asarray(ts), rtol=1e-8, atol=1e-10
    )
    diff = np.max(np.abs(np.asarray(ys) - ref.y.T))
    assert diff < 1e-3, f"Lorenz diff too large: {diff}"


def test_lorenz_under_jit():
    params = jnp.array([10.0, 28.0, 8.0 / 3.0])
    y0 = jnp.array([1.0, 0.0, 0.0])
    ode_problem = ODEProblem(lorenz, y0, params)
    jit_solver = jax.jit(lambda p: ode_problem.solve(y0, p, jnp.array([0.0, 2.0])))
    _, ys1 = jit_solver(params)
    _, ys2 = jit_solver(params * 1.01)
    assert ys1.shape == (200, 3)
    assert not jnp.allclose(ys1, ys2)
