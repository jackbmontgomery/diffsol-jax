import jax
import jax.numpy as jnp
import numpy as np
from diffsol_jax import make_diffsol_solver
from diffsol_jax.lowering import lorenz
from scipy.integrate import solve_ivp

jax.config.update("jax_enable_x64", True)


def test_lorenz_short_horizon():
    params = jnp.array([10.0, 28.0, 8.0 / 3.0])
    y0 = jnp.array([1.0, 0.0, 0.0])
    solver, src = make_diffsol_solver(
        lorenz,
        y0=y0,
        p_example=params,
        param_names=["sigma", "rho", "beta"],
        state_names=["x", "y", "z"],
        n_times=200,
    )
    print(src)
    ys, ts = solver(params, 2.0)

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
    solver, _ = make_diffsol_solver(
        lorenz,
        y0=y0,
        p_example=params,
        param_names=["sigma", "rho", "beta"],
        state_names=["x", "y", "z"],
    )
    jit_solver = jax.jit(lambda p: solver(p, 2.0))
    ys1, _ = jit_solver(params)
    ys2, _ = jit_solver(params * 1.01)
    assert ys1.shape == (200, 3)
    assert not jnp.allclose(ys1, ys2)
