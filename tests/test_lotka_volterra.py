import jax
import jax.numpy as jnp
import numpy as np
from diffsol_jax import make_diffsol_solver
from diffsol_jax.lowering import lotka_volterra
from scipy.integrate import solve_ivp

jax.config.update("jax_enable_x64", True)


def test_lv_matches_scipy():
    params = jnp.array([1.5, 1.0, 0.75, 3.0])
    y0 = jnp.array([1.0, 0.5])
    solver, src = make_diffsol_solver(
        lotka_volterra,
        y0=y0,
        p_example=params,
        param_names=["alpha", "beta", "delta", "gamma"],
        state_names=["x", "y"],
        n_times=200,
    )
    print(src)
    ys, ts = solver(params, 10.0)

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
    solver, _ = make_diffsol_solver(
        lotka_volterra,
        y0=y0,
        p_example=params,
        param_names=["alpha", "beta", "delta", "gamma"],
        state_names=["x", "y"],
    )
    jit_solver = jax.jit(lambda p: solver(p, 10.0))
    ys1, _ = jit_solver(params)
    ys2, _ = jit_solver(params * 1.01)
    assert ys1.shape == (200, 2)
    assert not jnp.allclose(ys1, ys2)
