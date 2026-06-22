import jax
import jax.numpy as jnp
import numpy as np
import pytest
from diffsol_jax import ODEProblem

PARAMS = [1.5, 1.0, 0.75, 3.0]
Y0 = [1.0, 0.5]
T_SPAN = [0.0, 5.0]
N_TIMES = 50
ODE_SOLVERS = ["tsit45", "bdf"]
DTYPES = [jnp.float32, jnp.float64]

SETTINGS = {
    jnp.float32: dict(rtol=1e-5, atol=1e-6, fd_eps=5e-3, jvp_tol=1e-1),
    jnp.float64: dict(rtol=1e-8, atol=1e-10, fd_eps=1e-4, jvp_tol=5e-2),
}


def lotka_volterra(t, y, p):
    x, yy = y[0], y[1]
    alpha, beta, delta, gamma = p[0], p[1], p[2], p[3]
    return (alpha * x - beta * x * yy, delta * x * yy - gamma * yy)


def _setup(dtype):
    params = jnp.array(PARAMS, dtype=dtype)
    y0 = jnp.array(Y0, dtype=dtype)
    t_span = jnp.array(T_SPAN, dtype=dtype)
    prob = ODEProblem(lotka_volterra, y0, params, n_times=N_TIMES)
    return prob, params, t_span


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("ode_solver", ODE_SOLVERS)
def test_jvp_matches_fd(ode_solver, dtype):
    cfg = SETTINGS[dtype]
    prob, params, t_span = _setup(dtype)

    def ys_of(p):
        return prob.solve(
            p, t_span, ode_solver=ode_solver, rtol=cfg["rtol"], atol=cfg["atol"]
        )[1]

    rng = np.random.default_rng(42)
    dp = jnp.array(rng.standard_normal(len(PARAMS)), dtype=dtype)

    _, dys = jax.jvp(ys_of, (params,), (dp,))
    assert dys.dtype == dtype

    eps = cfg["fd_eps"]
    fd = (ys_of(params + eps * dp) - ys_of(params - eps * dp)) / (2 * eps)

    # Compare in float64 so the error ratio isn't itself dominated by f32 noise.
    dys = np.asarray(dys, dtype=np.float64)
    fd = np.asarray(fd, dtype=np.float64)
    rel = np.max(np.abs(dys - fd) / (np.abs(fd) + 1e-6))
    assert rel < cfg["jvp_tol"], f"{ode_solver}/{dtype}: JVP rel err {rel:.2e}"


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("ode_solver", ODE_SOLVERS)
def test_jvp_zero_tangents(ode_solver, dtype):
    prob, params, t_span = _setup(dtype)

    def ys_of(p):
        return prob.solve(p, t_span, ode_solver=ode_solver)[1]

    dp = jnp.zeros(len(PARAMS), dtype=dtype)
    _, dys = jax.jvp(ys_of, (params,), (dp,))

    assert jnp.max(jnp.abs(dys)) == 0.0, (
        f"{ode_solver}/{dtype}: zero tangents should give exactly zero dys, "
        f"got max={float(jnp.max(jnp.abs(dys))):.2e}"
    )
