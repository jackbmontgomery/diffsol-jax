import time

import diffrax
import jax
import jax.numpy as jnp
from diffsol_jax import make_diffsol_solver

jax.config.update("jax_enable_x64", True)

PARAMS = jnp.array([1.5, 1.0, 0.75, 3.0])
Y0 = jnp.array([1.0, 0.5])
T_SPAN = jnp.array([0.0, 10.0])
N_TIMES = 200
N_WARMUP = 3
N_REPEAT = 20


def lotka_volterra(t, y, p):
    x, yy = y[0], y[1]
    alpha, beta, delta, gamma = p[0], p[1], p[2], p[3]
    return (alpha * x - beta * x * yy, delta * x * yy - gamma * yy)


# diffsol-jax

solver_ds, _ = make_diffsol_solver(
    lotka_volterra,
    y0=Y0,
    p_example=PARAMS,
    method="tsit45",
    param_names=["alpha", "beta", "delta", "gamma"],
    state_names=["x", "y"],
    n_times=N_TIMES,
)

diffsol_jit = jax.jit(lambda p: solver_ds(p, T_SPAN))

for _ in range(N_WARMUP):
    ys_ds, _ = diffsol_jit(PARAMS)
    ys_ds.block_until_ready()

t0 = time.perf_counter()
for _ in range(N_REPEAT):
    ys_ds, _ = diffsol_jit(PARAMS)
    ys_ds.block_until_ready()
ds_ms = (time.perf_counter() - t0) / N_REPEAT * 1e3


# diffrax (Dopri5)

ts_save = jnp.linspace(0.0, 10.0, N_TIMES)


def diffrax_solve(p):
    def rhs(t, y, args):
        x, yy = y[0], y[1]
        alpha, beta, delta, gamma = args[0], args[1], args[2], args[3]
        return jnp.array([alpha * x - beta * x * yy, delta * x * yy - gamma * yy])

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(rhs),
        diffrax.Tsit5(),
        t0=0.0,
        t1=10.0,
        dt0=0.05,
        y0=Y0,
        args=p,
        saveat=diffrax.SaveAt(ts=ts_save),
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8),
    )
    return sol.ys


diffrax_jit = jax.jit(diffrax_solve)

for _ in range(N_WARMUP):
    ys_dx = diffrax_jit(PARAMS)
    ys_dx.block_until_ready()

t0 = time.perf_counter()
for _ in range(N_REPEAT):
    ys_dx = diffrax_jit(PARAMS)
    ys_dx.block_until_ready()
dx_ms = (time.perf_counter() - t0) / N_REPEAT * 1e3


# results

max_diff = float(jnp.max(jnp.abs(ys_ds - ys_dx)))

print(f"Lotka-Volterra forward solve (n_times={N_TIMES}, n={N_REPEAT})")
print(f"  diffsol-jax (BDF, Cranelift): {ds_ms:.2f} ms/call")
print(f"  diffrax     (Dopri5):         {dx_ms:.2f} ms/call")
print(f"  speedup:                      {dx_ms / ds_ms:.2f}x")
print(f"  max |diff|:                   {max_diff:.2e}")
