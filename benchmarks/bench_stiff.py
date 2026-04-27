import time

import diffrax
import jax
import jax.numpy as jnp
from diffsol_jax import ODEProblem

jax.config.update("jax_enable_x64", True)

MU = 1000.0
PARAMS = jnp.array([MU])
Y0 = jnp.array([2.0, 0.0])
T_END = 2 * MU
T_SPAN = jnp.array([0.0, T_END])
N_TIMES = 200
N_WARMUP = 3
N_REPEAT = 10


def van_der_pol(t, y, p):
    mu = p[0]
    y1, y2 = y[0], y[1]
    return (y2, mu * (1.0 - y1 * y1) * y2 - y1)


# diffsol-jax (BDF)

ode_problem = ODEProblem(van_der_pol, Y0, PARAMS, N_TIMES)

diffsol_jit = jax.jit(lambda p: ode_problem.solve(Y0, p, T_SPAN))

for _ in range(N_WARMUP):
    _, ys_ds = diffsol_jit(PARAMS)
    ys_ds.block_until_ready()

t0 = time.perf_counter()
for _ in range(N_REPEAT):
    _, ys_ds = diffsol_jit(PARAMS)
    ys_ds.block_until_ready()
ds_ms = (time.perf_counter() - t0) / N_REPEAT * 1e3


# diffrax (Kvaerno5)

ts_save = jnp.linspace(0.0, T_END, N_TIMES)


def diffrax_solve(p):
    mu = p[0]

    def rhs(t, y, args):
        y1, y2 = y[0], y[1]
        return jnp.array([y2, args * (1.0 - y1 * y1) * y2 - y1])

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(rhs),
        diffrax.Kvaerno5(),
        t0=0.0,
        t1=T_END,
        dt0=1.0,
        y0=Y0,
        args=mu,
        saveat=diffrax.SaveAt(ts=ts_save),
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8),
        max_steps=500_000,
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

print(f"Van der Pol mu={MU:.0f} t=[0, {T_END:.0f}] (n_times={N_TIMES}, n={N_REPEAT})")
print(f"  diffsol-jax (BDF):      {ds_ms:.1f} ms/call")
print(f"  diffrax     (Kvaerno5): {dx_ms:.1f} ms/call")
print(f"  speedup:                {dx_ms / ds_ms:.2f}x")
print(f"  max |diff|:             {max_diff:.2e}")
