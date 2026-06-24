import time

import diffrax
import jax
import jax.numpy as jnp
from diffsol_jax import ODEProblem, OdeSolverType

PARAMS = jnp.array([1.5, 1.0, 0.75, 3.0])
Y0 = jnp.array([1.0, 0.5])
N_TIMES = 200
T_EVAL = jnp.linspace(0.0, 100.0, N_TIMES)
N_WARMUP = 3
N_REPEAT = 20


def lotka_volterra(t, y, p):
    x, yy = y[0], y[1]
    alpha, beta, delta, gamma = p[0], p[1], p[2], p[3]
    return (alpha * x - beta * x * yy, delta * x * yy - gamma * yy)


def diffrax_solve(p):
    def rhs(t, y, args):
        x, yy = y[0], y[1]
        alpha, beta, delta, gamma = args[0], args[1], args[2], args[3]
        return jnp.array([alpha * x - beta * x * yy, delta * x * yy - gamma * yy])

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(rhs),
        diffrax.Tsit5(),
        t0=T_EVAL[0],
        t1=T_EVAL[-1],
        dt0=0.05,
        y0=Y0,
        args=p,
        saveat=diffrax.SaveAt(ts=T_EVAL),
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8),
    )
    return sol.ys


def run():
    ode_problem = ODEProblem(lotka_volterra, Y0, PARAMS)
    diffsol_jit = jax.jit(
        lambda p: ode_problem.solve(
            p, T_EVAL, ode_solver=OdeSolverType.TSIT45, rtol=1e-8, atol=1e-8
        )
    )

    for _ in range(N_WARMUP):
        _, ys_ds = diffsol_jit(PARAMS)
        ys_ds.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(N_REPEAT):
        _, ys_ds = diffsol_jit(PARAMS)
        ys_ds.block_until_ready()
    ds_ms = (time.perf_counter() - t0) / N_REPEAT * 1e3

    diffrax_jit = jax.jit(diffrax_solve)
    for _ in range(N_WARMUP):
        ys_dx = diffrax_jit(PARAMS)
        ys_dx.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(N_REPEAT):
        ys_dx = diffrax_jit(PARAMS)
        ys_dx.block_until_ready()
    dx_ms = (time.perf_counter() - t0) / N_REPEAT * 1e3

    return {
        "label": "Lotka-Volterra (non-stiff)",
        "ds_ms": ds_ms,
        "dx_ms": dx_ms,
        "dx_solver": "Tsit5",
        "speedup": dx_ms / ds_ms,
        "max_diff": float(jnp.max(jnp.abs(ys_ds - ys_dx))),
    }


def main():
    r = run()
    print(f"Lotka-Volterra forward solve (n_times={N_TIMES}, n={N_REPEAT})")
    print(f"  diffsol-jax (Tsit45):   {r['ds_ms']:.2f} ms/call")
    print(f"  diffrax     (Tsit5):    {r['dx_ms']:.2f} ms/call")
    print(f"  speedup:                {r['speedup']:.2f}x")
    print(f"  max |diff|:             {r['max_diff']:.2e}")


if __name__ == "__main__":
    main()
