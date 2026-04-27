import time

import diffrax
import jax
import jax.numpy as jnp
import optax
from diffsol_jax import ODEProblem

jax.config.update("jax_enable_x64", True)

TRUE_P = jnp.array([1.5, 1.0, 0.75, 3.0])
Y0 = jnp.array([1.0, 0.5])
T_SPAN = jnp.array([0.0, 10.0])
N_TIMES = 50
N_STEPS = 500
LR = 1e-2
N_WARMUP = 2
N_REPEAT = 10

PERTURB = 0.2 * jax.random.normal(jax.random.key(0), (4,))
INIT_P = TRUE_P + PERTURB

ts_save = jnp.linspace(0.0, 10.0, N_TIMES)


def lotka_volterra_ds(t, y, p):
    x, yy = y[0], y[1]
    alpha, beta, delta, gamma = p[0], p[1], p[2], p[3]
    return (alpha * x - beta * x * yy, delta * x * yy - gamma * yy)


# --- diffsol-jax setup ---

ode_problem = ODEProblem(lotka_volterra_ds, Y0, TRUE_P, N_TIMES, ode_solver="tsit45")

_, target_ys = ode_problem.solve(TRUE_P, T_SPAN)


def loss_ds(p):
    _, ys = ode_problem.solve(p, T_SPAN)
    return jnp.mean((ys - target_ys) ** 2)


@jax.jit
def step_ds(p, state):
    loss_val, g = jax.value_and_grad(loss_ds)(p)
    updates, state = opt.update(g, state)
    return optax.apply_updates(p, updates), state, loss_val


def run_gd_ds():
    p = INIT_P
    opt_state = opt.init(p)
    for _ in range(N_STEPS):
        p, opt_state, _ = step_ds(p, opt_state)
    p.block_until_ready()
    return p


# --- diffrax setup ---


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


def loss_dx(p):
    ys = diffrax_solve(p)
    return jnp.mean((ys - target_ys) ** 2)


@jax.jit
def step_dx(p, state):
    loss_val, g = jax.value_and_grad(loss_dx)(p)
    updates, state = opt.update(g, state)
    return optax.apply_updates(p, updates), state, loss_val


def run_gd_dx():
    p = INIT_P
    opt_state = opt.init(p)
    for _ in range(N_STEPS):
        p, opt_state, _ = step_dx(p, opt_state)
    p.block_until_ready()
    return p


# --- warmup & benchmark ---

opt = optax.adam(LR)

print("Warming up diffsol-jax...")
for _ in range(N_WARMUP):
    run_gd_ds()

print("Warming up diffrax...")
for _ in range(N_WARMUP):
    run_gd_dx()

print(f"Benchmarking ({N_REPEAT} runs x {N_STEPS} adam steps)...")

t0 = time.perf_counter()
for _ in range(N_REPEAT):
    final_p_ds = run_gd_ds()
ds_ms = (time.perf_counter() - t0) / N_REPEAT * 1e3

t0 = time.perf_counter()
for _ in range(N_REPEAT):
    final_p_dx = run_gd_dx()
dx_ms = (time.perf_counter() - t0) / N_REPEAT * 1e3

# --- results ---

ds_err = float(jnp.max(jnp.abs(final_p_ds - TRUE_P)))
dx_err = float(jnp.max(jnp.abs(final_p_dx - TRUE_P)))

print()
print(
    f"Lotka-Volterra param fitting ({N_STEPS} adam steps, n_times={N_TIMES}, n={N_REPEAT})"
)
print(f"  diffsol-jax (BDF+AD):   {ds_ms:.1f} ms/run  |  max |p_err|: {ds_err:.4f}")
print(f"  diffrax     (Tsit5+AD): {dx_ms:.1f} ms/run  |  max |p_err|: {dx_err:.4f}")
print(f"  speedup:                {dx_ms / ds_ms:.2f}x")
