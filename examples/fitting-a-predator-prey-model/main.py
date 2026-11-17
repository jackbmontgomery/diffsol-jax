import pathlib
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax
from jaxtyping import Array, ArrayLike
from tqdm import trange

from diffsol_jax import Model, System

jax.config.update("jax_enable_x64", True)

BASE_DIR = pathlib.Path(__file__).resolve().parent

T_EVAL = jnp.linspace(0.0, 5.0, num=100)
NOISE_STD = 0.05
STEPS = 200
LR = 1e-2
NSTATES = 2


class LotkaVolterraParams(eqx.Module):
    alpha: Array
    beta: Array
    gamma: Array
    delta: Array

    def __init__(
        self,
        alpha: ArrayLike = 1.3,
        beta: ArrayLike = 0.9,
        gamma: ArrayLike = 0.8,
        delta: ArrayLike = 1.8,
    ):
        self.alpha = jnp.asarray(alpha)
        self.beta = jnp.asarray(beta)
        self.gamma = jnp.asarray(gamma)
        self.delta = jnp.asarray(delta)


class LotkaVolterra(System):
    y0: Array
    params: LotkaVolterraParams

    def __call__(self, x: Array, t: Array) -> Array:
        prey, predator = x[0], x[1]
        p = self.params
        return jnp.array(
            [
                p.alpha * prey - p.beta * predator * prey,
                p.gamma * prey * predator - p.delta * predator,
            ]
        )


def generate_data(system: LotkaVolterra, t_eval: Array, *, key: Array) -> Array:
    ys = Model(system, nstates=NSTATES, ode_solver="tsit45").solve(
        system.y0, system, t_eval
    )
    return ys + NOISE_STD * jr.normal(key=key, shape=ys.shape)


def fit(
    model: Model, system: LotkaVolterra, t_eval: Array, ys_data: Array
) -> LotkaVolterra:
    opt = optax.adam(LR)
    opt_state = opt.init(system)

    def loss_fn(system, ts_batch, ys_batch):
        return jnp.sum((model.solve(system.y0, system, ts_batch) - ys_batch) ** 2)

    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    @jax.jit
    def step(system, opt_state):
        loss_val, grads = value_and_grad(system, t_eval, ys_data)
        updates, opt_state = opt.update(grads, opt_state, system)
        return optax.apply_updates(system, updates), opt_state, loss_val

    pbar = trange(STEPS, desc="fit")
    for _ in pbar:
        system, opt_state, loss_val = step(system, opt_state)
        pbar.set_postfix(loss=f"{loss_val:.4f}")
    return system


def plot_fit(t_eval, ys_data, ys_true, ys_fit, path):
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for ax, name, i in zip(axes, ["prey", "predator"], range(2)):
        ax.plot(t_eval, ys_data[:, i], "o", ms=3, color="black", label="data")
        ax.plot(t_eval, ys_true[:, i], "--", color="gray", label="true")
        ax.plot(t_eval, ys_fit[:, i], "-", label="fit")
        ax.set_ylabel(name)
        ax.legend(fontsize="small")
    axes[-1].set_xlabel("t")
    fig.suptitle("fitting-a-predator-prey-model")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def time_call(fn, *args, repeats=20):
    jax.block_until_ready(fn(*args))
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - t0)
    return min(times)


def compare_ad_modes(system: LotkaVolterra, t_eval: Array, ys_data: Array):
    print(f"\n{'':20s}{'grad(fit loss)':>18s}{'jacrev(trajectory)':>22s}")
    for ad in ("adjoint", "forward"):
        model = Model(system, nstates=NSTATES, ode_solver="tsit45", ad=ad)

        def loss_fn(system):
            return jnp.sum((model.solve(system.y0, system, t_eval) - ys_data) ** 2)

        grad_time = time_call(jax.jit(jax.grad(loss_fn)), system)
        jacrev_time = time_call(
            jax.jit(jax.jacrev(lambda s: model.solve(s.y0, s, t_eval))), system
        )
        print(f"ad={ad!r:10s}{grad_time * 1e3:15.3f} ms{jacrev_time * 1e3:19.3f} ms")


def main():
    key = jr.key(0)
    y0 = 5.0 * jr.uniform(key=key, shape=(2,))
    system_true = LotkaVolterra(y0=y0, params=LotkaVolterraParams())
    ys_true = Model(system_true, nstates=NSTATES, ode_solver="tsit45").solve(
        system_true.y0, system_true, T_EVAL
    )
    ys_data = generate_data(system_true, T_EVAL, key=key)

    model = Model(system_true, nstates=NSTATES, ode_solver="tsit45")  # ad="adjoint"
    system_init = LotkaVolterra(
        y0=y0, params=LotkaVolterraParams(alpha=1.1, beta=1.0, gamma=0.5, delta=1.6)
    )
    system_fit = fit(model, system_init, T_EVAL, ys_data)
    print(system_fit.params)

    ys_fit = model.solve(system_fit.y0, system_fit, T_EVAL)
    plot_fit(
        T_EVAL, ys_data, ys_true, ys_fit, BASE_DIR / "fitting-a-predator-prey-model.png"
    )

    compare_ad_modes(system_fit, T_EVAL, ys_data)


if __name__ == "__main__":
    main()
