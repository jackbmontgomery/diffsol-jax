"""
Fitting a predator-prey model.

The Lotka-Volterra rates and the initial state are fitted together: both are leaves of the
`args` pytree that `solve` takes, so `jax.grad` returns a pytree of the same structure and optax
updates it directly.
"""

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

from diffsol_jax import ODEProblem, solve

jax.config.update("jax_enable_x64", True)

BASE_DIR = pathlib.Path(__file__).resolve().parent

T_EVAL = jnp.linspace(0.0, 5.0, num=100)
NOISE_STD = 0.05
STEPS = 200
LR = 1e-2


class Rates(eqx.Module):
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


class LotkaVolterra(eqx.Module):
    """Everything the fit varies: the rates and the initial state."""

    y0: Array
    rates: Rates


def rhs(x: Array, p: LotkaVolterra, t: Array) -> Array:
    prey, predator = x[0], x[1]
    r = p.rates
    return jnp.array(
        [
            r.alpha * prey - r.beta * predator * prey,
            r.gamma * prey * predator - r.delta * predator,
        ]
    )


PROBLEM = ODEProblem(rhs)


def trajectory(v: LotkaVolterra, t_eval: Array, **options) -> Array:
    return solve(PROBLEM, v.y0, t_eval, v, solver="tsit45", **options)


def generate_data(v: LotkaVolterra, t_eval: Array, *, key: Array) -> Array:
    ys = trajectory(v, t_eval)
    return ys + NOISE_STD * jr.normal(key=key, shape=ys.shape)


def fit(v: LotkaVolterra, t_eval: Array, ys_data: Array) -> LotkaVolterra:
    opt = optax.adam(LR)
    opt_state = opt.init(v)

    def loss_fn(v, ts_batch, ys_batch):
        return jnp.sum((trajectory(v, ts_batch) - ys_batch) ** 2)

    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    @jax.jit
    def step(v, opt_state):
        loss_val, grads = value_and_grad(v, t_eval, ys_data)
        updates, opt_state = opt.update(grads, opt_state, v)
        return optax.apply_updates(v, updates), opt_state, loss_val

    pbar = trange(STEPS, desc="fit")
    for _ in pbar:
        v, opt_state, loss_val = step(v, opt_state)
        pbar.set_postfix(loss=f"{loss_val:.4f}")
    return v


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


def compare_ad_modes(v: LotkaVolterra, t_eval: Array, ys_data: Array):
    """The two reverse modes solve the same problem: `ad` is a solve option, not another
    problem."""
    print(f"\n{'':20s}{'grad(fit loss)':>18s}{'jacrev(trajectory)':>22s}")
    for ad in ("adjoint", "forward"):

        def loss_fn(v):
            return jnp.sum((trajectory(v, t_eval, ad=ad) - ys_data) ** 2)

        grad_time = time_call(jax.jit(jax.grad(loss_fn)), v)
        jacrev_time = time_call(
            jax.jit(jax.jacrev(lambda v: trajectory(v, t_eval, ad=ad))), v
        )
        print(f"ad={ad!r:10s}{grad_time * 1e3:15.3f} ms{jacrev_time * 1e3:19.3f} ms")


def main():
    key = jr.key(0)
    y0 = 5.0 * jr.uniform(key=key, shape=(2,))
    v_true = LotkaVolterra(y0=y0, rates=Rates())
    ys_true = trajectory(v_true, T_EVAL)
    ys_data = generate_data(v_true, T_EVAL, key=key)

    v_init = LotkaVolterra(
        y0=y0, rates=Rates(alpha=1.1, beta=1.0, gamma=0.5, delta=1.6)
    )
    v_fit = fit(v_init, T_EVAL, ys_data)
    print(v_fit.rates)

    ys_fit = trajectory(v_fit, T_EVAL)
    plot_fit(
        T_EVAL, ys_data, ys_true, ys_fit, BASE_DIR / "fitting-a-predator-prey-model.png"
    )

    compare_ad_modes(v_fit, T_EVAL, ys_data)


if __name__ == "__main__":
    main()
