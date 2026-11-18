"""
Fitting a predator-prey model.
"""

import pathlib

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from jaxtyping import Array
import optax
from tqdm import trange

from diffsol_jax import ODEProblem, solve

jax.config.update("jax_enable_x64", True)

BASE_DIR = pathlib.Path(__file__).resolve().parent


STEPS = 200
LR = 1e-2


class LotkaVolterraParams(eqx.Module):
    alpha: Array
    beta: Array
    gamma: Array
    delta: Array


def rhs(x: Array, p: LotkaVolterraParams, _t: float) -> Array:
    prey, predator = x[0], x[1]
    return jnp.array(
        [
            p.alpha * prey - p.beta * predator * prey,
            p.gamma * prey * predator - p.delta * predator,
        ]
    )


def trajectory(
    prob: ODEProblem, y0: Array, p: LotkaVolterraParams, t_eval: Array, **options
) -> Array:
    return solve(prob, y0, p, t_eval, **options)


def generate_data(
    prob: ODEProblem,
    y0: Array,
    p: LotkaVolterraParams,
    t_eval: Array,
    *,
    key: Array,
    noise_std=0.05,
) -> Array:
    ys = trajectory(prob, y0, p, t_eval)
    return ys + noise_std * jr.normal(key=key, shape=ys.shape)


def fit(
    prob: ODEProblem, y0: Array, p: LotkaVolterraParams, t_eval: Array, ys_data: Array
) -> LotkaVolterraParams:
    opt = optax.adam(LR)
    opt_state = opt.init(p)

    def loss_fn(p, ys_batch):
        return jnp.sum((trajectory(prob, y0, p, t_eval) - ys_batch) ** 2)

    value_and_grad = jax.value_and_grad(loss_fn)

    @jax.jit
    def step(v, opt_state):
        loss_val, grads = value_and_grad(v, ys_data)
        updates, opt_state = opt.update(grads, opt_state, v)
        return optax.apply_updates(v, updates), opt_state, loss_val

    pbar = trange(STEPS, desc="fit")
    for _ in pbar:
        p, opt_state, loss_val = step(p, opt_state)
        pbar.set_postfix(loss=f"{loss_val:.4f}")
    return p


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


def main():
    key = jr.key(0)
    t_eval = jnp.linspace(0.0, 5.0, num=100)
    y0 = 5.0 * jr.uniform(key=key, shape=(2,))
    p_true = LotkaVolterraParams(
        alpha=jnp.asarray(1.3),
        beta=jnp.asarray(0.9),
        gamma=jnp.asarray(0.8),
        delta=jnp.asarray(1.8),
    )
    prob = ODEProblem(rhs, state=len(y0), params=p_true)
    ys_true = trajectory(prob, y0, p_true, t_eval)
    ys_data = generate_data(prob, y0, p_true, t_eval, key=key)

    p = LotkaVolterraParams(
        alpha=jnp.asarray(1.1),
        beta=jnp.asarray(1.0),
        gamma=jnp.asarray(0.5),
        delta=jnp.asarray(1.6),
    )
    p_fit = fit(prob, y0, p, t_eval, ys_data)
    print(jax.tree.leaves(p_fit))

    ys_fit = trajectory(prob, y0, p_fit, t_eval)
    plot_fit(
        t_eval, ys_data, ys_true, ys_fit, BASE_DIR / "fitting-a-predator-prey-model.png"
    )


if __name__ == "__main__":
    main()
