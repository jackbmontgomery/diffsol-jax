"""
Neural ODE weather prediction.

The MLP drives the right-hand side of an ODE, trained to fit the Delhi monthly climate data
(mean temperature, humidity, wind speed, mean pressure).
"""

import pathlib

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import optax
from diffsol_jax import ODEProblem, solve
from tqdm import trange

BASE_DIR = pathlib.Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "MonthlyDelhiClimate.csv"
PLOT_DIR = BASE_DIR / "plots"

DATA_DIM = 4
COLUMNS = ["meantemp", "humidity", "wind_speed", "meanpressure"]

BATCH_STEP = 4
STEPS_PER_ROUND = 250
LR = 1e-3
WEIGHT_DECAY = 1e-2


def load_data():
    data = np.loadtxt(DATA_PATH, delimiter=",", skiprows=1)
    return jnp.asarray(data[:, 0]), jnp.asarray(data[:, 1:])


class NeuralNetwork(eqx.Module):
    layers: list[eqx.nn.Linear]

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jax.nn.silu(self.layers[0](x))
        x = jax.nn.silu(self.layers[1](x))
        return self.layers[2](x)


def make_network(data_dim: int, key: jax.Array) -> NeuralNetwork:
    key1, key2, key3 = jax.random.split(key, 3)
    return NeuralNetwork(
        layers=[
            eqx.nn.Linear(data_dim, 64, key=key1),
            eqx.nn.Linear(64, 32, key=key2),
            eqx.nn.Linear(32, data_dim, key=key3),
        ]
    )


def train_one_round(prob, y0, net, ts_batch, ys_batch, round_idx):
    opt = optax.adamw(LR)
    opt_state = opt.init(net)

    def loss_fn(p, t_eval, ys):
        return jnp.sum(
            (solve(prob, y0, p, t_eval[0], t_eval[-1], saveat=t_eval) - ys) ** 2
        )

    value_and_grad = jax.value_and_grad(loss_fn)

    @jax.jit
    def step(p, opt_state, t_eval, ys_data):
        loss_val, grads = value_and_grad(p, t_eval, ys_data)
        updates, opt_state = opt.update(grads, opt_state, p)
        return optax.apply_updates(p, updates), opt_state, loss_val

    pbar = trange(STEPS_PER_ROUND, desc=f"round {round_idx}", leave=False)
    for _ in pbar:
        net, opt_state, loss_val = step(net, opt_state, ts_batch, ys_batch)
        pbar.set_postfix(loss=f"{loss_val:.4f}")
    return net


def plot_round(round_idx, batch_len, ts, ys, predict_ys, path):
    fig, axes = plt.subplots(DATA_DIM, 1, figsize=(8, 10), sharex=True)
    for i, (ax, name) in enumerate(zip(axes, COLUMNS)):
        ax.plot(ts, ys[:, i], "o", ms=3, color="black", label=f"{name}_data")
        ax.plot(ts[:batch_len], predict_ys[:batch_len, i], "-", label=f"{name}_train")
        ax.plot(
            ts[batch_len:], predict_ys[batch_len:, i], "--", label=f"{name}_predict"
        )
        ax.set_ylabel(name)
        ax.legend(fontsize="small")
    axes[-1].set_xlabel("month")
    fig.suptitle(f"neural-ode-weather-prediction, round {round_idx}")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    PLOT_DIR.mkdir(exist_ok=True)

    ts, ys = load_data()
    y0 = ys[0]
    net0 = make_network(DATA_DIM, key=jr.key(0))

    prob = ODEProblem(lambda x, net, t: net(x), state=y0, params=net0, solver="tsit45")

    net = net0
    ntimes = ts.shape[0]
    rounds = trange(1, ntimes // BATCH_STEP, desc="rounds")
    for i in rounds:
        batch_len = i * BATCH_STEP
        rounds.set_postfix(data_points=batch_len)
        ts_batch = ts[:batch_len]
        ys_batch = ys[:batch_len]
        net = train_one_round(prob, y0, net, ts_batch, ys_batch, i)

        predict_ys = solve(prob, y0, net, ts[0], ts[-1], saveat=ts)

        plot_round(
            i, batch_len, ts, ys, predict_ys, PLOT_DIR / f"neural-ode-weather_{i}.png"
        )


if __name__ == "__main__":
    main()
