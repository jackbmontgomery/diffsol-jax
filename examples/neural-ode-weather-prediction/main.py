"""
Neural ODE weather prediction.

The MLP drives the right hand side of an ODE, trained to fit the Delhi monthly climate data
(mean temperature, humidity, wind speed, mean pressure) with `diffsol_jax.Model`. The whole
problem is one `System`: the initial state and the network as fields (trained together, optax
updates the instance directly), the equations as methods.
"""

import pathlib

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import optax
from tqdm import trange

from diffsol_jax import Model, System

jax.config.update("jax_enable_x64", True)

BASE_DIR = pathlib.Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "MonthlyDelhiClimate.csv"
PLOT_DIR = BASE_DIR / "plots"

DATA_DIM = 4
COLUMNS = ["meantemp", "humidity", "wind_speed", "meanpressure"]

BATCH_STEP = 4
STEPS_PER_ROUND = 150
LR = 1e-2
WEIGHT_DECAY = 1e-2


def load_data():
    data = np.loadtxt(DATA_PATH, delimiter=",", skiprows=1)
    return jnp.asarray(data[:, 0]), jnp.asarray(data[:, 1:])


ts, ys = load_data()


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


class WeatherSystem(System):
    y0: jax.Array
    net: NeuralNetwork

    def __call__(self, x: jax.Array, t: jax.Array) -> jax.Array:
        return self.net(x)


system0 = WeatherSystem(y0=ys[0], net=make_network(DATA_DIM, jr.key(0)))

model = Model(system0, nstates=DATA_DIM, ode_solver="tsit45")


def loss_fn(system, ts_batch, ys_batch):
    return jnp.sum((model.solve(system.y0, system, ts_batch) - ys_batch) ** 2)


value_and_grad = jax.jit(jax.value_and_grad(loss_fn))


def train_one_round(system, ts_batch, ys_batch, round_idx):
    opt = optax.adamw(LR)
    opt_state = opt.init(system)
    pbar = trange(STEPS_PER_ROUND, desc=f"round {round_idx}", leave=False)
    for _ in pbar:
        loss, grads = value_and_grad(system, ts_batch, ys_batch)
        updates, opt_state = opt.update(grads, opt_state, system)
        system = optax.apply_updates(system, updates)
        pbar.set_postfix(loss=f"{loss:.4f}")
    return system


def plot_round(round_idx, batch_len, predict_ys, path):
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
    system = system0
    ntimes = ts.shape[0]
    rounds = trange(1, ntimes // BATCH_STEP, desc="rounds")
    for i in rounds:
        batch_len = i * BATCH_STEP
        rounds.set_postfix(data_points=batch_len)
        system = train_one_round(system, ts[:batch_len], ys[:batch_len], i)

        predict_ys = model.solve(system.y0, system, ts)
        plot_round(i, batch_len, predict_ys, PLOT_DIR / f"neural-ode-weather_{i}.png")


if __name__ == "__main__":
    main()
