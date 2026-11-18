"""
Neural ODE weather prediction.

The MLP drives the right hand side of an ODE, trained to fit the Delhi monthly climate data
(mean temperature, humidity, wind speed, mean pressure). The network is the `args` of the
solve: `jax.grad` returns a network of gradients, which optax updates directly.
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

from diffsol_jax import ODEProblem, solve

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


y0 = ys[0]
net0 = make_network(DATA_DIM, key=jr.key(0))

# the network is the parameters: the problem declares its structure, each solve passes the
# current weights
PROBLEM = ODEProblem(
    lambda x, net, t: net(x), nstates=DATA_DIM, params=net0, solver="tsit45"
)


def predict(net, t_eval):
    return solve(PROBLEM, y0, t_eval, net)


def loss_fn(net, ts_batch, ys_batch):
    return jnp.sum((predict(net, ts_batch) - ys_batch) ** 2)


value_and_grad = jax.jit(jax.value_and_grad(loss_fn))


def train_one_round(net, ts_batch, ys_batch, round_idx):
    opt = optax.adamw(LR)
    opt_state = opt.init(net)
    pbar = trange(STEPS_PER_ROUND, desc=f"round {round_idx}", leave=False)
    for _ in pbar:
        loss, grads = value_and_grad(net, ts_batch, ys_batch)
        updates, opt_state = opt.update(grads, opt_state, net)
        net = optax.apply_updates(net, updates)
        pbar.set_postfix(loss=f"{loss:.4f}")
    return net


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
    net = net0
    ntimes = ts.shape[0]
    rounds = trange(1, ntimes // BATCH_STEP, desc="rounds")
    for i in rounds:
        batch_len = i * BATCH_STEP
        rounds.set_postfix(data_points=batch_len)
        net = train_one_round(net, ts[:batch_len], ys[:batch_len], i)

        predict_ys = predict(net, ts)
        plot_round(i, batch_len, predict_ys, PLOT_DIR / f"neural-ode-weather_{i}.png")


if __name__ == "__main__":
    main()
