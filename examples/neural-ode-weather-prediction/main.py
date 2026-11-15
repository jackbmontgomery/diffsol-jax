"""
Neural ODE weather prediction.

A replica of diffsol's Rust `neural-ode-weather-prediction` example: an MLP drives the right
hand side of an ODE, trained to fit the Delhi monthly climate data (mean temperature, humidity,
wind speed, mean pressure) with `diffsol_jax.Model`. The training window grows round by round
(a standard trick to avoid getting stuck fitting noise before the broad trend)
"""

import pathlib

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import optax
from jax.flatten_util import ravel_pytree
from tqdm import trange

from diffsol_jax import Model
from model import NeuralNetwork

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
y0 = ys[0]

key = jr.key(0)
neural_net = NeuralNetwork(data_dim=DATA_DIM, key=key)
nn_params, static = eqx.partition(neural_net, eqx.is_array)
nn_p0, unravel_nn_params = ravel_pytree(nn_params)


def rhs(x, p, _t):
    params = unravel_nn_params(p[DATA_DIM:])
    net = eqx.combine(params, static)
    return net(x)


model = Model(
    rhs=rhs,
    init=lambda p, _t: p[:DATA_DIM],
    nstates=DATA_DIM,
    nparams=DATA_DIM + nn_p0.shape[0],
    ode_solver="tsit45",
)


def predict(nn_p, t_eval):
    full_p = jnp.concatenate([y0, nn_p])
    return model.solve(full_p, t_eval)


def loss_fn(nn_p, ts_batch, ys_batch):
    return jnp.sum((predict(nn_p, ts_batch) - ys_batch) ** 2)


value_and_grad = jax.jit(jax.value_and_grad(loss_fn))


def train_one_round(nn_p, ts_batch, ys_batch, round_idx):
    opt = optax.adamw(LR)
    opt_state = opt.init(nn_p)
    pbar = trange(STEPS_PER_ROUND, desc=f"round {round_idx}", leave=False)
    for _ in pbar:
        loss, grads = value_and_grad(nn_p, ts_batch, ys_batch)
        updates, opt_state = opt.update(grads, opt_state, nn_p)
        nn_p = optax.apply_updates(nn_p, updates)
        pbar.set_postfix(loss=f"{loss:.4f}")
    return nn_p


def plot_round(round_idx, batch_len, predict_ys, path):
    fig, axes = plt.subplots(DATA_DIM, 1, figsize=(8, 10), sharex=True)
    for i, (ax, name) in enumerate(zip(axes, COLUMNS)):
        ax.plot(ts, ys[:, i], "o", ms=3, color="black", label=f"{name}_data")
        ax.plot(ts[:batch_len], predict_ys[:batch_len, i], "-", label=f"{name}_train")
        ax.plot(
            ts[batch_len:], predict_ys[batch_len:, i], "--", label=f"{name}_predict"
        )
        ax.set_ylabel(name)
        ax.legend(loc="upper left", fontsize="small")
    axes[-1].set_xlabel("month")
    fig.suptitle(f"neural-ode-weather-prediction, round {round_idx}")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    PLOT_DIR.mkdir(exist_ok=True)
    nn_p = nn_p0
    ntimes = ts.shape[0]
    rounds = trange(1, ntimes // BATCH_STEP, desc="rounds")
    for i in rounds:
        batch_len = i * BATCH_STEP
        rounds.set_postfix(data_points=batch_len)
        nn_p = train_one_round(nn_p, ts[:batch_len], ys[:batch_len], i)

        predict_ys = predict(nn_p, ts)
        plot_round(i, batch_len, predict_ys, PLOT_DIR / f"neural-ode-weather_{i}.png")


if __name__ == "__main__":
    main()
