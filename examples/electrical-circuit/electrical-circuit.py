import pathlib

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from diffsol_jax import ODEProblem, solve

jax.config.update("jax_enable_x64", True)

NSTATES = 4
ALGEBRAIC = [True, False, True, False]


def rhs(x, p, t):
    i_r, i_l, i_c, v = x
    r, l_, c, v0, omega = p
    vs = v0 * jnp.sin(omega * t)
    return jnp.array(
        [
            v - r * i_r,
            (vs - v) / l_,
            i_l - i_r - i_c,
            i_c / c,
        ]
    )


problem = ODEProblem(rhs, nstates=NSTATES, params=5, algebraic=ALGEBRAIC, solver="bdf")

R, L, C, V0, OMEGA = 100.0, 1.0, 1e-3, 10.0, 100.0
params = jnp.array([R, L, C, V0, OMEGA])
t_eval = jnp.linspace(0.0, 1.0, 2000)

y0 = jnp.zeros(NSTATES)
ys = solve(problem, y0, t_eval, params)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t_eval, ys[:, 0], label="i_R")
ax.set_xlabel("t")
ax.set_ylabel("current")
ax.set_title("Low-pass LRC filter (DAE, diffsol_jax)")
ax.legend()
fig.tight_layout()

out_path = pathlib.Path(__file__).resolve().parent / "electrical-circuit.png"
fig.savefig(out_path)
