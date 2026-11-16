"""Low-pass LRC filter, a DAE.

a resistor R, inductor L and capacitor C across a sinusoidal source Vs.
The states are `[i_R, i_L, i_C, V]`, the current through the resistor, inductor,
capacitor, and the voltage across the resistor/capacitor.

Kirchhoff's current law and Ohm's law give two algebraic constraints (i_R, i_C have no time
derivative), so this is a semi-explicit index-1 DAE `M dy/dt = f(y, p, t)`:

    M = diag(0, 1, 0, 1)
    f = [V - R i_R, (Vs - V) / L, i_L - i_R - i_C, i_C / C]
    Vs(t) = V0 sin(omega t)

`diffsol_jax.Model`'s `algebraic=` is a shortcut for this diagonal mass: a boolean mask marking
which states have no time derivative.
"""

import pathlib

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from diffsol_jax import Model

jax.config.update("jax_enable_x64", True)

NSTATES = 4
# states: [i_R, i_L, i_C, V]; i_R and i_C are algebraic (no time derivative)
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


model = Model(
    rhs=rhs,
    init=lambda p, t: jnp.zeros(NSTATES),
    nstates=NSTATES,
    nparams=5,
    algebraic=ALGEBRAIC,
    ode_solver="bdf",
)

R, L, C, V0, OMEGA = 100.0, 1.0, 1e-3, 10.0, 100.0
params = jnp.array([R, L, C, V0, OMEGA])
t_eval = jnp.linspace(0.0, 1.0, 2000)

ys = model.solve(params, t_eval)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t_eval, ys[:, 0], label="i_R")
ax.set_xlabel("t")
ax.set_ylabel("current")
ax.set_title("Low-pass LRC filter (DAE, diffsol_jax)")
ax.legend()
fig.tight_layout()

out_path = pathlib.Path(__file__).resolve().parent / "electrical-circuit.png"
fig.savefig(out_path)
