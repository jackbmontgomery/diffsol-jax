"""
Robertson problem.

The classic stiff kinetics benchmark in two forms: as an ODE, and as a DAE with the
conservation law x + y + z = 1 in place of the third rate equation. The two are compared on
the trajectory, and on the gradients of z(T) with respect to the rate constants and to the
initial state.
"""

import pathlib

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from diffsol_jax import ODEProblem, solve

jax.config.update("jax_enable_x64", True)

BASE_DIR = pathlib.Path(__file__).resolve().parent

SPECIES = ["x", "y", "z"]
TOL = {"rtol": 1e-10, "atol": 1e-12}
TANGENT = jnp.array([-1.0, 0.0, 1.0])


def rober_ode(y: jax.Array, k: jax.Array, _t: float) -> jax.Array:
    return jnp.array(
        [
            -k[0] * y[0] + k[1] * y[1] * y[2],
            k[0] * y[0] - k[1] * y[1] * y[2] - k[2] * y[1] ** 2,
            k[2] * y[1] ** 2,
        ]
    )


def rober_dae(y: jax.Array, k: jax.Array, _t: float) -> jax.Array:
    return jnp.array(
        [
            -k[0] * y[0] + k[1] * y[1] * y[2],
            k[0] * y[0] - k[1] * y[1] * y[2] - k[2] * y[1] ** 2,
            jnp.sum(y) - 1,
        ]
    )


def z_end(
    prob: ODEProblem, y0: jax.Array, k: jax.Array, t_eval: jax.Array
) -> jax.Array:
    return solve(prob, y0, k, t_eval, **TOL)[-1, 2]


def grad_log_k(prob: ODEProblem, y0: jax.Array, k: jax.Array, t_eval: jax.Array):

    def f(scale):
        return z_end(prob, y0, scale * k, t_eval)

    one = jnp.ones_like(k)
    return jax.grad(f)(one), jax.jacfwd(f)(one)


def grad_y0(
    prob: ODEProblem, y0: jax.Array, k: jax.Array, t_eval: jax.Array
) -> jax.Array:
    return jax.grad(lambda y: z_end(prob, y, k, t_eval))(y0)


def directional_difference(
    prob: ODEProblem,
    y0: jax.Array,
    k: jax.Array,
    t_eval: jax.Array,
    v: jax.Array,
    h=1e-3,
) -> jax.Array:
    """The derivative along `v` by central differences, to check the gradients."""
    return (z_end(prob, y0 + h * v, k, t_eval) - z_end(prob, y0 - h * v, k, t_eval)) / (
        2 * h
    )


def fmt(v) -> str:
    return "[" + "  ".join(f"{x:+.6f}" for x in jnp.atleast_1d(v)) + "]"


def plot_forms(t_eval, ys_ode, ys_dae, path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, ys, title in zip(axes, [ys_ode, ys_dae], ["ODE form", "DAE form"]):
        for i, name in enumerate(SPECIES):
            scale = 1e4 if name == "y" else 1.0
            label = name if scale == 1.0 else f"{name} x 1e4"
            ax.plot(t_eval, scale * ys[:, i], label=label)
        ax.set_title(title)
        ax.legend(fontsize="small")
    axes[-1].plot(t_eval, jnp.abs(ys_ode - ys_dae), label=SPECIES)
    axes[-1].set_yscale("log")
    axes[-1].set_title("|ODE - DAE|")
    axes[-1].legend(fontsize="small")
    for ax in axes:
        ax.set_xlabel("t")
    fig.suptitle("robertson-problem")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    k = jnp.array([0.04, 1e4, 3e7])
    y0 = jnp.array([1.0, 0.0, 0.0])
    t_eval = jnp.linspace(0.0, 1000.0, num=100)

    prob_ode = ODEProblem(rober_ode, state=y0, params=k, solver="bdf")
    prob_dae = ODEProblem(rober_dae, state=y0, params=k, solver="bdf", algebraic=1)
    forms = (("ode", prob_ode), ("dae", prob_dae))

    ys = {name: solve(prob, y0, k, t_eval, **TOL) for name, prob in forms}
    plot_forms(t_eval, ys["ode"], ys["dae"], BASE_DIR / "robertson-problem.png")
    diff = jnp.abs(ys["ode"] - ys["dae"]).max()
    print(f"the two forms agree on the trajectory to {diff:.1e}")

    print("\nd z(T) / d log k")
    g_k = {}
    for name, prob in forms:
        g_k[name], forward = grad_log_k(prob, y0, k, t_eval)
        print(f"  {name}  adjoint {fmt(g_k[name])}  forward {fmt(forward)}")
    same = jnp.abs(g_k["ode"] - g_k["dae"]).max()
    print(f"  the same system, so the same gradient: max |ode - dae| = {same:.1e}")

    print("\nd z(T) / d y0")
    g_y0 = {name: grad_y0(prob, y0, k, t_eval) for name, prob in forms}
    for name, g in g_y0.items():
        print(f"  {name}  adjoint {fmt(g)}")
    print("  the DAE solves z at t0 from x + y + z = 1, so a change of the total does")
    print("  nothing there, and only directions that leave the total alone agree:")
    print(f"\nd z(T) / dv along v = {fmt(TANGENT)}")
    for name, prob in forms:
        fd = directional_difference(prob, y0, k, t_eval, TANGENT)
        adjoint = g_y0[name] @ TANGENT
        print(f"  {name}  adjoint {adjoint:+.8f}  finite difference {fd:+.8f}")


if __name__ == "__main__":
    main()
