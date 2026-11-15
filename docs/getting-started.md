# Getting started

## Installation

```bash
pip install diffsol-jax
```

or with [uv](https://docs.astral.sh/uv/):

```bash
uv add diffsol-jax
```

Requires Python $\geq$ 3.12 and JAX $\geq$ 0.12.

## The PJRT plugin

The equation callbacks are compiled and executed through the
[PJRT C API](https://openxla.org/xla/pjrt), so diffsol-jax needs a CPU PJRT plugin. jaxlib does
not ship one as a loadable library. A prebuilt plugin for linux/amd64 is inside
[gopjrt](https://github.com/gomlx/gopjrt)'s `gomlx_xlabuilder_linux_amd64.tar.gz` release asset
(at `lib/gomlx/pjrt/pjrt_c_api_cpu_plugin.so`; the macOS arm64 asset is
`gopjrt_cpu_darwin_arm64.tar.gz`), or you can build `pjrt_c_api_cpu_plugin.so` from
[openxla/xla](https://github.com/openxla/xla). Point the environment variable at it:

```bash
export DIFFSOL_JAX_PJRT_PLUGIN=/path/to/pjrt_c_api_cpu_plugin.so
```

!!! tip "Building from source"

    Working against the development version needs a Rust toolchain (for
    [diffsol-c](https://crates.io/crates/diffsol-c)), cmake, and a C++20 compiler. See the
    README's *Building (development)* section for the exact commands.

## Enable 64-bit floats

The solver uses `f64`. JAX defaults to `f32`, so enable x64 once at startup:

```python
import jax
import jax.numpy as jnp
from diffsol_jax import Model

jax.config.update("jax_enable_x64", True)
```

## First forward solve

A model is defined by two plain `jax.numpy` functions acting on one lane: the right-hand side
`rhs(x, p, t)` and the initial state `init(p, t)`, with `x` of shape `[nstates]` and `p` of shape
`[nparams]`. Anything JAX can trace is allowed — closures, `jnp` operations, neural networks:

```python
def lotka_volterra(x, p, t):
    alpha, beta, delta, gamma = p
    return jnp.array(
        [alpha * x[0] - beta * x[0] * x[1], delta * x[0] * x[1] - gamma * x[1]]
    )


model = Model(
    rhs=lotka_volterra,
    init=lambda p, t: jnp.array([1.0, 0.5]),
    nstates=2,
    nparams=4,
)
```

Solve over a vector of output times. `t_eval` lists the points at which the solution is returned:

```python
params = jnp.array([1.5, 1.0, 0.75, 3.0])
t_eval = jnp.linspace(0.0, 10.0, 200)
ys = model.solve(params, t_eval)  # float64[200, 2]
```

`solve` works under `jax.jit`, and under `jax.vmap`: a vmapped axis of `params` becomes more
lanes in **one** batched solve, with the equations of all the lanes evaluated together:

```python
ys = jax.jit(lambda p: model.solve(p, t_eval))(params)
many = jax.vmap(model.solve, (0, None))(param_batch, t_eval)  # [B, 200, 2]
```

## Computing gradients

`jax.grad` uses the diffsol adjoint by default; `jax.jacfwd`/`jax.jvp` use forward
sensitivities. Both account for `init` depending on the parameters:

```python
def loss(p):
    return jnp.sum(model.solve(p, t_eval) ** 2)


grad = jax.grad(loss)(params)
jac = jax.jacfwd(lambda p: model.solve(p, t_eval))(params)
```

The maths, the `ad=` option, and checkpointing are described in
[Computing gradients](gradients.md).

## Solver options

`Model` takes the solver configuration at construction:

| Argument            | Default            | Description                                                           |
| ------------------- | ------------------ | --------------------------------------------------------------------- |
| `ode_solver`        | `"bdf"`            | `"bdf"`, `"esdirk34"`, `"tr_bdf2"` (implicit), `"tsit45"` (explicit). |
| `rtol`, `atol`      | `1e-6`, `1e-6`     | Solver tolerances.                                                    |
| `ad`                | `"adjoint"`        | Reverse mode: `"adjoint"` or `"forward"` (transposed sensitivities).  |
| `max_checkpoints`   | `8`                | Adjoint checkpoints kept; controls memory only, not correctness.      |
| `out`, `nout`       | `None`, `0`        | Optional output function `out(x, p, t)`; `ys` then has `nout` columns.|
| `mass`, `algebraic` | `None`             | Mass matrix / DAE support — see [DAEs](dae.md).                       |
| `linear_solver`     | `"default"`        | `"lu"` or `"klu"` (with `matrix_type="faer_sparse"`).                 |
| `matrix_type`       | `"nalgebra_dense"` | `"faer_dense"` or `"faer_sparse"` for larger systems.                 |

## Where to go next

- [Computing gradients](gradients.md) — the adjoint and forward-sensitivity maths.
- [DAEs](dae.md) — algebraic constraints via singular mass matrices.
- [Architecture](architecture.md) — how a JAX function becomes a diffsol callback.
- [API reference](api/index.md) — full details on `Model`.
