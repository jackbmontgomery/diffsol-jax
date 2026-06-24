# Getting started

## Installation

```bash
pip install diffsol-jax
```

or with [uv](https://docs.astral.sh/uv/):

```bash
uv add diffsol-jax
```

Requires Python ≥ 3.11 and JAX.

!!! tip "Building from source"

    To work against the development version you need a stable Rust toolchain. Clone the repo and
    run:

    ```bash
    uv sync
    uv run maturin develop --release
    ```

## Enable 64-bit floats

`diffsol-jax` performs better on `f64`. JAX defaults to `f32`, so enable x64 once at startup:

```python
import jax
import jax.numpy as jnp
from diffsol_jax import ODEProblem

jax.config.update("jax_enable_x64", True)
```

## First forward solve

Define the right-hand side as a plain `jax.numpy` function `rhs(t, y, p)`. It may return either a
length-`n_state` vector or a tuple of `n_state` scalar components:

```python
def lotka_volterra(t, y, p):
    x, yy = y[0], y[1]
    alpha, beta, delta, gamma = p[0], p[1], p[2], p[3]
    return jnp.array([alpha * x - beta * x * yy, delta * x * yy - gamma * yy])
```

Build the problem. The values in `y0` are baked into the compiled DiffSL source; `params` is used
only to trace the RHS and set the default parameter values:

```python
params = jnp.array([1.5, 1.0, 0.75, 3.0])
y0     = jnp.array([1.0, 0.5])

problem = ODEProblem(lotka_volterra, y0=y0, params=params)
```

Solve over a vector of output times. `t_eval` lists the points at which the solution is returned -
its length sets the number of rows in `ys`:

```python
t_eval = jnp.linspace(0.0, 10.0, 200)
ts, ys = problem.solve(params, t_eval)
# ts: float64[200]    - the output times (equal to t_eval)
# ys: float64[200, 2] - the solution at each time
```

`solve` works under `jax.jit`. The first call compiles; later calls reuse the cached module and just
update the parameters:

```python
ts, ys = jax.jit(lambda p: problem.solve(p, t_eval))(params)
```

## Computing gradients

`jax.grad` works out of the box:

```python
def loss(p):
    _, ys = problem.solve(p, t_eval)
    return jnp.sum(ys ** 2)

grad = jax.grad(loss)(params)
```

Differentiation solves an augmented **forward-sensitivity** system with the same solver. There is no
separate adjoint pass. The compiled module is cached, so only the first gradient step pays a
compilation cost. The maths is outlined out in [Computing gradients](gradients.md).

!!! note

    Gradients w.r.t. `t_eval` are zero. The gradient w.r.t. `y0` is *not* returned — `y0` is baked
    into the DiffSL source at construction time.

## Where to go next

- [Computing gradients](gradients.md) — the forward-sensitivity maths behind `jax.grad`.
- [From Python to DiffSL](lowering.md) — how the RHS is traced and lowered to a compiled solver.
- [API reference](api/index.md) — full details on `ODEProblem`, solver options, and type aliases.
- [Benchmarks](benchmarks.md) — forward and gradient timing vs diffrax.
