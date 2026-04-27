# Getting started

## Requirements

- Rust toolchain (stable)
- Python >= 3.12
- C++17 compiler
- LLVM 20 (required for adjoint/gradient support)

On macOS with Homebrew:

```bash
brew install llvm
```

## Installation

```bash
uv sync
uv run maturin develop --release
```

!!! note
    `.cargo/config.toml` sets `LLVM_SYS_201_PREFIX` to `/opt/homebrew/opt/llvm`.
    If LLVM 20 is installed elsewhere, update that path before running `maturin develop`.

## First forward solve

Enable 64-bit floats — diffsol-jax requires f64:

```python
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from diffsol_jax import ODEProblem
```

Define the RHS as a plain Python function returning a tuple of scalars, one per state component:

```python
def lotka_volterra(t, y, p):
    x, yy = y[0], y[1]
    alpha, beta, delta, gamma = p[0], p[1], p[2], p[3]
    return (alpha * x - beta * x * yy, delta * x * yy - gamma * yy)
```

Create example arrays for the initial state and parameters, then build the solver.
The values in `y0` are baked into the compiled DiffSL source; `params` is used only to
trace the RHS and set default parameter values:

```python
params = jnp.array([1.5, 1.0, 0.75, 3.0])
y0     = jnp.array([1.0, 0.5])

problem = ODEProblem(lotka_volterra, y0=y0, params=params)
```

Call `problem.solve` under `jax.jit`. The first call compiles; subsequent calls reuse
the cached module and update parameters via `set_params`:

```python
t_span = jnp.array([0.0, 10.0])
ts, ys = jax.jit(lambda p: problem.solve(p, t_span))(params)
# ts: float64[200]    — uniformly spaced output times
# ys: float64[200, 2] — solution at each time
```

## Computing gradients

`jax.grad` works out of the box. The VJP uses diffsol's discrete adjoint; the compiled
LLVM module is cached, so only the first gradient step pays compilation cost:

```python
def loss(p):
    ts, ys = problem.solve(p, t_span)
    return jnp.sum(ys ** 2)

grad = jax.grad(loss)(params)
```

Gradients w.r.t. `t_span` return zeros. Gradient w.r.t. `y0` is computed internally
but not returned — `y0` is baked into the DiffSL source at construction time.

## Where to go next

- [API reference](api/index.md) — full details on `ODEProblem`, solver options, and type aliases.
- [Benchmarks](benchmarks.md) — forward and gradient timing vs diffrax.
