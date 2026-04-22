# diffsol-jax

JAX wrapper around [diffsol](https://github.com/martinjrobins/diffsol), a Rust ODE solver library.
Exposes diffsol's BDF solver via `jax.ffi` so it can be called from inside `jax.jit`.

The user writes an RHS function in Python, which gets lowered to a
[DiffSL](https://martinjrobins.github.io/diffsl/) source string and compiled by diffsol's Cranelift
JIT backend at call time.

## Architecture

```
Python rhs fn  ->  DiffSL string  ->  XLA FFI call
                                          |
                                    C++ shim (wrapper.cc)
                                          |
                                    Rust (lib.rs)
                                          |
                                    diffsol BDF solver
```

The C++ shim is a thin layer that decodes the XLA CallFrame and forwards to a Rust function via
`extern "C"`. All solver logic lives in Rust.

## Requirements

- Rust toolchain (stable)
- Python >= 3.12
- C++17 compiler

## Install

```bash
uv sync
uv run maturin develop --release
```

## Usage

```python
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from diffsol_jax import make_diffsol_solver

def lotka_volterra(t, y, p):
    x, yy = y[0], y[1]
    alpha, beta, delta, gamma = p[0], p[1], p[2], p[3]
    return (alpha * x - beta * x * yy, delta * x * yy - gamma * yy)

params = jnp.array([1.5, 1.0, 0.75, 3.0])
y0    = jnp.array([1.0, 0.5])

solver, src = make_diffsol_solver(
    lotka_volterra,
    y0=y0,
    p_example=params,
    param_names=["alpha", "beta", "delta", "gamma"],
    state_names=["x", "y"],
)

ys, ts = jax.jit(lambda p: solver(p, 10.0))(params)
# ys: float64[200, 2],  ts: float64[200]
```

The RHS must return a tuple of scalars (one per state component). It can use standard JAX
operations; the lowerer handles elementwise arithmetic, unary functions, and parameter/state
indexing via Python-level unpacking.

## Tests

```bash
uv run pytest tests/ -v
```

## Limitations

- CPU only, f64 only.
- No gradients (diffsol has adjoint support but it is not wired up here yet).
- No vmap batching rule; `vmap_method="sequential"` gives correct results via a Python loop.
- The DiffSL lowerer handles elementwise ops and the common ODE patterns (Lotka-Volterra, Lorenz).
  Operations like `dot_general`, `reduce_sum`, and `concatenate` are not supported.
- DiffSL is compiled fresh on every call; there is no caching of compiled modules across calls.
