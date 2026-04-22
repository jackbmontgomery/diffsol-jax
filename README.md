# diffsol-jax

JAX wrapper around [diffsol](https://github.com/martinjrobins/diffsol), a Rust ODE solver library.
Exposes diffsol's BDF solver via `jax.ffi` so it can be called from inside `jax.jit` and
differentiated with `jax.grad`.

The user writes an RHS function in Python, which gets lowered to a
[DiffSL](https://martinjrobins.github.io/diffsl/) source string and compiled at call time.
Gradients are computed via diffsol's discrete adjoint, wrapped with `jax.custom_vjp`.

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

The C++ shim decodes the XLA CallFrame and forwards to Rust via `extern "C"`. All solver logic
lives in Rust.

Two backends are used internally:
- **Forward solve**: Cranelift JIT (fast compilation, no LLVM overhead)
- **Adjoint (VJP)**: LLVM (required for sensitivity gradient code generation; Cranelift does not
  emit the `*_sgrad` symbols needed for discrete adjoint)

## Requirements

- Rust toolchain (stable)
- Python >= 3.12
- C++17 compiler
- LLVM 20 (for adjoint support)

On macOS with Homebrew: `brew install llvm`

## Install

```bash
uv sync
uv run maturin develop --release
```

LLVM 20 is required for the adjoint. `.cargo/config.toml` sets `LLVM_SYS_201_PREFIX` to
`/opt/homebrew/opt/llvm`. If LLVM is installed elsewhere, update that path.

## Usage

### Forward solve

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

t_span = jnp.array([0.0, 10.0])
ys, ts = jax.jit(lambda p: solver(p, t_span))(params)
# ys: float64[200, 2],  ts: float64[200]
```

The RHS must return a tuple of scalars (one per state component). It can use standard JAX
operations; the lowerer handles elementwise arithmetic, unary functions, and parameter/state
indexing via Python-level unpacking.

### Gradients

`jax.grad` works out of the box:

```python
def loss(p):
    ys, _ = solver(p, t_span)
    return jnp.sum(ys ** 2)

grad = jax.grad(loss)(params)
```

The gradient is computed via diffsol's discrete adjoint (stateless: the VJP re-runs the forward
solve with checkpointing internally, costing ~2x a forward pass).

### Parameter fitting

```python
import optax

opt = optax.adam(1e-2)
state = opt.init(params)

@jax.jit
def step(p, state):
    loss_val, g = jax.value_and_grad(loss)(p)
    updates, state = opt.update(g, state)
    return optax.apply_updates(p, updates), state, loss_val

for _ in range(500):
    params, state, loss_val = step(params, state)
```

## Tests

```bash
uv run pytest tests/ -v
```

## Limitations

- CPU only, f64 only.
- No vmap batching rule; `vmap_method="sequential"` gives correct results via a Python loop.
- The DiffSL lowerer handles elementwise ops and the common ODE patterns (Lotka-Volterra, Lorenz).
  Operations like `dot_general`, `reduce_sum`, and `concatenate` are not supported (needed for
  neural ODE case).
- DiffSL is compiled fresh on every call; no caching of compiled modules across calls.
- Gradient wrt `y0` is computed internally but not returned; `y0` is baked into the DiffSL source.
- Gradient wrt `t_span` returns zeros.
