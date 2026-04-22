# diffsol-jax

JAX wrapper around [diffsol](https://github.com/martinjrobins/diffsol), a Rust ODE solver library.
Exposes diffsol's ODE solvers via `jax.ffi` so they can be called from inside `jax.jit` and
differentiated with `jax.grad`.

The user writes an RHS function in Python, which gets lowered to a
[DiffSL](https://martinjrobins.github.io/diffsl/) source string and compiled on first call.
Gradients are computed via diffsol's discrete adjoint, wrapped with `jax.custom_vjp`.

## Architecture

```
Python rhs fn  ->  DiffSL string  ->  XLA FFI call
                                          |
                                    C++ shim (wrapper.cc)
                                          |
                                    Rust (lib.rs)
                                          |
                                    diffsol solver (BDF / Tsit45 / ESDIRK34 / TR-BDF2)
```

The C++ shim decodes the XLA CallFrame and forwards to Rust via `extern "C"`. All solver logic lives
in Rust.

Two backends are used internally:

- **Forward solve**: Cranelift JIT (fast compilation, no LLVM overhead). The compiled module is
  cached per DiffSL source string (thread-local), so repeated calls with different parameters skip
  recompilation.
- **Adjoint (VJP)**: LLVM (required for sensitivity gradient code generation; Cranelift does not
  emit the `*_sgrad` symbols needed for discrete adjoint). Compiled fresh each VJP call — caching is
  not safe because the checkpointer returned by the forward pass is tied to the problem object's
  mutable state.

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

### Solver selection

Four solvers are available via the `method=` argument:

| `method=` | Type | Adjoint |
|---|---|---|
| `"bdf"` (default) | BDF (implicit, variable-order) | BDF |
| `"tsit45"` | Tsitouras 4(5) (explicit) | Tsit45 |
| `"esdirk34"` | ESDIRK3(4) (implicit) | BDF (fallback) |
| `"tr_bdf2"` | TR-BDF2 (implicit) | BDF (fallback) |

```python
solver, _ = make_diffsol_solver(rhs, y0=y0, p_example=params, method="tsit45")
```

BDF and Tsit45 use their own solver for both forward and backward passes. ESDIRK34 and TR-BDF2
fall back to BDF for the adjoint — their implicit adjoint solvers fail to converge on non-trivial
problems (the adjoint ODEs are typically harder to integrate than the forward).

### Gradients

`jax.grad` works out of the box:

```python
def loss(p):
    ys, _ = solver(p, t_span)
    return jnp.sum(ys ** 2)

grad = jax.grad(loss)(params)
```

The gradient is computed via diffsol's discrete adjoint. The VJP re-runs the forward solve with
checkpointing internally (costs ~2x a forward pass) and recompiles the LLVM-backed DiffSL module
each call.

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

## Benchmarks

Forward solve vs diffrax, Apple M-series, `rtol=atol=1e-8`, 200 output times.

| System                     | diffsol-jax | diffrax          | Speedup |
| -------------------------- | ----------- | ---------------- | ------- |
| Lotka-Volterra (non-stiff) | 0.08 ms     | 0.23 ms (Dopri5) | 2.92x   |
| Van der Pol μ=1000 (stiff) | 0.5 ms      | 65 ms (Kvaerno5) | ~120x   |

BDF's variable-order stepping gives a large advantage on stiff problems. On non-stiff systems
explicit methods like Dopri5 are competitive.

```bash
uv run python benchmarks/bench_forward.py
uv run python benchmarks/bench_stiff.py
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
- Forward DiffSL is cached per source string (thread-local); first call compiles, subsequent calls
  reuse the compiled Cranelift module. VJP recompiles LLVM each call (caching interacts unsafely
  with the adjoint checkpointer).
- Gradient wrt `y0` is computed internally but not returned; `y0` is baked into the DiffSL source.
- Gradient wrt `t_span` returns zeros.
- ESDIRK34 and TR-BDF2 fall back to BDF for the adjoint; their own adjoint solvers fail to
  converge on non-trivial problems.
