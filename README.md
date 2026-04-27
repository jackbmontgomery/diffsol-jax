# diffsol-jax

JAX wrapper around [diffsol](https://github.com/martinjrobins/diffsol), a Rust ODE solver library.
Exposes diffsol's ODE solvers via `jax.ffi` so they can be called from inside `jax.jit` and
differentiated with `jax.grad`.

The user writes an RHS function in Python, which gets lowered to a
[DiffSL](https://martinjrobins.github.io/diffsl/) source string and compiled on first call.

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

## Limitations

- CPU only, f64 only.
- No vmap batching rule; `vmap_method="sequential"` gives correct results via a Python loop.
- The DiffSL lowerer handles elementwise ops and the common ODE patterns (Lotka-Volterra, Lorenz).
  Operations like `dot_general`, `reduce_sum`, and `concatenate` are not supported (needed for
  neural ODE case).
- Forward and adjoint DiffSL modules are cached per source string (thread-local); first call
  compiles, subsequent calls reuse the compiled module and update parameters via `set_params`.
- Gradient wrt `y0` is computed internally but not returned; `y0` is baked into the DiffSL source.
- Gradient wrt `t_span` returns zeros.
- ESDIRK34 and TR-BDF2 fall back to BDF for the adjoint; their own adjoint solvers fail to converge
  on non-trivial problems.
- Does not work with `jax.pmap`
