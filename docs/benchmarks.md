# Benchmarks

All benchmarks compare diffsol-jax against [diffrax](https://github.com/patrick-kidger/diffrax)
on the same ODE problems, same tolerances (`rtol=atol=1e-8`), and the same hardware
(Apple M-series). Each benchmark picks the most appropriate diffrax solver for the problem type.

## Forward solve

200 output times, wall-clock time per solve call after JIT warm-up.

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

## Gradient / parameter fitting

500 Adam steps, `n_times=50`, Lotka-Volterra with Tsit45 + discrete adjoint.

| System                     | diffsol-jax          | diffrax               | Speedup |
| -------------------------- | -------------------- | --------------------- | ------- |
| Lotka-Volterra (Tsit45+AD) | 184.9 ms / 500 steps | 2882.4 ms / 500 steps | 15.59x  |

Both solvers converge to the same parameters (`max |p_err| = 0.0175`). The speedup comes from
diffsol running entirely in compiled Rust/LLVM — gradients avoid JAX's tracing overhead on the
inner solver loop.

```bash
uv run python benchmarks/bench_grad.py
```
