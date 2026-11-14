# Benchmarks

The architecture was benchmarked against hand-written native C callbacks, the diffsl language
backend, and [diffrax](https://github.com/patrick-kidger/diffrax), on Robertson (stiff),
Lotka–Volterra, and chains of nonlinear oscillators at batch sizes 1 to 1024. The full tables,
methodology, and a profile of where the time goes are in
[Architecture](architecture.md) (measured on Apple M-series, CPU).

The headlines:

- **Stiff problems**: diffsol-jax is 14–16× faster than diffrax at batch sizes ≥ 64 (solve and
  gradient), and within ~1.3× of hand-written native C callbacks at batch 1024.
- **Small non-stiff problems**: diffrax wins at batch 1 (for example 0.08 ms against 4 ms for
  one Lotka–Volterra lane). An explicit method makes ~6 callback calls per step and each call
  carries ~2.5 µs of fixed PJRT overhead, while diffrax compiles the whole loop with XLA.
- **Larger non-stiff systems**: comparable, with diffsol-jax ahead on gradients at large batch.

The results are checked for correctness as part of the benchmark: native C agrees with
diffsol-jax to 1e-11 or better, diffsl and diffrax to the solver tolerances.

## Running the benchmark

`bench/bench.py` needs `diffrax`, a diffsol-c built with the LLVM diffsl backend (for the diffsl
comparison), and the native models compiled from `bench/models_native.c` — see the build notes
in [Architecture](architecture.md#build).
