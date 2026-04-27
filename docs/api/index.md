# API overview

The main entry point is `ODEProblem`. Construct it with your RHS function, an initial state,
and an example parameter vector; it compiles the problem once and exposes a `solve` callable
that is compatible with `jax.jit` and `jax.grad`.

## Public symbols

| Symbol | Kind | Description |
|--------|------|-------------|
| [`ODEProblem`](diffsol_jax.md#diffsol_jax.ODEProblem) | class | Compiled ODE problem; exposes a `solve` method. |
| [`ODE_RHS`](diffsol_jax.md#diffsol_jax.ODE_RHS) | type alias | `Callable` — type annotation for RHS functions. |
| [`ODE_SOLVER`](diffsol_jax.md#diffsol_jax.ODE_SOLVER) | type alias | `Literal["bdf", "tsit45", "esdirk34", "tr_bdf2"]` — solver names. |

## Solvers

Pass the `ode_solver=` argument to `ODEProblem` to select a solver.

| `ode_solver=`     | Type                           | Adjoint        |
| ----------------- | ------------------------------ | -------------- |
| `"bdf"` (default) | BDF (implicit, variable-order) | BDF            |
| `"tsit45"`        | Tsitouras 4(5) (explicit)      | Tsit45         |
| `"esdirk34"`      | ESDIRK3(4) (implicit)          | BDF (fallback) |
| `"tr_bdf2"`       | TR-BDF2 (implicit)             | BDF (fallback) |

BDF and Tsit45 use their own solver for both forward and backward passes. ESDIRK34 and
TR-BDF2 fall back to BDF for the adjoint — their implicit adjoint solvers fail to converge
on non-trivial problems.

```python
problem = ODEProblem(rhs, y0=y0, params=params, ode_solver="tsit45")
```
