# API overview

There are two public pieces: [`ODEProblem`](diffsol_jax.md#diffsol_jax.ODEProblem),
which holds the equations, and [`solve`](diffsol_jax.md#diffsol_jax.solve), which
solves one. `solve` is compatible with `jax.jit`, `jax.grad`, `jax.jacrev`,
`jax.jvp`, `jax.jacfwd`, and `jax.vmap`.

## Public symbols

| Symbol                                                           | Kind      | Description                                   |
| ---------------------------------------------------------------- | --------- | --------------------------------------------- |
| [`ODEProblem`](diffsol_jax.md#diffsol_jax.ODEProblem)            | class     | The equations of an ODE or index-1 DAE.       |
| [`solve`](diffsol_jax.md#diffsol_jax.solve)                      | function  | Solve a problem; differentiable and batchable.|
| [`Model`](diffsol_jax.md#diffsol_jax.Model)                      | class     | The built diffsol solver behind a solve, from `ODEProblem.compile` — its counters and statistics. |
| [`clear_cache`](diffsol_jax.md#diffsol_jax.clear_cache)          | function  | Drop the built solvers.                       |
| [`DiffsolJaxError`](diffsol_jax.md#diffsol_jax.DiffsolJaxError)  | exception | Raised on configuration and solver errors.    |

## The equations

All the equations act on **one lane**: `x` (or `v`) has shape `[nstates]`, `t` is a scalar, and
`p` is the parameters — the `args` of the solve, so either an array `[nparams]` or the pytree
rebuilt. Any traceable JAX code is allowed.

| Function    | Signature      | Required | Description                                             |
| ----------- | -------------- | -------- | ------------------------------------------------------- |
| `rhs`       | `(x, p, t)`    | yes      | The right-hand side, returning `[nstates]`.             |
| `out`       | `(x, p, t)`    | no       | An output function returning `[nout]`; `ys` then has `nout` columns instead of `nstates`. |
| `mass`      | `(v, p, t)`    | no       | The mass matrix action `M(t) v`, linear in `v`, not using `p`. See [DAEs](../dae.md). |

`algebraic=` (an int, or a boolean mask over the states) is a shortcut for the diagonal `mass`
of a semi-explicit DAE.

## Solving

```python
problem = ODEProblem(rhs)                                  # equations only
problem = ODEProblem(rhs, out=out, algebraic=1)            # an output, and a DAE
ys = solve(problem, y0, t_eval, args, solver="bdf", rtol=1e-6)
```

`solve(problem, y0, t_eval, args)` takes `y0` of shape `[..., nstates]`, `t_eval` of shape
`[nt]`, and `args` either of shape `[..., nparams]` or as a pytree, and returns `ys` of shape
`[..., nt, nout]` (`nout = nstates` if the problem has no `out`). Leading axes of `y0` and of an
array `args` are broadcast together and solved as lanes of one batched solve; a pytree `args`
batches by `jax.vmap` over its leaves. The sizes are read from the arguments: `nstates` from
`y0`, `nparams` from `args`, `nout` by shape-tracing `out`.

The solve is differentiable in `y0` as well as in `args`: internally `y0` is appended to the
parameter vector, so it reaches the diffsol sensitivity and adjoint equations by the same path.
An initial state that is a function of the parameters needs no special support — pass
`solve(problem, f(params), t_eval, params)` and JAX composes the derivatives.

## Caching

The diffsol solver for a problem is built on the first solve that needs it and reused by every
solve that matches on the problem, `nstates`, the parameter layout, and the compile-time options
(`solver`, `linear_solver`, `matrix_type`, `max_checkpoints`). `rtol`, `atol` and `ad` are per
solve and do not build another one. `ODEProblem.compile(y0, args, **options)` returns that same
`Model` — use it to pay the construction cost before the first solve, or to read its callback
statistics and checkpoint counters. `clear_cache()` drops them all.

## Solvers

| `solver=`         | Type                             | Mass matrices / DAEs |
| ----------------- | -------------------------------- | -------------------- |
| `"bdf"` (default) | BDF (implicit, variable-order)   | yes                  |
| `"esdirk34"`      | ESDIRK3(4) (implicit)            | yes                  |
| `"tr_bdf2"`       | TR-BDF2 (implicit)               | yes                  |
| `"tsit45"`        | Tsitouras 4(5) (explicit)        | no                   |

Reverse mode uses the diffsol adjoint (`ad="adjoint"`, the default) or the transpose of the
forward sensitivities (`ad="forward"`); forward mode always uses forward sensitivities. See
[Computing gradients](../gradients.md).
