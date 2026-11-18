# API overview

There are two public pieces: [`ODEProblem`](diffsol_jax.md#diffsol_jax.ODEProblem),
which holds the equations and the diffsol solver built from them, and
[`solve`](diffsol_jax.md#diffsol_jax.solve), which solves one. `solve` is
compatible with `jax.jit`, `jax.grad`, `jax.jacrev`, `jax.jvp`, `jax.jacfwd`,
and `jax.vmap`.

## Public symbols

| Symbol                                                           | Kind      | Description                                   |
| ---------------------------------------------------------------- | --------- | --------------------------------------------- |
| [`ODEProblem`](diffsol_jax.md#diffsol_jax.ODEProblem)            | class     | An ODE or index-1 DAE, and the solver for it.  |
| [`solve`](diffsol_jax.md#diffsol_jax.solve)                      | function  | Solve a problem; differentiable and batchable.|
| [`DiffsolJaxError`](diffsol_jax.md#diffsol_jax.DiffsolJaxError)  | exception | Raised on configuration and solver errors.    |

## The equations

All the equations act on **one lane**: `x` (or `v`) has shape `[nstates]`, `t` is a scalar, and
`p` is the parameters — an array `[nparams]`, or the `params` pytree rebuilt. Any traceable JAX
code is allowed.

| Function    | Signature      | Required | Description                                             |
| ----------- | -------------- | -------- | ------------------------------------------------------- |
| `rhs`       | `(x, p, t)`    | yes      | The right-hand side, returning `[nstates]`.             |
| `out`       | `(x, p, t)`    | no       | An output function returning `[nout]`; `ys` then has `nout` columns instead of `nstates`. `nout` is inferred by shape tracing. |
| `mass`      | `(v, p, t)`    | no       | The mass matrix action `M(t) v`, linear in `v`, not using `p`. See [DAEs](../dae.md). |

`algebraic=` (an int, or a boolean mask over the states) is a shortcut for the diagonal `mass`
of a semi-explicit DAE.

## The parameter layout

`params=` declares what the `args` of a solve will be:

| `params=`        | The equations get      | A solve passes                                     |
| ---------------- | ---------------------- | -------------------------------------------------- |
| an int `n`       | `p [n]`                | an array `[..., n]`; its leading axes are lanes    |
| a template pytree| the pytree, rebuilt    | a pytree of the same structure and leaf shapes     |
| `None`           | `None`                 | nothing                                            |

In a template, the float-array leaves are the parameters (differentiated, in flatten order) and
every other leaf is captured by the equations as a constant — so the values of those leaves come
from the template, not from `args`. Only shapes and dtypes are read from the parameter leaves,
so `params=jax.eval_shape(make_net, key)` is a complete declaration.

## Solving

```python
problem = ODEProblem(rhs, nstates=2, params=4)                   # a flat parameter vector
problem = ODEProblem(rhs, nstates=4, params=net, solver="tsit45")  # a pytree of weights
ys = solve(problem, y0, t_eval, args, rtol=1e-6, ad="adjoint")
```

`solve(problem, y0, t_eval, args)` takes `y0` of shape `[..., nstates]`, `t_eval` of shape
`[nt]`, and `args` in the declared layout, and returns `ys` of shape `[..., nt, nout]`
(`nout = nstates` if the problem has no `out`). Leading axes of `y0` and of a flat `args` are
broadcast together and solved as lanes of one batched solve; a pytree `args` batches by
`jax.vmap` over its leaves.

The solve is differentiable in `y0` as well as in `args`: internally `y0` is appended to the
parameter vector, so it reaches the diffsol sensitivity and adjoint equations by the same path.
An initial state that is a function of the parameters needs no special support — pass
`solve(problem, f(params), t_eval, params)` and JAX composes the derivatives.

## What belongs where

Constructing an `ODEProblem` builds the diffsol solver: its native state, the callbacks compiled
for it on demand, and its adjoint checkpoint registry. The solver lives as long as the problem
and is freed with it, so build the problem once and reuse it.

The four options that the solver is built with — `solver`, `linear_solver`, `matrix_type`,
`max_checkpoints` — are therefore arguments of `ODEProblem`, and changing one means another
solver: `problem.replace(solver="tsit45")` returns a second problem. `rtol`, `atol` and `ad` are
per solve and are arguments of `solve`.

`ODEProblem` also exposes the solver's counters: `num_recomputes()`, `num_checkpoints()`,
`clear_checkpoints()`, and `callback_stats(nbatch)`.

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
