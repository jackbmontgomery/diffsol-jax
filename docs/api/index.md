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

All the equations act on **one lane**: `y` (or `v`) is in the state structure, `p` is in the
parameter structure, and `t` is a scalar. Any traceable JAX code is allowed.

| Function    | Signature      | Required | Description                                             |
| ----------- | -------------- | -------- | ------------------------------------------------------- |
| `rhs`       | `(y, p, t)`    | yes      | The right-hand side, returning the state structure.     |
| `out`       | `(y, p, t)`    | no       | An output function returning any pytree; `ys` comes back in its structure instead of the state's. Inferred by shape tracing. |
| `mass`      | `(v, p, t)`    | no       | The mass matrix action `M(t) v`, with `v` in the state structure, linear in `v` and not using `p`. See [DAEs](../dae.md). |

`algebraic=` is a shortcut for the diagonal `mass` of a semi-explicit DAE: an int (the last
entries of the flat state), a sequence of `nstates` bools, or a pytree of bools matching the
state.

## The layouts

`state=` and `params=` declare the pytrees that the equations, `y0`, `ys` and the parameter
values share:

| Declared as       | The equations get      | A solve passes                                  |
| ----------------- | ---------------------- | ----------------------------------------------- |
| an int `n`        | one flat `f64[n]`      | one array of shape `(n,)`                       |
| a template pytree | the pytree, rebuilt    | a pytree of the same structure and leaf shapes   |
| `None` (`params`) | `None`                 | nothing                                         |

Only the shapes and dtypes of the float-array leaves are read, so
`params=jax.eval_shape(make_net, key)` is a complete declaration. Every leaf of a `state`
template must be a float array, since every leaf is integrated. In a `params` template, any
other leaf is a constant that the equations capture — its value comes from the template, not
from what a solve passes.

## Solving

```python
problem = ODEProblem(rhs, state=2, params=4)                        # both flat
problem = ODEProblem(rhs, state=4, params=net, solver="tsit45")     # a pytree of weights
ys = solve(problem, y0, params, t_eval, rtol=1e-6, ad="adjoint")
```

`solve(problem, y0, params, t_eval)` takes `y0` and `params` in the declared layouts and
`t_eval` of shape `[nt]`, and returns `ys` in the output layout, each leaf `[nt, *leaf_shape]`
— for a flat `state=n` and no `out`, one array `[nt, n]`.

Batching is `jax.vmap`, and a vmapped axis of `y0` or of `params` becomes more lanes of one
batched solve. The leaves carry exactly the declared shapes, so an extra axis is a shape error
rather than a lane count.

The solve is differentiable in `y0` as well as in `params`: internally `y0` is appended to the
parameter vector, so it reaches the diffsol sensitivity and adjoint equations by the same path.
An initial state that is a function of the parameters needs no special support — pass
`solve(problem, f(params), params, t_eval)` and JAX composes the derivatives.

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
