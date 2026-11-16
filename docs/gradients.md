# Computing gradients

`jax.grad`, `jax.jacrev`, `jax.jvp`, and `jax.jacfwd` all work on `model.solve`.
This page explains the maths behind them.

## TL;DR

Forward mode (`jax.jvp`, `jax.jacfwd`) always uses diffsol
**forward sensitivities**: an augmented system solved alongside the state,
efficient when the number of parameters is small to moderate. Reverse mode
(`jax.grad`, `jax.jacrev`) depends on the model's `ad=` option:

- `ad="adjoint"` (the default) solves the diffsol **adjoint** equations
  backwards in time — one backward solve per cotangent regardless of the number
  of parameters;
- `ad="forward"` transposes the forward sensitivities — cheaper for few
  parameters and many outputs, and it avoids the backward solve entirely.

Both are exact derivatives of the numerical scheme's continuous problem,
computed to the solver tolerances.

## The problem

`Model` integrates an initial value problem

$$ M \frac{\mathrm{d}u}{\mathrm{d}t} = f(u, p, t), \qquad u(t_0) = u_0(p), $$

where $u(t) \in \mathbb{R}^{n}$ is the state, $p \in \mathbb{R}^{m}$ the
parameters, and $M$ an optional [mass matrix](dae.md). `solve` returns the
solution sampled at the times in `t_eval`.

## Forward sensitivities

At a fixed time define the sensitivity matrix
$S(t) = \partial u(t) / \partial p \in \mathbb{R}^{n \times m}$. Differentiating
the ODE with respect to $p$:

$$ M \frac{\mathrm{d}S}{\mathrm{d}t} = \frac{\partial f}{\partial u}\, S + \frac{\partial f}{\partial p}, \qquad S(t_0) = \frac{\partial u_0}{\partial p}. $$

diffsol solves this augmented system together with the state, using the same
solver and tolerances. The Jacobians are never formed: the $j$-th column of the
right-hand side is a single Jacobian–vector product of $f$, which diffsol-jax
derives from your `rhs` with `jax.jvp` and compiles like any other callback. The
initial condition comes from differentiating `init`, so parameters of the
initial state are included. Given a parameter tangent $\dot p$, forward mode
returns $\dot u = S \dot p$.

## The adjoint

For `jax.grad` of a scalar loss built from the solution, the adjoint method
solves

$$ M^{\top} \frac{\mathrm{d}\lambda}{\mathrm{d}t} = -\frac{\partial f}{\partial u}^{\top} \lambda, $$

backwards from $t_N$ to $t_0$, with $\lambda$ receiving a jump
$\partial g_i/\partial u$ at each output time $t_i$ (the cotangent that JAX
propagates in). The parameter gradient accumulates the quadrature of
$\partial f/\partial p^{\top} \lambda$ along the way, plus the initial-state
term $\partial u_0/\partial p^{\top} \lambda(t_0)$. One backward solve produces
the gradient with respect to **all** parameters; `jax.jacrev` batches the
cotangent directions through a single backward solve.

### Checkpointing

The backward solve needs the forward trajectory. The forward pass keeps a
checkpoint in C++ and hands JAX an opaque `int64` handle; the backward solve
interpolates from the checkpoint. A model keeps at most `max_checkpoints`
checkpoints. The backward solve also receives `params` and `t_eval`, so if its
checkpoint has been evicted (for example in `jax.grad` of a `lax.scan` with many
solves) it **re-runs the forward solve** and the gradient is unchanged.

## Limitations

- **No higher-order derivatives.** `grad(grad(...))` is not supported - the
  sensitivity and adjoint solves are not themselves differentiated.
- **No gradient with respect to `t_eval`.** Output times are constants; their
  cotangent is a symbolic zero.
- **No `dM/dp`.** A mass matrix must not depend on the parameters — see
  [DAEs](dae.md).
