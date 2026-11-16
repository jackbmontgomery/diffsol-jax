# DAEs

diffsol-jax solves differential-algebraic equations of the form

$$
M(t)\, \frac{\mathrm{d}u}{\mathrm{d}t} = f(u, p, t),
$$

where the mass matrix $M$ may be **singular**: rows of $M$ that are zero turn the corresponding
rows of $f$ into algebraic constraints $0 = f_i(u, p, t)$. This is the semi-explicit,
**index-1** class of DAEs. Pure-JAX solvers (diffrax) do not support it; diffsol's implicit
solvers (`bdf`, the default, `esdirk34`, and `tr_bdf2`) handle it directly, and the solves stay
differentiable like any other model.

## Marking algebraic states

Write the algebraic equations as rows of `rhs` and mark them with `algebraic=` — an int (the
last states are algebraic) or a boolean mask:

```python
# y' = -p0 * y   and   0 = z - p1 * y
model = Model(
    rhs=lambda x, p, t: jnp.array([-p[0] * x[0], x[1] - p[1] * x[0]]),
    init=lambda p, t: jnp.array([1.0, p[1]]),
    nstates=2,
    nparams=2,
    algebraic=1,  # or algebraic=[False, True]
)
```

`algebraic=` combines with a [`System`](getting-started.md#systems-classes-instead-of-flat-vectors)
the same way: `Model(system, algebraic=1)`.

A classic example: the Robertson system with its conservation law as a constraint,

```python
def rhs(x, p, t):
    k1, k2, k3 = p
    r1, r2 = k1 * x[0] - k3 * x[1] * x[2], k2 * x[1] ** 2
    return jnp.array([-r1, r1 - r2, x[0] + x[1] + x[2] - 1.0])


robertson = Model(
    rhs=rhs,
    init=lambda p, t: jnp.array([1.0, 0.0, 0.0]),
    nstates=3,
    nparams=3,
    algebraic=1,
)
```

## Consistent initial conditions

The initial state must satisfy the algebraic constraints. diffsol does this for you: at $t_0$ it
detects the algebraic rows from the mass matrix and Newton-solves the algebraic variables, so an
`init` that is inconsistent in its algebraic entries is projected onto the constraints before
the solve. The sensitivity and adjoint systems get the matching consistent initialisation, so
the gradients account for the projection.

## General mass matrices

For a non-diagonal or time-dependent mass, give `mass=lambda v, p, t: ...`, the action
$v \mapsto M(t)\,v$ — or override `mass(self, v, t)` on a `System`. It must be **linear in
`v`** and must **not use the parameters** (for a `System`: the trainable fields) — both are
checked once at model construction. The parameter restriction exists because diffsol's
sensitivity and adjoint equations have no $\partial M/\partial p$ term, so the gradient of a
parameter-dependent mass would silently miss it; fold such parameters into the right-hand
side instead (for example divide the differential equations by them, or scale the algebraic
equations).

## Limits

- **Index 1 only.** Higher-index DAEs (the full pendulum in Cartesian coordinates is index 3)
  must be reduced to index 1 first, as with every production DAE solver.
- **Linear in $u'$.** Fully implicit systems $F(u', u, p, t) = 0$ that cannot be written with a
  state-independent mass matrix are not supported.
- **Implicit solvers only.** `tsit45` is explicit and is rejected for models with a mass.
