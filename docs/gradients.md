# Computing gradients

`jax.grad`, `jax.jvp`, `jax.jacfwd`, and `jax.vjp` all work on `problem.solve`. This page explains
the maths behind them.

## TL;DR

`diffsol-jax` differentiates by solving an augmented **forward-sensitivity** system with the same
solver as the forward pass so there is no separate adjoint solve. This means it is extremely fast
for small to medium sized parameter and state spaces. But the advantage reduces as we increase the
dimensionality.

## The problem

`ODEProblem` integrates an initial value problem

$$
\frac{\mathrm{d}y}{\mathrm{d}t} = f(t, y, p), \qquad y(t_0) = y_0,
$$

where $y(t) \in \mathbb{R}^{n}$ is the state, $p \in \mathbb{R}^{m}$ the parameters, and $f$ the
right-hand side you wrote in Python. `solve` returns the solution sampled at the times in `t_eval`.

For training and fitting we need the derivative of the solution with respect to the parameters. At a
fixed time $t$ define the **sensitivity matrix**

$$
S(t) \;=\; \frac{\partial y(t)}{\partial p} \;\in\; \mathbb{R}^{n \times m},
\qquad S_{ij} = \frac{\partial y_i(t)}{\partial p_j}.
$$

## The sensitivity equations

Differentiate the ODE with respect to $p$ and exchange the order of the $t$- and $p$-derivatives:

$$ \frac{\mathrm{d}}{\mathrm{d}t}\,\frac{\partial y}{\partial p} = \frac{\partial f}{\partial
y}\,\frac{\partial y}{\partial p} - \frac{\partial f}{\partial p}. $$

So $S$ obeys its the ODE, driven by the Jacobians of $f$:

$$ \boxed{\; \frac{\mathrm{d}S}{\mathrm{d}t} = \underbrace{\frac{\partial f}{\partial y}}_{n \times
n}\, S - \underbrace{\frac{\partial f}{\partial p}}_{n \times m}, \qquad S(t_0) = \frac{\partial
y_0}{\partial p} = 0. \;} $$

The initial condition is **zero** because `y0` is fixed at construction time and does not depend on
$p$. (This is also why gradients w.r.t. `y0` are not available.)

## The augmented system

The state ODE and the sensitivity ODE are solved together as one augmented system. Stacking $y$ with
the columns $S_{:,j}$ of the sensitivity matrix gives a state of length $n(1 + m)$:

$$
z = \begin{bmatrix} y \\ S_{:,0} \\ \vdots \\ S_{:,m-1} \end{bmatrix},
\qquad
\frac{\mathrm{d}z}{\mathrm{d}t} =
\begin{bmatrix}
f(t, y, p) \\
\dfrac{\partial f}{\partial y} S_{:,0} + \dfrac{\partial f}{\partial p}\, e_0 \\
\vdots \\
\dfrac{\partial f}{\partial y} S_{:,m-1} + \dfrac{\partial f}{\partial p}\, e_{m-1}
\end{bmatrix},
$$

where $e_j$ is the $j$-th unit vector in parameter space. `diffsol-jax` never forms the Jacobians
$\partial f/\partial y$ and $\partial f/\partial p$ explicitly. Instead it notes that the $j$-th
sensitivity block is exactly a **Jacobian–vector product** of $f$ in the direction $(S_{:,j},\,
e_j)$:

$$
\frac{\partial f}{\partial y} S_{:,j} + \frac{\partial f}{\partial p}\, e_j
= \mathrm{jvp}\big(f;\,(y, p),\,(S_{:,j}, e_j)\big).
$$

Each block is produced with a single `jax.jvp`, the augmented RHS is lowered to DiffSL just like any
other model, and diffsol integrates $z$ with the same solver and tolerances as the forward solve.
The sensitivity columns are read back out of $z$ and reshaped into the Jacobian

$$
J(t) = \frac{\partial y(t)}{\partial p} \in \mathbb{R}^{n \times m}.
$$

## Forward and reverse mode

With $J$ in hand, both AD directions are simple linear algebra applied at each output time.

**Forward mode** (`jax.jvp`, `jax.jacfwd`). Given a parameter tangent $\dot p$, the directional
derivative of the solution is the matrix–vector product

$$
\dot y = J\,\dot p.
$$

**Reverse mode** (`jax.grad`, `jax.vjp`). The map $\dot p \mapsto J\,\dot p$ is linear, so JAX
obtains the vector–Jacobian product for free by automatically transposing it:

$$
\bar p = J^{\top}\,\bar y.
$$

`diffsol-jax` only defines the forward (`custom_jvp`) rule; JAX's
[automatic transposition](https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html)
derives reverse mode from it. This is why `jax.grad` works without a hand-written adjoint.

!!! note "Cost"

    The augmented state has length $n(1 + m)$, so the forward-sensitivity solve scales with the
    number of parameters $m$. This is efficient when $m$ is small to moderate. The augmented solver
    is compiled and cached on first differentiation, so repeated gradient steps pay no further
    compilation cost.

## Limitations

- **No higher-order derivatives.** `grad(grad(...))` is not supported in general — the sensitivity
  solve itself is not re-differentiated.
- **No gradient w.r.t. `y0`.** The initial state is baked into the compiled DiffSL source, fixing
  $S(t_0) = 0$.
- **No gradient w.r.t. `t_eval`.** Output times are treated as constants; their tangents are zero.
