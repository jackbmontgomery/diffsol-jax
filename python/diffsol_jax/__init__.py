r"""JAX bindings for the diffsol ODE solver.

A user writes the ODE right-hand side as a plain Python function ``rhs(t, y, p)``.
``ODEProblem`` lowers it to diffsl, JIT-compiles it with diffsol's Cranelift backend,
and exposes a ``ODEProblem.solve`` method that is compatible with
``jax.jit``, ``jax.grad``, ``jax.jvp``/``jax.jacfwd``, and ``jax.vmap``.

Differentiation strategy:
    Cranelift gives us a fast primal solve but, unlike the LLVM/Enzyme backend, it does
    not emit the reverse/sensitivity kernels that diffsol's built-in adjoint needs. So
    all derivatives are obtained by *forward sensitivity analysis done at the JAX level*:

    * Alongside the primal RHS we build an **augmented RHS** for the state
      ``[y; S]`` where ``S = \partial y / \partial p``. Its sensitivity block is
      ``dS_:,j/dt = \partial f/\partial y * S_:,j + \partial f/\partial p * e_j``, which we obtain column-by-column with
      ``jax.jvp`` of the user RHS at trace time. The whole augmented system lowers to a
      single DiffSL string and is solved as an ordinary primal solve on Cranelift.
    * Solving the augmented system materialises the full Jacobian
      ``J[t] = \partial y(t)/\partial p`` of shape ``(n_times, n_state, n_params)``.
    * ``jax.custom_jvp`` then defines the tangent as ``dys = J * dp``. Because ``J``
      is a constant w.r.t. the linearisation, JAX transposes this contraction for free,
      giving reverse-mode (``grad``/``vjp``) without any adjoint solve.

    This means there is exactly one Rust/FFI entry point - the primal dense solve - used
    for both the value and (via the augmented system) its derivatives.

    The augmented solver is built lazily, only the first time a problem is
    differentiated, so plain forward evaluation never pays for it.

Module layout:
    * ``problem``:  the ``ODEProblem`` class
    * ``solver_type``:  the ``OdeSolverType`` solver selection enum
    * ``sensitivity``:  augmented RHS + ``custom_jvp`` solver factory
    * ``ffi``:  the XLA custom-call bridge
    * ``lowering``: Python RHS -> DiffSL source lowering
"""

import jax

from .problem import ODEProblem
from .solver_type import OdeSolverType

jax.config.update("jax_enable_x64", True)

__all__ = ["ODEProblem", "OdeSolverType"]
