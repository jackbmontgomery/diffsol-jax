"""The `System` class, and pytree variables as a boundary layer over the flat `f64[nparams]`
vector.

`System` is the abstract base for an ODE system: subclasses declare fields (the initial state,
network weights, rate constants, ...) and define the equations as methods over those fields.
Instances are frozen dataclasses registered as pytrees, so `jax.grad`, `jax.vmap` and optax
work on them directly, and `Model(system)` derives the equations and sizes from them.

`ParamSpec(variables_like)` splits the leaves of the example pytree (a `System` instance, or
any other pytree) into trainable (inexact-dtype arrays) and static (everything else, fixed at
model construction). `ravel` takes a pytree of the same structure to the flat vector at the
solve boundary; `bind` rebuilds the pytree from the flat vector inside the traced callbacks,
where its slices and reshapes fuse away. The ravel is built from
`reshape`/`concatenate`/`astype`, so `jax.grad` returns a pytree of the same structure (each
leaf's gradient in that leaf's own dtype) and `jax.vmap` over any leaf batches the flat
vector.
"""

import abc
import dataclasses
import math
from typing import Any, Self, cast, dataclass_transform

import jax
import jax.numpy as jnp
import numpy as np

from ._lib import DiffsolJaxError


@dataclass_transform(frozen_default=True, field_specifiers=(dataclasses.field,))
class System(abc.ABC):
    """An ODE system: its differentiable inputs as fields, its equations as methods.

        class WeatherSystem(System):
            y0: jax.Array
            net: NeuralNetwork              # any pytree, e.g. an eqx.Module

            def init(self, t):
                return self.y0

            def __call__(self, x, t):
                return self.net(x)

        system = WeatherSystem(y0=y0, net=net)
        model = Model(system, ode_solver="tsit45")   # nstates inferred from init
        ys = model.solve(system, t_eval)

    Subclasses are frozen dataclasses registered as pytrees. `init(t)` gives the initial state
    and `__call__(x, t)` the right-hand side; override `out(x, t)` for an output function and
    `mass(v, t)` for a mass matrix action (see `Model`). Field leaves that are float (or
    complex) arrays are trainable: `model.solve` differentiates with respect to them and
    `jax.grad` returns a `WeatherSystem` of gradients, so an optax update applies directly.
    Every other field (ints, bools, callables) is fixed when the model is built. Mark a field
    static with `dataclasses.field(metadata={"static": True})` to keep it out of the pytree
    leaves entirely (it must then be hashable). Field names must not shadow the methods
    (`init`, `out`, `mass`, `replace`).
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        datacls: Any = dataclasses.dataclass(frozen=True)(cls)
        fields = dataclasses.fields(datacls)
        static = [f.name for f in fields if f.metadata.get("static", False)]
        data = [f.name for f in fields if not f.metadata.get("static", False)]
        jax.tree_util.register_dataclass(cls, data_fields=data, meta_fields=static)

    def replace(self, **changes: Any) -> Self:
        """A copy with the given fields replaced (instances are frozen)."""
        return dataclasses.replace(cast(Any, self), **changes)

    @abc.abstractmethod
    def init(self, t: jax.Array) -> jax.Array:
        """The initial state `[nstates]` at the start time `t`."""

    @abc.abstractmethod
    def __call__(self, x: jax.Array, t: jax.Array) -> jax.Array:
        """The right-hand side `[nstates]` at state `x [nstates]` and time `t`."""

    def out(self, x: jax.Array, t: jax.Array) -> jax.Array:
        """Override: the output `[nout]` at state `x` and time `t`."""
        raise NotImplementedError

    def mass(self, v: jax.Array, t: jax.Array) -> jax.Array:
        """Override: the mass matrix action `M(t) v`, linear in `v`. It must not use the
        trainable fields (checked at model construction): the diffsol sensitivities have no
        dM/dp term, so the gradient of a trainable mass would be wrong."""
        raise NotImplementedError


def _is_trainable(leaf):
    return isinstance(leaf, (jax.Array, np.ndarray)) and jnp.issubdtype(
        jnp.result_type(leaf), jnp.inexact
    )


class ParamSpec:
    def __init__(self, variables_like):
        leaves, self.treedef = jax.tree_util.tree_flatten_with_path(variables_like)
        self.paths = [jax.tree_util.keystr(path) for path, _ in leaves]
        self.trainable = [_is_trainable(leaf) for _, leaf in leaves]
        self.statics = [
            None if t else leaf for t, (_, leaf) in zip(self.trainable, leaves)
        ]
        self.shapes = [
            jnp.shape(leaf) if t else None
            for t, (_, leaf) in zip(self.trainable, leaves)
        ]
        self.nparams = sum(math.prod(s) for s in self.shapes if s is not None)
        if self.nparams == 0:
            raise DiffsolJaxError(
                "variables_like has no trainable leaves: at least one leaf must be an "
                "array with a float (or complex) dtype"
            )

    def ravel(self, variables):
        """The trainable leaves of `variables`, flattened and concatenated to
        `f64[nparams]`. The values of the static leaves are ignored: they were fixed when
        the model was built."""
        leaves, treedef = jax.tree_util.tree_flatten_with_path(variables)
        if treedef != self.treedef:
            raise DiffsolJaxError(
                f"variables does not match variables_like: got {treedef}, expected "
                f"{self.treedef}"
            )
        flat = []
        for t, shape, path, (_, leaf) in zip(
            self.trainable, self.shapes, self.paths, leaves
        ):
            if not t:
                if _is_trainable(leaf):
                    raise DiffsolJaxError(
                        f"variables{path} is an inexact array, but the matching leaf of "
                        "variables_like was not, so the model treats it as fixed. Build "
                        "the model with it trainable in variables_like, or pass the "
                        "construction-time value"
                    )
                continue
            leaf = jnp.asarray(leaf)
            if leaf.shape != shape:
                raise DiffsolJaxError(
                    f"variables{path} has shape {leaf.shape}, expected {shape} (batch "
                    "with jax.vmap, not with extra axes)"
                )
            flat.append(leaf.ravel().astype(jnp.float64))
        return jnp.concatenate(flat)

    def bind_equations(self, rhs, init, out, mass):
        """The user functions over the pytree, as functions of the flat `f64[nparams]`
        vector, with f64 outputs whatever the leaf dtypes."""

        def wrap(f):
            if f is None:
                return None
            return lambda x, p, t: jnp.asarray(f(x, self.bind(p), t), jnp.float64)

        def rhs_flat(x, p, t):
            return jnp.asarray(rhs(x, self.bind(p), t), jnp.float64)

        def init_flat(p, t):
            return jnp.asarray(init(self.bind(p), t), jnp.float64)

        return rhs_flat, init_flat, wrap(out), wrap(mass)

    def bind(self, flat):
        """The pytree of `f64[nparams]`: the trainable leaves carved out of `flat`, the
        static leaves from construction."""
        leaves, i = [], 0
        for t, shape, static in zip(self.trainable, self.shapes, self.statics):
            if t:
                assert shape is not None  # set for every trainable leaf
                n = math.prod(shape)
                leaves.append(flat[i : i + n].reshape(shape))
                i += n
            else:
                leaves.append(static)
        return jax.tree_util.tree_unflatten(self.treedef, leaves)
