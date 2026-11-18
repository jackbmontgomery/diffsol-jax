"""The parameter layout: how the `args` of a solve become the flat `f64[nparams]` vector that
the diffsol equations take, and back."""

import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ._lib import DiffsolJaxError

F64 = jnp.float64


def is_array(x) -> bool:
    return isinstance(x, (jax.Array, np.ndarray))


def _is_trainable(leaf) -> bool:
    """A leaf is a parameter if it is an array with a float (or complex) dtype; every other leaf
    is captured by the equations as a constant."""
    return is_array(leaf) and jnp.issubdtype(jnp.result_type(leaf), jnp.inexact)


def _constant_key(leaf, path):
    """A hashable key for a leaf that the equations capture as a constant: its value, since the
    value is compiled into the callbacks."""
    if not is_array(leaf):
        try:
            hash(leaf)
        except TypeError:
            raise DiffsolJaxError(
                f"args{path} is not an array and is not hashable, so it cannot be captured "
                "by the equations as a constant. Close over it in the equations instead"
            ) from None
        return leaf
    try:
        value = np.asarray(leaf)
    except Exception:
        raise DiffsolJaxError(
            f"args{path} has the dtype {jnp.result_type(leaf)}, which is not differentiable, "
            "so the equations capture its value -- but here it is a tracer (inside jax.jit or "
            "another transformation), so there is no value to capture. Close over it in the "
            "equations, or give it a float dtype to make it a parameter"
        ) from None
    return (value.dtype.str, value.shape, value.tobytes())


class _Layout:
    """The shared part: the user equations take `args`, the diffsol equations take the flat
    vector."""

    nparams: int
    key: Any

    def bind(self, flat):
        raise NotImplementedError

    def ravel(self, args) -> jax.Array:
        raise NotImplementedError

    def bind_equations(self, *fns):
        """The user functions of `args`, as functions of the flat vector."""

        def wrap(f):
            if f is None:
                return None
            return lambda a, p, t: jnp.asarray(f(a, self.bind(p), t), F64)

        return tuple(wrap(f) for f in fns)


class Flat(_Layout):
    """`args` is one array `[..., nparams]`: the equations get it as `p [nparams]`, and its
    leading axes are lanes of a batched solve."""

    def __init__(self, args):
        self.nparams = int(jnp.shape(args)[-1])
        self.key = ("flat", self.nparams)

    def bind(self, flat):
        return flat

    def ravel(self, args) -> jax.Array:
        return jnp.asarray(args, F64)


class Tree(_Layout):
    """`args` is a pytree: the float-array leaves are the parameters (in flatten order), every
    other leaf is a constant of the equations. Batch with `jax.vmap`, not with extra axes."""

    def __init__(self, args):
        leaves, self.treedef = jax.tree_util.tree_flatten_with_path(args)
        self.trainable, self.constants, self.shapes, key = [], [], [], []
        for path, leaf in leaves:
            trainable = _is_trainable(leaf)
            self.trainable.append(trainable)
            self.constants.append(None if trainable else leaf)
            self.shapes.append(jnp.shape(leaf) if trainable else None)
            key.append(
                ("p", jnp.shape(leaf), jnp.result_type(leaf).str)
                if trainable
                else ("c", _constant_key(leaf, jax.tree_util.keystr(path)))
            )
        self.nparams = sum(math.prod(s) for s in self.shapes if s is not None)
        self.key = (self.treedef, tuple(key))

    def bind(self, flat):
        """The pytree rebuilt from the flat `f64[nparams]` vector."""
        leaves, i = [], 0
        for trainable, shape, constant in zip(
            self.trainable, self.shapes, self.constants
        ):
            if not trainable:
                leaves.append(constant)
                continue
            assert shape is not None
            n = math.prod(shape)
            leaves.append(flat[i : i + n].reshape(shape))
            i += n
        return jax.tree_util.tree_unflatten(self.treedef, leaves)

    def ravel(self, args) -> jax.Array:
        """The parameter leaves of `args` as one `f64[nparams]` vector."""
        leaves = jax.tree_util.tree_leaves(args)
        if len(leaves) != len(self.trainable):
            raise DiffsolJaxError(
                f"args has {len(leaves)} leaves, expected {len(self.trainable)}"
            )
        flat = [
            jnp.asarray(leaf).ravel().astype(F64)
            for trainable, leaf in zip(self.trainable, leaves)
            if trainable
        ]
        return jnp.concatenate(flat) if flat else jnp.zeros((0,), F64)


def layout(args) -> _Layout:
    """The parameter layout of `args`: one array is the parameter vector itself, anything else is
    a pytree of parameters (`None` for a problem with no parameters)."""
    if is_array(args) and jnp.ndim(args) >= 1:
        return Flat(args)
    return Tree(args)
