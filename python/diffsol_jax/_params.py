"""The parameter layout: how the `args` of a solve become the flat `f64[nparams]` vector that
the diffsol equations take, and back. The layout is declared once, by the `params` of an
`ODEProblem`: an int (a flat vector of that width) or a template pytree."""

import math

import jax
import jax.numpy as jnp
import numpy as np

from ._lib import DiffsolJaxError

F64 = jnp.float64


def is_array(x) -> bool:
    """An array, or a shape and dtype standing in for one (`jax.eval_shape` of a network)."""
    return isinstance(x, (jax.Array, np.ndarray, jax.ShapeDtypeStruct))


def _is_param(leaf) -> bool:
    """A leaf is a parameter if it is an array with a float (or complex) dtype; every other leaf
    is captured by the equations as a constant."""
    return is_array(leaf) and jnp.issubdtype(jnp.result_type(leaf), jnp.inexact)


class _Layout:
    """The shared part: the user equations take `args`, the diffsol equations take the flat
    vector."""

    nparams: int

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
    """`params=<int>`: the equations get the parameters as one vector `p [nparams]`, and the
    leading axes of the `args` of a solve are lanes of a batched solve."""

    def __init__(self, nparams: int):
        if nparams < 0:
            raise DiffsolJaxError(f"params must not be negative, got {nparams}")
        self.nparams = int(nparams)

    def __repr__(self):
        return f"params={self.nparams}"

    def bind(self, flat):
        return flat

    def ravel(self, args) -> jax.Array:
        args = jnp.asarray(args, F64)
        if args.ndim < 1 or args.shape[-1] != self.nparams:
            raise DiffsolJaxError(
                f"args must have the shape [..., {self.nparams}], got {args.shape}"
            )
        return args


class Tree(_Layout):
    """`params=<pytree>`: a template. Its float-array leaves are the parameters (in flatten
    order), and its other leaves are captured by the equations as constants. The `args` of a
    solve must match it; batch with `jax.vmap`, not with extra axes."""

    def __init__(self, template):
        leaves, self.treedef = jax.tree_util.tree_flatten_with_path(template)
        self.paths = [jax.tree_util.keystr(path) for path, _ in leaves]
        self.is_param = [_is_param(leaf) for _, leaf in leaves]
        self.shapes = [
            jnp.shape(leaf) if p else None
            for p, (_, leaf) in zip(self.is_param, leaves)
        ]
        self.constants = []
        for p, path, (_, leaf) in zip(self.is_param, self.paths, leaves):
            if p:
                self.constants.append(None)
            elif isinstance(leaf, jax.ShapeDtypeStruct):
                raise DiffsolJaxError(
                    f"params{path} has the dtype {leaf.dtype}, which is not differentiable, so "
                    "the equations capture its value -- but a ShapeDtypeStruct has no value. "
                    "Give the value itself, or a float dtype to make it a parameter"
                )
            else:
                self.constants.append(leaf)
        self.nparams = sum(math.prod(s) for s in self.shapes if s is not None)

    def __repr__(self):
        return f"nparams={self.nparams} (a pytree)"

    def bind(self, flat):
        """The pytree rebuilt from the flat `f64[nparams]` vector."""
        leaves, i = [], 0
        for param, shape, constant in zip(self.is_param, self.shapes, self.constants):
            if not param:
                leaves.append(constant)
                continue
            assert shape is not None
            n = math.prod(shape)
            leaves.append(flat[i : i + n].reshape(shape))
            i += n
        return jax.tree_util.tree_unflatten(self.treedef, leaves)

    def ravel(self, args) -> jax.Array:
        """The parameter leaves of `args` as one `f64[nparams]` vector."""
        leaves, treedef = jax.tree_util.tree_flatten_with_path(args)
        if treedef != self.treedef:
            raise DiffsolJaxError(
                f"args does not match the params template: got {treedef}, expected "
                f"{self.treedef}"
            )
        flat = []
        for param, shape, path, (_, leaf) in zip(
            self.is_param, self.shapes, self.paths, leaves
        ):
            if not param:
                if _is_param(leaf):
                    raise DiffsolJaxError(
                        f"args{path} is an inexact array, but the matching leaf of the params "
                        "template was not, so the equations captured it as a constant. Build "
                        "the problem with a float leaf there to make it a parameter, or pass "
                        "the value it was built with"
                    )
                continue
            leaf = jnp.asarray(leaf)
            if leaf.shape != shape:
                raise DiffsolJaxError(
                    f"args{path} has the shape {leaf.shape}, expected {shape}: batch a pytree "
                    "of parameters with jax.vmap, or declare params=<int> for a flat parameter "
                    "vector whose leading axes are lanes"
                )
            flat.append(leaf.ravel().astype(F64))
        return jnp.concatenate(flat) if flat else jnp.zeros((0,), F64)


def layout(params) -> _Layout:
    """The layout declared by `params`: an int is a flat parameter vector of that width, `None`
    is no parameters, and anything else is a template pytree."""
    if isinstance(params, bool):
        raise DiffsolJaxError("params must be an int or a pytree, not a bool")
    if isinstance(params, (int, np.integer)):
        return Flat(params)
    return Tree(params)
