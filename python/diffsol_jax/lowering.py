from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.extend import core as jex_core

# skip 'o' (looks like zero), reserve 't'
_INDEX_LETTERS = "ijklmnpqrs"

_UNARY_FNS = {
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "exp": "exp",
    "log": "log",
    "sqrt": "sqrt",
    "tanh": "tanh",
    "sinh": "sinh",
    "cosh": "cosh",
    "abs": "abs",
    "logistic": "sigmoid",
}

_BINARY_OPS = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
}


def _subs_for_rank(rank: int) -> str:
    if rank == 0:
        return ""
    if rank > len(_INDEX_LETTERS):
        raise NotImplementedError(f"rank {rank} exceeds subscript pool")
    return "_" + _INDEX_LETTERS[:rank]


def _fmt_float(x: float) -> str:
    if x == int(x):
        return f"{int(x)}.0"
    return repr(x)


class Env:
    def __init__(self):
        self._by_var: dict = {}
        self._counter = 0

    def fresh_name(self, hint: str = "v") -> str:
        name = f"{hint}{self._counter}"
        self._counter += 1
        return name

    def bind(self, var, name: str, subs: str):
        self._by_var[var] = (name, subs)

    def get(self, atom):
        if isinstance(atom, jex_core.Literal):
            val = atom.val
            if hasattr(val, "shape") and val.shape != ():
                raise NotImplementedError("non-scalar inline literal not supported")
            return _fmt_float(float(val)), ""
        return self._by_var[atom]


def _bind_const(var, val, env: Env, lines: list[str]):
    shape = getattr(val, "shape", ())
    if shape == ():
        name = env.fresh_name("c")
        env.bind(var, name, "")
        lines.append(f"{name} {{ {_fmt_float(float(val))} }}")
    else:
        raise NotImplementedError(
            f"non-scalar constvar (shape={shape}); pass as a parameter instead"
        )


def _emit_eqn(eqn, env: Env, lines: list[str]):
    prim_name = eqn.primitive.name

    if prim_name == "broadcast_in_dim":
        (in_var,) = eqn.invars
        in_name, in_subs = env.get(in_var)
        out_var = eqn.outvars[0]
        out_shape = eqn.params["shape"]
        bcast_dims = eqn.params["broadcast_dimensions"]
        out_rank = len(out_shape)
        out_subs_full = _subs_for_rank(out_rank)[1:] if out_rank else ""
        if in_subs == "" and out_rank >= 1:
            out_name = env.fresh_name("b")
            env.bind(out_var, out_name, out_subs_full)
            if out_rank == 0:
                lines.append(f"{out_name} {{ {in_name} }}")
            else:
                lines.append(f"{out_name}_{out_subs_full} {{ {in_name} }}")
            return
        if len(in_subs) == out_rank and tuple(bcast_dims) == tuple(range(out_rank)):
            env.bind(out_var, in_name, in_subs)
            return
        raise NotImplementedError(
            f"broadcast_in_dim shape={out_shape} bcast_dims={bcast_dims} not handled"
        )

    if prim_name == "convert_element_type":
        (in_var,) = eqn.invars
        new_dtype = eqn.params["new_dtype"]
        if jnp.dtype(new_dtype) != jnp.float64:
            raise NotImplementedError(f"convert_element_type to {new_dtype}: f64 only")
        in_name, in_subs = env.get(in_var)
        env.bind(eqn.outvars[0], in_name, in_subs)
        return

    if prim_name in ("pjit", "closed_call"):
        inner = eqn.params["jaxpr"]
        if hasattr(inner, "jaxpr"):
            inner_jaxpr = inner.jaxpr
            inner_consts = inner.consts if hasattr(inner, "consts") else inner.literals
        else:
            inner_jaxpr = inner
            inner_consts = []
        for inner_var, outer_atom in zip(inner_jaxpr.invars, eqn.invars):
            name, subs = env.get(outer_atom)
            env.bind(inner_var, name, subs)
        for cv, cval in zip(inner_jaxpr.constvars, inner_consts):
            _bind_const(cv, cval, env, lines)
        for inner_eqn in inner_jaxpr.eqns:
            _emit_eqn(inner_eqn, env, lines)
        for outer_var, inner_atom in zip(eqn.outvars, inner_jaxpr.outvars):
            if isinstance(inner_atom, jex_core.Literal):
                env.bind(outer_var, _fmt_float(float(inner_atom.val)), "")
            else:
                name, subs = env._by_var[inner_atom]
                env.bind(outer_var, name, subs)
        return

    out_var = eqn.outvars[0]
    out_rank = len(out_var.aval.shape)
    out_subs_letters = _INDEX_LETTERS[:out_rank] if out_rank else ""
    out_subs_suffix = f"_{out_subs_letters}" if out_subs_letters else ""
    out_name = env.fresh_name("v")

    def _ref(atom) -> str:
        name, subs = env.get(atom)
        if isinstance(atom, jex_core.Literal) or subs == "":
            return name
        return f"{name}_{subs}"

    if prim_name in _UNARY_FNS:
        (a,) = eqn.invars
        expr = f"{_UNARY_FNS[prim_name]}({_ref(a)})"
    elif prim_name in _BINARY_OPS:
        a, b = eqn.invars
        expr = f"{_ref(a)} {_BINARY_OPS[prim_name]} {_ref(b)}"
    elif prim_name == "neg":
        (a,) = eqn.invars
        expr = f"-{_ref(a)}"
    elif prim_name == "pow":
        a, b = eqn.invars
        expr = f"pow({_ref(a)}, {_ref(b)})"
    elif prim_name == "integer_pow":
        (a,) = eqn.invars
        expr = f"pow({_ref(a)}, {_fmt_float(float(eqn.params['y']))})"
    elif prim_name == "square":
        (a,) = eqn.invars
        r = _ref(a)
        expr = f"{r} * {r}"
    else:
        raise NotImplementedError(f"primitive '{prim_name}' not supported")

    env.bind(out_var, out_name, out_subs_letters)
    lines.append(f"{out_name}{out_subs_suffix} {{ {expr} }}")


def make_diffsl_tuple(
    rhs_tuple,
    y0,
    p_example,
    t_example: float = 0.0,
    *,
    param_names: list[str] | None = None,
    state_names: list[str] | None = None,
) -> str:
    """Trace rhs_tuple and emit a DiffSL source string.

    rhs_tuple(t, y, p) must return a tuple of scalars, one per state component.
    p and y are indexed at the Python level (p[0], y[1], ...) before any JAX ops.
    """
    y0 = jnp.asarray(y0, dtype=jnp.float64)
    p_example = jnp.asarray(p_example, dtype=jnp.float64)
    t_example = jnp.float64(t_example)

    n_state = int(y0.shape[0]) if y0.ndim == 1 else 1
    n_param = int(p_example.shape[0]) if p_example.ndim == 1 else 1

    if state_names is None:
        state_names = [f"y{i}" for i in range(n_state)]
    if param_names is None:
        param_names = [f"p{i}" for i in range(n_param)]

    closed = jax.make_jaxpr(rhs_tuple)(t_example, y0, p_example)
    jaxpr = closed.jaxpr
    consts = closed.literals

    env = Env()
    lines: list[str] = []

    in_body = ", ".join(
        f"{param_names[i]} = {_fmt_float(float(p_example[i]))}" for i in range(n_param)
    )
    in_subs = "_i" if n_param > 1 else ""
    lines.append(f"in{in_subs} {{ {in_body} }}")

    t_var, y_var, p_var = jaxpr.invars
    env.bind(t_var, "t", "")
    env.bind(y_var, "__u_vec", "i" if n_state > 1 else "")
    env.bind(p_var, "__p_vec", "i" if n_param > 1 else "")
    env._param_names_by_index = {i: param_names[i] for i in range(n_param)}
    env._state_names_by_index = {i: state_names[i] for i in range(n_state)}
    env._p_var = p_var
    env._y_var = y_var

    for cv, cval in zip(jaxpr.constvars, consts):
        _bind_const(cv, cval, env, lines)

    for eqn in jaxpr.eqns:
        pn = eqn.primitive.name
        if pn == "slice" and eqn.invars[0] in (env._p_var, env._y_var):
            start = eqn.params["start_indices"][0]
            limit = eqn.params["limit_indices"][0]
            if limit - start == 1:
                if eqn.invars[0] is env._p_var:
                    sym = env._param_names_by_index[start]
                else:
                    # diffsl 0.9.4 cranelift rejects u_i[k] in expressions;
                    # use the declared state label name directly.
                    sym = env._state_names_by_index[start]
                env.bind(eqn.outvars[0], sym, "")
                continue
        if pn == "squeeze":
            (a,) = eqn.invars
            name, subs = env.get(a)
            env.bind(eqn.outvars[0], name, "")
            continue
        _emit_eqn(eqn, env, lines)

    if n_state == 1:
        u_line = f"u {{ {state_names[0]} = {_fmt_float(float(y0))} }}"
    else:
        inits = ", ".join(
            f"{state_names[i]} = {_fmt_float(float(y0[i]))}" for i in range(n_state)
        )
        u_line = f"u_i {{ {inits} }}"

    if len(jaxpr.outvars) != n_state:
        raise ValueError(
            f"rhs_tuple returned {len(jaxpr.outvars)} outputs, state has {n_state}"
        )
    if n_state == 1:
        name, _ = env.get(jaxpr.outvars[0])
        f_line = f"F {{ {name} }}"
    else:
        comps = [env.get(out)[0] for out in jaxpr.outvars]
        f_line = "F_i {\n  " + ",\n  ".join(comps) + ",\n}"

    header = lines[0]
    body = "\n".join(lines[1:])
    return f"{header}\n{u_line}\n{body}\n{f_line}\n"


def lotka_volterra(t, y, p):
    x, yy = y[0], y[1]
    alpha, beta, delta, gamma = p[0], p[1], p[2], p[3]
    return (alpha * x - beta * x * yy, delta * x * yy - gamma * yy)


def lorenz(t, y, p):
    x, yy, z = y[0], y[1], y[2]
    sigma, rho, beta = p[0], p[1], p[2]
    return (sigma * (yy - x), x * (rho - z) - yy, x * yy - beta * z)


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    lv_src = make_diffsl_tuple(
        lotka_volterra,
        y0=jnp.array([1.0, 0.5]),
        p_example=jnp.array([1.5, 1.0, 0.75, 3.0]),
        param_names=["alpha", "beta", "delta", "gamma"],
        state_names=["x", "y"],
    )
    print(lv_src)

    lz_src = make_diffsl_tuple(
        lorenz,
        y0=jnp.array([1.0, 0.0, 0.0]),
        p_example=jnp.array([10.0, 28.0, 8.0 / 3.0]),
        param_names=["sigma", "rho", "beta"],
        state_names=["x", "y", "z"],
    )
    print(lz_src)
