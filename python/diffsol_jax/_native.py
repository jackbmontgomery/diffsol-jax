import ctypes as c
import os
import sys
import weakref
from pathlib import Path

import jax

from . import _lower
from ._equations import Equations
from ._types import F32, F64, DiffsolJaxError

_ERR_LEN = 4096
_lib = None

# The field of `DiffsolJaxApi` (`cpp/c_api.cc`) that holds each XLA FFI custom-call handler's
# address.
FFI_TARGETS = {
    "diffsol_solve_dense": "solve_dense",
    "diffsol_solve_fwd_sens": "solve_fwd_sens",
    "diffsol_solve_vjp_fwd": "solve_vjp_fwd",
    "diffsol_solve_vjp_bkwd": "solve_vjp_bkwd",
}

# The C enum values of `diffsol.h`.
ODE_SOLVERS = {"bdf": 0, "esdirk34": 1, "tr_bdf2": 2, "tsit45": 3}
LINEAR_SOLVERS = {"default": 0, "lu": 1, "klu": 2}
MATRIX_TYPES = {"nalgebra_dense": 0, "faer_dense": 1, "faer_sparse": 2}
DTYPE_CODES = {F32: 0, F64: 1}

_ERR = [c.c_char_p, c.c_size_t]
_U64 = c.POINTER(c.c_uint64)


class _Api(c.Structure):
    """Mirrors `DiffsolJaxApi` in `cpp/c_api.cc`, field for field and in the same order."""

    _fields_ = [
        ("init", c.CFUNCTYPE(c.c_int, c.c_char_p, *_ERR)),
        (
            "model_new",
            c.CFUNCTYPE(
                c.c_int64, *([c.c_size_t] * 3 + [c.c_int32] * 4 + [c.c_size_t] + _ERR)
            ),
        ),
        ("model_free", c.CFUNCTYPE(None, c.c_int64)),
        (
            "model_add_variant",
            c.CFUNCTYPE(
                c.c_int,
                *(
                    [c.c_int64, c.c_size_t, c.c_int32, c.c_size_t]
                    + [c.c_char_p, c.c_size_t] * 2
                    + _ERR
                ),
            ),
        ),
        (
            "model_has_variant",
            c.CFUNCTYPE(c.c_int, c.c_int64, c.c_size_t, c.c_int32, c.c_size_t),
        ),
        ("model_clear_checkpoints", c.CFUNCTYPE(None, c.c_int64)),
        ("model_num_checkpoints", c.CFUNCTYPE(c.c_size_t, c.c_int64)),
        ("model_num_recomputes", c.CFUNCTYPE(c.c_uint64, c.c_int64)),
        (
            "model_stats",
            c.CFUNCTYPE(c.c_int, c.c_int64, c.c_size_t, _U64, _U64, _U64, c.c_int),
        ),
        # raw addresses: `jax.ffi.pycapsule` only needs the address, not a typed signature.
        ("solve_dense", c.c_void_p),
        ("solve_fwd_sens", c.c_void_p),
        ("solve_vjp_fwd", c.c_void_p),
        ("solve_vjp_bkwd", c.c_void_p),
    ]


def _default_lib_path() -> Path:
    ext = ".dylib" if sys.platform == "darwin" else ".so"
    bundled = Path(__file__).resolve().parent / f"libdiffsol_jax{ext}"
    if bundled.exists():
        return bundled
    return (
        Path(__file__).resolve().parents[2]
        / "target"
        / "release"
        / f"libdiffsol_jax{ext}"
    )


def _find_plugin() -> str:
    env = os.environ.get("DIFFSOL_JAX_PJRT_PLUGIN")
    if env:
        return env
    try:
        import diffsol_jax_pjrt_cpu
    except ImportError:
        pass
    else:
        return str(diffsol_jax_pjrt_cpu.library_path())
    name = "pjrt_c_api_cpu_plugin.so"
    here = Path(__file__).resolve()
    for candidate in (
        here.parent / name,
        Path.home() / ".local" / "share" / "pjrt" / name,
        here.parents[2] / ".pjrt" / name,
    ):
        if candidate.exists():
            return str(candidate)
    raise DiffsolJaxError(
        'no CPU PJRT plugin found: install one with `pip install "diffsol-jax[cpu]"`, '
        f"set DIFFSOL_JAX_PJRT_PLUGIN to the path of a {name}, or put one in "
        "~/.local/share/pjrt"
    )


def lib() -> _Api:
    global _lib
    if _lib is not None:
        return _lib
    path = os.environ.get("DIFFSOL_JAX_LIB", str(_default_lib_path()))
    plugin = _find_plugin()
    # the plugin's XLA logs spurious error-level messages on every compilation. solver failures
    # reach the user through DiffsolJaxError, so quieten the plugin's stderr.
    if os.environ.get("TF_CPP_MIN_LOG_LEVEL") in (None, "1"):
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    so = c.CDLL(path)
    so.diffsol_jax_get_api.restype = c.POINTER(_Api)
    api = so.diffsol_jax_get_api().contents

    err = c.create_string_buffer(_ERR_LEN)
    if not api.init(plugin.encode(), err, _ERR_LEN):
        raise DiffsolJaxError(err.value.decode())
    for target, field in FFI_TARGETS.items():
        jax.ffi.register_ffi_target(
            target, jax.ffi.pycapsule(getattr(api, field)), platform="cpu"
        )
    _lib = api
    return api


def call(fn, *args):
    """Call a library function whose last two arguments are `err, err_len`."""
    err = c.create_string_buffer(_ERR_LEN)
    result = fn(*args, err, _ERR_LEN)
    if not result:
        raise DiffsolJaxError(err.value.decode())
    return result


def one_of(name, value, table):
    if value not in table:
        raise DiffsolJaxError(
            f"{name} must be one of {', '.join(table)}, got {value!r}"
        )
    return table[value]


class Model:
    def __init__(
        self, eq: Equations, *, solver, linear_solver, matrix_type, max_checkpoints
    ):
        self.eq = eq
        api = lib()
        self.id = call(
            api.model_new,
            eq.nstates,
            eq.nparams,
            eq.nout,
            DTYPE_CODES[eq.dtype],
            one_of("matrix_type", matrix_type, MATRIX_TYPES),
            one_of("linear_solver", linear_solver, LINEAR_SOLVERS),
            one_of("solver", solver, ODE_SOLVERS),
            max_checkpoints,
        )
        self._finalizer = weakref.finalize(self, api.model_free, self.id)

    def _add(self, cb, nbatch, p):
        api = lib()
        if api.model_has_variant(self.id, nbatch, int(cb), p):
            return
        code = _lower.lower(self.eq, cb, nbatch, p)
        opts = _lower.compile_options()
        call(
            api.model_add_variant,
            self.id,
            nbatch,
            int(cb),
            p,
            code,
            len(code),
            opts,
            len(opts),
        )

    def ensure_callbacks(self, kind, nbatch, ndirections=1):
        available = self.eq.available()
        for cb in available:
            self._add(cb, nbatch, 1)
        multi_p = ()
        if kind == "fwd_sens" and self.eq.nparams > 1:
            multi_p = [(cb, self.eq.nparams) for cb in _lower.FWD_SENS_MULTI_P]
        elif kind == "vjp_bkwd" and ndirections > 1:
            multi_p = [(cb, ndirections) for cb in _lower.VJP_BKWD_MULTI_P]
        for cb, p in multi_p:
            if cb in available:
                self._add(cb, nbatch, p)

    def clear_checkpoints(self):
        lib().model_clear_checkpoints(self.id)

    def num_checkpoints(self) -> int:
        return lib().model_num_checkpoints(self.id)

    def num_recomputes(self) -> int:
        return lib().model_num_recomputes(self.id)

    def callback_stats(self, nbatch: int, reset: bool = False):
        ncalls, nstaged, nviews = (c.c_uint64(), c.c_uint64(), c.c_uint64())
        ok = lib().model_stats(
            self.id,
            nbatch,
            c.byref(ncalls),
            c.byref(nstaged),
            c.byref(nviews),
            int(reset),
        )
        return (ncalls.value, nstaged.value, nviews.value) if ok else (0, 0, 0)
