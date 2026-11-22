"""The native side: loading `libdiffsol_jax` and the PJRT plugin, and the diffsol solver of one
equation set with its cache of compiled callbacks."""

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

FFI_TARGETS = {
    "diffsol_solve_dense": "DiffsolSolveDense",
    "diffsol_solve_fwd_sens": "DiffsolSolveFwdSens",
    "diffsol_solve_vjp_fwd": "DiffsolSolveVjpFwd",
    "diffsol_solve_vjp_bkwd": "DiffsolSolveVjpBkwd",
}

# The C enum values of `diffsol.h`.
ODE_SOLVERS = {"bdf": 0, "esdirk34": 1, "tr_bdf2": 2, "tsit45": 3}
LINEAR_SOLVERS = {"default": 0, "lu": 1, "klu": 2}
MATRIX_TYPES = {"nalgebra_dense": 0, "faer_dense": 1, "faer_sparse": 2}
# diffsol-c's `ScalarType`, by the solver dtype it stands for.
DTYPE_CODES = {F32: 0, F64: 1}

# The trailing `err, err_len` of a function that can fail; `call` supplies them.
_ERR = [c.c_char_p, c.c_size_t]
_U64 = c.POINTER(c.c_uint64)

# name: (argtypes, restype)
_SIGNATURES = {
    "diffsol_jax_init": ([c.c_char_p] + _ERR, c.c_int),
    # nstates, nparams, nout, dtype, matrix_type, linear_solver, ode_solver,
    # max_checkpoints
    "diffsol_jax_model_new": (
        [c.c_size_t] * 3 + [c.c_int32] * 4 + [c.c_size_t] + _ERR,
        c.c_int64,
    ),
    "diffsol_jax_model_free": ([c.c_int64], None),
    "diffsol_jax_model_add_variant": (
        [c.c_int64, c.c_size_t, c.c_int32, c.c_size_t]
        + [c.c_char_p, c.c_size_t] * 2
        + _ERR,
        c.c_int,
    ),
    "diffsol_jax_model_has_variant": (
        [c.c_int64, c.c_size_t, c.c_int32, c.c_size_t],
        c.c_int,
    ),
    "diffsol_jax_model_clear_checkpoints": ([c.c_int64], None),
    "diffsol_jax_model_num_checkpoints": ([c.c_int64], c.c_size_t),
    "diffsol_jax_model_num_recomputes": ([c.c_int64], c.c_uint64),
    "diffsol_jax_model_stats": (
        [c.c_int64, c.c_size_t, _U64, _U64, _U64, c.c_int],
        c.c_int,
    ),
}


def _default_lib_path() -> Path:
    ext = ".dylib" if sys.platform == "darwin" else ".so"
    bundled = Path(__file__).resolve().parent / f"libdiffsol_jax{ext}"
    if bundled.exists():
        return bundled
    return Path(__file__).resolve().parents[2] / "build" / f"libdiffsol_jax{ext}"


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


def lib() -> c.CDLL:
    """The library, loaded and set up on the first call.

    Environment variables:
    - `DIFFSOL_JAX_LIB`: the path of `libdiffsol_jax` (default: bundled, else `build/` at the
      repository root).
    - `DIFFSOL_JAX_PJRT_PLUGIN`: the path of the CPU PJRT plugin (default: the
      `diffsol-jax-pjrt-cpu` package of the `cpu` extra, else a bundled
      `pjrt_c_api_cpu_plugin.so`, else `~/.local/share/pjrt`, else `.pjrt` at the
      repository root).
    """
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
    for name, (argtypes, restype) in _SIGNATURES.items():
        fn = getattr(so, name)
        fn.argtypes = argtypes
        fn.restype = restype

    err = c.create_string_buffer(_ERR_LEN)
    if not so.diffsol_jax_init(plugin.encode(), err, _ERR_LEN):
        raise DiffsolJaxError(err.value.decode())
    for target, symbol in FFI_TARGETS.items():
        jax.ffi.register_ffi_target(
            target, jax.ffi.pycapsule(getattr(so, symbol)), platform="cpu"
        )
    _lib = so
    return so


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
    """The diffsol solver of one equation set, and the callbacks compiled for it.

    The executables depend on the number of lanes `B`, which is known only after `jax.vmap`, so
    the lowering rule of each solve compiles the callbacks for its own `B` (`ensure_callbacks`).
    The native model is freed when the last reference to this object goes.
    """

    def __init__(
        self, eq: Equations, *, solver, linear_solver, matrix_type, max_checkpoints
    ):
        self.eq = eq
        so = lib()
        self.id = call(
            so.diffsol_jax_model_new,
            eq.nstates,
            eq.nparams,
            eq.nout,
            DTYPE_CODES[eq.dtype],
            one_of("matrix_type", matrix_type, MATRIX_TYPES),
            one_of("linear_solver", linear_solver, LINEAR_SOLVERS),
            one_of("solver", solver, ODE_SOLVERS),
            max_checkpoints,
        )
        self._finalizer = weakref.finalize(self, so.diffsol_jax_model_free, self.id)

    def _add(self, cb, nbatch, p):
        so = lib()
        if so.diffsol_jax_model_has_variant(self.id, nbatch, int(cb), p):
            return
        code = _lower.lower(self.eq, cb, nbatch, p)
        opts = _lower.compile_options()
        call(
            so.diffsol_jax_model_add_variant,
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
        """Compile the callbacks that a solve of `kind` with `nbatch` lanes needs. Every
        callback gets a `P = 1` executable, so the set of callbacks diffsol is given does not
        change between solve kinds -- which it may not, once the first solve has built the
        diffsol model."""
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

    # ---- counters ----------------------------------------------------------------------------

    def clear_checkpoints(self):
        lib().diffsol_jax_model_clear_checkpoints(self.id)

    def num_checkpoints(self) -> int:
        return lib().diffsol_jax_model_num_checkpoints(self.id)

    def num_recomputes(self) -> int:
        return lib().diffsol_jax_model_num_recomputes(self.id)

    def callback_stats(self, nbatch: int, reset: bool = False):
        ncalls, nstaged, nviews = (c.c_uint64(), c.c_uint64(), c.c_uint64())
        ok = lib().diffsol_jax_model_stats(
            self.id,
            nbatch,
            c.byref(ncalls),
            c.byref(nstaged),
            c.byref(nviews),
            int(reset),
        )
        return (ncalls.value, nstaged.value, nviews.value) if ok else (0, 0, 0)
