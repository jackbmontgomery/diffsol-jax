"""Tests of the CPU PJRT plugin discovery, without loading a library."""

import sys
import types

import pytest

from diffsol_jax import _native


def test_environment_variable_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("DIFFSOL_JAX_PJRT_PLUGIN", "/some/plugin.so")
    module = types.ModuleType("diffsol_jax_pjrt_cpu")
    module.library_path = lambda: tmp_path / "libpjrt_cpu.so"
    monkeypatch.setitem(sys.modules, "diffsol_jax_pjrt_cpu", module)
    assert _native._find_plugin() == "/some/plugin.so"


def test_installed_package(monkeypatch, tmp_path):
    monkeypatch.delenv("DIFFSOL_JAX_PJRT_PLUGIN", raising=False)
    module = types.ModuleType("diffsol_jax_pjrt_cpu")
    module.library_path = lambda: tmp_path / "libpjrt_cpu.so"
    monkeypatch.setitem(sys.modules, "diffsol_jax_pjrt_cpu", module)
    assert _native._find_plugin() == str(tmp_path / "libpjrt_cpu.so")


def test_error_names_the_extra(monkeypatch, tmp_path):
    monkeypatch.delenv("DIFFSOL_JAX_PJRT_PLUGIN", raising=False)
    # `None` in `sys.modules` makes the import raise, as if the extra were not installed.
    monkeypatch.setitem(sys.modules, "diffsol_jax_pjrt_cpu", None)
    # point the fallback directories somewhere empty
    monkeypatch.setattr(_native.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(
        _native, "__file__", str(tmp_path / "a" / "b" / "c" / "_native.py")
    )
    with pytest.raises(_native.DiffsolJaxError, match=r"diffsol-jax\[cpu\]"):
        _native._find_plugin()
