import subprocess
import sys
import sysconfig
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CargoBuildHook(BuildHookInterface):
    """Build the native library with cargo and put it in the wheel.

    The library has no CPython ABI (the package binds it with ctypes), so the
    wheel is tagged py3-none-<platform>: one wheel per platform, every Python.
    Editable installs skip this; `_native.py` falls back to target/release.
    """

    def initialize(self, version, build_data):
        if self.target_name != "wheel" or version == "editable":
            return
        root = Path(self.root)
        subprocess.run(["cargo", "build", "--release"], cwd=root, check=True)
        ext = ".dylib" if sys.platform == "darwin" else ".so"
        lib = root / "target" / "release" / f"libdiffsol_jax{ext}"
        build_data["force_include"][str(lib)] = f"diffsol_jax/libdiffsol_jax{ext}"
        platform = sysconfig.get_platform().replace("-", "_").replace(".", "_")
        build_data["tag"] = f"py3-none-{platform}"
        build_data["pure_python"] = False
