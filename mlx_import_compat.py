"""Import the pinned MLX-LM package on Windows without POSIX resource limits."""
import builtins
import ctypes
from importlib import import_module
from importlib.machinery import PathFinder
import sys
import threading


_import_lock = threading.RLock()


class _WindowsFileLimits:
    """The one resource operation used by mlx_lm.utils on import."""
    RLIMIT_NOFILE = object()

    @classmethod
    def setrlimit(cls, resource, limits):
        if resource is not cls.RLIMIT_NOFILE:
            raise ValueError("Only the MLX file descriptor limit is supported")
        # Windows maintains a per-CRT stdio limit, without a POSIX hard limit.
        crt = ctypes.CDLL("ucrtbase")
        if crt._getmaxstdio() < limits[0] and crt._setmaxstdio(limits[0]) == -1:
            raise OSError("Cannot raise the Windows CRT stdio limit for MLX")


class _WindowsUtilsLoader:
    def __init__(self, loader):
        self.loader = loader

    def create_module(self, spec):
        return self.loader.create_module(spec)

    def exec_module(self, module):
        def import_for_utils(name, *args, **kwargs):
            if name == "resource":
                return _WindowsFileLimits
            return builtins.__import__(name, *args, **kwargs)

        # Keep the platform adapter local to mlx_lm.utils. Registering a fake
        # process-wide resource module can break other Windows dependencies.
        module.__builtins__ = {**vars(builtins), "__import__": import_for_utils}
        self.loader.exec_module(module)


class _WindowsUtilsFinder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mlx_lm.utils":
            spec = PathFinder.find_spec(fullname, path)
            if spec and hasattr(spec.loader, "exec_module"):
                spec.loader = _WindowsUtilsLoader(spec.loader)
                return spec
        return None


def load_mlx_lm():
    """Return MLX-LM, adapting its import-time file limit for Windows."""
    if sys.platform != "win32":
        return import_module("mlx_lm")
    with _import_lock:
        finder = _WindowsUtilsFinder()
        sys.meta_path.insert(0, finder)
        try:
            return import_module("mlx_lm")
        finally:
            sys.meta_path.remove(finder)
