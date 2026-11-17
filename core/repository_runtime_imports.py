"""Temporarily bind a model repository's own runtime modules without global leaks."""
from __future__ import annotations

from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
import threading

_REPO_IMPORT_LOCK = threading.RLock()
_MISSING = object()


@contextmanager
def repository_runtime_imports(runtime_directory):
    """Avoid collisions with unrelated installed packages called `runtime`.

    Only explicitly selected, locally downloaded model code is executed here.
    Restore the process-global import table even if model initialization fails.
    """
    directory = Path(runtime_directory)
    modules = [(name, directory / (name + ".py")) for name in ("codec", "runtime")]
    with _REPO_IMPORT_LOCK:
        previous = {name: sys.modules.get(name, _MISSING) for name, _ in modules}
        try:
            for name, path in modules:
                if not path.is_file():
                    continue
                spec = importlib.util.spec_from_file_location(name, path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"Cannot import selected model runtime file: {path}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)
            yield
        finally:
            for name, value in previous.items():
                if value is _MISSING:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value
