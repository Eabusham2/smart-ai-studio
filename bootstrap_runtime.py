"""Release entry point: bundled Python, offline portable launch, private setup.

The release builder validates this same entry point. No model weights are loaded
or downloaded during dependency installation or the --smoke-test command.
"""
from __future__ import annotations
import contextlib
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def requirements_file(root):
    locked = root / 'requirements.lock.txt'
    return locked if locked.is_file() else root / 'requirements.txt'


def fingerprint(root):
    req = requirements_file(root)
    return hashlib.sha256(req.read_bytes()).hexdigest()


def python_path(runtime):
    return runtime / ('python.exe' if os.name == 'nt' else 'bin/python3')


def ready(runtime, root):
    try:
        return (runtime / '.smartai-deps-ready').read_text(encoding='utf-8').strip() == fingerprint(root)
    except (OSError, ValueError):
        return False


def pip_install(runtime, root):
    python = python_path(runtime)
    wheel_kind = 'metal' if platform.system() == 'Darwin' and platform.machine() == 'arm64' else 'cpu'
    options = ['--prefer-binary', '--only-binary=llama-cpp-python',
               '--extra-index-url', f'https://abetlen.github.io/llama-cpp-python/whl/{wheel_kind}']
    # Default dependencies/sampling are untouched. On setup, use the exact
    # platform lock produced and imported by the native build, not future latest.
    subprocess.run([str(python), '-m', 'pip', 'install', '--no-cache-dir', *options,
                    '-r', str(requirements_file(root))], check=True)
    subprocess.run([str(python), '-c', 'import tkinter,sqlite3,torch,transformers,llama_cpp; tkinter.Tcl()'], check=True)
    (runtime / '.smartai-deps-ready').write_text(fingerprint(root)+'\n', encoding='utf-8')


def private_root():
    if os.name == 'nt':
        return Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData/Local')) / 'SmartAI'
    if sys.platform == 'darwin':
        return Path.home() / 'Library/Application Support/SmartAI'
    return Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local/share')) / 'SmartAI'


@contextlib.contextmanager
def setup_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as handle:
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            handle.write(b'0'); handle.flush(); handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == 'nt':
                handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def ensure_runtime(root):
    bundled = root / 'runtime/python'
    if not python_path(bundled).is_file():
        raise RuntimeError('This release is missing its bundled Python runtime. Re-extract the complete package.')
    if ready(bundled, root):
        return bundled
    private = private_root() / 'runtimes' / (platform.system()+'-'+platform.machine()+'-'+fingerprint(root)[:20])
    with setup_lock(private.parent / (private.name+'.lock')):
        if ready(private, root):
            return private
        if not python_path(private).exists():
            temporary = private.with_name(private.name+'.installing')
            if temporary.exists():
                shutil.rmtree(temporary)
            shutil.copytree(bundled, temporary, symlinks=False)
            temporary.rename(private)
        print('Installing Smart AI Studio dependencies. Model weights are not downloaded.', flush=True)
        pip_install(private, root)
    return private


def main(argv=None):
    root = Path(__file__).resolve().parent
    args = list(sys.argv[1:] if argv is None else argv)
    runtime = ensure_runtime(root)
    if '--install-only' in args:
        print('Smart AI Studio dependencies are ready.', flush=True)
        return 0
    target = root / ('package_smoke.py' if '--smoke-test' in args else 'app_gui.py')
    args = [a for a in args if a != '--smoke-test']
    env = dict(os.environ)
    env.pop('PYTHONHOME', None); env.pop('PYTHONPATH', None)
    env['PYTHONNOUSERSITE'] = '1'
    # Run the owned interpreter, not a machine/global Python discovered on PATH.
    return subprocess.call([str(python_path(runtime)), str(target), *args], cwd=str(root), env=env)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print('Smart AI Studio setup failed: '+str(exc), file=sys.stderr, flush=True)
        raise SystemExit(1)
