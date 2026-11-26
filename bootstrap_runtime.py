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
import plistlib
import shutil
import subprocess
import sys


def requirements_file(root):
    locked = root / 'requirements.lock.txt'
    return locked if locked.is_file() else root / 'requirements.txt'


def fingerprint(root):
    req = requirements_file(root)
    payload = req.read_bytes()
    mlx_setup = root / 'mlx_setup.py'
    if platform.system() in {'Linux', 'Windows'} and mlx_setup.is_file():
        payload += mlx_setup.read_bytes()
        mlx_compat = root / 'mlx_import_compat.py'
        if mlx_compat.is_file():
            payload += mlx_compat.read_bytes()
        payload += os.environ.get('SMARTAI_MLX_BACKEND', 'auto').encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def python_path(runtime):
    return runtime / ('python.exe' if os.name == 'nt' else 'bin/python3')


def ready(runtime, root):
    try:
        return (runtime / '.smartai-deps-ready').read_text(encoding='utf-8').strip() == fingerprint(root)
    except (OSError, ValueError):
        return False


def pip_install(runtime, root):
    python = python_path(runtime)
    if platform.system() in {'Linux', 'Windows'}:
        (runtime / '.smartai-deps-ready').unlink(missing_ok=True)
    wheel_kind = 'metal' if platform.system() == 'Darwin' and platform.machine() == 'arm64' else 'cpu'
    options = ['--prefer-binary', '--only-binary=llama-cpp-python',
               '--extra-index-url', f'https://abetlen.github.io/llama-cpp-python/whl/{wheel_kind}']
    wheelhouse = root / 'wheelhouse'
    manifest = wheelhouse / 'manifest.json'
    if manifest.is_file():
        rows = json.loads(manifest.read_text(encoding='utf-8')).get('files', [])
        if len(rows) != 1:
            raise RuntimeError('Invalid bundled native wheel manifest')
        row = rows[0]
        name = row['name']
        if Path(name).name != name or not name.startswith('llama_cpp_python-') or not name.endswith('.whl'):
            raise RuntimeError('Invalid bundled native wheel path')
        wheel = wheelhouse / name
        if wheel.stat().st_size != row['size'] or hashlib.sha256(wheel.read_bytes()).hexdigest() != row['sha256']:
            raise RuntimeError('Bundled native wheel checksum mismatch')
        subprocess.run([str(python), '-m', 'pip', 'install', '--no-cache-dir',
                        '--no-deps', str(wheel)], check=True)
        options = ['--prefer-binary', '--only-binary=llama-cpp-python', '--find-links', str(wheelhouse)]
    # On setup, install the exact tested lock; preserve installed native wheel.
    subprocess.run([str(python), '-m', 'pip', 'install', '--no-cache-dir', *options,
                    '-r', str(requirements_file(root))], check=True)
    if platform.system() in {'Linux', 'Windows'}:
        from mlx_setup import install as install_native_mlx
        install_native_mlx(python, os.environ.get('SMARTAI_MLX_BACKEND', 'auto'))
    subprocess.run([str(python), '-c', 'import tkinter,sqlite3,torch,transformers,llama_cpp; tkinter.Tcl()'], check=True)
    (runtime / '.smartai-deps-ready').write_text(fingerprint(root)+'\n', encoding='utf-8')


def private_root():
    # The release smoke test must not accidentally reuse a runner's previous
    # installation. Also permit an explicit user-owned setup destination.
    override = os.environ.get('SMARTAI_SETUP_HOME')
    if override:
        return Path(override).expanduser().resolve()
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


def macos_native_launcher(root):
    """Find this app's native entry point, including a renamed/relocated app."""
    if sys.platform != 'darwin' or root.name != 'app' or root.parent.name != 'Resources' or root.parent.parent.name != 'Contents':
        return None
    bundle = root.parent.parent.parent
    if bundle.suffix != '.app':
        return None
    try:
        with (bundle / 'Contents/Info.plist').open('rb') as handle:
            info = plistlib.load(handle)
    except (OSError, ValueError):
        return None
    if info.get('SmartAINativeLauncher') != 1:
        return None
    executable = bundle / 'Contents/MacOS/applet'
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise RuntimeError('This release is missing its native Mac launcher. Re-extract the complete app.')
    return executable


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
    # The updater must also wait for this parent to release the bundled runtime.
    env['SMARTAI_BOOTSTRAP_PID'] = str(os.getpid())
    native = macos_native_launcher(root)
    if native is not None:
        # Keep the actual GUI executable inside SmartAI.app on first launch,
        # private Setup reuse, and the updater's direct bootstrap restart path.
        os.execve(str(native), [str(native), '--smartai-run-python',
                               str(runtime.resolve()), str(target), *args], env)
        return 0  # execve replaces this process; useful to mocked launch tests.
    # Run the owned interpreter, not a machine/global Python discovered on PATH.
    return subprocess.call([str(python_path(runtime)), str(target), *args], cwd=str(root), env=env)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print('Smart AI Studio setup failed: '+str(exc), file=sys.stderr, flush=True)
        raise SystemExit(1)
