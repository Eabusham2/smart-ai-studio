"""Build a relocatable portable + first-launch setup package on its native OS."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

BUILD_RECORD = '.smartai-build.json'
_USER_PATHS = {
    'data', 'SmartAI_Data', 'preloaded_models', 'models', 'cache', 'caches',
    '.cache', '.env', '.git', '__pycache__', '.pytest_cache',
}


def release_identity():
    """Identify rolling releases by their actual build, not the reused tag."""
    commit = os.environ.get('GITHUB_SHA', '').strip()
    if not commit:
        commit = subprocess.check_output(
            ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    if not re.fullmatch(r'[0-9a-fA-F]{40}', commit):
        raise RuntimeError('A full source commit is required for release identity')
    ref = os.environ.get('GITHUB_REF', '')
    name = os.environ.get('GITHUB_REF_NAME', '') or ref.removeprefix('refs/tags/')
    is_tag = os.environ.get('GITHUB_REF_TYPE') == 'tag' or ref.startswith('refs/tags/')
    return {'schema': 1, 'repository': 'Eabusham2/smart-ai-studio',
            'commit': commit.lower(), 'tag': name if is_tag and name.startswith('v') else 'latest'}


def write_build_record(code, app, target, package_kind, identity):
    """Stamp the exact package flavor while keeping user paths out of ownership."""
    code, app = Path(code), Path(app)
    if package_kind not in {'portable', 'setup'}:
        raise ValueError('Unknown package kind')
    managed = {p.name for p in code.iterdir() if p.name not in _USER_PATHS}
    managed.add(BUILD_RECORD)
    installed = {p.name for p in app.iterdir() if p.name not in _USER_PATHS}
    if code == app:
        installed.add(BUILD_RECORD)
    record = dict(identity, target=target, package_kind=package_kind,
                  managed_paths=sorted(managed), install_managed_paths=sorted(installed))
    (code / BUILD_RECORD).write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    return record


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def target_name():
    machine=platform.machine().lower()
    arch='arm64' if machine in ('arm64','aarch64') else 'x86_64' if machine in ('amd64','x86_64') else machine
    return {'Darwin':'macOS','Windows':'Windows','Linux':'Linux'}[platform.system()]+'-'+arch


def extract_python(archive, destination):
    """Extract a checksum-pinned runtime without version-dependent tar filters.

    Validate every member and link before writing. Never extract through a
    symlink, apply archive ownership, or require Windows symlink privileges.
    """
    root = Path(destination).resolve()
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        if sum(m.size for m in members) > 2 * 1024**3:
            raise RuntimeError('Python archive exceeds extraction limit')
        entries = {}
        links = {}
        for member in members:
            name = PurePosixPath(member.name)
            if (name.is_absolute() or '..' in name.parts or '\\' in member.name
                    or ':' in member.name or not name.parts or name.parts[0] != 'python'):
                raise RuntimeError('Unsafe Python archive path')
            key = str(name)
            if key in entries:
                raise RuntimeError('Duplicate Python archive path')
            if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
                raise RuntimeError('Unsupported Python archive entry')
            target = root.joinpath(*name.parts)
            if not target.resolve().is_relative_to(root / 'python'):
                raise RuntimeError('Python archive path escapes runtime')
            entries[key] = member
            if member.issym() or member.islnk():
                link = PurePosixPath(member.linkname)
                parent = target.parent if member.issym() else root
                resolved = (parent / str(link)).resolve()
                if (link.is_absolute() or '\\' in member.linkname or ':' in member.linkname
                        or not resolved.is_relative_to(root / 'python')):
                    raise RuntimeError('Python archive link escapes runtime')
                links[key] = resolved
        # Reject writes through even an internal alias; real runtime archives
        # place aliases after their actual files instead of writing under them.
        for name in entries:
            if any(str(parent) in links for parent in PurePosixPath(name).parents):
                raise RuntimeError('Python archive member has a link parent')
        for member in members:
            if member.issym() or member.islnk():
                continue
            target = root.joinpath(*PurePosixPath(member.name).parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, target.open('xb') as output:
                    shutil.copyfileobj(source, output, 1024 * 1024)
                target.chmod(member.mode & 0o777)
        # Resolve aliases in dependency order; never follow a missing external
        # target or cyclic link. Windows packages need no administrator rights.
        pending = dict(links)
        while pending:
            progress = False
            for name, resolved in list(pending.items()):
                if not resolved.exists():
                    continue
                target = root.joinpath(*PurePosixPath(name).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                if os.name == 'nt':
                    if resolved.is_dir():
                        shutil.copytree(resolved, target)
                    else:
                        shutil.copy2(resolved, target)
                elif entries[name].islnk():
                    os.link(resolved, target)
                else:
                    target.symlink_to(os.path.relpath(resolved, target.parent), target_is_directory=resolved.is_dir())
                del pending[name]
                progress = True
            if not progress:
                raise RuntimeError('Python archive has missing or cyclic link targets')


def fetch_runtime(target, destination):
    record=json.loads((ROOT/'packaging/python-runtimes.json').read_text(encoding='utf-8'))['assets'][target]
    dest=Path(destination);dest.mkdir(parents=True,exist_ok=True)
    archive=dest/'python.tar.gz'
    with urllib.request.urlopen(urllib.request.Request(record['url'],headers={'User-Agent':'SmartAI-release-builder'}),timeout=120) as response, archive.open('wb') as out:
        shutil.copyfileobj(response,out,8*1024*1024)
    if archive.stat().st_size!=record['size'] or sha(archive)!=record['sha256']:
        raise RuntimeError('Python runtime checksum or size mismatch')
    extract_python(archive,dest);archive.unlink()
    return dest/'python'


def archive_tree(source, destination, kind='zip'):
    source=Path(source);destination=Path(destination)
    if kind=='zip':
        with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED,compresslevel=4,allowZip64=True) as z:
            for p in sorted(source.rglob('*')):
                if p.is_file():z.write(p,source.name+'/'+str(p.relative_to(source)))
    else:
        with tarfile.open(destination,'w:gz',compresslevel=4) as t:
            t.add(source,arcname=source.name,recursive=True)


def split_large(path, limit=1_850_000_000):
    path=Path(path)
    if path.stat().st_size<=limit:return [path]
    rows=[]
    with path.open('rb') as src:
        i=0
        while True:
            chunk=src.read(min(8*1024*1024,limit))
            if not chunk:break
            part=path.with_name(path.name+f'.part{i:03d}')
            with part.open('wb') as out:
                written=0
                while chunk:
                    out.write(chunk);written+=len(chunk)
                    if written>=limit:break
                    chunk=src.read(min(8*1024*1024,limit-written))
            rows.append({'name':part.name,'size':part.stat().st_size,'sha256':sha(part)})
            i+=1
    manifest=path.with_name(path.name+'.parts.json')
    full_hash=sha(path)
    manifest.write_text(json.dumps({'archive':path.name,'sha256':full_hash,'parts':rows},indent=2)+'\n',encoding='utf-8')
    path.unlink()
    return [path.parent/r['name'] for r in rows]+[manifest]


def portable_dependency_lock(frozen, native_version):
    """Do not retain the CI worker's local wheel URL in a shipped setup lock."""
    rows = []
    for line in frozen.splitlines():
        if re.match(r'^llama[-_]cpp[-_]python\s*@', line, flags=re.I):
            line = 'llama-cpp-python==' + native_version
        rows.append(line)
    return '\n'.join(rows) + '\n'


def smoke_app(code, *, setup_home=None, smoke_data_dir=None, launch=1):
    """Launch the packaged interpreter with no host Python/user site fallback."""
    from bootstrap_runtime import macos_native_launcher, python_path
    code = Path(code)
    env = dict(os.environ, PYTHONNOUSERSITE='1', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', RUN_REAL_WEIGHTS_EVAL='0')
    env.pop('PYTHONHOME', None)
    env.pop('PYTHONPATH', None)
    if setup_home is not None:
        env['SMARTAI_SETUP_HOME'] = str(setup_home)
    if smoke_data_dir is not None:
        env['SMARTAI_SMOKE_DATA_DIR'] = str(Path(smoke_data_dir).resolve())
    env['SMARTAI_SMOKE_LAUNCH'] = str(launch)
    native = macos_native_launcher(code.resolve()) if launch == 1 else None
    command = ([str(native), '--smoke-test'] if native is not None else
               [str(python_path(code / 'runtime/python')),
                str(code / 'bootstrap_runtime.py'), '--smoke-test'])
    subprocess.run(command, env=env, check=True)


def smoke_bonsai_runtime(code, cache_dir):
    """Verify each native OS can install and start the pinned MTP executable.

    Only the small publisher runtime is downloaded; --help exits before loading
    weights. The temporary CI cache is outside the package and never ships.
    """
    from bootstrap_runtime import python_path
    code = Path(code).resolve()
    env = dict(os.environ, PYTHONNOUSERSITE='1', SMARTAI_DATA_DIR=str(cache_dir),
               SMARTAI_BONSAI_NATIVE_VARIANT='cpu', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', RUN_REAL_WEIGHTS_EVAL='0')
    for key in ('PYTHONHOME', 'PYTHONPATH', 'SMARTAI_BONSAI_SERVER'):
        env.pop(key, None)
    script = '''
from pathlib import Path
import subprocess
from core.bonsai_native_runtime import ensure_bonsai_server, RELEASE
from core.engines.bonsai25_engine import Bonsai25ReasoningBackend, PUBLISHER_CONTRACT
from core.speculative_runtime import SpeculativePlan
server = ensure_bonsai_server()
backend = Bonsai25ReasoningBackend('no-model-weights.gguf', server_path=server,
                                  n_ctx=4096, n_gpu_layers=0, max_draft_tokens=2,
                                  mmproj_path='no-projector-weights.gguf')
backend._plan = SpeculativePlan('mtp', 'mtp', '0'*64, None, 2,
                                runtime_contract=PUBLISHER_CONTRACT)
result = subprocess.run(backend._build_command(8711) + ['--help'],
                        env=backend._runtime_env(), capture_output=True,
                        text=True, timeout=30)
help_text = result.stdout + result.stderr
required = ('--spec-type', 'draft-mtp', '--spec-draft-n-max',
            '--spec-draft-n-min', '--spec-draft-p-min', '--jinja', '--mmproj')
if result.returncode or not all(flag in help_text for flag in required):
    raise RuntimeError('Pinned Bonsai MTP runtime did not start: ' + help_text[-2000:])
assert ensure_bonsai_server() == server, 'Verified runtime cache could not be reopened'
print('BONSAI_NATIVE_SMOKE', RELEASE, 'passed; no model weights loaded')
'''
    subprocess.run([str(python_path(code / 'runtime/python')), '-c', script],
                   cwd=code, env=env, check=True, timeout=300)


def smoke_bonsai_image_runtime(code, cache_dir):
    """Test private image setup using the relocated Mac app's actual Python."""
    from bootstrap_runtime import python_path
    code = Path(code).resolve()
    env = dict(os.environ, PYTHONNOUSERSITE='1', SMARTAI_DATA_DIR=str(Path(cache_dir).resolve()),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', RUN_REAL_WEIGHTS_EVAL='0')
    for key in ('PYTHONHOME', 'PYTHONPATH'):
        env.pop(key, None)
    script = '''
from importlib.metadata import version
from core.bonsai_image_runtime import ensure_bonsai_image_runtime
before = version('mlx')
assert before == '0.32.0', 'Unexpected chat MLX runtime'
runtime = ensure_bonsai_image_runtime(force_probe=True)
assert version('mlx') == before, 'Private image setup changed the chat MLX runtime'
assert ensure_bonsai_image_runtime() == runtime, 'The private image runtime could not be reused'
print('BONSAI_IMAGE_NATIVE_SMOKE passed; publisher imports and 2-bit Metal, no model weights')
'''
    subprocess.run([str(python_path(code / 'runtime/python')), '-c', script],
                   cwd=code, env=env, check=True, timeout=1800)


def smoke_restored_media_runtimes(code, cache_dir, *, apple_silicon=False):
    """Exercise packaged native code and tiny operators, without model weights."""
    from bootstrap_runtime import python_path
    code = Path(code).resolve()
    env = dict(os.environ, PYTHONNOUSERSITE='1', SMARTAI_DATA_DIR=str(Path(cache_dir).resolve()),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', RUN_REAL_WEIGHTS_EVAL='0')
    for key in ('PYTHONHOME', 'PYTHONPATH', 'SMARTAI_GGUF_MEDIA_DIR'):
        env.pop(key, None)
    python = str(python_path(code / 'runtime/python'))
    subprocess.run([python, '-m', 'core.gguf_image_runtime', '--prepare', '--probe'],
                   cwd=code, env=env, check=True, timeout=240)
    if apple_silicon:
        script = '''
from importlib.metadata import version
from core.mlx_video_runtime import ensure_mlx_video_runtime
before = version('mlx')
for backend in ('ltx_mlx', 'minimax_mlx'):
    runtime = ensure_mlx_video_runtime(backend, force_probe=True)
    assert ensure_mlx_video_runtime(backend) == runtime
    assert version('mlx') == before, 'Video runtime setup changed the main chat environment'
print('RESTORED_MLX_VIDEO_SMOKE passed; native code only, no pretrained model weights')
'''
        subprocess.run([python, '-c', script], cwd=code, env=env, check=True, timeout=1800)


def package(target, output, *, prepare_only=False):
    from bootstrap_runtime import pip_install, fingerprint, python_path
    from tools.bundled_native import embed_native_runtime
    from tools.portable_download import write_linux_downloader
    from tools.build_gguf_media_runtime import build_gguf_media_runtime
    import build_app
    if target!=target_name():raise RuntimeError('Packages must be built on their matching native OS/architecture')
    identity = release_identity()
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    build=output.parent/('package-work-'+target)
    if build.exists():shutil.rmtree(build)
    build.mkdir()
    base=fetch_runtime(target,build/'base')
    # All packages contain the verified private Python runtime.
    os.environ['SMARTAI_EMBEDDED_RUNTIME']=str(base)
    os.environ['SMARTAI_PACKAGE_ASSEMBLY_ONLY']='1'
    os.environ['SMARTAI_MODEL_BUNDLE_RELEASE']='1'
    if target.startswith('macOS'):
        build_app.create_macos_bundle(str(build),'SmartAI')
        app=build/'SmartAI.app';code=app/'Contents/Resources/app'
    elif target.startswith('Windows'):
        build_app.create_windows_bundle(str(build),'SmartAI')
        app=build/'SmartAI-Windows';code=app
    else:
        build_app.create_linux_bundle(str(build),'SmartAI')
        app=build/'SmartAI-Linux';code=app/'SmartAI.AppDir/usr/bin'
    bundled_native = embed_native_runtime(code, target)
    # A matching native build avoids new-OS-only publisher binaries and keeps
    # compilers off users' machines. The same files ship in portable and Setup.
    # Ubuntu 22's SDK predates required Vulkan structures. A deliberate CPU
    # build preserves this package's OS floor; Metal stays native on Mac.
    build_gguf_media_runtime(code / 'runtime/gguf-media',
                            backend='cpu' if target.startswith('Linux') else 'auto',
                            work_dir=build / 'gguf-media-build')
    runtime=code/'runtime/python'
    # Every declared app dependency is installed; no models or inference.
    if target.startswith('macOS'):
        wheelhouse = code / 'wheelhouse'
        wheelhouse.mkdir()
        env = dict(os.environ)
        env['CMAKE_ARGS'] = '-DGGML_METAL=on -DGGML_NATIVE=OFF'
        subprocess.run([str(python_path(runtime)), '-m', 'pip', 'wheel', '--no-cache-dir',
                        '--no-deps', '--no-binary=llama-cpp-python',
                        '--wheel-dir', str(wheelhouse), 'llama-cpp-python>=0.3.0'],
                       env=env, check=True)
        wheels = list(wheelhouse.glob('llama_cpp_python-*.whl'))
        if len(wheels) != 1:
            raise RuntimeError('Expected one locally built Metal wheel')
        with zipfile.ZipFile(wheels[0]) as wheel:
            bad = wheel.testzip()
            if bad:
                raise RuntimeError('Locally built Metal wheel failed integrity: ' + bad)
        (wheelhouse / 'manifest.json').write_text(json.dumps({
            'files': [{'name': wheels[0].name, 'sha256': sha(wheels[0]), 'size': wheels[0].stat().st_size}]
        }, indent=2) + '\n', encoding='utf-8')
    pip_install(runtime, code)
    lock=subprocess.check_output([str(python_path(runtime)),'-m','pip','freeze'],text=True)
    native_version=subprocess.check_output([
        str(python_path(runtime)), '-c',
        'from importlib.metadata import version; print(version("llama-cpp-python"))'
    ], text=True).strip()
    lock=portable_dependency_lock(lock, native_version)
    (code/'requirements.lock.txt').write_text(lock,encoding='utf-8')
    (code/'runtime/python/.smartai-deps-ready').write_text(fingerprint(code)+'\n',encoding='utf-8')
    # Prove the installed interpreter survives relocation and an empty user home.
    relocated=build/'relocated with spaces'/app.name
    relocated.parent.mkdir();shutil.move(str(app),str(relocated));app=relocated
    code=app/'Contents/Resources/app' if target.startswith('macOS') else app/'SmartAI.AppDir/usr/bin' if target.startswith('Linux') else app
    write_build_record(code, app, target, 'portable', identity)
    portable_smoke_data = build / 'portable launch preferences'
    smoke_app(code, smoke_data_dir=portable_smoke_data)
    smoke_app(code, smoke_data_dir=portable_smoke_data, launch=2)
    smoke_bonsai_runtime(code, build / 'bonsai-runtime-smoke')
    if target.startswith('macOS'):
        smoke_bonsai_image_runtime(code, build / 'bonsai-image-runtime-smoke')
    smoke_restored_media_runtimes(code, build / 'restored-media-runtime-smoke',
                                 apple_silicon=target.startswith('macOS'))
    if prepare_only:return app
    portable=output/f'SmartAI-{target}-portable.zip'
    archive_tree(app,portable)
    files=split_large(portable)
    if target.startswith('Linux') and not portable.exists():
        files.append(write_linux_downloader(
            portable.with_name(portable.name+'.parts.json'), output,
            repository=identity['repository'], tag=identity['tag']))
    # Setup is the same app and pinned Python, with packages installed on demand
    # into a user-owned directory, never into /Applications or system Python.
    shutil.rmtree(code/'runtime/python');shutil.copytree(base,code/'runtime/python',symlinks=False)
    write_build_record(code, app, target, 'setup', identity)
    # The private Setup environment is intentionally empty: exercise its real
    # first-launch installation before certifying the non-portable format.
    setup_home=build/'clean setup home with spaces'
    if setup_home.exists():
        raise RuntimeError('Setup smoke directory is not initially empty')
    try:
        setup_smoke_data = build / 'setup launch preferences'
        smoke_app(code, setup_home=setup_home, smoke_data_dir=setup_smoke_data)
        markers = list(setup_home.glob('runtimes/*/.smartai-deps-ready'))
        if len(markers) != 1 or markers[0].read_text(encoding='utf-8').strip() != fingerprint(code):
            raise RuntimeError('Setup did not initialize its isolated private runtime')
        first_marker = markers[0].stat().st_mtime_ns
        # A second launch must reuse the complete private environment.
        smoke_app(code, setup_home=setup_home, smoke_data_dir=setup_smoke_data, launch=2)
        if markers[0].stat().st_mtime_ns != first_marker:
            raise RuntimeError('Second Setup launch unexpectedly reinstalled dependencies')
    finally:
        if setup_home.exists():
            shutil.rmtree(setup_home)
    setup=output/f'SmartAI-{target}-Setup.zip'
    archive_tree(app,setup);files+=split_large(setup)
    receipt={'schema':1,'target':target,'commit':identity['commit'],
             'model_inference_run':False,'gui_smoke':'passed','setup_smoke':'passed',
             'relaunch_smoke':'passed',
             'macos_app_identity':'passed' if target.startswith('macOS') else 'not_applicable',
             'bonsai_native_smoke':'passed',
             'bonsai_image_runtime_smoke':'passed' if target.startswith('macOS') else 'not_applicable',
             'gguf_media_runtime_smoke':'passed',
             'mlx_video_runtime_smoke':'passed' if target.startswith('macOS') else 'not_applicable',
             'bundled_native_verified':'passed' if bundled_native else 'not_applicable',
             'files':[{'name':p.name,'size':p.stat().st_size,'sha256':sha(p)} for p in files]}
    (output/f'SmartAI-{target}.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print('PACKAGES_VERIFIED',target,len(files),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--target',default=target_name());parser.add_argument('--output',default='release-packages')
    opts=parser.parse_args();package(opts.target,opts.output)
