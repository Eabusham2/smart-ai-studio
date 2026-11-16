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
    checksums=path.with_name(path.name+'.sha256')
    checksums.write_text(''.join(row['sha256']+'  '+row['name']+'\n' for row in rows),encoding='utf-8')
    sh=path.with_name('Reassemble-'+path.name+'.sh')
    sh.write_text('#!/bin/sh\nset -eu\ncd "$(dirname "$0")"\n'
                  'if command -v sha256sum >/dev/null 2>&1; then SHA="sha256sum"; else SHA="shasum -a 256"; fi\n'
                  f'$SHA -c "{checksums.name}"\ncat "{path.name}".part[0-9][0-9][0-9] > "{path.name}"\n'
                  f"printf '%s\\n' '{full_hash}  {path.name}' | $SHA -c -\n"
                  f'echo "Verified: {path.name}. Extract normally."\n',encoding='utf-8')
    sh.chmod(0o755)
    ps=path.with_name('Reassemble-'+path.name+'.ps1')
    ps.write_text("$ErrorActionPreference = 'Stop'\nSet-Location $PSScriptRoot\n"
                  f"$m = Get-Content -LiteralPath '{manifest.name}' -Raw | ConvertFrom-Json\n"
                  "$dst = [IO.File]::Create((Join-Path $PSScriptRoot $m.archive))\n"
                  "try { foreach ($part in $m.parts) {\n"
                  "  $p = Join-Path $PSScriptRoot $part.name\n"
                  "  if ((Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLower() -ne $part.sha256) { throw 'Part checksum mismatch' }\n"
                  "  $src = [IO.File]::OpenRead($p); try { $src.CopyTo($dst) } finally { $src.Dispose() }\n"
                  "} } finally { $dst.Dispose() }\n"
                  "if ((Get-FileHash -LiteralPath $m.archive -Algorithm SHA256).Hash.ToLower() -ne $m.sha256) { throw 'Archive checksum mismatch' }\n"
                  "Write-Host ('Verified: ' + $m.archive + '. Extract normally.')\n",encoding='utf-8')
    path.unlink()
    return [path.parent/r['name'] for r in rows]+[manifest,checksums,sh,ps]


def portable_dependency_lock(frozen, native_version):
    """Do not retain the CI worker's local wheel URL in a shipped setup lock."""
    rows = []
    for line in frozen.splitlines():
        if re.match(r'^llama[-_]cpp[-_]python\s*@', line, flags=re.I):
            line = 'llama-cpp-python==' + native_version
        rows.append(line)
    return '\n'.join(rows) + '\n'


def smoke_app(code, *, setup_home=None):
    """Launch the packaged interpreter with no host Python/user site fallback."""
    from bootstrap_runtime import python_path
    code = Path(code)
    env = dict(os.environ, PYTHONNOUSERSITE='1', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', RUN_REAL_WEIGHTS_EVAL='0')
    env.pop('PYTHONHOME', None)
    env.pop('PYTHONPATH', None)
    if setup_home is not None:
        env['SMARTAI_SETUP_HOME'] = str(setup_home)
    subprocess.run([str(python_path(code / 'runtime/python')),
                    str(code / 'bootstrap_runtime.py'), '--smoke-test'],
                   env=env, check=True)


def verified_dmg(app, destination):
    """Package the .app directory itself, then test it from the mounted image."""
    app, destination = Path(app), Path(destination)
    # -srcfolder copies a directory's contents. Its source must contain the
    # .app, rather than point inside the .app and put Contents at volume root.
    subprocess.run(['hdiutil', 'create', '-volname', 'SmartAI', '-srcfolder',
                    str(app.parent), '-ov', '-format', 'UDZO', str(destination)], check=True)
    mount = app.parent.parent / 'mounted-dmg-check'
    mount.mkdir()
    attached = False
    try:
        subprocess.run(['hdiutil', 'attach', '-readonly', '-nobrowse', '-mountpoint',
                        str(mount), str(destination)], check=True)
        attached = True
        mounted_code = mount / 'SmartAI.app/Contents/Resources/app'
        if not (mounted_code / 'bootstrap_runtime.py').is_file():
            raise RuntimeError('DMG does not contain the complete SmartAI.app')
        smoke_app(mounted_code)
    finally:
        if attached:
            subprocess.run(['hdiutil', 'detach', str(mount)], check=True)


def package(target, output, *, prepare_only=False):
    from bootstrap_runtime import pip_install, fingerprint, python_path
    import build_app
    if target!=target_name():raise RuntimeError('Packages must be built on their matching native OS/architecture')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    build=output.parent/('package-work-'+target)
    if build.exists():shutil.rmtree(build)
    build.mkdir()
    base=fetch_runtime(target,build/'base')
    # Copy the verified runtime while assembling; all archives, including DMG,
    # are made only after this runtime has been placed in the app.
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
    smoke_app(code)
    if prepare_only:return app
    portable=output/f'SmartAI-{target}-portable.zip'
    archive_tree(app,portable)
    files=split_large(portable)
    if target.startswith('Linux'):
        linux_tar=output/'SmartAI-Linux-x86_64.tar.gz'
        archive_tree(app,linux_tar,'tar');files+=split_large(linux_tar)
    if target=='macOS-arm64':
        # Publisher combines these exact, tested app bytes with the existing
        # model bundles. Keep that packaging step separate from model inference.
        archive_tree(app,output/'SmartAI.app.tar.gz','tar')
    if target.startswith('macOS'):
        dmg=output/f'SmartAI-{target}.dmg'
        verified_dmg(app, dmg)
        files+=split_large(dmg)
    # Setup is the same app and pinned Python, with packages installed on demand
    # into a user-owned directory, never into /Applications or system Python.
    shutil.rmtree(code/'runtime/python');shutil.copytree(base,code/'runtime/python',symlinks=False)
    # The private Setup environment is intentionally empty: exercise its real
    # first-launch installation before certifying the non-portable format.
    setup_home=build/'clean setup home with spaces'
    if setup_home.exists():
        raise RuntimeError('Setup smoke directory is not initially empty')
    try:
        smoke_app(code, setup_home=setup_home)
        markers = list(setup_home.glob('runtimes/*/.smartai-deps-ready'))
        if len(markers) != 1 or markers[0].read_text(encoding='utf-8').strip() != fingerprint(code):
            raise RuntimeError('Setup did not initialize its isolated private runtime')
        first_marker = markers[0].stat().st_mtime_ns
        # A second launch must reuse the complete private environment.
        smoke_app(code, setup_home=setup_home)
        if markers[0].stat().st_mtime_ns != first_marker:
            raise RuntimeError('Second Setup launch unexpectedly reinstalled dependencies')
    finally:
        if setup_home.exists():
            shutil.rmtree(setup_home)
    setup=output/f'SmartAI-{target}-Setup.zip'
    archive_tree(app,setup);files+=split_large(setup)
    receipt={'schema':1,'target':target,'commit':os.environ.get('GITHUB_SHA','local'),
             'model_inference_run':False,'gui_smoke':'passed','setup_smoke':'passed',
             'dmg_smoke':'passed' if target.startswith('macOS') else 'not_applicable',
             'files':[{'name':p.name,'size':p.stat().st_size,'sha256':sha(p)} for p in files]}
    (output/f'SmartAI-{target}.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print('PACKAGES_VERIFIED',target,len(files),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--target',default=target_name());parser.add_argument('--output',default='release-packages')
    opts=parser.parse_args();package(opts.target,opts.output)
