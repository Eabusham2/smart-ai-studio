"""Build a relocatable portable + first-launch setup package on its native OS."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
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
    """Validate all paths and internal links before writing executable bytes."""
    root=Path(destination).resolve()
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers()
        if sum(m.size for m in members)>2*1024**3:
            raise RuntimeError('Python archive exceeds extraction limit')
        seen=set()
        for m in members:
            path=PurePosixPath(m.name)
            if path.is_absolute() or '..' in path.parts or '\\' in m.name or ':' in m.name or not path.parts or path.parts[0]!='python':
                raise RuntimeError('Unsafe Python archive path')
            if m.name in seen:raise RuntimeError('Duplicate Python archive path')
            seen.add(m.name)
            if not (m.isfile() or m.isdir() or m.issym() or m.islnk()):
                raise RuntimeError('Unsupported Python archive entry')
            if m.issym() or m.islnk():
                link=PurePosixPath(m.linkname)
                parent=(root/str(path.parent)) if m.issym() else root
                resolved=(parent/str(link)).resolve()
                if link.is_absolute() or not resolved.is_relative_to(root/'python'):
                    raise RuntimeError('Python archive link escapes runtime')
        # All inputs are from a checksum-pinned upstream archive, prevalidated.
        tar.extractall(root, members=members, filter='data')


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
    pip_install(runtime,ROOT)
    lock=subprocess.check_output([str(python_path(runtime)),'-m','pip','freeze'],text=True)
    (code/'requirements.lock.txt').write_text(lock,encoding='utf-8')
    (code/'runtime/python/.smartai-deps-ready').write_text(fingerprint(code)+'\n',encoding='utf-8')
    # Prove the installed interpreter survives relocation and an empty user home.
    relocated=build/'relocated with spaces'/app.name
    relocated.parent.mkdir();shutil.move(str(app),str(relocated));app=relocated
    code=app/'Contents/Resources/app' if target.startswith('macOS') else app/'SmartAI.AppDir/usr/bin' if target.startswith('Linux') else app
    env=dict(os.environ, PYTHONNOUSERSITE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', RUN_REAL_WEIGHTS_EVAL='0')
    env.pop('PYTHONHOME',None);env.pop('PYTHONPATH',None)
    subprocess.run([str(python_path(code/'runtime/python')),str(code/'bootstrap_runtime.py'),'--smoke-test'],env=env,check=True)
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
        subprocess.run(['hdiutil','create','-volname','SmartAI','-srcfolder',str(app),'-ov','-format','UDZO',str(dmg)],check=True)
        files+=split_large(dmg)
    # Setup is the same app and pinned Python, with packages installed on demand
    # into a user-owned directory, never into /Applications or system Python.
    shutil.rmtree(code/'runtime/python');shutil.copytree(base,code/'runtime/python',symlinks=False)
    setup=output/f'SmartAI-{target}-Setup.zip'
    archive_tree(app,setup);files+=split_large(setup)
    receipt={'schema':1,'target':target,'commit':os.environ.get('GITHUB_SHA','local'),
             'model_inference_run':False,'gui_smoke':'passed',
             'files':[{'name':p.name,'size':p.stat().st_size,'sha256':sha(p)} for p in files]}
    (output/f'SmartAI-{target}.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print('PACKAGES_VERIFIED',target,len(files),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--target',default=target_name());parser.add_argument('--output',default='release-packages')
    opts=parser.parse_args();package(opts.target,opts.output)
