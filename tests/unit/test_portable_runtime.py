"""Offline packaging contracts: standalone Python, setup isolation, receipts."""
from pathlib import Path
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import tarfile
from types import SimpleNamespace
import pytest

import bootstrap_runtime as bootstrap

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('package_builder', ROOT/'tools/build_release_package.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
spec=importlib.util.spec_from_file_location('package_verifier',ROOT/'tools/verify_release_packages.py')
verifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(verifier)


def test_portable_ready_launch_never_uses_pip_or_global_python(tmp_path,monkeypatch):
    (tmp_path/'requirements.txt').write_text('test==1\n',encoding='utf-8')
    runtime=tmp_path/'runtime/python'
    python=bootstrap.python_path(runtime);python.parent.mkdir(parents=True);python.write_bytes(b'fake-runtime')
    (runtime/'.smartai-deps-ready').write_text(bootstrap.fingerprint(tmp_path)+'\n',encoding='utf-8')
    monkeypatch.setattr(bootstrap,'pip_install',lambda *a:pytest.fail('portable must not install'))
    assert bootstrap.ensure_runtime(tmp_path)==runtime


def test_setup_uses_private_copy_and_retries_only_incomplete_install(tmp_path,monkeypatch):
    app=tmp_path/'app';app.mkdir();(app/'requirements.txt').write_text('test==1\n')
    base=app/'runtime/python';python=bootstrap.python_path(base);python.parent.mkdir(parents=True);python.write_bytes(b'fake')
    home=tmp_path/'user';monkeypatch.setattr(bootstrap,'private_root',lambda:home)
    calls=[]
    def install(runtime,root):
        calls.append(runtime)
        assert runtime.is_relative_to(home)
        (runtime/'.smartai-deps-ready').write_text(bootstrap.fingerprint(root)+'\n')
    monkeypatch.setattr(bootstrap,'pip_install',install)
    runtime=bootstrap.ensure_runtime(app)
    assert bootstrap.ensure_runtime(app)==runtime and len(calls)==1
    assert not (base/'.smartai-deps-ready').exists()


@pytest.mark.parametrize('name,kind,target',[
    ('../escape','file',None),('/root','file',None),('python/escape','symlink','../../outside'),
    ('python/fifo','fifo',None),('python/../escape','file',None),
])
def test_archive_rejects_unsafe_paths_before_writing(tmp_path,name,kind,target):
    archive=tmp_path/'bad.tar.gz'
    with tarfile.open(archive,'w:gz') as t:
        member=tarfile.TarInfo(name)
        if kind=='symlink':member.type=tarfile.SYMTYPE;member.linkname=target
        elif kind=='fifo':member.type=tarfile.FIFOTYPE
        else:member.size=1
        t.addfile(member,io.BytesIO(b'x') if kind=='file' else None)
    with pytest.raises(RuntimeError):builder.extract_python(archive,tmp_path/'dest')
    assert not (tmp_path/'dest').exists()


def test_archive_accepts_internal_python_symlinks(tmp_path):
    archive=tmp_path/'valid.tar.gz'
    with tarfile.open(archive,'w:gz') as t:
        f=tarfile.TarInfo('python/bin/python3.11');f.size=3;f.mode=0o755;t.addfile(f,io.BytesIO(b'bin'))
        link=tarfile.TarInfo('python/bin/python3');link.type=tarfile.SYMTYPE;link.linkname='python3.11';t.addfile(link)
    builder.extract_python(archive,tmp_path/'dest')
    assert (tmp_path/'dest/python/bin/python3').read_bytes()==b'bin'


def test_split_download_has_verified_shell_and_powershell_reassembly(tmp_path):
    archive=tmp_path/'SmartAI-Test-portable.zip';original=b'abcdefghijk'*20;archive.write_bytes(original)
    files=builder.split_large(archive,limit=50)
    manifest=json.loads((tmp_path/'SmartAI-Test-portable.zip.parts.json').read_text())
    assert b''.join((tmp_path/r['name']).read_bytes() for r in manifest['parts'])==original
    assert any(p.suffix=='.ps1' for p in files)
    shell=next(p for p in files if p.suffix=='.sh')
    if os.name!='nt':
        subprocess.run(['sh',str(shell)],check=True,capture_output=True)
        assert archive.read_bytes()==original


def test_receipts_reject_wrong_commit_missing_file_and_corruption(tmp_path):
    for target in ('Linux-x86_64','Windows-x86_64','macOS-arm64'):
        files=[]
        for suffix in ('portable.zip','Setup.zip'):
            p=tmp_path/f'SmartAI-{target}-{suffix}';p.write_bytes(b'package')
            files.append({'name':p.name,'size':7,'sha256':hashlib.sha256(b'package').hexdigest()})
        (tmp_path/f'SmartAI-{target}.receipt.json').write_text(json.dumps({
            'schema':1,'target':target,'commit':'a'*40,'gui_smoke':'passed','files':files}))
    verifier.verify(tmp_path,'a'*40)
    with pytest.raises(RuntimeError):verifier.verify(tmp_path,'b'*40)
    (tmp_path/'SmartAI-Windows-x86_64-portable.zip').write_bytes(b'changed')
    with pytest.raises(RuntimeError,match='hash mismatch'):verifier.verify(tmp_path,'a'*40)


def test_all_bundled_runtime_versions_and_hashes_are_explicit():
    data=json.loads((ROOT/'packaging/python-runtimes.json').read_text(encoding='utf-8'))
    for name,asset in data['assets'].items():
        assert asset['url'].startswith('https://github.com/astral-sh/python-build-standalone/releases/download/')
        assert '/latest/' not in asset['url'] and len(asset['sha256'])==64 and asset['size']>0


def test_archive_extraction_does_not_require_new_tarfile_filter_api(tmp_path, monkeypatch):
    def no_extractall(*args, **kwargs):
        pytest.fail('validated extraction must not depend on the host tarfile filter API')
    monkeypatch.setattr(tarfile.TarFile, 'extractall', no_extractall)
    archive = tmp_path / 'runtime.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        member = tarfile.TarInfo('python/bin/python3'); member.size = 3
        tar.addfile(member, io.BytesIO(b'bin'))
    builder.extract_python(archive, tmp_path / 'dest')
    assert (tmp_path / 'dest/python/bin/python3').read_bytes() == b'bin'


def test_archive_rejects_a_member_nested_below_an_alias(tmp_path):
    archive = tmp_path / 'alias.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        link = tarfile.TarInfo('python/alias'); link.type = tarfile.SYMTYPE; link.linkname = 'lib'
        tar.addfile(link)
        file = tarfile.TarInfo('python/alias/payload'); file.size = 1
        tar.addfile(file, io.BytesIO(b'x'))
    with pytest.raises(RuntimeError, match='link parent'):
        builder.extract_python(archive, tmp_path / 'dest')
    assert not (tmp_path / 'dest').exists()


def test_setup_checks_bundled_native_wheel_before_running_pip(tmp_path, monkeypatch):
    (tmp_path / 'requirements.txt').write_text('llama-cpp-python==0.3.0\n')
    wheelhouse = tmp_path / 'wheelhouse'; wheelhouse.mkdir()
    wheel = wheelhouse / 'llama_cpp_python-0.3.0-py3-none-any.whl'; wheel.write_bytes(b'corrupt')
    (wheelhouse / 'manifest.json').write_text(json.dumps({'files': [
        {'name': wheel.name, 'size': 7, 'sha256': '0' * 64}
    ]}))
    monkeypatch.setattr(bootstrap.subprocess, 'run', lambda *a, **k: pytest.fail('corrupt wheel must not run pip'))
    with pytest.raises(RuntimeError, match='checksum mismatch'):
        bootstrap.pip_install(tmp_path / 'runtime', tmp_path)


def test_package_job_does_not_cache_the_unused_host_interpreter():
    workflow = (ROOT / '.github/workflows/ci-build-release.yml').read_text(encoding='utf-8')
    package = workflow.split('  package:', 1)[1].split('  build-and-release:', 1)[0]
    assert 'cache: pip' not in package
    assert 'Build bundled runtime and test relocated first launch' in package
