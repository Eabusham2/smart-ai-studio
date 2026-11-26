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


def test_split_download_keeps_only_parts_and_verified_manifest(tmp_path):
    archive=tmp_path/'SmartAI-Test-portable.zip';original=b'abcdefghijk'*20;archive.write_bytes(original)
    files=builder.split_large(archive,limit=50)
    manifest=json.loads((tmp_path/'SmartAI-Test-portable.zip.parts.json').read_text())
    assert b''.join((tmp_path/r['name']).read_bytes() for r in manifest['parts'])==original
    assert len(files) == len(manifest['parts']) + 1
    assert all(p.name.endswith('.parts.json') or '.part' in p.name for p in files)
    assert manifest['sha256'] == hashlib.sha256(original).hexdigest()
    assert not archive.exists()


def test_receipts_reject_wrong_commit_missing_file_and_corruption(tmp_path):
    for target in ('Linux-x86_64','Windows-x86_64','macOS-arm64'):
        files=[]
        for suffix in ('portable.zip','Setup.zip'):
            p=tmp_path/f'SmartAI-{target}-{suffix}';p.write_bytes(b'package')
            files.append({'name':p.name,'size':7,'sha256':hashlib.sha256(b'package').hexdigest()})
        (tmp_path/f'SmartAI-{target}.receipt.json').write_text(json.dumps({
            'schema':1,'target':target,'commit':'a'*40,'gui_smoke':'passed','setup_smoke':'passed',
            'bonsai_native_smoke':'passed',
            'bundled_native_verified':'not_applicable' if target.startswith('Windows') else 'passed',
            'files':files}))
    verifier.verify(tmp_path,'a'*40)
    receipt_path = tmp_path / 'SmartAI-Windows-x86_64.receipt.json'
    receipt = json.loads(receipt_path.read_text())
    receipt.pop('bonsai_native_smoke')
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError, match='Bonsai MTP runtime launch'):
        verifier.verify(tmp_path, 'a'*40)
    receipt['bonsai_native_smoke'] = 'passed'
    receipt_path.write_text(json.dumps(receipt))
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



def test_shipped_setup_lock_does_not_reference_the_ci_wheel_directory():
    original = 'torch==2.2.0\nllama_cpp_python @ file:///Users/runner/build/wheelhouse/native.whl#sha256=abc\ntransformers==4.40.0\n'
    result = builder.portable_dependency_lock(original, '0.3.37')
    assert result == 'torch==2.2.0\nllama-cpp-python==0.3.37\ntransformers==4.40.0\n'
    assert 'file:' not in result


def test_setup_smoke_uses_a_private_clean_home_and_ignores_host_python(tmp_path, monkeypatch):
    monkeypatch.setenv('PYTHONHOME', '/wrong-host-python')
    monkeypatch.setenv('PYTHONPATH', '/wrong-host-packages')
    calls = []
    monkeypatch.setattr(builder.subprocess, 'run', lambda cmd, **kwargs:calls.append((cmd,kwargs)))
    builder.smoke_app(tmp_path / 'code', setup_home=tmp_path / 'clean setup',
                      smoke_data_dir=tmp_path / 'saved preferences', launch=2)
    command, options = calls[0]
    assert command[-1] == '--smoke-test'
    assert str(tmp_path / 'code/runtime/python') in command[0]
    assert options['check'] is True
    assert options['env']['SMARTAI_SETUP_HOME'] == str(tmp_path / 'clean setup')
    assert options['env']['SMARTAI_SMOKE_DATA_DIR'] == str(tmp_path / 'saved preferences')
    assert options['env']['SMARTAI_SMOKE_LAUNCH'] == '2'
    assert options['env']['RUN_REAL_WEIGHTS_EVAL'] == '0'
    assert 'PYTHONHOME' not in options['env'] and 'PYTHONPATH' not in options['env']


def test_bonsai_image_probe_uses_relocated_app_python_and_an_unshipped_private_cache(tmp_path, monkeypatch):
    monkeypatch.setenv('PYTHONHOME', '/wrong-host-python')
    monkeypatch.setenv('PYTHONPATH', '/wrong-host-packages')
    calls = []
    monkeypatch.setattr(builder.subprocess, 'run', lambda cmd, **kwargs: calls.append((cmd, kwargs)))
    code = tmp_path / 'relocated with spaces' / 'app'
    cache = tmp_path / 'separate probe cache'
    builder.smoke_bonsai_image_runtime(code, cache)
    command, options = calls[0]
    assert str(code / 'runtime/python') in command[0]
    assert options['cwd'] == code
    assert options['env']['SMARTAI_DATA_DIR'] == str(cache)
    assert options['env']['HF_HUB_OFFLINE'] == options['env']['TRANSFORMERS_OFFLINE'] == '1'
    assert 'PYTHONHOME' not in options['env'] and 'PYTHONPATH' not in options['env']
    assert 'force_probe=True' in command[-1]
    assert 'no model weights' in command[-1]


def test_mac_smoke_checks_native_entry_and_direct_bootstrap_relaunch(tmp_path, monkeypatch):
    code = tmp_path / 'Renamed app.app/Contents/Resources/app'
    native = code.parent.parent / 'MacOS/applet'
    monkeypatch.setattr(bootstrap, 'macos_native_launcher', lambda root: native)
    calls = []
    monkeypatch.setattr(builder.subprocess, 'run', lambda cmd, **kwargs: calls.append((cmd, kwargs)))
    data = tmp_path / 'unshipped saved preferences'
    builder.smoke_app(code, smoke_data_dir=data)
    builder.smoke_app(code, smoke_data_dir=data, launch=2)
    assert calls[0][0] == [str(native), '--smoke-test']
    assert calls[1][0] == [str(bootstrap.python_path(code / 'runtime/python')),
                            str(code / 'bootstrap_runtime.py'), '--smoke-test']
    assert calls[0][1]['env']['SMARTAI_SMOKE_DATA_DIR'] == calls[1][1]['env']['SMARTAI_SMOKE_DATA_DIR']


@pytest.mark.parametrize('missing', ['setup_smoke', 'bundled_native_verified'])
def test_release_refuses_package_without_format_specific_launch_proof(tmp_path, missing):
    for target in ('Linux-x86_64','Windows-x86_64','macOS-arm64'):
        rows = []
        for suffix in ('portable.zip', 'Setup.zip'):
            path = tmp_path / f'SmartAI-{target}-{suffix}'
            path.write_bytes(b'package')
            rows.append({'name':path.name,'size':7,'sha256':hashlib.sha256(b'package').hexdigest()})
        receipt={'schema':1,'target':target,'commit':'a'*40,'gui_smoke':'passed',
                 'setup_smoke':'passed','bonsai_native_smoke':'passed',
                 'bundled_native_verified':'not_applicable' if target.startswith('Windows') else 'passed','files':rows}
        if target == 'macOS-arm64':
            receipt.pop(missing)
        (tmp_path/f'SmartAI-{target}.receipt.json').write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError, match='was not verified'):
        verifier.verify(tmp_path,'a'*40)



def test_setup_home_override_controls_actual_installation_and_reuse(tmp_path, monkeypatch):
    app = tmp_path / 'app'; app.mkdir()
    (app / 'requirements.txt').write_text('test==1\n', encoding='utf-8')
    base = app / 'runtime/python'
    executable = bootstrap.python_path(base)
    executable.parent.mkdir(parents=True); executable.write_bytes(b'fake-python')
    isolated = tmp_path / 'clean setup home with spaces'
    monkeypatch.setenv('SMARTAI_SETUP_HOME', str(isolated))
    calls = []
    def install(runtime, root):
        calls.append(runtime)
        assert runtime.is_relative_to(isolated)
        (runtime / '.smartai-deps-ready').write_text(bootstrap.fingerprint(root)+'\n', encoding='utf-8')
    monkeypatch.setattr(bootstrap, 'pip_install', install)
    assert bootstrap.private_root() == isolated.resolve()
    installed = bootstrap.ensure_runtime(app)
    assert installed.is_relative_to(isolated)
    assert bootstrap.ensure_runtime(app) == installed
    assert len(calls) == 1
    assert not (base / '.smartai-deps-ready').exists()


def test_native_package_gate_checks_actual_private_installation():
    source = (ROOT / 'tools/build_release_package.py').read_text(encoding='utf-8')
    assert "setup_home.glob('runtimes/*/.smartai-deps-ready')" in source
    assert "Setup did not initialize its isolated private runtime" in source
    assert "Second Setup launch unexpectedly reinstalled dependencies" in source
