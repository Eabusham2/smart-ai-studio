"""No-admin macOS portable release kit and synthetic end-to-end local assembly."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys
import zipfile

import pytest

from tools.build_macos_noadmin_portable import (
    ARCHIVE, ASSEMBLER, KIT, LAUNCHER, MANIFEST, REPOS, build, verify,
    verify_manifest,
)


def _sample_bundle(tmp_path):
    packed = tmp_path / ARCHIVE
    with zipfile.ZipFile(packed, 'w') as archive:
        root = 'SmartAI.app/Contents/Resources/app/'
        runtime_bin = zipfile.ZipInfo(root + 'runtime/python/bin/python3')
        runtime_bin.create_system = 3
        runtime_bin.external_attr = (stat.S_IFREG | 0o755) << 16
        archive.writestr(runtime_bin, '#!/bin/sh\nexit 0\n')
        archive.writestr(root + 'runtime/python/.smartai-deps-ready', 'synthetic-test\n')
        for repo in sorted(REPOS):
            folder = root + 'preloaded_models/' + repo.replace('/', '--') + '/'
            archive.writestr(folder + 'config.json', '{"model_type":"test_only"}')
            archive.writestr(folder + 'model.safetensors', b'test-bytes-not-a-real-model')
    payload = packed.read_bytes()
    part_bytes = max(100, len(payload) // 2)
    rows = []
    for i, index in enumerate(range(0, len(payload), part_bytes)):
        body = payload[index:index+part_bytes]
        name = f'{ARCHIVE}.part{i:03d}'
        (tmp_path / name).write_bytes(body)
        rows.append({'name':name,'size':len(body),'sha256':hashlib.sha256(body).hexdigest()})
    models = [{'repo_id':repo,'revision':'1'*40,'files':2,'expected_bytes':23} for repo in sorted(REPOS)]
    data = {'format':1, 'archive_name':ARCHIVE,'archive_size':len(payload),
            'archive_sha256':hashlib.sha256(payload).hexdigest(),
            'parts':rows,'models':models}
    (tmp_path / MANIFEST).write_text(json.dumps(data),encoding='utf-8')
    (tmp_path / ASSEMBLER).write_text('#!/bin/bash\n/usr/bin/shasum -a 256\n',encoding='utf-8')
    # The synthetic app archive is retained here for integrity proof, but the
    # no-admin test starts from split parts alone.
    packed.unlink()
    return data


def _kit(tmp_path):
    data = _sample_bundle(tmp_path)
    dest = build(tmp_path / MANIFEST, tmp_path / ASSEMBLER, tmp_path,
                 repo='Eabusham2/smart-ai-studio',tag='latest',commit='2'*40)
    return dest, data


def test_no_admin_kit_contains_two_models_manifest_and_executable_scripts(tmp_path):
    path, data = _kit(tmp_path)
    assert path.name == KIT
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert (tmp_path / (KIT + '.sha256')).read_text().strip() == f'{actual}  {KIT}'
    assert verify(path,expected=data,repo='Eabusham2/smart-ai-studio',tag='latest',commit='2'*40) == data
    with zipfile.ZipFile(path) as archive:
        launcher = archive.read(LAUNCHER).decode()
        readme = archive.read('README-NoAdmin-Portable.txt').decode()
        assert 'SmartAI-Portable' in launcher
        assert 'HOME' in launcher and 'NoAdmin' in launcher
        assert 'both' in readme.lower() and 'Qwen3.8' in readme
        assert 'system' in readme.lower() and 'python' in readme.lower()
        assert archive.getinfo(LAUNCHER).external_attr >> 16 & stat.S_IXUSR
        assert 'sudo ' not in launcher and '/usr/bin/sudo' not in launcher
        assert 'curl' in launcher and '/usr/bin/shasum' in launcher
        assert 'SMARTAI_PORTABLE_OFFLINE' in launcher
        script = tmp_path / LAUNCHER
        script.write_text(launcher, encoding='utf-8')
        subprocess.run(['/bin/bash', '-n', str(script)], check=True)


def test_rejects_missing_model_or_bad_part_hash(tmp_path):
    _sample_bundle(tmp_path)
    data=json.loads((tmp_path/MANIFEST).read_text())
    data['models']=data['models'][:1]
    with pytest.raises(ValueError,match='both original Bonsai'):
        verify_manifest(data)
    data=json.loads((tmp_path/MANIFEST).read_text())
    data['parts'][0]['sha256']='invalid'
    with pytest.raises(ValueError,match='Missing model part hash'):
        verify_manifest(data)
    data=json.loads((tmp_path/MANIFEST).read_text())
    data['parts'][0]['name']='../../evil'
    with pytest.raises(ValueError,match='ordered'):
        verify_manifest(data)


@pytest.mark.skipif(sys.platform != 'darwin' or platform.machine().lower() != 'arm64',
                    reason='offline portable assembly uses macOS built-in JXA, ditto and shasum')
def test_full_portable_install_succeeds_offline_without_admin_or_python(tmp_path):
    # Use the actual tested project's reassembly script: no downloads, no model
    # inference, no root privileges and no system Python. Tiny fake weights only.
    from build_bonsai2_bundled_zip import _write_reassembler
    path, data = _kit(tmp_path)
    _write_reassembler(tmp_path, ARCHIVE, MANIFEST)
    kit = build(tmp_path/MANIFEST, tmp_path/ASSEMBLER, tmp_path,
                repo='Eabusham2/smart-ai-studio',tag='latest',commit='2'*40)
    extract=tmp_path/'kit'
    with zipfile.ZipFile(kit) as source:
        source.extractall(extract)
    script=extract/LAUNCHER
    script.chmod(0o755)
    cache=tmp_path/'cache'
    cache.mkdir()
    for row in data['parts']:
        shutil.copy2(tmp_path/row['name'],cache/row['name'])
    env=dict(os.environ, SMARTAI_PORTABLE_CACHE=str(cache),
             SMARTAI_PORTABLE_DEST=str(tmp_path/'installed with spaces'),
             SMARTAI_PORTABLE_OFFLINE='1',SMARTAI_PORTABLE_INSTALL_ONLY='1',
             PATH='/usr/bin:/bin')
    # Deny even accidental usage of a globally provisioned Python; the script
    # requires only stock system executables and its bundled app runtime.
    result=subprocess.run(['/bin/bash',str(script)],env=env,capture_output=True,
                          text=True,timeout=90)
    assert result.returncode == 0, (result.stdout,result.stderr)
    app=tmp_path/'installed with spaces'/'2'*12/'SmartAI.app'/'Contents'/'Resources'/'app'
    assert (app/'runtime/python/bin/python3').exists()
    for repo in REPOS:
        base=app/'preloaded_models'/repo.replace('/','--')
        assert (base/'config.json').is_file() and (base/'model.safetensors').is_file()
    # Second run reuses the already extracted model package even when the
    # downloaded parts are gone. This is genuinely offline after installation.
    for part in cache.glob('*.part*'):
        part.unlink()
    again=subprocess.run(['/bin/bash',str(script)],env=env,capture_output=True,
                         text=True,timeout=30)
    assert again.returncode == 0, (again.stdout,again.stderr)
