"""Exercise update identity through small, real portable and Setup archives."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import bootstrap_runtime as bootstrap
import build_app
from tools import build_release_package as builder


COMMIT = 'a' * 40


class UpdatePackagingTests(unittest.TestCase):
    def test_rolling_and_version_tags_keep_full_source_identity(self):
        examples = [
            ({'GITHUB_REF': 'refs/heads/master', 'GITHUB_REF_NAME': 'master'}, 'latest'),
            ({'GITHUB_REF_TYPE': 'branch', 'GITHUB_REF_NAME': 'v-next'}, 'latest'),
            ({'GITHUB_REF_TYPE': 'tag', 'GITHUB_REF_NAME': 'v2.1.0'}, 'v2.1.0'),
            ({'GITHUB_REF': 'refs/tags/v2.2.0'}, 'v2.2.0'),
        ]
        for environment, tag in examples:
            with self.subTest(tag=tag, environment=environment), patch.dict(
                os.environ, {'GITHUB_SHA': COMMIT, **environment}, clear=True
            ):
                record = builder.release_identity()
                self.assertEqual(record['commit'], COMMIT)
                self.assertEqual(record['tag'], tag)
                self.assertEqual(record['repository'], 'Eabusham2/smart-ai-studio')

    def test_local_build_reads_git_and_malformed_commit_is_rejected(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(
            builder.subprocess, 'check_output', return_value=COMMIT + '\n'
        ) as git:
            self.assertEqual(builder.release_identity()['commit'], COMMIT)
            self.assertEqual(git.call_args.args[0],
                             ['git', '-C', str(builder.ROOT), 'rev-parse', 'HEAD'])
        with patch.dict(os.environ, {'GITHUB_SHA': 'short-sha'}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'full source commit'):
                builder.release_identity()

    def test_owned_paths_exclude_models_data_and_user_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            code = Path(directory)
            for name in ('data', 'SmartAI_Data', 'preloaded_models', 'models',
                         'cache', 'caches', '.cache', '.git', '__pycache__', 'core', 'runtime'):
                (code / name).mkdir()
            (code / '.env').write_text('PRIVATE_SETTING=1\n', encoding='utf-8')
            (code / 'app_gui.py').write_text('# application\n', encoding='utf-8')
            record = builder.write_build_record(
                code, code, 'Windows-x86_64', 'portable',
                {'schema': 1, 'repository': 'Eabusham2/smart-ai-studio',
                 'commit': COMMIT, 'tag': 'latest'})
            self.assertEqual(record['managed_paths'],
                             ['.smartai-build.json', 'app_gui.py', 'core', 'runtime'])
            self.assertEqual(record['install_managed_paths'], record['managed_paths'])
            self.assertEqual(json.loads((code / builder.BUILD_RECORD).read_text()), record)

    def _package_fixture(self, directory, target, *, prepare_only=False):
        directory = Path(directory)
        output = directory / 'release'
        smoke_kinds = []

        def code_root(app):
            if target.startswith('macOS'):
                return app / 'Contents/Resources/app'
            if target.startswith('Linux'):
                return app / 'SmartAI.AppDir/usr/bin'
            return app

        def fetch_runtime(_target, destination):
            runtime = Path(destination) / 'python'
            executable = bootstrap.python_path(runtime)
            executable.parent.mkdir(parents=True)
            executable.write_bytes(b'fixture interpreter')
            return runtime

        def create_bundle(destination, _name):
            name = ('SmartAI.app' if target.startswith('macOS') else
                    'SmartAI-Linux' if target.startswith('Linux') else 'SmartAI-Windows')
            app = Path(destination) / name
            code = code_root(app)
            code.mkdir(parents=True)
            (code / 'app_gui.py').write_text('# fixture app\n', encoding='utf-8')
            (code / 'requirements.txt').write_text('fixture==1\n', encoding='utf-8')
            (code / 'bootstrap_runtime.py').write_text('# fixture bootstrap\n', encoding='utf-8')
            (code / 'core').mkdir()
            shutil.copytree(os.environ['SMARTAI_EMBEDDED_RUNTIME'], code / 'runtime/python')
            if target.startswith('Linux'):
                (app / 'SmartAI.sh').write_text('#!/bin/sh\n', encoding='utf-8')

        def check_output(command, **_kwargs):
            return 'fixture==1\n' if command[-1] == 'freeze' else '0.3.37\n'

        def run(command, **_kwargs):
            # Native wheel-building is not part of this metadata contract.
            wheelhouse = Path(command[command.index('--wheel-dir') + 1])
            with zipfile.ZipFile(wheelhouse / 'llama_cpp_python-0.3.37-fixture.whl', 'w') as wheel:
                wheel.writestr('fixture', b'native wheel fixture')
            return subprocess.CompletedProcess(command, 0)

        def smoke(code, *, setup_home=None):
            record = json.loads((code / builder.BUILD_RECORD).read_text(encoding='utf-8'))
            smoke_kinds.append(record['package_kind'])
            self.assertEqual(record['commit'], COMMIT)
            if setup_home is not None:
                self.assertEqual(record['package_kind'], 'setup')
                self.assertFalse((code / 'runtime/python/.smartai-deps-ready').exists())
                marker = setup_home / 'runtimes/fixture/.smartai-deps-ready'
                if not marker.exists():
                    marker.parent.mkdir(parents=True)
                    marker.write_text(bootstrap.fingerprint(code) + '\n', encoding='utf-8')
            else:
                self.assertEqual(record['package_kind'], 'portable')
                self.assertTrue(bootstrap.ready(code / 'runtime/python', code))

        def dmg(app, destination):
            record = json.loads((code_root(app) / builder.BUILD_RECORD).read_text())
            self.assertEqual(record['package_kind'], 'portable')
            destination.write_bytes(b'fixture dmg')

        with patch.dict(os.environ, {'GITHUB_SHA': COMMIT, 'GITHUB_REF_TYPE': 'tag',
                                     'GITHUB_REF_NAME': 'v2.3.0'}, clear=True), \
             patch.object(builder, 'target_name', return_value=target), \
             patch.object(builder, 'fetch_runtime', side_effect=fetch_runtime), \
             patch.object(bootstrap, 'pip_install'), \
             patch.object(build_app, 'create_macos_bundle', side_effect=create_bundle), \
             patch.object(build_app, 'create_windows_bundle', side_effect=create_bundle), \
             patch.object(build_app, 'create_linux_bundle', side_effect=create_bundle), \
             patch.object(builder.subprocess, 'check_output', side_effect=check_output), \
             patch.object(builder.subprocess, 'run', side_effect=run), \
             patch.object(builder, 'smoke_app', side_effect=smoke), \
             patch.object(builder, 'verified_dmg', side_effect=dmg):
            result = builder.package(target, output, prepare_only=prepare_only)
        return output, result, smoke_kinds

    def test_each_platform_archives_its_own_portable_and_setup_identity(self):
        for target in ('macOS-arm64', 'Windows-x86_64', 'Linux-x86_64'):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                output, _, smoke_kinds = self._package_fixture(directory, target)
                self.assertEqual(smoke_kinds, ['portable', 'setup', 'setup'])
                for suffix, expected in (('portable', 'portable'), ('Setup', 'setup')):
                    with zipfile.ZipFile(output / f'SmartAI-{target}-{suffix}.zip') as archive:
                        stamp = next(name for name in archive.namelist()
                                     if name.endswith('/' + builder.BUILD_RECORD))
                        record = json.loads(archive.read(stamp))
                        self.assertEqual(record['package_kind'], expected)
                        self.assertEqual(record['target'], target)
                        self.assertEqual(record['commit'], COMMIT)
                        self.assertEqual(record['tag'], 'v2.3.0')
                        self.assertIn('requirements.lock.txt', record['managed_paths'])
                receipt = json.loads((output / f'SmartAI-{target}.receipt.json').read_text())
                self.assertEqual(receipt['commit'], COMMIT)
                if target.startswith('macOS'):
                    # This exact portable app is later reused for the model bundle.
                    with tarfile.open(output / 'SmartAI.app.tar.gz') as archive:
                        stamp = archive.extractfile('SmartAI.app/Contents/Resources/app/' + builder.BUILD_RECORD)
                        self.assertEqual(json.load(stamp)['package_kind'], 'portable')

    def test_prepare_only_returns_a_stamped_portable_app(self):
        with tempfile.TemporaryDirectory() as directory:
            output, app, smoke_kinds = self._package_fixture(
                directory, 'Windows-x86_64', prepare_only=True)
            self.assertEqual(smoke_kinds, ['portable'])
            self.assertEqual(json.loads((app / builder.BUILD_RECORD).read_text())['package_kind'], 'portable')
            self.assertEqual(list(output.iterdir()), [])

    def test_bootstrap_passes_its_actual_pid_to_the_gui(self):
        with patch.object(bootstrap, 'ensure_runtime', return_value=Path('/fixture/runtime')), \
             patch.object(bootstrap.subprocess, 'call', return_value=0) as launch, \
             patch.dict(os.environ, {'SMARTAI_BOOTSTRAP_PID': 'stale-parent'}):
            self.assertEqual(bootstrap.main(['--example']), 0)
        options = launch.call_args.kwargs
        self.assertEqual(options['env']['SMARTAI_BOOTSTRAP_PID'], str(os.getpid()))
        self.assertEqual(launch.call_args.args[0][-1], '--example')


if __name__ == '__main__':
    unittest.main()
