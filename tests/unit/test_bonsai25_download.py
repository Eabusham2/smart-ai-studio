"""Exercise preset download/cached callbacks without fetching model weights."""
from copy import deepcopy
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.parse import unquote, urlparse

import pytest

from config.bonsai_models import bonsai_model_info, MLX_MTP_FILE
from core.hf_downloader import download_model_from_hf
from core.model_import_download import download_imported_model
from tests.unit.test_gui_model_import import gui, run_worker, drain


def _fake_snapshot_api(monkeypatch, snapshot_download):
    from core import hf_transfer
    for name in ('_hub', '_original_snapshot', '_original_file'):
        monkeypatch.setattr(hf_transfer, name, None)
    # Policy installation also updates the child modules. Keep those isolated
    # so a fake transfer cannot replace the real APIs used by later tests.
    for name in ('huggingface_hub', 'huggingface_hub._snapshot_download',
                 'huggingface_hub.file_download'):
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(snapshot_download=snapshot_download))


@pytest.mark.parametrize('crack', [False, True])
@pytest.mark.parametrize('apple_silicon', [False, True])
def test_preset_download_keeps_exact_revision_files_projector_and_mtp(gui, tmp_path, monkeypatch,
                                                                  crack, apple_silicon):
    # The transfer is a local fixture, but must exercise the online API branch
    # even when the release gate sets Hub's offline switches for real services.
    for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'OFFLINE'):
        monkeypatch.delenv(name, raising=False)
    info = bonsai_model_info(crack=crack, is_apple_silicon=apple_silicon)
    before = deepcopy(info)
    gui.models_config['preset'] = info
    gui.active_tab_id = 'preset'
    # Cancelling a previous generation must not cancel an independent download.
    gui.cancel_event.set()
    gui.env['download_model_from_hf'] = download_model_from_hf
    gui.env['download_imported_model'] = Mock(wraps=download_imported_model)
    calls = []
    def snapshot_download(**kwargs):
        calls.append(kwargs)
        directory = tmp_path / kwargs['repo_id'].replace('/', '--') / kwargs['revision']
        if kwargs.get('local_files_only'):
            if not directory.is_dir():
                raise FileNotFoundError('Pinned fixture snapshot is not cached')
            return str(directory)
        for name in kwargs['allow_patterns']:
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'small offline artifact fixture')
        return str(directory)
    _fake_snapshot_api(monkeypatch, snapshot_download)
    gui._on_download_hf_model()
    download_cancel = gui._hf_download_events['preset']
    assert download_cancel is not gui.cancel_event and not download_cancel.is_set()
    gui._on_download_hf_model()
    assert len(gui.workers) == 1
    assert gui.btn_load_unload.cget('text') == 'Cancel download'
    run_worker(gui)
    assert gui._hf_download_in_progress == {'preset'}
    gui._on_download_hf_completed.assert_not_called()
    drain(gui)
    assert not gui._hf_download_events and not gui._hf_download_in_progress
    assert gui.btn_load_unload.cget('state') == 'normal'
    assert gui.btn_load_unload.cget('text') == '⚡ Load'
    gui.env['purge_local_model_cache'].assert_not_called()
    gui._on_download_hf_failed.assert_not_called()
    gui._on_download_hf_completed.assert_called_once_with(info)
    gui.env['download_imported_model'].assert_not_called()
    gui.env['make_imported_model_entry'].assert_not_called()
    gui._save_custom_models.assert_not_called()
    primary = tmp_path / info['repo_id'].replace('/', '--') / info['repo_revision']
    assert info == {**before, 'path': str(primary)} and gui.models_config['preset'] is info
    assert info['bonsai25_mtp'] is True
    transfers = [call for call in calls if not call.get('local_files_only')]
    for call in transfers:
        directory = tmp_path / call['repo_id'].replace('/', '--') / call['revision']
        assert all((directory / name).is_file() for name in call['allow_patterns'])
    if apple_silicon:
        assert len(transfers) == 1
        assert transfers[0]['repo_id'] == info['repo_id']
        assert transfers[0]['revision'] == info['repo_revision']
        assert transfers[0]['allow_patterns'] == info['download_required_files']
        assert MLX_MTP_FILE in transfers[0]['allow_patterns']
        assert 'runtime/mtp_artifact.py' in transfers[0]['allow_patterns']
        assert info['mtp_sha256'] == before['mtp_sha256']
        # A newly created preset after restart has no remembered local path.
        restarted = bonsai_model_info(crack=crack)
        assert not restarted.get('path')
        assert gui._model_is_cached(restarted) is True
        assert gui.btn_load_unload.cget('text') == '⚡ Load'
    else:
        expected = [(info['repo_id'], info['repo_revision'], [info['gguf_file']])]
        if info['mmproj_repo_id'] == info['repo_id']:
            expected[0][2].append(info['mmproj_file'])
        else:
            expected.append((info['mmproj_repo_id'], info['mmproj_repo_revision'], [info['mmproj_file']]))
        assert [(call['repo_id'], call['revision'], call['allow_patterns']) for call in transfers] == expected


def _select_preset(gui):
    info = bonsai_model_info(crack=False, is_apple_silicon=False)
    gui.models_config['preset'] = info
    gui.active_tab_id = 'preset'
    return info


@pytest.mark.parametrize('raises', [False, True])
def test_preset_download_failure_restores_controls_on_ui_thread(gui, tmp_path, raises):
    info = _select_preset(gui)
    completed_file = tmp_path / 'completed-model.gguf'
    completed_file.write_bytes(b'keep a completed cached file')

    def fail_download(_repo_id, **kwargs):
        kwargs['progress_callback']('Downloading', 25)
        if raises:
            raise RuntimeError('connection lost')
        return {'status': 'error', 'error': 'connection lost'}

    gui.env['download_model_from_hf'].side_effect = fail_download
    gui._on_download_hf_model()
    run_worker(gui)
    gui._on_download_hf_failed.assert_not_called()
    drain(gui)
    gui._on_download_hf_failed.assert_called_once_with(info, 'connection lost')
    gui._on_download_hf_completed.assert_not_called()
    assert not gui._hf_download_events and not gui._hf_download_in_progress
    assert gui.btn_load_unload.cget('state') == 'normal'
    assert gui.btn_load_unload.cget('text') == '⬇️ Install'
    assert completed_file.read_bytes() == b'keep a completed cached file'
    gui.env['purge_local_model_cache'].assert_not_called()


def test_preset_cancel_after_transfer_keeps_cache_and_does_not_report_success(gui, tmp_path):
    _select_preset(gui)
    completed_file = tmp_path / 'completed-model.gguf'
    gui.cancel_event.set()

    def finish_transfer(_repo_id, **kwargs):
        assert kwargs['cancel_event'] is gui._hf_download_events['preset']
        assert not kwargs['cancel_event'].is_set()
        completed_file.write_bytes(b'completed before cancellation')
        return {'status': 'success', 'local_dir': str(tmp_path)}

    gui.env['download_model_from_hf'].side_effect = finish_transfer
    gui._on_download_hf_model()
    download_cancel = gui._hf_download_events['preset']
    run_worker(gui)
    gui.btn_load_unload.options['command']()
    assert download_cancel.is_set()
    assert gui.btn_load_unload.cget('state') == 'disabled'
    drain(gui)
    assert not gui._hf_download_events and not gui._hf_download_in_progress
    assert gui.btn_load_unload.cget('state') == 'normal'
    assert gui.btn_load_unload.cget('text') == '⬇️ Install'
    assert gui.lbl_model_status.cget('text') == '○ Download cancelled'
    gui._on_download_hf_completed.assert_not_called()
    gui._on_download_hf_failed.assert_not_called()
    assert completed_file.read_bytes() == b'completed before cancellation'
    gui.env['purge_local_model_cache'].assert_not_called()


@pytest.mark.parametrize('change', ['switch', 'replace', 'remove'])
def test_preset_download_callbacks_do_not_touch_another_or_removed_selection(gui, change):
    info = _select_preset(gui)

    def finish_transfer(_repo_id, **kwargs):
        kwargs['progress_callback']('Downloading', 90)
        return {'status': 'success'}

    gui.env['download_model_from_hf'].side_effect = finish_transfer
    gui._on_download_hf_model()
    run_worker(gui)
    if change == 'switch':
        gui.active_tab_id = 'model_1'
    elif change == 'replace':
        gui.models_config['preset'] = dict(info, name='Replacement selection')
    else:
        del gui.models_config['preset']
        gui.active_tab_id = 'model_1'
    gui._update_model_action_buttons()
    gui.lbl_model_status.configure(text='Current selection is unchanged')
    button_state = dict(gui.btn_load_unload.options)
    drain(gui)
    assert gui.lbl_model_status.cget('text') == 'Current selection is unchanged'
    if change == 'replace':
        assert gui.btn_load_unload.cget('text') == '⬇️ Install'
        assert gui.btn_load_unload.cget('state') == 'normal'
    else:
        assert gui.btn_load_unload.options == button_state
    assert not gui._hf_download_events and not gui._hf_download_in_progress
    gui._on_download_hf_completed.assert_not_called()
    gui._on_download_hf_failed.assert_not_called()
    if change == 'remove':
        assert 'preset' not in gui.models_config


def test_unavailable_builtin_download_reports_reason_without_starting_transfer(gui):
    info = _select_preset(gui)
    info['download_unavailable_reason'] = 'This published model needs an unsupported runtime.'
    gui._on_download_hf_model()
    gui._on_download_hf_failed.assert_called_once_with(info, info['download_unavailable_reason'])
    assert not gui.workers
    gui.env['download_model_from_hf'].assert_not_called()
    gui.env['download_imported_model'].assert_not_called()
    assert not getattr(gui, '_hf_download_in_progress', set())


@pytest.mark.parametrize('crack', [False, True])
def test_preset_cache_requires_every_pinned_file_without_network(gui, tmp_path, monkeypatch, crack):
    info = bonsai_model_info(crack=crack)
    snapshot = tmp_path / info['repo_revision']
    for name in info['download_required_files']:
        path = snapshot / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'offline fixture')
    lookups = []
    def snapshot_download(**kwargs):
        lookups.append(kwargs)
        assert kwargs == {'repo_id': info['repo_id'], 'revision': info['repo_revision'],
                          'allow_patterns': info['download_allow_patterns'], 'local_files_only': True}
        return str(snapshot)
    _fake_snapshot_api(monkeypatch, snapshot_download)
    head = snapshot / MLX_MTP_FILE
    head.unlink()
    assert gui._model_is_cached(info) is False
    head.write_bytes(b'offline MTP fixture')
    assert gui._model_is_cached(info) is True
    assert len(lookups) == 2
    gui.env['is_model_cached_locally'].assert_not_called()


@pytest.fixture
def real_hub_selection(tmp_path, monkeypatch):
    """Use Hub's actual filtered snapshot/tree/blob cache; fake only HTTP I/O."""
    import huggingface_hub as hub
    from huggingface_hub import constants, file_download
    from huggingface_hub.hf_api import RepoFile
    from core import hf_transfer

    # Earlier transfer-fault tests replace Hub modules. Restore the real
    # implementation rather than inheriting a fake in the global policy aliases;
    # register every alias the policy mutates so fixture teardown restores it.
    snapshots = importlib.import_module('huggingface_hub._snapshot_download')
    for module, names in ((hub, ('hf_hub_download', 'snapshot_download')),
                          (file_download, ('hf_hub_download',)),
                          (snapshots, ('snapshot_download', 'hf_hub_download'))):
        for name in names:
            monkeypatch.setattr(module, name, getattr(module, name))
    for name in ('_hub', '_original_snapshot', '_original_file'):
        monkeypatch.setattr(hf_transfer, name, None)
    file_download = importlib.reload(file_download)
    snapshots = importlib.reload(snapshots)
    hub.hf_hub_download = file_download.hf_hub_download
    hub.snapshot_download = snapshots.snapshot_download

    cache = tmp_path / 'actual hub cache'
    monkeypatch.setattr(constants, 'HF_HUB_CACHE', str(cache))
    monkeypatch.setenv('HF_HUB_CACHE', str(cache))  # Fresh Eval child sees the same cache.
    for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'OFFLINE'):
        monkeypatch.delenv(name, raising=False)
    downloaded = []

    def prepare(info):
        files = {name: b'{}' if name.endswith('.json') else ('tiny fixture ' + name).encode()
                 for name in info['download_required_files']}
        files['README.md'] = b'This optional model card is intentionally not downloaded.'
        rows = [RepoFile(path=name, size=len(data), oid=hashlib.sha1(data).hexdigest())
                for name, data in files.items()]
        monkeypatch.setattr(hub.HfApi, 'list_repo_tree', lambda *args, **kwargs: iter(rows))
        monkeypatch.setattr(hub.HfApi, 'repo_info', lambda *args, **kwargs:
                            SimpleNamespace(sha=info['repo_revision'], siblings=rows))

        def filename(url):
            return unquote(urlparse(url).path.split('/resolve/' + info['repo_revision'] + '/', 1)[1])

        def metadata(url, **kwargs):
            data = files[filename(url)]
            return file_download.HfFileMetadata(info['repo_revision'], hashlib.sha1(data).hexdigest(),
                                                url, len(data), None)

        def transfer(url, destination, **kwargs):
            name = filename(url)
            downloaded.append(name)
            destination.write(files[name])

        monkeypatch.setattr(file_download, 'get_hf_file_metadata', metadata)
        monkeypatch.setattr(file_download, 'http_get', transfer)
        return files

    return SimpleNamespace(hub=hub, cache=cache, prepare=prepare, downloaded=downloaded)


def _extended_download_gui(gui, monkeypatch):
    # The old fixture extracted methods from the AST, skipping installed GUI
    # extensions. Keep its strict widgets/worker queue, but use the real class.
    import app_gui
    app = app_gui.SmartAIChatbotApp.__new__(app_gui.SmartAIChatbotApp)
    app.__dict__.update(gui.__dict__)
    for name in ('_on_download_hf_completed', '_on_download_hf_failed'):
        app.__dict__.pop(name)
        setattr(app, name, Mock(wraps=getattr(app, name)))
    monkeypatch.setattr(app_gui, 'threading', gui.env['threading'])
    return app


_EVAL_CACHE_CHILD = r'''
import json, sys
from pathlib import Path
from types import SimpleNamespace
from core import cross_platform_mlx, pro_engine
from eval.app_cross_platform_bridge import AppEvalEngineAdapter

config = json.loads(Path(sys.argv[1]).read_text())
expected = Path(sys.argv[2])
allocated = []
class FixtureMLX:
    def __init__(self, model_path, adapter_path=None, model_info=None):
        self.model_path, self.adapter_path = model_path, adapter_path
        self.model_info = model_info or {}
        self.model = self.tokenizer = None
        self.is_mlx_available = False
    def load_model(self):
        assert Path(self.model_path) == expected, self.model_path
        assert all((expected / name).stat().st_size > 0 for name in self.model_info['download_required_files'])
        allocated.append(self.model_path)
        self.model, self.tokenizer = object(), SimpleNamespace()
        self.is_mlx_available = True
        return True
    def unload_model(self):
        self.model = self.tokenizer = None
        self.is_mlx_available = False

pro_engine.MLXReasoningBackend = FixtureMLX
pro_engine.platform = SimpleNamespace(system=lambda: 'Darwin', machine=lambda: 'arm64')
cross_platform_mlx.platform = pro_engine.platform
engine = AppEvalEngineAdapter(config, str(Path(sys.argv[1]).parent))
assert engine.pro.settings.auto_download and not engine.pro.settings.use_mock
assert allocated == [str(expected)], allocated
assert engine.pro.active_model_path == engine.backend.model_path == engine.settings.mlx_model_path == str(expected)
print('EVAL_SELECTED_CACHE_PASS', json.dumps({'path': allocated[0], 'revision': engine.model_info['repo_revision']}))
engine.unload_model()
'''


@pytest.mark.parametrize('crack', [False, True])
def test_real_hub_download_restart_and_eval_share_complete_pinned_selection(
        gui, real_hub_selection, tmp_path, monkeypatch, crack):
    import app_gui
    from core import gui_eval_panel

    app = _extended_download_gui(gui, monkeypatch)
    info = bonsai_model_info(crack=crack, is_apple_silicon=True)
    original = deepcopy(info)
    files = real_hub_selection.prepare(info)
    app.models_config['preset'], app.active_tab_id = info, 'preset'
    app._update_model_action_buttons()
    assert app.btn_load_unload.cget('text') == '⬇️ Install'
    app.btn_load_unload.invoke()
    run_worker(app)
    drain(app)

    assert not app._on_download_hf_failed.called, app._on_download_hf_failed.call_args
    snapshot = Path(info['path'])
    assert app.btn_load_unload.cget('text') == '⚡ Load'
    assert snapshot.name == info['repo_revision']
    assert info == {**original, 'path': str(snapshot)}
    assert all((snapshot / name).read_bytes() == files[name] for name in info['download_required_files'])
    assert not (snapshot / 'README.md').exists()
    assert not (snapshot.parent.parent / 'refs/main').exists()
    before = {name: ((snapshot / name).read_bytes(), (snapshot / name).stat().st_mtime_ns)
              for name in info['download_required_files']}
    transfers = list(real_hub_selection.downloaded)

    # Recreated built-ins have no remembered path. Only the exact selected
    # pinned files are required, not README/other repo formats or refs/main.
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    monkeypatch.setenv('TRANSFORMERS_OFFLINE', '1')
    restarted = bonsai_model_info(crack=crack, is_apple_silicon=True)
    app.models_config['preset'] = restarted
    assert not restarted.get('path') and app._model_is_cached(restarted)
    app._update_model_action_buttons()
    assert app.btn_load_unload.cget('text') == '⚡ Load'

    # Exercise the installed Eval extension's real config writer, then load
    # that config in a fresh process with only heavyweight allocation replaced.
    app._eval_proc = None
    app._eval_output_queue = queue.Queue()
    app._eval_mem_enabled_var = SimpleNamespace(get=lambda: False)
    app._eval_mem_gb_var = SimpleNamespace(get=lambda: '16')
    app._eval_deepswe_var = SimpleNamespace(get=lambda: False)
    monkeypatch.setattr(app_gui, 'get_portable_data_dir', lambda: str(tmp_path / 'desktop data'))
    launch = []
    def capture(command, **kwargs):
        launch.append(command)
        return SimpleNamespace(pid=999999999, poll=lambda: None)
    monkeypatch.setattr(gui_eval_panel, 'subprocess', SimpleNamespace(
        Popen=capture, PIPE=subprocess.PIPE, STDOUT=subprocess.STDOUT, DEVNULL=subprocess.DEVNULL))
    monkeypatch.setattr(gui_eval_panel, 'threading', gui.env['threading'])
    start_eval = inspect.getclosurevars(inspect.unwrap(app_gui.SmartAIChatbotApp._open_eval_window)).nonlocals['_start_eval']
    start_eval(app)
    assert launch[0][1:4] == ['-u', '-m', 'eval.app_eval_runner']
    config_path = Path(launch[0][-1])
    config = json.loads(config_path.read_text())
    assert config['model_info'] == restarted
    child = subprocess.run([sys.executable, '-c', _EVAL_CACHE_CHILD, str(config_path), str(snapshot)],
                           cwd=Path(app_gui.__file__).parent, env=dict(os.environ),
                           capture_output=True, text=True, timeout=30)
    assert child.returncode == 0, child.stdout + child.stderr
    assert 'EVAL_SELECTED_CACHE_PASS' in child.stdout
    assert real_hub_selection.downloaded == transfers
    assert before == {name: ((snapshot / name).read_bytes(), (snapshot / name).stat().st_mtime_ns)
                      for name in info['download_required_files']}


@pytest.mark.parametrize('missing', [True, False])
def test_successful_transfer_with_missing_or_empty_mtp_cannot_claim_download_complete(gui, tmp_path, missing):
    info = bonsai_model_info(is_apple_silicon=True)
    snapshot = tmp_path / info['repo_revision']
    for name in info['download_required_files']:
        path = snapshot / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'completed fixture')
    head = snapshot / MLX_MTP_FILE
    head.unlink() if missing else head.write_bytes(b'')
    gui.models_config['preset'], gui.active_tab_id = info, 'preset'
    gui.env['download_model_from_hf'].return_value = {'status': 'success', 'local_dir': str(snapshot)}
    gui._on_download_hf_model()
    run_worker(gui)
    drain(gui)
    assert gui.btn_load_unload.cget('text') == '⬇️ Install'
    gui._on_download_hf_failed.assert_called_once()
    assert not any('Download Complete' in call.args[0] for call in gui._append_ai_message.call_args_list)
    assert (snapshot / info['download_required_files'][0]).read_bytes() == b'completed fixture'
