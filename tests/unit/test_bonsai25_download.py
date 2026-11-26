"""Exercise preset download/cached callbacks without fetching model weights."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from config.bonsai_models import bonsai_model_info, MLX_MTP_FILE
from core.hf_downloader import download_model_from_hf
from core.model_import_download import download_imported_model
from tests.unit.test_gui_model_import import gui, run_worker, drain


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
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(snapshot_download=snapshot_download))
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
    assert info == before and gui.models_config['preset'] is info
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
        assert kwargs == {'repo_id': info['repo_id'], 'revision': info['repo_revision'], 'local_files_only': True}
        return str(snapshot)
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(snapshot_download=snapshot_download))
    head = snapshot / MLX_MTP_FILE
    head.unlink()
    assert gui._model_is_cached(info) is False
    head.write_bytes(b'offline MTP fixture')
    assert gui._model_is_cached(info) is True
    assert len(lookups) == 2
    gui.env['is_model_cached_locally'].assert_not_called()
