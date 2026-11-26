"""Exercise real media routing, component resolution and codecs without model weights."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def media(tmp_path, monkeypatch):
    # Import the small backend without booting the unrelated text model package.
    core = ModuleType('core')
    core.__path__ = [str(ROOT / 'core')]
    monkeypatch.setitem(sys.modules, 'core', core)
    spec = importlib.util.spec_from_file_location('core.media_engine', ROOT / 'core/media_engine.py')
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    from config.media_models import media_download_metadata

    downloads, loads, calls, vae_loads = [], [], [], []
    state = SimpleNamespace(fail=None, invalid=False, video=False)
    components = {'scheduler': ['diffusers', 'Scheduler'], 'text_encoder': ['transformers', 'TextEncoder'],
                  'tokenizer': ['transformers', 'Tokenizer'], 'transformer': ['diffusers', 'Transformer'],
                  'vae': ['diffusers', 'AutoencoderKLWan']}

    def snapshot_download(**kwargs):
        downloads.append(kwargs)
        directory = tmp_path / kwargs['repo_id'].replace('/', '--') / kwargs['revision']
        files = kwargs['allow_patterns']
        for name in files:
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'offline component fixture')
        if 'model_index.json' in files:
            class_name = 'WanPipeline' if 'Wan' in kwargs['repo_id'] else 'Flux2KleinPipeline'
            manifest = dict(components, _class_name=class_name)
            if 'RealVis' in kwargs['repo_id']:
                manifest.update(_class_name='StableDiffusionXLPipeline', unet=['diffusers', 'UNet'],
                                text_encoder_2=['transformers', 'TextEncoder'], tokenizer_2=['transformers', 'Tokenizer'])
                manifest.pop('transformer')
            if kwargs['repo_id'] == 'fixture/packed':
                manifest = {'_class_name': 'Flux2KleinPipeline'}
            (directory / 'model_index.json').write_text(json.dumps(manifest))
        return str(directory)

    transfer = SimpleNamespace(snapshot_download=snapshot_download, hf_hub_download=Mock(),
                               run_hf_operation=lambda operation, **_: operation(),
                               hf_subprocess_command=lambda command: command)
    monkeypatch.setitem(sys.modules, 'core.hf_transfer', transfer)
    monkeypatch.setitem(sys.modules, 'core.media_learning', SimpleNamespace(
        apply_saved_media_adapter=Mock(), get_saved_media_adapter=lambda _: None))
    torch = SimpleNamespace(float16='fp16', float32='fp32', bfloat16='bf16',
                            Generator=lambda device: SimpleNamespace(manual_seed=lambda seed: ('seed', device, seed)),
                            cuda=SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None),
                            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setitem(sys.modules, 'torch', torch)

    pixels = np.zeros((16, 16, 3), dtype=np.uint8)
    pixels[:, :8, 0], pixels[:, 8:, 1] = 220, 180

    class Pipe:
        def to(self, device):
            self.device = device

        def enable_model_cpu_offload(self):
            self.device = 'cuda-offload'

        def __call__(self, prompt, num_inference_steps, callback_on_step_end=None, **kwargs):
            calls.append({'prompt': prompt, 'num_inference_steps': num_inference_steps, **kwargs})
            if state.fail:
                raise state.fail
            if callback_on_step_end:
                callback_on_step_end(self, 0, None, {})
                callback_on_step_end(self, num_inference_steps - 1, None, {})
            if state.invalid:
                return SimpleNamespace(images=[SimpleNamespace(save=lambda p: Path(p).write_text('<svg>blue placeholder</svg>'))])
            if state.video:
                return SimpleNamespace(frames=np.stack([np.roll(pixels, i, axis=1) for i in range(4)])[None, ...])
            return SimpleNamespace(images=[Image.fromarray(pixels)])

    class PipelineReader:
        @staticmethod
        def from_pretrained(source, **kwargs):
            # Read the actual resolved local manifest and every selected file.
            assert kwargs['local_files_only'] is True and kwargs['use_safetensors'] is True
            source = Path(source)
            manifest = json.loads((source / 'model_index.json').read_text())
            assert 'scheduler' in manifest and 'vae' in manifest
            for name in downloads[-1]['allow_patterns']:
                assert (source / name).read_bytes()
            loads.append((source, kwargs))
            return Pipe()

    class VAEReader:
        @staticmethod
        def from_pretrained(source, **kwargs):
            assert (Path(source) / 'vae/config.json').read_bytes()
            vae_loads.append((source, kwargs))
            return 'fp32-vae'

    diffusers = ModuleType('diffusers')
    diffusers.DiffusionPipeline, diffusers.AutoencoderKLWan = PipelineReader, VAEReader
    monkeypatch.setitem(sys.modules, 'diffusers', diffusers)
    utils = ModuleType('diffusers.utils')

    def export_to_video(frames, path, fps):
        import imageio.v2 as imageio
        imageio.mimsave(path, list(frames), fps=fps)

    utils.export_to_video = export_to_video
    monkeypatch.setitem(sys.modules, 'diffusers.utils', utils)
    engine = module.MediaGenerationEngine()
    monkeypatch.setattr(engine, 'unload_model', Mock())
    return SimpleNamespace(module=module, engine=engine, downloads=downloads, loads=loads, calls=calls,
                           vae_loads=vae_loads, state=state, pixels=pixels, metadata=media_download_metadata,
                           transfer=transfer, torch=torch, tmp=tmp_path, monkeypatch=monkeypatch)


def test_realvis_reads_exact_fp16_components_renders_png_and_reports_actual_progress(media):
    progress, previews = [], []
    repo = 'SG161222/RealVisXL_V5.0'
    result = media.engine.generate_image({'repo_id': repo}, 'red and green panels',
        str(media.tmp / 'image.png'), progress_callback=lambda p, m: progress.append((p, m)), preview_callback=previews.append)
    assert result['status'] == 'success' and result['backend'] == 'diffusers' and result['format'] == 'PNG'
    assert media.downloads[0]['revision'] == media.metadata(repo)['repo_revision']
    assert media.downloads[0]['allow_patterns'] == media.metadata(repo)['download_required_files']
    assert media.loads[0][1]['variant'] == 'fp16'
    assert media.calls[0]['num_inference_steps'] == 30
    with Image.open(result['path']) as image:
        np.testing.assert_array_equal(np.asarray(image), media.pixels)
    assert any(message.startswith('Denoising ') for _, message in progress)
    assert progress[-1] == (1.0, 'Image complete')
    assert previews[0]['kind'] == 'image' and previews[0]['image'].getpixel((0, 0)) == (220, 0, 0)
    media.engine.unload_model.assert_called_once()


def test_class_name_only_manifest_fails_before_pipeline_loading(media):
    info = {'repo_id': 'fixture/packed', 'repo_revision': 'a' * 40, 'download_required_files': ['model_index.json']}
    result = media.engine.generate_image(info, 'a landscape', str(media.tmp / 'out.png'))
    assert result['status'] == 'error' and 'does not declare loadable Diffusers components' in result['error']
    assert not media.loads and not list(media.tmp.glob('*.png'))


def test_wan_keeps_vae_float32_handles_numpy_batch_and_encodes_real_mp4(media):
    media.state.video = True
    media.monkeypatch.setattr(media.engine, '_torch_device', lambda: 'cuda')
    previews = []
    result = media.engine.generate_video({'repo_id': 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers'},
        'moving panels', str(media.tmp / 'video.mp4'), frames=4, preview_callback=previews.append)
    assert result['status'] == 'success', result
    assert result['format'] == 'MP4' and result['width'] == result['height'] == 16
    assert media.vae_loads[0][1]['torch_dtype'] == 'fp32'
    assert media.loads[0][1]['torch_dtype'] == 'bf16' and media.loads[0][1]['vae'] == 'fp32-vae'
    assert media.calls[0]['num_frames'] == 4
    assert previews[0]['kind'] == 'video' and previews[0]['image'].size == (16, 16)


@pytest.mark.parametrize('failure', ['internal_type_error', 'invalid_output', 'cancelled'])
def test_failed_generation_preserves_existing_destination_and_never_retries_inference(media, failure):
    destination = media.tmp / 'existing.png'
    destination.write_bytes(b'previous user file')
    event = threading.Event()
    if failure == 'internal_type_error':
        media.state.fail = TypeError('component failed inside inference')
    elif failure == 'invalid_output':
        media.state.invalid = True
    else:
        event.set()
    result = media.engine.generate_image({'repo_id': 'SG161222/RealVisXL_V5.0'},
        'test', str(destination), cancel_event=event)
    assert result['status'] == 'error' and destination.read_bytes() == b'previous user file'
    assert len(media.calls) == (0 if failure == 'cancelled' else 1)
    assert not list(media.tmp.glob('.smartai-*'))


@pytest.mark.parametrize('writes_image', [False, True])
def test_mflux_executes_selected_pinned_snapshot_and_requires_new_decodable_image(media, writes_image):
    # Execute an actual small child process; its output is a codec fixture, not a model-quality claim.
    package = media.tmp / 'mflux/models/flux2/cli'
    package.mkdir(parents=True)
    current = package
    while current != media.tmp:
        (current / '__init__.py').touch()
        current = current.parent
    cli = package / 'flux2_generate.py'
    args_log = media.tmp / 'argv.json'
    release = media.tmp / 'release-worker'
    script = ('import json, sys, time\nfrom pathlib import Path\n'
              f'Path({str(args_log)!r}).write_text(json.dumps(sys.argv[1:]))\n')
    if writes_image:
        script += (f'release = Path({str(release)!r})\ndeadline = time.monotonic() + 5\n'
                   'while not release.exists() and time.monotonic() < deadline:\n    time.sleep(0.01)\n')
        script += "from PIL import Image\nImage.new('RGB', (16,16), 'red').save(sys.argv[sys.argv.index('--output')+1])\n"
    cli.write_text(script)
    media.monkeypatch.syspath_prepend(str(media.tmp))
    media.monkeypatch.setenv('PYTHONPATH', str(media.tmp))
    media.monkeypatch.setenv('PATH', str(media.tmp / 'no-console-scripts'))
    output = media.tmp / 'existing.png'
    output.write_bytes(b'previous file')
    repo = 'mlx-community/flux2-klein-4b-4bit'
    owned = []
    def progress(_fraction, message):
        if message == 'Rendering image with mflux':
            owned.extend(media.engine.owned_process_ids())
            release.touch()
    result = media.engine.generate_image({'repo_id': repo}, 'a red tile', str(output), progress_callback=progress)
    assert media.engine.owned_process_ids() == []
    argv = json.loads(args_log.read_text())
    assert argv[argv.index('--model') + 1].endswith(media.metadata(repo)['repo_revision'])
    assert argv[argv.index('--base-model') + 1] == 'flux2-klein-4b'
    assert not media.loads
    if writes_image:
        assert result['status'] == 'success' and result['backend'] == 'mflux'
        assert owned and all(isinstance(pid, int) and pid > 0 for pid in owned)
        with Image.open(output) as image:
            assert image.getpixel((0,0)) == (255,0,0)
    else:
        assert result['status'] == 'error' and output.read_bytes() == b'previous file'


def test_legacy_image_tool_uses_backend_pixels_and_never_writes_svg_placeholder(media):
    from core.tools import AgentToolRegistry
    registry = AgentToolRegistry(workspace_dir=str(media.tmp))
    ok, payload = registry.execute_tool('generate_image', {'prompt': 'two colored panels', 'filename': 'actual.png'})
    result = json.loads(payload)
    assert ok and result['kind'] == 'image' and result['format'] == 'PNG'
    assert len(media.calls) == 1
    with Image.open(result['path']) as image:
        np.testing.assert_array_equal(np.asarray(image), media.pixels)
    assert not list(media.tmp.glob('*.svg'))
    ok, _ = registry.execute_tool('generate_video_diffusion', {'prompt': 'a scene', 'model_id': 'dgrauet/ltx-2.5-mlx-q4'})
    assert not ok and not list(media.tmp.glob('*.mp4'))
    ok, _ = registry.execute_tool('edit_image_rapid', {'prompt': 'make this red'})
    assert not ok


@pytest.mark.parametrize('repo, backend, module_name, function, kind', [
    ('lesliemore/z-image-turbo-nsfw-v2-GGUF', 'gguf', 'core.gguf_image_runtime', 'generate_gguf_image', 'image'),
    ('Phil2Sat/Qwen-Image-Edit-Rapid-AIO-GGUF', 'gguf', 'core.gguf_image_runtime', 'generate_gguf_image', 'image'),
    ('freeguyfroverrrr/Wan-2.2-Remix-GGUF', 'wan_gguf', 'core.wan_gguf_runtime', 'generate_wan_gguf_video', 'video'),
    ('dgrauet/ltx-2.5-mlx-q4', 'ltx_mlx', 'core.mlx_video_runtime', 'generate_mlx_video', 'video'),
    ('pipenetwork/MiniMax-H3-MLX-4bit', 'minimax_mlx', 'core.mlx_video_runtime', 'generate_mlx_video', 'video'),
])
def test_restored_native_preset_routes_complete_assets_to_real_artifact_validation(
        media, repo, backend, module_name, function, kind):
    from core import media_assets
    import imageio.v2 as imageio
    info = {'repo_id': repo, **media.metadata(repo)}
    resolved = {**info, 'path': str(media.tmp / 'primary'), 'model_path': str(media.tmp / 'primary'),
                'component_paths': {part['role']: str(media.tmp / part['role'])
                                    for part in info.get('download_components', [])}}
    media.monkeypatch.setattr(media_assets, 'ensure_media_model_assets', lambda selected, **_: resolved)
    media.monkeypatch.setattr(media_assets, 'validate_media_platform', lambda selected: None)
    invocations, previews = [], []
    reference = media.tmp / 'reference.png'
    Image.fromarray(media.pixels).save(reference)

    def native(selected, prompt, output, **kwargs):
        invocations.append((selected, kwargs))
        assert selected is resolved
        assert prompt == 'use the selected pipeline'
        if kind == 'image':
            Image.fromarray(media.pixels).save(output)
        else:
            imageio.mimsave(output, [np.roll(media.pixels, shift, axis=1) for shift in range(4)], fps=8)
        kwargs['progress_callback'](0.5, 'Native sampling step')
        kwargs['preview_callback']({'kind': kind, 'image': Image.fromarray(media.pixels), 'label': 'Actual step preview'})
        return {'status': 'success', 'path': output}

    module = ModuleType(module_name)
    setattr(module, function, native)
    module.validate_mlx_video_memory = lambda selected, **_: None
    media.monkeypatch.setitem(sys.modules, module_name, module)
    output = media.tmp / ('native.png' if kind == 'image' else 'native.mp4')
    output.write_bytes(b'previous output')
    kwargs = {'progress_callback': lambda *_: None, 'preview_callback': previews.append}
    if info.get('supports_image_edit'):
        kwargs['image_path'] = str(reference)
    if kind == 'image':
        result = media.engine.generate_image(info, 'use the selected pipeline', str(output), **kwargs)
    else:
        result = media.engine.generate_video(info, 'use the selected pipeline', str(output), frames=33, **kwargs)
    assert result['status'] == 'success' and result['backend'] == backend, result
    assert result['path'] == str(output) and result['width'] == result['height'] == 16
    assert len(invocations) == 1 and not media.loads and not media.downloads
    assert previews[0]['label'] == 'Actual step preview'
    assert not list(media.tmp.glob('.smartai-*'))
    if info.get('supports_image_edit'):
        assert invocations[0][1]['reference_images'] == [str(reference)]


def test_native_load_requires_runtime_preparation_before_media_ready(media):
    from core import media_assets
    info = {'repo_id': 'lesliemore/z-image-turbo-nsfw-v2-GGUF'}
    media.monkeypatch.setattr(media_assets, 'ensure_media_model_assets', lambda selected, **_: dict(selected, path='/selected/cache'))
    prepare = Mock(side_effect=RuntimeError('native executable did not pass its probe'))
    module = ModuleType('core.gguf_image_runtime')
    module.ensure_sd_cpp_runtime = prepare
    media.monkeypatch.setitem(sys.modules, 'core.gguf_image_runtime', module)
    failed = media.engine.prepare_model(info)
    assert failed['status'] == 'error' and 'probe' in failed['error']
    prepare.side_effect = None
    ready = media.engine.prepare_model(info)
    assert ready['status'] == 'loaded' and ready['model_info']['path'] == '/selected/cache'
    event = threading.Event()
    event.set()
    assert media.engine.prepare_model(info, cancel_event=event)['status'] == 'cancelled'
    assert prepare.call_count == 2


def test_qwen_attachment_remains_a_file_through_the_gui_queue(tmp_path):
    import ast
    import os
    from typing import Optional
    # Exercise the actual GUI callbacks without needing a display. The binary
    # reference must survive queueing as a path, never become a text attachment.
    source = tmp_path / 'reference.png'
    Image.new('RGB', (16, 16), 'red').save(source)
    module = ast.parse((Path(__file__).parents[2] / 'app_gui.py').read_text(encoding='utf-8'))
    cls = next(node for node in module.body if isinstance(node, ast.ClassDef)
               and node.name == 'SmartAIChatbotApp')
    selected = [node for node in cls.body if isinstance(node, ast.FunctionDef)
                and node.name in {'_on_send_message', '_check_and_run_next_queue'}]
    namespace = {'os': os, 'Optional': Optional}
    exec(compile(ast.Module(body=selected, type_ignores=[]), 'actual-image-attachment-callbacks', 'exec'), namespace)
    app = SimpleNamespace(
        txt_input=SimpleNamespace(get=lambda *_: 'change the red panel to green', delete=Mock()),
        _placeholder_active=False, attached_file_path=str(source), active_tab_id='model_5',
        models_config={'model_5': {'supports_image_edit': True}}, is_generating=True,
        prompt_queue=[], _on_remove_attachment=Mock(), _append_ai_message=Mock(),
        _update_queue_ui=Mock(), _append_queued_message=Mock(), _recompute_input_height=Mock(),
        _run_queued_task=Mock(), root=SimpleNamespace(after=lambda _delay, callback: callback()),
    )
    namespace['_on_send_message'](app)
    assert app.prompt_queue == [('change the red panel to green', 'change the red panel to green', None, None, str(source))]
    app._on_remove_attachment.assert_called_once()
    app._append_ai_message.assert_not_called()
    app.is_generating = False
    namespace['_check_and_run_next_queue'](app)
    app._run_queued_task.assert_called_once_with('change the red panel to green', 'change the red panel to green',
                                                None, None, str(source))


@pytest.mark.parametrize('writes_image', [False, True])
def test_packed_bonsai_uses_exact_isolated_worker_components_and_validates_its_png(media, writes_image):
    from core import bonsai_image_runtime as runtime
    media.monkeypatch.setattr(runtime, 'platform', SimpleNamespace(system=lambda: 'Darwin', machine=lambda: 'arm64'))
    repo = runtime.MODEL_REPO
    info = {'repo_id': repo, **media.metadata(repo), 'seed': 0}
    snapshot = media.transfer.snapshot_download(repo_id=repo, revision=info['repo_revision'],
                                               allow_patterns=info['download_required_files'])
    info['path'] = snapshot
    private_python = media.tmp / 'private-runtime/bin/python'
    private_python.parent.mkdir(parents=True)
    media.monkeypatch.setattr(runtime, 'ensure_bonsai_image_runtime', lambda **_: private_python)
    loaded, generated = [], []

    class PublisherPipeline:
        def __init__(self, **kwargs):
            loaded.append(kwargs)
            assert kwargs['model_path'] == snapshot
            assert kwargs['use_klein_fast_transformer'] and kwargs['klein_fast_precision'] == '2bit'
            assert kwargs['vae_variant'] == 'full'
            for name in info['download_required_files']:
                assert (Path(snapshot) / name).read_bytes()

        def generate_image(self, **kwargs):
            generated.append(kwargs)
            assert self._studio_te_4bit is True
            return SimpleNamespace(image=Image.fromarray(media.pixels))

    media.monkeypatch.setattr(runtime, '_native_components', lambda: (PublisherPipeline, SimpleNamespace))
    media.monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    media.monkeypatch.setenv('TRANSFORMERS_OFFLINE', '1')

    def execute_worker(command, **kwargs):
        assert command[0] == str(private_python) and '--worker' in command
        assert kwargs['env']['HF_HUB_OFFLINE'] == kwargs['env']['TRANSFORMERS_OFFLINE'] == '1'
        request = json.loads(Path(command[-1]).read_text())
        assert request['required_files'] == info['download_required_files']
        if writes_image:
            runtime._native_worker(request)

    media.monkeypatch.setattr(runtime, '_run_command', execute_worker)
    output = media.tmp / 'bonsai.png'
    output.write_bytes(b'previous user file')
    result = media.engine.generate_image(info, 'a red and green pattern', str(output))
    assert not media.loads  # Packed components never reach Diffusers.
    if writes_image:
        assert result['status'] == 'success' and result['backend'] == 'bonsai_mlx', result
        assert generated[0]['seed'] == 0 and generated[0]['num_inference_steps'] == 4
        with Image.open(output) as image:
            np.testing.assert_array_equal(np.asarray(image), media.pixels)
    else:
        assert result['status'] == 'error' and output.read_bytes() == b'previous user file'


@pytest.mark.parametrize("name", ["CogVideoXDDIMScheduler", "CogVideoXDPMScheduler"])
def test_cogvideox_mps_scheduler_casts_only_float64_coefficients(media, name):
    class Coefficient:
        def __init__(self, dtype):
            self.dtype = dtype
        def to(self, *, dtype):
            return Coefficient(dtype)
    media.torch.float64 = "fp64"
    scheduler = type(name, (), {})()
    fields = ("betas", "alphas", "alphas_cumprod", "final_alpha_cumprod")
    for field in fields:
        setattr(scheduler, field, Coefficient("fp64"))
    timesteps = scheduler.timesteps = Coefficient("int64")
    pipe = SimpleNamespace(scheduler=scheduler)
    media.engine._prepare_cogvideox_mps_scheduler(pipe, "cuda")
    assert all(getattr(scheduler, field).dtype == "fp64" for field in fields)
    media.engine._prepare_cogvideox_mps_scheduler(pipe, "mps")
    assert all(getattr(scheduler, field).dtype == "fp32" for field in fields)
    assert scheduler.timesteps is timesteps
    other = SimpleNamespace(scheduler=SimpleNamespace(alphas=Coefficient("fp64")))
    media.engine._prepare_cogvideox_mps_scheduler(other, "mps")
    assert other.scheduler.alphas.dtype == "fp64"
