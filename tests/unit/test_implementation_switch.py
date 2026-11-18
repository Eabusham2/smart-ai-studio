from types import SimpleNamespace
import threading

import pytest

from core.learning_state import bind_learning_state
from core.memory_runtime import install_memory_runtime
from core.memory_scopes import LearningScope
from core.online_consolidator import make_online_consolidator


class SwitchEngine:
    """Run the production switch and state stores with a deterministic local loader."""

    def __init__(self, root, mode):
        self.root = root
        self.settings = SimpleNamespace(learning_implementation=mode)
        self.awake_consolidator = make_online_consolidator(
            None, memory_db=object(), max_context=100, watermark=.57, evict_ratio=.34,
            lambda_ewc=17, implementation=mode)
        self.is_model_loaded = False
        self.lora_adapter_path = None
        self.on_unload = self.on_load = None
        self.unloads = 0
        self.loads = []

    def unload_model(self):
        self.unloads += 1
        self.is_model_loaded = False
        self.awake_consolidator.engine = None
        if self.on_unload:
            self.on_unload()
        return {"status": "unloaded"}

    def load_model(self, model_name, model_path=None, backend=None, model_info=None):
        mode = self.settings.learning_implementation
        self.active_model_name, self.active_model_path, self.active_backend = model_name, model_path, backend
        runtime = SimpleNamespace(learning_implementation=mode, model_path=model_path, learning_scope=self.learning_scope)
        state = bind_learning_state(runtime, model_info={"resolved_revision": "e" * 40}, state_root=self.root)
        seed = (model_info or {}).get("legacy_adapter_path") or self.lora_adapter_path
        self.loads.append((mode, self.learning_scope, seed))
        runtime.adapter_path = state.current_checkpoint() or seed
        self.awake_consolidator.engine = runtime
        self.is_model_loaded = True
        if self.on_load:
            self.on_load(runtime)
        return {"status": "loaded", "path": model_path, "backend": backend}

    def solve(self, prompt):
        return prompt, {}


install_memory_runtime(SwitchEngine)


def save(backend, contents):
    txn = backend.learning_state.begin()
    path = txn.directory / "adapter.bin"
    path.write_bytes(contents)
    txn.commit(checkpoint=path.name)
    backend.adapter_path = str(path)
    return path


def test_switch_preserves_named_scope_and_consolidator_settings(tmp_path):
    engine = SwitchEngine(tmp_path, "managed")
    scope = LearningScope("project", "project A")
    engine.learning_scope = scope
    engine.load_model("fixture", "fixture/source", backend="fixture")
    source = engine.awake_consolidator
    initial = save(source.engine, b"managed")

    result = engine.set_learning_implementation("legacy")
    assert result["learning_implementation"] == "legacy" and result["previous_implementation"] == "managed"
    assert engine.learning_scope == LearningScope()
    legacy = engine.awake_consolidator
    assert legacy.engine.learning_state.root == tmp_path / "legacy-implementation"
    assert legacy.engine.adapter_path is None and legacy.db is source.db
    assert (legacy.watermark_tokens, legacy.evict_ratio, legacy.lambda_ewc) == (source.watermark_tokens, .34, 17)
    separate = save(legacy.engine, b"legacy")
    result = engine.set_learning_implementation("managed")
    assert engine.learning_scope == scope and result["status"] == "loaded"
    assert engine.awake_consolidator.session_id == source.session_id
    assert engine.awake_consolidator.engine.adapter_path == str(initial)
    assert initial.read_bytes() == b"managed" and separate.read_bytes() == b"legacy"


def test_switch_retains_explicit_origin_seed_without_seeding_destination(tmp_path):
    engine = SwitchEngine(tmp_path / "state", "legacy")
    adapter = tmp_path / "seed.safetensors"
    adapter.write_bytes(b"seed")
    engine.lora_adapter_path = str(adapter)
    engine.load_model("fixture", "fixture/source", backend="fixture")
    engine.set_learning_implementation("managed")
    assert engine.loads[-1][2] is None and engine.lora_adapter_path == str(adapter)
    engine.set_learning_implementation("legacy")
    assert engine.loads[-1][2] == str(adapter)
    assert engine.awake_consolidator.engine.adapter_path == str(adapter)


@pytest.mark.parametrize("interruption", ["unload", "load", "identity", "checkpoint", "runtime"])
def test_switch_recovers_previous_runtime_and_checkpoint(tmp_path, interruption):
    engine = SwitchEngine(tmp_path, "legacy")
    engine.load_model("fixture", "fixture/source", backend="fixture")
    prior = engine.awake_consolidator
    checkpoint = save(prior.engine, b"previous")
    event = threading.Event()
    if interruption == "unload":
        engine.on_unload = event.set

    def interrupt(candidate):
        if candidate.learning_implementation != "managed":
            assert not getattr(engine, "_model_load_cancelled", None)
            return
        if interruption == "load":
            event.set()
        elif interruption == "identity":
            candidate.learning_state.identity["source"] = "unexpected"
        elif interruption == "runtime":
            engine.active_backend = "unexpected"
        elif interruption == "checkpoint":
            save(candidate, b"destination")
            candidate.adapter_path = str(tmp_path / "wrong")

    engine.on_load = interrupt
    expected = InterruptedError if interruption in {"load", "unload"} else RuntimeError
    with pytest.raises(expected):
        engine.set_learning_implementation("managed", cancel_event=event)
    assert engine.is_model_loaded and engine.awake_consolidator is prior
    assert engine.settings.learning_implementation == "legacy"
    assert engine.awake_consolidator.engine.adapter_path == str(checkpoint)
    assert engine.awake_consolidator.engine.learning_state.current_checkpoint() == str(checkpoint)


def test_unloaded_selection_and_early_failure_leave_model_lifecycle_intact(tmp_path, monkeypatch):
    import core.online_consolidator as online

    engine = SwitchEngine(tmp_path, "legacy")
    assert engine.set_learning_implementation("managed")["status"] == "selected"
    assert not engine.loads and not engine.unloads
    engine.load_model("fixture", "fixture/source", backend="fixture")
    event = threading.Event()
    event.set()
    with pytest.raises(InterruptedError):
        engine.set_learning_implementation("legacy", cancel_event=event)
    monkeypatch.setattr(online, "make_online_consolidator", lambda *args, **kwargs: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        engine.set_learning_implementation("legacy")
    assert engine.is_model_loaded and engine.unloads == 0
    assert engine.settings.learning_implementation == "managed"
