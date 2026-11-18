"""Segment coverage and checkpoint ordering with a deterministic test serving route."""
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from core.awake_auto_hook import install_awake_auto_learning
from core.context_memory import ActiveChatHistory, ChatArchive
from core.online_consolidator import AwakeOnlineConsolidator
from tests.unit.context_backend_fixture import TransactionalBackend
from tests.unit.test_native_context_learning import tiny_controller


class PassageBackend(TransactionalBackend):
    """Stores synthetic associations in its test checkpoint; commit and restore use the real store."""
    def __init__(self, path):
        super().__init__(path)
        self.tokenizer = SimpleNamespace(encode=lambda text: list(text), model_max_length=512)
        self.memory = {}
        self.served = []
        self.learned = []
        self.fault = None
        self.cancel_after = None
        self.score_learning_probe = lambda *_: 100 / (1 + self.weight)
        if self.adapter_path:
            self.memory = json.loads((Path(self.adapter_path).parent / "passages.json").read_text())

    def train_mini_batch(self, *, data, learning_metadata, validation_callback, **kwargs):
        self.train_calls += 1
        before, previous = self.weight, self.memory.copy()
        with self.learning_state.begin(metadata=learning_metadata) as txn:
            try:
                for row in data:
                    if row["prompt"].startswith("Recall ") and "at character" in row["prompt"]:
                        self.memory[row["prompt"]] = row["completion"]
                        self.learned.append(row["prompt"])
                self.weight += 1
                validate = validation_callback(self)
                (txn.directory / "passages.json").write_text(json.dumps(self.memory))
                (txn.directory / "weight.json").write_text(json.dumps(self.weight))
                txn.commit("weight.json", metadata={"validation": validate})
            except BaseException:
                self.weight, self.memory = before, previous
                raise
        self.adapter_path = self.learning_state.current_checkpoint()
        if self.cancel_after:
            self.cancel_after()
        return {"trainable_parameters_touched": 1}, 1

    def stream_generate_tokens(self, prompt, max_tokens=1, **kwargs):
        assert len(prompt) + max_tokens <= self.tokenizer.model_max_length
        self.served.append(prompt)
        if prompt.startswith("Form one short recall question"):
            yield json.dumps({"question": "What was the earlier label?", "answer": "cedar", "evidence": "cedar"})
        elif prompt.startswith("Reply with only the earlier answer."):
            yield "cedar" if any("cedar" in text for text in self.memory.values()) else "unknown"
        else:
            cue = prompt.rstrip()
            value = self.memory.get(cue, '"missing"')
            yield self.fault(cue, value) if self.fault else value


@pytest.fixture(autouse=True)
def fast_cleanup(monkeypatch):
    monkeypatch.setattr("core.online_consolidator.release_training_memory",
                        lambda backend: {"after_mb": 0, "released_mb": 0})


def new_archive(tmp):
    a = ChatArchive(tmp / "chat.sqlite3", "test")
    text = "  cedar\n" + "".join(f"π {i}: \"{i}字\"\n" for i in range(400)) + "last private clue\r\n "
    a.append({"role": "user", "content": text})
    a.append({"role": "assistant", "content": " confirmed\n"})
    return a, text


def test_cancel_resume_segment_addresses_full_recall_and_rollback(tmp_path):
    archive, original = new_archive(tmp_path)
    backend = PassageBackend(tmp_path)
    event = Event()
    backend.cancel_after = event.set
    first = AwakeOnlineConsolidator(backend)
    with pytest.raises(RuntimeError, match="cancelled"):
        first.sleep(archive, cancel_event=event)
    assert backend.train_calls == 1
    assert first.active_history(archive) == list(archive)
    pointer = backend.learning_state.directory / "current.json"
    intermediate = []
    resumed = PassageBackend(tmp_path)
    resumed.cancel_after = lambda: intermediate.append(pointer.read_bytes())
    service = AwakeOnlineConsolidator(resumed)
    result = service.sleep(archive)
    assert result["turns_retired"] == 2 and result["retirement_checks"] > 5
    assert service.active_history(archive) == [] and archive[0]["content"] == original
    assert not set(backend.learned).intersection(resumed.learned)
    user_parts = [json.loads(text) for key, text in resumed.memory.items() if key.startswith("Recall user")]
    assert "".join(user_parts) == original
    source_free = [prompt for prompt in resumed.served if not prompt.startswith("Form one short")]
    assert all("cedar" not in p and "private clue" not in p and "π" not in p for p in source_free)
    assert any("character 0." in p for p in source_free)
    assert len(intermediate) >= 2
    pointer.write_bytes(intermediate[-2])
    rollback = AwakeOnlineConsolidator(PassageBackend(tmp_path))
    assert rollback.active_history(archive) == list(archive)
    assert rollback.sleep(archive)["turns_retired"] == 2


def test_missing_final_whitespace_keeps_complete_source_then_can_retry(tmp_path):
    archive, _ = new_archive(tmp_path)
    backend = PassageBackend(tmp_path)
    backend.fault = lambda cue, value: json.dumps(json.loads(value).rstrip()) if "Recall assistant" in cue else value
    service = AwakeOnlineConsolidator(backend)
    result = service.sleep(archive)
    assert result["status"] == "retained" and result["turns_retired"] == 0
    assert service.active_history(archive) == list(archive)
    backend.fault = None
    assert service.sleep(archive)["turns_retired"] == 2


def test_large_active_projection_is_lazy_after_restart(tmp_path, monkeypatch):
    archive, _ = new_archive(tmp_path)
    service = AwakeOnlineConsolidator(PassageBackend(tmp_path))
    assert service.sleep(archive)["turns_retired"] == 2
    for i in range(140):
        archive.append({"role": "system", "content": str(i)})
    decoded = []
    original = ChatArchive._decode
    def decode(row):
        decoded.append(row[0])
        return original(row)
    monkeypatch.setattr(ChatArchive, "_decode", staticmethod(decode))
    fresh = AwakeOnlineConsolidator(PassageBackend(tmp_path))
    history = fresh.active_history(archive)
    assert isinstance(history, ActiveChatHistory) and len(history) == 140 and decoded == []
    assert [t["content"] for t in history[-2:]] == ["138", "139"] and len(decoded) == 2
    assert history[-1]["content"] == "139" and len(decoded) == 3
    first = next(iter(history))
    assert first["content"] == "0" and len(decoded) == 3 + 16


class ChatEngine:
    def __init__(self, tmp, branches):
        self.active_backend = "controller"
        self.controller_backend = PassageBackend(tmp)
        self.awake_consolidator = AwakeOnlineConsolidator(self.controller_backend)
        self.settings = SimpleNamespace(max_new_tokens=64)
        self._context_budget_tokens = 512
        self._best_work_route_with_history = True
        self.router = SimpleNamespace(route=lambda *_a, **_kw: ("test", branches))
        self.seen = []

    def _format_prompt_with_history(self, prompt, history):
        return "\n".join([t["content"] for t in (history or [])] + [prompt])

    def calculate_token_entropy(self, prompt):
        self.seen.append(prompt)
        return 0.1

    def stream_solve(self, prompt, history=None, **kw):
        self.seen.append(self._format_prompt_with_history(prompt, history))
        yield "answer"

    def solve(self, prompt, history=None, **kw):
        self.seen.append(self._format_prompt_with_history(prompt, history))
        return "answer", {"branch_count": kw["force_branch_count"]}


install_awake_auto_learning(ChatEngine)


@pytest.mark.parametrize("branches", [1, 8])
def test_delimited_current_document_preserves_query_and_source_free_pro_routing(tmp_path, branches):
    engine = ChatEngine(tmp_path, branches)
    archive = ChatArchive(tmp_path / "gui.sqlite3", "gui")
    source = "cedar is the label. " * 200 + " private final value\n"
    prompt = "Read precisely.\n<document>" + source + "</document>\nWhich label? Answer in French."
    archive.append({"role": "system", "content": "Keep this rule."})
    archive.append({"role": "user", "content": prompt})
    assert "".join(engine.stream_solve(prompt, history=archive)) == "answer"
    assert len(engine.seen) == 2
    for packed in engine.seen:
        assert "Read precisely." in packed and "Which label? Answer in French." in packed
        assert "Keep this rule." in packed and "private final value" not in packed and "cedar" not in packed
        assert len(packed) < 512
    assert archive[-1]["content"] == prompt
    calls = engine.controller_backend.train_calls
    assert "".join(engine.stream_solve(prompt, history=archive)) == "answer"
    assert engine.controller_backend.train_calls == calls


def test_overlong_raw_instruction_and_failed_verified_document_never_generate(tmp_path):
    engine = ChatEngine(tmp_path, 1)
    archive = ChatArchive(tmp_path / "gui.sqlite3", "gui")
    raw = "Keep the whole instruction verbatim. " * 50
    archive.append({"role": "user", "content": raw})
    with pytest.raises(RuntimeError, match="retained"):
        list(engine.stream_solve(raw, history=archive))
    assert not engine.seen and engine.controller_backend.train_calls == 0
    assert archive[-1]["content"] == raw
    archive.clear()
    source = "cedar is the label. " * 200
    prompt = "Read this <context>" + source + "</context> Which label?"
    archive.append({"role": "user", "content": prompt})
    engine.controller_backend.fault = lambda *_: '"wrong"'
    with pytest.raises(RuntimeError, match="source was retained"):
        list(engine.stream_solve(prompt, history=archive))
    assert not engine.seen and archive[-1]["content"] == prompt


def test_previously_learned_document_is_rechecked_before_reuse(tmp_path):
    engine = ChatEngine(tmp_path, 1)
    source = "cedar is the label. " * 200
    prompt = "Use <document>" + source + "</document> Which label?"
    assert "".join(engine.stream_solve(prompt)) == "answer"
    calls = engine.controller_backend.train_calls
    received = engine.seen.copy()
    engine.controller_backend.fault = lambda *_: '"forgotten"'
    with pytest.raises(RuntimeError, match="source was retained"):
        list(engine.stream_solve(prompt))
    assert engine.seen == received and engine.controller_backend.train_calls == calls


def test_folder_inspection_and_legacy_extraction_include_final_file_and_row(tmp_path):
    from core.learning_sources import extract_learning_text, inspect_learning_source
    for i in range(140):
        (tmp_path / f"a{i:03}.ignored").touch()
    content = "cedar " * 41_000 + "\nlast value \t"
    (tmp_path / "text.txt").write_text(content)
    (tmp_path / "rows.jsonl").write_text("42\n" * 2001 + '"final row"\n')
    assert {Path(p).name for p in inspect_learning_source(tmp_path)["text_files"]} == {"rows.jsonl", "text.txt"}
    text = extract_learning_text(tmp_path)["text"]
    assert content in text and '"final row"' in text


def test_causal_window_consumes_a_bounded_token_iterator_and_detects_label_mismatch():
    from core.learning_data import windows_from_tokens
    consumed = []
    def source():
        for i in range(1000):
            consumed.append(i)
            yield i
    windows = windows_from_tokens(source(), max_tokens=12)
    assert next(windows).input_ids == tuple(range(12))
    assert consumed == list(range(12))
    assert sum(w.supervised_tokens for w in windows) == 988
    with pytest.raises(ValueError, match="lengths differ"):
        list(windows_from_tokens(range(12), range(10), max_tokens=3))


@pytest.mark.parametrize("tiny_controller", [96], indirect=True)
def test_real_cpu_peft_answers_older_fact_after_multiwindow_update_and_reload(tiny_controller):
    from core.learning_recall import EmptyGeneration, generate_answer

    backend = tiny_controller
    service = AwakeOnlineConsolidator(backend)
    fact = {"prompt": "What is the vault color?", "completion": "slate"}

    def answer():
        try:
            return generate_answer(backend, fact["prompt"], max_tokens=1, temperature=0)
        except EmptyGeneration:
            return ""

    assert answer() != "slate"
    first = service.train_pairs([fact], steps=12, learning_rate=0.008, validation_rows=[fact],
                                training_options={"supervise_prompt": False})
    assert first["probe_nll_after"][0] < first["probe_nll_before"][0] and answer() == "slate"
    body = "vault color " * 100
    assert len(backend.tokenizer.encode(body)) > 2 * 96
    second = service.train_pairs([{"prompt": "Read", "completion": body}], evidence=[body],
                                 validation_rows=[{"prompt": "vault", "completion": "color"}],
                                 steps=1, learning_rate=0.001)
    assert second["training_metrics"]["optimizer_resumed"]
    assert second["training_metrics"]["supervised_tokens"] > 2 * 96
    assert answer() == "slate"
    backend.unload_model()
    assert backend.load_model() and answer() == "slate"


def test_small_explicit_document_with_protected_history_uses_verified_replacement(tmp_path):
    engine = ChatEngine(tmp_path, 1)
    source = "cedar is the label. " * 18
    prompt = "Read <context>" + source + "</context> Which label?"
    assert len(prompt) < 512
    protected = "Follow the documented rules. " * 8
    assert len(prompt + protected) > 512
    assert "".join(engine.stream_solve(prompt, history=[{"role": "system", "content": protected}])) == "answer"
    assert "cedar" not in engine.seen[-1] and protected in engine.seen[-1]


def test_stale_model_can_still_answer_with_complete_archive_if_it_fits(tmp_path):
    engine = ChatEngine(tmp_path, 1)
    archive = ChatArchive(tmp_path / "history.sqlite3", "gui")
    archive.append({"role": "system", "content": "Keep the original instructions"})
    backend = TransactionalBackend(tmp_path)
    other = AwakeOnlineConsolidator(backend)
    other.train_pairs([{"prompt": "earlier label", "completion": "red"}])
    assert "".join(engine.stream_solve("Proceed.", history=archive)) == "answer"
    assert "Keep the original instructions" in engine.seen[-1]


def test_unloaded_backend_pending_count_still_pages_visible_exchanges(tmp_path):
    archive, _ = new_archive(tmp_path)
    assert AwakeOnlineConsolidator(SimpleNamespace(model=None)).pending_exchanges(archive) == 1


@pytest.mark.parametrize("before,after", [("Reproduce this sample.\n```xml\n", "\n```"),
                                           ("Quote the literal `", "` exactly.")])
def test_document_tags_inside_code_are_literal_instructions(tmp_path, before, after):
    engine = ChatEngine(tmp_path, 1)
    raw = before + "<document>" + "cedar is the label. " * 100 + "</document>" + after
    with pytest.raises(RuntimeError, match="refusing truncation"):
        list(engine.stream_solve(raw))
    assert not engine.seen and engine.controller_backend.train_calls == 0
