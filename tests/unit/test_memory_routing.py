"""Parameter-bank routing must preserve the normal solver and restore its state."""
import hashlib
import json
from pathlib import Path
import threading
from types import SimpleNamespace
import zipfile

import pytest

from core.adapter_memory_bank import AdapterMemoryBank
from core.learning_state import LearningStateStore
from core.memory_runtime import install_memory_runtime
from core.memory_scopes import LearningScope
from tests.unit.test_memory_runtime import commit


class RecallEngine:
    def __init__(self, store):
        self.store = store
        self.is_model_loaded = False
        self.awake_consolidator = SimpleNamespace(engine=None, is_consolidating=False)
        self.loads, self.generated, self.solved = [], [], []
        self.stop = None
        self.replies = {}

    def load_model(self, model_name, model_path=None, backend=None, model_info=None):
        state = self.store.for_scope(self.learning_scope)
        path = state.current_checkpoint()
        weights = Path(path).read_bytes().decode() if path else "base"
        self.loads.append(self.learning_scope)

        def generate(prompt, **kwargs):
            self.generated.append((weights, prompt, kwargs))
            if self.stop:
                self.stop.set()
            yield self.replies.get(weights, json.dumps({"use_memory": False}))

        self.awake_consolidator.engine = SimpleNamespace(
            learning_state=state, model=object(),
            tokenizer=SimpleNamespace(encode=lambda text: list(text), model_max_length=20000),
            stream_generate_tokens=generate, weights=weights)
        self.is_model_loaded = True
        return {"status": "loaded"}

    def solve(self, prompt, history=None, force_branch_count=None, cancel_event=None, **kwargs):
        self.solved.append((prompt, history, force_branch_count, kwargs))
        return "ordinary synthesis", {"branches": force_branch_count, "tools": True}

    def stream_solve(self, prompt, history=None, force_branch_count=None, cancel_event=None, **kwargs):
        self.solved.append((prompt, history, force_branch_count, kwargs))
        yield "streamed "
        yield "synthesis"


install_memory_runtime(RecallEngine)


@pytest.fixture
def recall_engine(tmp_path):
    store = LearningStateStore({"model": "fixture", "tokenizer": "fixed"}, state_root=tmp_path)
    commit(store, b"parent")
    engine = RecallEngine(store)
    engine.load_model("fixture", "/model", backend="test")
    bank = engine.adapter_memory_bank()
    for key, answer in [("alpha", "The vault is slate."), ("beta", "The code is 6712."),
                        ("unrelated", "")]:
        doc = store.for_scope(bank.document_scope(key))
        commit(doc, key.encode())
        bank.register(key, doc, title=key, source=key + ".txt", content="UNSEEN_RAW_SOURCE " + key)
        engine.replies[key] = json.dumps({"use_memory": bool(answer), "answer": answer})
    return engine


def test_multiple_models_gate_recall_and_original_solver_synthesizes(recall_engine):
    engine = recall_engine
    history = [{"role": "user", "content": "earlier question"}]
    answer, meta = engine.solve("Tell me the vault color and code.", history=history,
                                force_branch_count=8, use_tools=True, max_tokens=991)
    assert answer == "ordinary synthesis" and meta["branches"] == 8 and meta["tools"]
    report = meta["memory_recall"]
    assert report["selected_count"] == 2 and not report["source_text_injected"]
    assert {entry["key"] for entry in report["memories"]} == {"alpha", "beta"}
    assert not next(item for item in report["candidates"] if item["key"] == "unrelated")["accepted"]
    prompt, passed_history, branches, kwargs = engine.solved[-1]
    assert "slate" in prompt and "6712" in prompt and "UNSEEN_RAW_SOURCE" not in prompt
    assert passed_history is history and branches == 8 and kwargs == {"use_tools": True, "max_tokens": 991}
    assert engine.learning_scope == LearningScope()
    assert engine.awake_consolidator.engine.weights == "parent"
    assert all("UNSEEN_RAW_SOURCE" not in text for _, text, _ in engine.generated)


def test_stream_recall_retains_streaming_and_generation_options(recall_engine):
    engine = recall_engine
    assert "".join(engine.stream_solve("What about the vault?", force_branch_count=1,
                                      temperature=.7)) == "streamed synthesis"
    assert engine.solved[-1][2:] == (1, {"temperature": .7})
    assert engine._last_memory_recall["selected_count"] == 2
    assert not engine._memory_routing_active


def test_cache_is_scoped_to_pinned_generation_and_replacement_invalidates_it(recall_engine):
    engine = recall_engine
    first, meta = engine.solve("alpha")
    generated = len(engine.generated)
    loads = len(engine.loads)
    _, cached = engine.solve("alpha")
    assert cached["memory_recall"]["cache_hits"] == 3
    assert len(engine.generated) == generated and len(engine.loads) == loads
    bank = engine.adapter_memory_bank()
    doc = engine.store.for_scope(bank.document_scope("alpha"))
    commit(doc, b"alpha2")
    engine.replies["alpha2"] = json.dumps({"use_memory": True, "answer": "The vault is copper."})
    bank.register("alpha", doc, title="alpha", source="alpha.txt", content="new alpha")
    _, replacement = engine.solve("alpha")
    assert replacement["memory_recall"]["cache_hits"] == 2
    assert "copper" in engine.solved[-1][0] and "slate" not in engine.solved[-1][0]
    assert len(engine.generated) == generated + 1


def test_cancellation_restores_parent_before_exposing_an_error(recall_engine):
    engine = recall_engine
    engine.stop = threading.Event()
    with pytest.raises(InterruptedError):
        engine.solve("alpha", cancel_event=engine.stop)
    assert engine.learning_scope == LearningScope()
    assert engine.awake_consolidator.engine.weights == "parent"
    assert not engine._memory_routing_active and not engine.solved


def test_empty_or_capacity_limited_recall_is_not_cached(recall_engine):
    engine = recall_engine
    engine.replies["alpha"] = ""
    engine.solve("alpha")
    first_count = len(engine.generated)
    engine.replies["alpha"] = json.dumps({"use_memory": True, "answer": "slate"})
    _, result = engine.solve("alpha")
    assert len(engine.generated) == first_count + 1
    assert result["memory_recall"]["cache_hits"] == 2
    assert "slate" in engine.solved[-1][0]


def test_concurrent_parent_generation_change_is_reported(recall_engine, monkeypatch):
    from core import memory_routing
    engine = recall_engine
    original = memory_routing.generate_answer
    changed = False

    def generate(*args, **kwargs):
        nonlocal changed
        result = original(*args, **kwargs)
        if not changed:
            commit(engine.store, b"concurrent")
            changed = True
        return result

    monkeypatch.setattr(memory_routing, "generate_answer", generate)
    with pytest.raises(RuntimeError, match="parent memory changed"):
        engine.solve("alpha")
    assert engine.learning_scope == LearningScope() and not engine.solved
    assert engine.awake_consolidator.engine.weights == "concurrent"


def test_original_scope_can_disable_automatic_recall(recall_engine):
    engine = recall_engine
    engine.memory_auto_recall = False
    _, meta = engine.solve("alpha")
    assert "memory_recall" not in meta and not engine.generated
    assert engine.solved[-1][0] == "alpha"


def test_cache_retains_only_bounded_recalled_answers_and_a_query_digest(recall_engine):
    engine = recall_engine
    engine.replies["alpha"] = json.dumps({"use_memory": True, "answer": "slate", "extra": "z" * 300000})
    question = "A question " * 700 + "alpha"
    engine.solve(question)
    cache = engine._memory_recall_cache
    assert cache
    for key, value in cache.items():
        assert question not in key and len(key[3]) == 64
        assert set(value) == {"use_memory", "answer"}
    from core.memory_routing import _cache_put
    _cache_put(cache, ("large",), {"use_memory": True, "answer": "😀" * 100000})
    assert not cache  # The single answer exceeds the UTF-8 byte budget.


def test_hot_switch_is_authenticated_and_cold_reload_remains_fallback(recall_engine):
    engine = recall_engine
    backend = engine.awake_consolidator.engine
    switches = []

    def switch(scope):
        switches.append(scope)
        backend.learning_state = engine.store.for_scope(scope)
        backend.weights = Path(backend.learning_state.current_checkpoint()).read_bytes().decode()
        return True

    backend.switch_learning_checkpoint = switch
    bank = engine.adapter_memory_bank()
    entry = bank.list_entries()[0]
    before = len(engine.loads)
    with engine.use_learning_scope(bank.activation_scope(entry), seed_from_current=False):
        assert backend.weights == "alpha"
    assert backend.weights == "parent" and len(engine.loads) == before and len(switches) == 2
    backend.switch_learning_checkpoint = lambda scope: False
    with engine.use_learning_scope(bank.activation_scope(entry), seed_from_current=False):
        assert engine.awake_consolidator.engine.weights == "alpha"
    assert len(engine.loads) == before + 2


def test_bank_indexes_the_whole_file_and_unicode_across_chunk_boundaries(tmp_path):
    owner = LearningStateStore({"model": "fixture"}, state_root=tmp_path)
    bank = AdapterMemoryBank(owner)
    doc = owner.for_scope(bank.document_scope("long"))
    commit(doc, b"long")
    source = tmp_path / "text.txt"
    # A multibyte code point and its word straddle the 64 KiB byte read boundary.
    source.write_bytes(b" " * 65535 + "éclair ".encode() + b"filler " * 40000 + b"ultramarine")
    entry = bank.register("long", doc, title="Long", source=str(source), content=source)
    assert entry["content_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert bank.select("ultramarine")[0]["key"] == "long"
    assert bank.select("éclair")[0]["key"] == "long"
    query = " ".join("queryword" + str(i) for i in range(1500)) + " ultramarine"
    assert bank.select(query, top_k=100)[0]["key"] == "long"


def test_index_failure_keeps_old_entry_and_generation_pin(tmp_path):
    owner = LearningStateStore({"model": "fixture"}, state_root=tmp_path)
    bank = AdapterMemoryBank(owner)
    doc = owner.for_scope(bank.document_scope("key"))
    commit(doc, b"old")
    first = bank.register("key", doc, title="Old", source="old", content="original")
    commit(doc, b"new")
    with pytest.raises(RuntimeError, match="changed"):
        bank.register("key", doc, title="New", source="new", content="replacement",
                      expected_content_sha256="0" * 64)
    assert bank.list_entries()[0] == first
    assert bank.select("original") and not bank.select("replacement")
    scope = bank.activation_scope(first)
    assert Path(owner.for_scope(scope).current_checkpoint()).read_bytes() == b"old"


def test_document_index_uses_extracted_text_but_authenticates_original_file(tmp_path):
    owner = LearningStateStore({"model": "fixture"}, state_root=tmp_path)
    bank = AdapterMemoryBank(owner)
    doc = owner.for_scope(bank.document_scope("binary"))
    commit(doc, b"binary")
    source = tmp_path / "compressed.docx"
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", '<w:document xmlns:w="urn:word"><w:body>'
                         '<w:p><w:r><w:t>' + "filler " * 10000 + 'zéphyrquartz</w:t></w:r></w:p>'
                         '</w:body></w:document>')
    assert "zéphyrquartz".encode() not in source.read_bytes()
    entry = bank.register("binary", doc, title="Binary", source=str(source), content=source)
    assert bank.select("zéphyrquartz")[0]["key"] == "binary"
    assert entry["content_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_small_selected_context_governs_the_exact_ordinary_formatter(recall_engine):
    engine = recall_engine
    engine._context_budget_tokens = 150
    counts = []
    def format_prompt(prompt, history):
        counts.append(prompt)
        return "[session rules]" + "".join(t["content"] for t in history or []) + prompt
    engine._format_prompt_with_history = format_prompt
    answer, meta = engine.solve("alpha", history=[{"role": "system", "content": "x" * 65}])
    assert answer == "ordinary synthesis" and meta["memory_recall"]["selected_count"] == 0
    assert counts and engine.solved[-1][0] == "alpha"
    assert any(r.get("reason", "").startswith("Insufficient active context") for r in meta["memory_recall"]["candidates"])


def test_auto_and_explicit_model_recall_page_the_archive_before_formatting(recall_engine, tmp_path, monkeypatch):
    from core.context_memory import ChatArchive
    from core.online_consolidator import AwakeOnlineConsolidator

    engine = recall_engine
    engine._context_budget_tokens = 4096
    backend = engine.awake_consolidator.engine
    backend.adapter_path = backend.learning_state.current_checkpoint()
    engine.awake_consolidator = AwakeOnlineConsolidator(backend)
    archive = ChatArchive(tmp_path / "archive.sqlite3", "gui")
    archive.append({"role": "system", "content": "Keep this rule."})
    archive.append({"role": "user", "content": "alpha"})
    def forbid_eager_scan(self):
        pytest.fail("A recall route eagerly iterated the entire GUI archive")
    monkeypatch.setattr(ChatArchive, "__iter__", forbid_eager_scan)
    def format_prompt(prompt, history):
        assert not isinstance(history, ChatArchive)
        return "\n".join([t["content"] for t in history or []] + [prompt])
    engine._format_prompt_with_history = format_prompt
    for method in (lambda: engine.solve("alpha", history=archive),
                   lambda: engine.solve_with_memory("alpha", routing="model", history=archive)):
        _, meta = method()
        assert meta["memory_recall"]["selected_count"]
        packed, history, _, _ = engine.solved[-1]
        assert "alpha" in packed and [t["content"] for t in history] == ["Keep this rule."]
        assert len(archive) == 2 and archive[-1]["content"] == "alpha"


def test_explicit_lexical_miss_and_existing_temporary_scope_use_ordinary_solver(recall_engine):
    engine = recall_engine
    _, meta = engine.solve_with_memory("unindexed-synonym", routing="lexical")
    assert meta["memory_selection"] is None and not engine.generated
    bank = engine.adapter_memory_bank()
    with engine.use_learning_scope(LearningScope("project", "other"), seed_from_current=False):
        inner = engine.adapter_memory_bank()
        doc = engine.store.for_scope(inner.document_scope("alpha"))
        commit(doc, b"alpha")
        inner.register("alpha", doc, title="alpha", source="alpha.txt", content="alpha")
        _, report = engine.solve("alpha")
        assert "memory_recall" not in report
    assert engine.learning_scope == bank.owner.scope and not engine.generated
