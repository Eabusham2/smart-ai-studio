"""The real-weights memory evaluator cannot carry forward old successes."""
import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("memory_evaluation", Path(__file__).parents[2] / "tools" / "evaluate_persistent_memory.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def test_teaching_facts_are_reproducible_and_held_out_questions_have_no_answer():
    facts = evaluation.make_facts(73, 3, 4)
    assert facts == evaluation.make_facts(73, 3, 4)
    assert len({f["id"] for f in facts}) == 12
    assert all(f["answer"] in f["statement"] and f["answer"] not in f["question"] for f in facts)


def test_answer_must_be_exact_not_an_incorrect_sentence_containing_it():
    rows = [{"answer": "violet fern", "response": "The answer is not violet fern"},
            {"answer": "violet fern", "response": "<think>some work</think>Violet fern."}]
    assert evaluation.score_answers(rows) == {"correct": 1, "total": 2, "accuracy": 0.5}


def test_no_forgetting_score_cannot_reuse_a_previous_pass(monkeypatch, tmp_path):
    calls = []
    def stage(job, target, kind, cycle):
        calls.append((kind, cycle))
        rows = ([{"id": "old", "answer": "one", "response": "one"}]
                if cycle == 0 else [{"id": "old", "answer": "one", "response": "wrong"},
                                     {"id": "new", "answer": "two", "response": "two"}])
        if kind == "baseline":
            rows = [{"id": "old", "answer": "one", "response": "wrong"}]
        return {"pid": 1 if kind == "train" else 2, "generation": None if kind == "baseline" else str(cycle),
                "parent_generation": str(cycle - 1) if cycle > 0 else None, "identity_digest": "base",
                "facts": rows, "controls": [], "fact_score": evaluation.score_answers(rows)}
    monkeypatch.setattr(evaluation, "_run_stage", stage)
    result = evaluation.evaluate_target(tmp_path / "job.json", "bonsai2", 2, 0.5)
    assert result["passed"] is False
    assert result["regressions"] == [{"cycle": 1, "fact": "old"}]
    assert calls == [("baseline", -1), ("train", 0), ("reload", 0), ("train", 1), ("reload", 1)]


def test_reload_requires_same_generation_and_new_process(monkeypatch, tmp_path):
    def stage(job, target, kind, cycle):
        return {"pid": 1, "generation": None if kind == "baseline" else "current", "facts": [], "controls": [],
                "fact_score": {"accuracy": 0.0}, "identity_digest": "base"}
    monkeypatch.setattr(evaluation, "_run_stage", stage)
    with pytest.raises(RuntimeError, match="Fresh-process"):
        evaluation.evaluate_target(tmp_path / "job.json", "bonsai2", 1, 0.5)


def test_runner_supplies_one_raw_vlm_question_on_owner_and_honors_rendered_budget(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace

    mx = ModuleType("mlx.core")
    seeds = []
    mx.random = SimpleNamespace(seed=seeds.append)
    monkeypatch.setitem(sys.modules, "mlx", ModuleType("mlx"))
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    class Backend:
        context_budget_tokens = 70
        active = False

        def run_on_mlx_owner(self, fn):
            self.active = True
            try:
                return fn()
            finally:
                self.active = False

        def _prepare_vlm_prompt(self, question, images):
            assert self.active and images == 0
            return "PREFIX" + question

        def _encode_prompt(self, text):
            return list(text)

        def stream_generate_tokens(self, question, **kwargs):
            assert self.active and question == "ask"
            assert kwargs["max_tokens"] == 3
            assert self.context_budget_tokens == len(question) + 3
            yield "ok"

    backend = Backend()
    engine = SimpleNamespace(mlx_backend=backend, _context_budget_tokens=12)
    assert evaluation._answer(engine, "ask", 20, 2**32 + 7) == "ok"
    assert backend.context_budget_tokens == 70 and seeds == [7]


def test_worker_settings_cannot_inherit_user_database_or_adapter(monkeypatch, tmp_path):
    from config import settings as settings_module
    from core import pro_engine

    monkeypatch.setattr(settings_module, "_global_settings", settings_module._global_settings)
    state_directory = tmp_path / "isolated-state"
    monkeypatch.setenv("SMARTAI_DATA_DIR", str(state_directory))
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "user-memory.sqlite3"))
    monkeypatch.setenv("LORA_ADAPTER_PATH", str(tmp_path / "user-adapter.safetensors"))

    class Engine:
        def __init__(self, settings):
            self.settings = settings

        def load_model(self, *args, **kwargs):
            return {"status": "loaded", "backend": "mlx"}

    monkeypatch.setattr(pro_engine, "ProReasoningEngine", Engine)
    engine = evaluation._load_engine({"paths": {"bonsai2": str(tmp_path)},
                                      "context_tokens": 4096}, "bonsai2")
    assert engine.settings.database_path == str(state_directory / "evaluation.sqlite3")
    assert engine.settings.lora_adapter_path is None
