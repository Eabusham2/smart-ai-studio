"""Bounded explicit adaptation, restart semantics, and answer-blind retirement gates."""
import json
import threading

import pytest

from core.context_memory import ChatArchive
from core.online_consolidator import ManagedAwakeOnlineConsolidator
from core.test_time_learning import OnlineLearning
from tests.unit.context_backend_fixture import TransactionalBackend
from tests.unit.test_native_context_learning import tiny_controller


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    monkeypatch.setattr("core.online_consolidator.release_training_memory",
                        lambda backend: {"after_mb": 0, "released_mb": 0})
    return OnlineLearning(ManagedAwakeOnlineConsolidator(TransactionalBackend(tmp_path / "state")))


def test_stream_resume_seeks_exact_utf8_boundaries_and_all_files(workflow, tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    texts = []
    for i in range(134):
        text = f"class café{i}:\r\n    名称 = 'octopus{i}'\n"
        (root / f"{i:03}.py").write_bytes(text.encode())
        texts.append(text)
    request = {"method": "ingest", "source": str(root), "chunk_chars": 32, "max_chunks": 3}
    first = workflow.run_request(request)
    assert first["status"] == "paused" and first["next"] == 3
    resumed = OnlineLearning(ManagedAwakeOnlineConsolidator(TransactionalBackend(tmp_path / "state")))
    del request["max_chunks"]
    second = resumed.run_request(request)
    assert second["status"] == "completed" and second["source_chars"] == sum(map(len, texts))
    assert second["files"] == 134
    batches = workflow.backend.received_batches + resumed.backend.received_batches
    assert "".join(batch[0]["completion"] for batch in batches) == "".join(texts)
    final = resumed.run_request(request)
    assert final["chunks_this_run"] == 0
    assert not any("[" + "...truncated" in str(batch) for batch in batches)


def test_cancel_and_resume_does_not_reapply_any_committed_source(workflow, tmp_path):
    path = tmp_path / "long.txt"
    text = "A sample with an apple. " * 17 + "Final evidence is forest."
    path.write_text(text)
    event = threading.Event()
    with pytest.raises(InterruptedError):
        workflow.ingest(path, chunk_chars=32, cancel_event=event, progress=lambda update: event.set())
    assert len(workflow.backend.received_batches) == 1
    result = workflow.ingest(path, chunk_chars=32)
    assert result["chunks_this_run"] == result["next"] - 1
    assert "".join(batch[0]["completion"] for batch in workflow.backend.received_batches) == text


def test_self_guided_span_budget_rejects_hallucinations_and_resumes_skips(workflow, tmp_path):
    path = tmp_path / "evidence.txt"
    first = "X" * 32
    last = "The label is green."
    path.write_text(first + last)
    seen = []
    def propose(prompt, **options):
        seen.append(prompt)
        return json.dumps({"spans": ["made up", last, "green", "green"]})
    workflow.generate = propose
    request = dict(source=path, strategy="evidence", question="Which label?", chunk_chars=32, span_chars=8)
    assert workflow.ingest(**request, max_chunks=1)["trained_chunks"] == 0
    result = workflow.ingest(**request)
    assert len(seen) == 2 and result["selected_chars"] == 5 and result["trained_chunks"] == 1
    assert workflow.backend.received_batches[0][0]["completion"] == "green"


def test_elicit_already_known_and_unverified_pairs_are_skipped(workflow, tmp_path):
    row = {"question": "What color is the vault?", "answer": "slate", "evidence": "vault is slate"}
    path = tmp_path / "knowledge.md"
    path.write_text("The vault is slate. " * 3)
    workflow.generate = lambda prompt, **kw: json.dumps(row) if "Elicit" in prompt else "slate"
    assert workflow.ingest(path, strategy="elicitation")["trained_chunks"] == 0
    workflow.generate = lambda prompt, **kw: json.dumps(row) if "Elicit" in prompt else "unknown"
    result = workflow.ingest(path, strategy="elicitation", question="vault")
    assert result["trained_chunks"] == 1 and workflow.backend.received_batches[-1][0] == {
        "prompt": row["question"], "completion": row["answer"]}


def test_few_shot_holds_out_each_demonstration_and_unanswered_query(workflow):
    prompts = []
    def generate(prompt, **kw):
        prompts.append(prompt)
        return "third"
    workflow.generate = generate
    demos = [{"prompt": "apple", "completion": "first"}, {"prompt": "plum", "completion": "second"}]
    outcome = workflow.few_shot(demos, "peach", votes=5)
    assert outcome["votes"] == {"third": 2} and outcome["training_rows"] == 2
    first, second = workflow.backend.received_batches[0][:2]
    assert "first" not in first["prompt"] and "second" not in second["prompt"]
    assert all("peach" not in str(row) for row in workflow.backend.received_batches[0])
    assert len(set(prompts)) == 2


def test_consensus_and_task_verifier_never_enter_generation_prompt(workflow, monkeypatch):
    monkeypatch.setattr(workflow, "_require", lambda objective: None)
    answers = iter(["x", "y", "x", "y", "z", "x", "y", "y"])
    prompts = []
    def generate(prompt, **kwargs):
        prompts.append(prompt)
        return next(answers)
    workflow.generate = generate
    assert workflow.reinforce("solve", mode="consensus", rollouts=4)["updates"] == 0
    def score(prompt, completion):
        return 5.0 - workflow.backend.weight * (2 if completion == "z" else 1)
    monkeypatch.setattr(workflow.backend, "score_learning_probe", score)
    result = workflow.reinforce("solve", mode="verifier", verifier={"kind": "exact", "expected": "z"}, rollouts=4)
    assert result["updates"] == 1 and result["reward_source"] == "task_verifier"
    rows = workflow.backend.received_batches[-1]
    assert [r["reward"] for r in rows] == [1.0, 0.0, 0.0, 0.0]
    assert prompts == ["solve"] * 8 and all(row["prompt"] == "solve" for row in rows)


def test_full_reconstruction_guards_each_exchange_and_earlier_receipts(workflow, tmp_path):
    archive = ChatArchive(str(tmp_path / "chat.sqlite3"), "a")
    def append(index):
        archive.append({"role": "user", "content": f"The archive label for sample {index} is cedar{index}."})
        archive.append({"role": "assistant", "content": f"I recorded sample {index} as cedar{index}."})
    append(0)
    service = workflow.service
    assert service.sleep(archive)["turns_retired"] == 2
    original_generate = workflow.backend.stream_generate_tokens
    def forget_new_turn(prompt, **kwargs):
        if "Quote the entire original assistant" in prompt and "sample 1" in prompt:
            yield "I recorded something."
        else:
            yield from original_generate(prompt, **kwargs)
    workflow.backend.stream_generate_tokens = forget_new_turn
    append(1)
    result = service.sleep(archive)
    assert result["status"] == "retained" and result["parameter_drift_l2"] > 0
    assert service.active_history(archive) == archive[2:]
    pointer = (workflow.backend.learning_state.directory / "current.json").read_bytes()
    workflow.backend.stream_generate_tokens = lambda *args, **kwargs: iter(["unknown"])
    with pytest.raises(RuntimeError, match="previously retired"):
        service.sleep(archive)
    assert (workflow.backend.learning_state.directory / "current.json").read_bytes() == pointer


def test_candidate_distillation_preserves_distinct_consensus_traces(workflow):
    generated = iter(["A calculation.\nFinal answer: 36", "Another route.\nFinal answer: 36",
                      "A guess.\nFinal answer: 20", "A fresh synthesis: 36"])
    workflow.generate = lambda prompt, **kw: next(generated)
    result = workflow.consolidate_candidates("Square six", rollouts=3)
    assert result["synthesized"] and result["answer"] == "A fresh synthesis: 36"
    data = workflow.backend.received_batches[-1]
    assert result["updates"] == 1 and result["reward_source"] == "model_consensus"
    assert len([r for r in data if r["prompt"] == "Square six"]) == 2
    assert all("20" not in r["completion"] for r in data)
    assert all("reward" not in r for r in data)


@pytest.mark.parametrize("tiny_controller", [96], indirect=True)
def test_real_cpu_rollouts_and_verifier_policy_checkpoint(tiny_controller):
    import torch
    service = ManagedAwakeOnlineConsolidator(tiny_controller, lambda_ewc=0)
    workflow = OnlineLearning(service)
    with torch.random.fork_rng():
        torch.manual_seed(91)
        result = workflow.run_request({"method": "reinforce", "prompt": "vault color", "rollouts": 8,
            "max_tokens": 2, "learning_rate": 0.002, "mode": "verifier",
            "verifier": lambda answer: float(any(x in answer for x in ("slate", "red", "color")))})
    assert result["updates"] == 1
    update = result["rounds"][0]["update"]
    assert update["parameter_drift_l2"] > 0 and update["reward_logprob_contrast_gain"] > 0
    expected = tiny_controller.learning_state.current_manifest()
    tiny_controller.unload_model()
    assert tiny_controller.load_model() and tiny_controller.learning_state.current_manifest() == expected


def test_replay_bank_is_bounded_and_follows_seeded_branch(tmp_path):
    from core.learning_recall import ReplayBank
    parent = tmp_path / "parent"
    child = tmp_path / "child"
    parent.mkdir()
    child.mkdir()
    bank = ReplayBank(parent)
    def row(i):
        return {"prompt": "question", "completion": str(i)}
    for i in range(65):
        bank.stage(str(i), str(i - 1) if i else None, [row(i)])
    bank.stage("sibling", "20", [row("sibling")])
    restored = ReplayBank(child)
    restored.inherit(parent, "sibling", None)
    for i in range(16):
        sample = restored.sample("sibling", str(i))
        assert row(0) in sample and row("sibling") in sample and row(64) not in sample
        assert len(sample) <= 3 and sample == bank.sample("sibling", str(i))
    with pytest.raises(RuntimeError, match="missing"):
        restored.sample("64", "any")


def test_scope_directory_inside_source_never_becomes_training_text(workflow, tmp_path):
    path = tmp_path / "example.txt"
    path.write_text("A single verified source.", encoding="utf-8")
    result = workflow.ingest(tmp_path)
    assert result["files"] == 1 and result["trained_chunks"] == 1


def test_python_verifier_checks_only_candidate_training_cases_and_reports_timing(workflow):
    pytest.importorskip("resource")
    correct = "def square(value):\n    return value * value"
    wrong = "def square(value):\n    return value + value"
    generated = iter([correct, wrong, correct])
    seen = []
    def generate(prompt, **kwargs):
        seen.append(prompt)
        return next(generated)
    workflow.generate = generate
    result = workflow.consolidate_candidates("Implement square(value).", mode="verifier",
        verifier={"kind": "python_tests", "function": "square", "cases": [{"args": [7], "expected": 49}],
                  "baseline_ms": 1}, rollouts=2)
    measured = result["verifier_measurements"]
    assert measured[0]["passed"] and measured[0]["median_cases_ms"] >= 0 and not measured[1]["passed"]
    assert 0 < result["rounds"][0]["rewards"][0] <= 1 and result["rounds"][0]["rewards"][1] == 0
    assert seen == ["Implement square(value)."] * 3
    assert workflow.backend.received_batches[0][0] == {"prompt": seen[0], "completion": correct}


def test_python_verifier_rejects_unbounded_program_shapes_and_data():
    from core.learning_code_verifier import PythonTaskVerifier, pure_function
    assert not pure_function("def f(a):\n    import sys\n    return a", "f")
    assert not pure_function("def f(a):\n    return a.__class__", "f")
    assert not pure_function("def f(a):\n    return open(a)", "f")
    with pytest.raises(ValueError, match="at least one"):
        PythonTaskVerifier({"function": "f", "cases": []})


def test_source_cursor_seeding_rollback_and_many_job_manifest_size(workflow, tmp_path):
    from core.learning_recall import StreamCursorBank
    store = workflow.backend.learning_state
    for i in range(9):
        source = tmp_path / f"source{i}.txt"
        source.write_text(f"The source value is word{i}.")
        workflow.ingest(source)
    memory = store.current_manifest()["metadata"]["memory"]
    assert "streams" not in memory and len(memory["stream_head"]) == 64
    parent = StreamCursorBank(store.directory)
    head = parent.record(memory["stream_head"], "job", {"next": 4})
    fork = parent.record(head, "job", {"next": 5})
    other = parent.record(head, "job", {"next": 55})
    child = tmp_path / "request"
    child.mkdir()
    seeded = StreamCursorBank(child)
    seeded.inherit(store.directory, fork)
    assert seeded.lookup(head, "job") == {"next": 4}
    assert seeded.lookup(fork, "job") == {"next": 5}
    assert parent.lookup(other, "job") == {"next": 55}
    with pytest.raises(RuntimeError, match="missing"):
        seeded.lookup(other, "job")
