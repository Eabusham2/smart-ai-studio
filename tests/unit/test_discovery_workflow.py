"""PUCT reuse, fresh entropic rollout groups and validated adapter publication."""
import math
import threading

import pytest

from core.discovery_search import DiscoveryArchive
from core.online_consolidator import AwakeOnlineConsolidator, LearningValidationError
from core.test_time_learning import OnlineLearning
from tests.unit.context_backend_fixture import TransactionalBackend
from tests.unit.test_native_context_learning import tiny_controller


def test_ranked_puct_tracks_direct_child_quality_and_ancestor_visits_after_eviction():
    archive = DiscoveryArchive([("seed", .4)], capacity=3, exploration=1.0)
    assert archive.select()[0].id == 0
    assert archive.expand(0, [("mid", .6), ("poor", .2), ("worse", .1)]) == [1, 2]
    parent, info = archive.select()
    assert parent.id == 1 and info["q"] == .6 and info["prior"] == .5
    assert info["bonus"] == pytest.approx(.4 * .5 * math.sqrt(2))
    assert archive.expand(1, [("best", .9), ("better", .8)]) == [3, 4]
    assert archive.active == {0, 3, 4}
    assert archive.states[1].text is None
    assert archive.states[0].q == .6 and archive.states[1].q == .9
    archive.expand(3, [("next", .85)])
    assert [archive.states[i].visits for i in (0, 1, 3)] == [3, 2, 1]
    assert archive.states[3].q == .85 and archive.states[0].q == .6
    assert archive.expansions == 3 and 0 in archive.active


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    backend = TransactionalBackend(tmp_path)
    service = AwakeOnlineConsolidator(backend, lambda_ewc=0)
    runner = OnlineLearning(service)
    monkeypatch.setattr(runner, "_require", lambda method: None)
    monkeypatch.setattr("core.online_consolidator.release_training_memory",
                        lambda backend: {"after_mb": 0, "released_mb": 0})
    return runner


def test_discover_scores_fresh_groups_and_reuses_only_the_parent_candidate(workflow, monkeypatch):
    values = {"seed": .1, "A": .3, "B": .4, "C": .5, "D": .9}
    answers = iter(["A", "B", "C", "D"])
    prompts = []
    def generate(prompt, **kwargs):
        prompts.append(prompt)
        return next(answers)
    workflow.generate = generate
    monkeypatch.setattr(workflow.backend, "score_learning_probe",
                        lambda p, text: 8.0 - workflow.backend.weight * values.get(text, .01))
    result = workflow.discover("Optimize the function", seed="seed", verifier=values.__getitem__,
                               rounds=2, rollouts=2, archive_size=3, entropic_kl_target=.15)
    assert result["updates"] == 2 and result["best_sample"] == "D"
    assert result["reference_kl_coefficient"] == 0 and result["expansions"] == 2
    assert prompts[0] == prompts[1] and prompts[2] == prompts[3] and prompts[2] != prompts[0]
    assert "Previous candidate:\nB" in prompts[2]
    assert "0.4" not in str(prompts) and "D" not in prompts[2]
    assert [[r["completion"] for r in batch] for batch in workflow.backend.received_batches] == [["A","B"],["C","D"]]
    second = result["rounds"][1]["update"]
    assert second["reward_weighting"] == "entropic" and second["entropic_kl_achieved"] == pytest.approx(.15)


def test_discover_no_signal_and_cancel_do_not_mutate_parameters(workflow):
    workflow.generate = lambda *args, **kwargs: "some text"
    result = workflow.discover("task", verifier=lambda x: 0.0, rounds=2, rollouts=2)
    assert result["updates"] == 0 and result["status"] == "skipped" and result["expansions"] == 2
    assert all(r["reason"] == "uniform rollout rewards" for r in result["rounds"])
    stop = threading.Event()
    def verify(text):
        stop.set()
        return .3
    with pytest.raises(InterruptedError):
        workflow.discover("task", verifier=verify, cancel_event=stop)
    assert workflow.backend.train_calls == 0 and workflow.backend.learning_state.current_checkpoint() is None


def test_rejected_discovery_candidate_preserves_measured_solution(workflow, monkeypatch):
    answers = iter(["candidate one", "candidate two"])
    workflow.generate = lambda *args, **kwargs: next(answers)
    def reject(*args, **kwargs):
        assert kwargs["steps"] == 1 and kwargs["training_options"]["reward_weighting"] == "entropic"
        raise LearningValidationError("regressed on retention")
    monkeypatch.setattr(workflow.service, "train_pairs", reject)
    result = workflow.discover("task", verifier=lambda text: .9 if text.endswith("two") else .1,
                               rounds=1, rollouts=2)
    assert result["best_sample"] == "candidate two" and result["updates"] == 0
    assert "regressed" in result["rounds"][0]["reason"]


@pytest.mark.parametrize("tiny_controller", [96], indirect=True)
def test_real_discovery_gradient_on_fresh_cpu_rollouts(tiny_controller):
    import torch
    runner = OnlineLearning(AwakeOnlineConsolidator(tiny_controller, lambda_ewc=0))
    def reward(answer):
        return .1 + .3 * ("slate" in answer) + .2 * ("color" in answer) + .25 * ("red" in answer)
    with torch.random.fork_rng():
        torch.manual_seed(91)
        result = runner.run_request({"method": "discover", "prompt": "vault color", "verifier": reward,
                                    "rounds": 1, "rollouts": 8, "max_tokens": 2, "learning_rate": .002,
                                    "entropic_kl_target": .15})
    assert result["updates"] == 1, result["rounds"]
    update = result["rounds"][0]["update"]
    assert update["entropic_kl_achieved"] == pytest.approx(.15)
    assert update["parameter_drift_l2"] > 0 and update["reward_logprob_contrast_gain"] > 0
    assert result["best_reward"] == reward(result["best_sample"])
    current = tiny_controller.learning_state.current_manifest()
    assert current["metadata"]["metrics"]["reward_weighting"] == "entropic"
    tiny_controller.unload_model()
    assert tiny_controller.load_model() and tiny_controller.learning_state.current_manifest() == current


def test_tiny_continuous_reward_gap_still_runs_an_entropic_update(workflow, monkeypatch):
    answers = iter(["lower", "higher"])
    workflow.generate = lambda *args, **kwargs: next(answers)
    monkeypatch.setattr(workflow.backend, "score_learning_probe",
                        lambda prompt, text: 8.0 - workflow.backend.weight * (2 if text == "higher" else 1))
    result = workflow.discover("task", verifier=lambda x: .5 + 1e-10 * (x == "higher"),
                               rounds=1, rollouts=2, entropic_kl_target=.15)
    assert result["updates"] == 1
    assert result["rounds"][0]["update"]["entropic_kl_achieved"] == pytest.approx(.15)


@pytest.mark.parametrize("favor_loser", [False, True])
def test_entropic_validation_requires_relative_gain_on_preferred_actions(workflow, monkeypatch, favor_loser):
    answers = iter(["lower", "higher"])
    workflow.generate = lambda *args, **kwargs: next(answers)
    monkeypatch.setattr(workflow.backend, "score_learning_probe",
                        lambda prompt, text: 8.0 - workflow.backend.weight * (2 if favor_loser and text == "lower" else 1))
    result = workflow.discover("task", verifier=lambda x: .8 if x == "higher" else .2,
                               rounds=1, rollouts=2, entropic_kl_target=.15)
    assert result["updates"] == 0 and result["best_sample"] == "higher"
    assert "relative likelihood" in result["rounds"][0]["reason"]
    assert workflow.backend.learning_state.current_checkpoint() is None


def test_discovery_python_verifier_uses_measured_time_without_a_supplied_baseline(workflow, monkeypatch):
    pytest.importorskip("resource")
    correct = "def square(n):\n    return n * n"
    wrong = "def square(n):\n    return n"
    generated = iter([correct, wrong])
    workflow.generate = lambda prompt, **kwargs: next(generated)
    monkeypatch.setattr(workflow.backend, "score_learning_probe",
                        lambda prompt, text: 8.0 - workflow.backend.weight * (2 if text == correct else 1))
    result = workflow.discover("Implement square(n).", verifier={"kind": "python_tests", "function": "square",
                               "cases": [{"args": [4], "expected": 16}]}, rollouts=2, rounds=1)
    assert result["updates"] == 1 and result["best_sample"] == correct and result["runtime_reference_ms"] == 1
    elapsed = result["verifier_measurements"][0]["median_cases_ms"]
    assert result["best_reward"] == pytest.approx(1 / (1 + elapsed))
    assert result["rounds"][0]["rewards"][1] == 0
    assert all("16" not in str(row["prompt"]) for row in workflow.backend.received_batches[-1])


def test_float32_noise_cannot_certify_a_better_policy(workflow, monkeypatch):
    generated = iter(["winner", "other"])
    workflow.generate = lambda *args, **kwargs: next(generated)
    def score(prompt, text):
        if text in ("winner", "other"):
            tiny = 1e-8 if text == "winner" else 0
            return 4.13 - workflow.backend.weight * (.1 + tiny)
        return 3.3
    monkeypatch.setattr(workflow.backend, "score_learning_probe", score)
    result = workflow.discover("task", verifier=lambda text: .9 if text == "winner" else .1,
                               rounds=1, rollouts=2)
    assert result["updates"] == 0 and "relative likelihood" in result["rounds"][0]["reason"]
