"""The Studio verified-episode queue, its active adapter and recovery boundaries."""
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from core.online_consolidator import ManagedAwakeOnlineConsolidator, LearningValidationError
from memory.anchor_dataset import get_anchor_dataset
from memory.db import EpisodicMemoryDB
from tests.unit.context_backend_fixture import TransactionalBackend
from tests.unit.test_native_context_learning import tiny_controller


SETTINGS = SimpleNamespace(ewc_lambda=1., consolidation_lr=0.001, episodic_replay_ratio=.25)


def add(db, prompt, score, *, reward=1., completion="slate"):
    return db.log_interaction(prompt, completion, verified_reward=reward, surprise_score=score)


def test_sleep_uses_original_verified_queue_order_and_shared_scope(tmp_path):
    db = EpisodicMemoryDB(tmp_path / "episodes.sqlite")
    add(db, "least surprise", .01)
    add(db, "unverified", .99, reward=.5)
    add(db, "middle surprise", .20)
    add(db, "most surprise", .90)
    already = add(db, "previously consolidated", 1.)
    db.mark_consolidated([already])
    backend = TransactionalBackend(tmp_path / "weights")
    service = ManagedAwakeOnlineConsolidator(backend)
    assert service.pending_verified_episodes(db) == 3
    assert service.sleep([])["status"] == "idle"
    with patch.object(backend, "train_mini_batch", wraps=backend.train_mini_batch) as train:
        result = service.sleep_verified_episodes(db, settings=SETTINGS, limit=2)
    options = train.call_args.kwargs
    assert options["supervise_prompt"] is False and options["lambda_ewc"] == 1.
    assert len(options["anchor_texts"]) == len(get_anchor_dataset()) - 1
    assert options["data"][0]["prompt"] == "most surprise"
    assert options["data"][4]["prompt"] == "middle surprise"
    assert get_anchor_dataset()[2] not in options["data"]
    assert result["identity_digest"] == backend.learning_state.identity_digest
    assert result["scope"] == backend.learning_state.scope.record
    assert Path(result["adapter_saved_to"]).is_file() and result["parameter_drift_l2"] > 0
    assert service.pending_verified_episodes(db) == 1 and db.get_stats()["consolidation_cycles"] == 1
    assert [row["prompt"] for row in db.fetch_surprise_replay_data()] == ["least surprise"]
    assert service.sleep_verified_episodes(db, settings=SETTINGS)["memories_consolidated"] == 1
    assert service.sleep_verified_episodes(db, settings=SETTINGS)["status"] == "idle"


def test_interrupted_ack_recovers_scope_bound_receipt_after_later_update(tmp_path, monkeypatch):
    db = EpisodicMemoryDB(tmp_path / "episodes.sqlite")
    add(db, "old", .5)
    backend = TransactionalBackend(tmp_path / "weights")
    service = ManagedAwakeOnlineConsolidator(backend)
    with monkeypatch.context() as m:
        def interrupted(*args):
            raise InterruptedError("interrupted after durable commit")
        m.setattr(db, "finish_verified_sleep", interrupted)
        with pytest.raises(InterruptedError, match="after durable commit"):
            service.sleep_verified_episodes(db, settings=SETTINGS)
    assert db.get_stats()["consolidation_cycles"] == 0 and backend.train_calls == 1
    # The active reservation is excluded from the legacy standalone worker until acknowledged.
    assert not db.fetch_surprise_replay_data()
    add(db, "newer", .95)
    service.train_pairs([{"prompt": "q", "completion": "a"}])
    other = ManagedAwakeOnlineConsolidator(TransactionalBackend(tmp_path / "other-state"))
    with pytest.raises(RuntimeError, match="original model and memory scope"):
        other.sleep_verified_episodes(db, settings=SETTINGS)
    restored = ManagedAwakeOnlineConsolidator(TransactionalBackend(tmp_path / "weights"))
    recovered = restored.sleep_verified_episodes(db, settings=SETTINGS)
    assert recovered["recovered"] and recovered["memories_consolidated"] == 1
    assert restored.engine.train_calls == 0 and db.get_stats()["consolidation_cycles"] == 1
    assert [row["prompt"] for row in db.fetch_surprise_replay_data()] == ["newer"]


def test_concurrent_consumers_train_a_batch_once_and_release_writers(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    db = EpisodicMemoryDB(tmp_path / "episodes.sqlite")
    add(db, "once", .9)
    first = ManagedAwakeOnlineConsolidator(TransactionalBackend(tmp_path / "first"))
    second = ManagedAwakeOnlineConsolidator(TransactionalBackend(tmp_path / "second"))
    started, release = threading.Event(), threading.Event()

    def pause():
        started.set()
        assert release.wait(10)

    first.engine.before_validation = pause
    with ThreadPoolExecutor(max_workers=2) as pool:
        update = pool.submit(first.sleep_verified_episodes, db, settings=SETTINGS)
        try:
            assert started.wait(10)
            follow = pool.submit(second.sleep_verified_episodes, db, settings=SETTINGS)
            # A concurrent chat can log to the same database during gradient computation.
            add(db, "not verified", 1., reward=0.)
        finally:
            release.set()
        assert update.result()["memories_consolidated"] == 1
        assert follow.result()["status"] == "idle"
    assert first.engine.train_calls == 1 and second.engine.train_calls == 0


def test_mutated_episode_cannot_be_acknowledged_with_old_checkpoint(tmp_path):
    db = EpisodicMemoryDB(tmp_path / "episodes.sqlite")
    episode_id = add(db, "original", .9)
    backend = TransactionalBackend(tmp_path / "weights")

    def edit():
        with db._get_connection() as conn:
            conn.execute("UPDATE interactions SET completion='edited' WHERE id=?", (episode_id,))

    backend.before_validation = edit
    service = ManagedAwakeOnlineConsolidator(backend)
    with pytest.raises(RuntimeError, match="content changed"):
        service.sleep_verified_episodes(db, settings=SETTINGS)
    with pytest.raises(RuntimeError, match="content changed"):
        service.sleep_verified_episodes(db, settings=SETTINGS)
    assert backend.train_calls == 1 and service.pending_verified_episodes(db) == 1
    assert db.get_stats()["consolidation_cycles"] == 0


@pytest.mark.parametrize("cancel", [True, False])
def test_cancelled_or_rejected_sleep_releases_episodes(tmp_path, cancel):
    db = EpisodicMemoryDB(tmp_path / "episodes.sqlite")
    add(db, "keep", .99)
    backend = TransactionalBackend(tmp_path / "weights")
    event = threading.Event()
    if cancel:
        backend.before_validation = event.set
    else:
        backend.regress = True
    with pytest.raises(InterruptedError if cancel else LearningValidationError):
        ManagedAwakeOnlineConsolidator(backend).sleep_verified_episodes(db, settings=SETTINGS, cancel_event=event)
    assert db.get_stats()["consolidation_cycles"] == 0 and backend.learning_state.current_manifest() is None
    assert [row["prompt"] for row in db.fetch_surprise_replay_data()] == ["keep"]


@pytest.mark.parametrize("tiny_controller", [96], indirect=True)
def test_active_controller_verified_sleep_real_ewc_and_checkpoint(tiny_controller, tmp_path):
    db = EpisodicMemoryDB(tmp_path / "episodes.sqlite")
    for i in range(4):
        add(db, "What is the vault color?", .9 - i / 10, completion="slate")
    service = ManagedAwakeOnlineConsolidator(tiny_controller)
    result = service.sleep_verified_episodes(db, settings=SETTINGS)
    assert result["ewc_active"] and result["parameter_drift_l2"] > 0
    assert result["memories_consolidated"] == 4 and service.pending_verified_episodes(db) == 0
    manifest = tiny_controller.learning_state.current_manifest()
    assert manifest["metadata"]["mode"] == "verified_sleep"
    assert manifest["metadata"]["metrics"]["supervise_prompt"] is False
