"""Repeat real tiny MLX updates and ensure discarded tensors/optimizers die.

The fixture runs native MLX autodiff on CPU or Metal, never a 27B download.
Weak references catch retained Python owners; allocator samples catch native
growth for this bounded workload. These are regression limits, not a proof
about all model shapes or the user's machine.
"""
import gc
import weakref

import pytest

from core.learning_state import bind_learning_state
from core.training_memory import release_training_memory
from tests.unit.test_mlx_learning_implementations import tiny_mlx


@pytest.mark.parametrize("mode", ["legacy", "managed", "hybrid"])
def test_repeated_consolidation_releases_old_parameters_and_optimizer(tiny_mlx, tmp_path, monkeypatch, mode):
    import mlx.core as mx
    import mlx.optimizers as optim

    owners = []
    original = optim.AdamW

    class TrackedAdamW(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            owners.append(weakref.ref(self))

    monkeypatch.setattr(optim, "AdamW", TrackedAdamW)
    backend = tiny_mlx()
    backend.learning_implementation = mode
    store = bind_learning_state(backend, model_info={"resolved_revision": "c" * 40}, state_root=tmp_path)
    model = backend.model
    rows = [{"prompt": "why?", "completion": "yes"}]
    samples = []
    for _ in range(12):
        previous = [weakref.ref(value) for value in backend.adapters.values()]
        candidate, drift = backend.train_mini_batch({}, rows, steps=1, lambda_ewc=0,
                                                    validation_callback=lambda _: True)
        assert drift > 0
        del candidate
        release_training_memory(backend)
        mx.synchronize()
        gc.collect()
        assert all(owner() is None for owner in owners)
        assert all(value() is None for value in previous)
        assert backend.model is model and backend.model.training is False
        samples.append(mx.get_active_memory())
    # Compilation/bootstrap is excluded; no per-cycle persistent tensor growth.
    assert max(samples[4:]) - min(samples[4:]) <= 64 * 1024
    assert store.current_manifest()["generation"] == backend._mlx_parent_generation


def test_rejected_consolidation_frees_candidate_and_preserves_other_model(tiny_mlx, tmp_path):
    import mlx.core as mx

    first, second = tiny_mlx(), tiny_mlx()
    first.model_path, second.model_path = "fixture/normal", "fixture/crack"
    for backend in (first, second):
        backend.learning_implementation = "hybrid"
        bind_learning_state(backend, model_info={"resolved_revision": "d" * 40}, state_root=tmp_path)
        backend.train_mini_batch({}, [{"prompt": "a", "completion": "b"}], steps=1, lambda_ewc=0)
    assert first.learning_state.directory != second.learning_state.directory
    second_checkpoint = second.learning_state.current_checkpoint()
    second_params = dict(second.adapters)
    first_checkpoint = first.learning_state.current_checkpoint()
    rejected = []

    def reject(candidate):
        rejected.extend(weakref.ref(value) for value in candidate.adapters.values())
        return False

    for _ in range(8):
        with pytest.raises(RuntimeError, match="rejected"):
            first.train_mini_batch({}, [{"prompt": "a", "completion": "c"}], steps=1,
                                   lambda_ewc=0, validation_callback=reject)
        release_training_memory(first)
        gc.collect()
        assert all(value() is None for value in rejected)
        assert first.learning_state.current_checkpoint() == first_checkpoint
        assert second.learning_state.current_checkpoint() == second_checkpoint
        assert all(mx.array_equal(value, second.adapters[key]).item() for key, value in second_params.items())


@pytest.mark.parametrize("mode", ["managed", "hybrid"])
@pytest.mark.parametrize("operation", ["awake", "verified_sleep"])
def test_public_consolidation_and_sleep_release_validation_context(tiny_mlx, tmp_path, mode, operation):
    import functools
    from types import SimpleNamespace
    import mlx.core as mx
    from core.online_consolidator import ManagedAwakeOnlineConsolidator, HybridAwakeOnlineConsolidator, LearningValidationError
    from memory.db import EpisodicMemoryDB

    backend = tiny_mlx()
    backend.learning_implementation = mode
    # A complete role-prefixed template provides an exact assistant boundary
    # for the public service's completion-only training and validation calls.
    backend.tokenizer.apply_chat_template = lambda turns, tokenize=False, add_generation_prompt=False: (
        "".join(turn["role"][0] + ":" + turn["content"] for turn in turns)
        + ("a:" if add_generation_prompt else ""))
    bind_learning_state(backend, model_info={"resolved_revision": "e" * 40}, state_root=tmp_path)
    cls = HybridAwakeOnlineConsolidator if mode == "hybrid" else ManagedAwakeOnlineConsolidator
    service = cls(backend, lambda_ewc=0)
    db = EpisodicMemoryDB(tmp_path / "episodes.db")
    original = backend.train_mini_batch
    callbacks, samples = [], []

    @functools.wraps(original)
    def tracked(*args, **kwargs):
        callbacks.append(weakref.ref(kwargs["validation_callback"]))
        return original(*args, **kwargs)

    backend.train_mini_batch = tracked
    rows = [{"prompt": "why?", "completion": "yes"}]
    accepted = 0
    for _ in range(5):
        previous = [weakref.ref(value) for value in backend.adapters.values()]
        checkpoint = backend.learning_state.current_checkpoint()
        result = None
        if operation == "awake":
            result = service.train_pairs(rows, mode="awake", steps=1, validation_rows=rows,
                                         training_options={"supervise_prompt": False})
        else:
            db.log_interaction("why?", "yes", verified_reward=1)
            try:
                result = service.sleep_verified_episodes(db, settings=SimpleNamespace(
                    ewc_lambda=0, episodic_replay_ratio=.25, consolidation_lr=.0001))
            except LearningValidationError:
                # Repeated rehearsal can stop improving this nine-token toy.
                # Rejected real training must roll back and release its context
                # just as an accepted update does; never weaken validation.
                assert backend.learning_state.current_checkpoint() == checkpoint
                assert db.fetch_surprise_replay_data()
            else:
                assert not db.fetch_surprise_replay_data()
        if result is not None:
            assert result["status"] == "success" and result["parameter_drift_l2"] > 0
            assert "ram_after_cleanup_mb" in result
            accepted += 1
        assert not service.is_consolidating
        gc.collect()
        mx.synchronize()
        assert all(ref() is None for ref in callbacks)
        if result is not None:
            assert all(ref() is None for ref in previous)
        else:
            assert {id(ref()) for ref in previous} == {id(value) for value in backend.adapters.values()}
        samples.append(mx.get_active_memory())
    assert 1 <= accepted == service.consolidation_count <= 5
    assert max(samples[2:]) - min(samples[2:]) <= 64 * 1024
