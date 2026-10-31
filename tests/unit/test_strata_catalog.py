"""Registered Strata identities and launch policy without acquiring models."""
import pytest

from config.strata_models import (
    STRATA_PRESETS, get_strata_preset, strata_launch_options,
    strata_memory_policy, validate_strata_selection,
)


@pytest.mark.parametrize("key,preset_id,family,native,download", [
    ("gyro-s", "strata_gyro_s", "gyro", "Gyro-S", 58_487_015_808),
    ("q2_0", "strata_q2_0", "qwen", "Q2_0", 66_423_878_624),
])
def test_two_presets_have_exact_publisher_identity_and_honest_storage(key, preset_id, family, native, download):
    info = get_strata_preset(key)
    assert info == get_strata_preset(preset_id) == get_strata_preset("strata://" + key)
    assert info["strata_family"] == family
    assert info["native_strata_model"] == native
    assert info["download_bytes"] == download == sum(part["bytes"] for part in info["strata_files"])
    assert info["storage_gb_estimate"] > info["download_gb"] + 25
    assert info["repo_id"] in info["model_card_url"]
    assert len(info["repo_revision"]) == 40
    assert all(len(part["sha256"]) == 64 for part in info["strata_files"])
    assert info["ternary"] is False
    assert info["training_supported"] is False
    assert info["input_modalities"] == ["text"]
    assert "inference only" in info["name"]
    assert "nonternary" in info["precision"]


def test_mutating_registry_or_returned_metadata_cannot_change_registered_identity(monkeypatch):
    original = get_strata_preset("gyro-s")
    returned = get_strata_preset("gyro-s")
    returned["strata_files"][0]["path"] = "other.gguf"
    returned["input_modalities"].append("image")
    monkeypatch.setitem(STRATA_PRESETS, "strata_gyro_s", {"model_key": "foreign"})
    assert get_strata_preset("gyro-s") == original
    assert validate_strata_selection(original) == original


@pytest.mark.parametrize("bad", [
    "other", "Q2_0; command", "strata://unknown", None, {},
    {"model_key": "gyro-s", "repo_id": "someone/foreign"},
    {"model_key": "gyro-s", "model_path": "strata://q2_0"},
    {"model_key": "gyro-s", "native_strata_model": "Gyro-M"},
    {"model_key": "gyro-s", "strata_family": "coder"},
    {"model_key": "q2_0", "ternary": True},
    {"model_key": "q2_0", "training_supported": True},
    {"model_key": "q2_0", "controller_runtime": "gguf"},
])
def test_unknown_or_conflicting_models_do_not_gain_strata_admission(bad):
    with pytest.raises(ValueError):
        validate_strata_selection(bad)


@pytest.mark.parametrize("key", ["gyro-s", "q2_0"])
def test_launch_preserves_context_and_disables_output_altering_shortcuts(key):
    info = get_strata_preset(key)
    options = strata_launch_options(info, context_tokens=65536, memory_limit_gb=32)
    pairs = list(zip(options[::2], options[1::2]))
    assert ("--context", "65536") in pairs
    assert ("--family", info["strata_family"]) in pairs
    assert ("--model", info["native_strata_model"]) in pairs
    assert ("--kv", "fp16") in pairs
    assert ("--mtp", "off") in pairs
    assert ("--drafting", "off") in pairs
    assert ("--set", "spec=0") in pairs
    assert ("--set", "STRATA_ROUTE_PRIOR=0") in pairs
    assert ("--set", "STRATA_ROUTE_TAIL_SKIP=0") in pairs
    assert ("--set", "STRATA_KVA=0") in pairs
    assert not any("resident-budget" in value or "memory-limit-mib" in value for value in options)
    assert not any("expert-cache" in value or "prefill=" in value for value in options)


def test_explicit_kv_choice_is_respected_without_mutating_the_default():
    chosen = get_strata_preset("gyro-s")
    chosen["strata_kv_mode"] = "k8v4"
    assert "k8v4" in strata_launch_options(chosen, context_tokens=32768)
    assert "int8" in strata_launch_options(chosen, context_tokens=32768, kv_mode="int8")
    assert get_strata_preset("gyro-s")["strata_kv_mode"] == "fp16"


def test_unrelated_runtime_overrides_are_not_passed_to_the_official_cli():
    chosen = get_strata_preset("gyro-s")
    chosen.update({"extra": {"STRATA_ROUTE_PRIOR": "1.5"}, "strata_args": ["--access", "lan"]})
    assert strata_launch_options(chosen, context_tokens=32768) == strata_launch_options("gyro-s", context_tokens=32768)


@pytest.mark.parametrize("context", [True, 512, 262145, "32768", 32768.0, None])
def test_invalid_context_is_not_silently_reduced_or_coerced(context):
    with pytest.raises(ValueError):
        strata_launch_options("gyro-s", context_tokens=context)


@pytest.mark.parametrize("kv", ["q2", "auto", "fp16 --restart", 8, True])
def test_invalid_kv_cannot_inject_runtime_flags(kv):
    with pytest.raises(ValueError):
        strata_launch_options("q2_0", context_tokens=32768, kv_mode=kv)


@pytest.mark.parametrize("limit", [-1, 0, True, "32", float("nan"), float("inf"), 10**400])
def test_bad_memory_limits_fail_without_a_misleading_cap(limit):
    with pytest.raises(ValueError):
        strata_launch_options("q2_0", context_tokens=32768, memory_limit_gb=limit)


def test_small_ram_budget_does_not_reject_oversized_weights_or_promise_hard_cap():
    # Tiered SSD inference is the purpose of this backend; model bytes are not
    # an all-in-RAM admission test, even when the app's watchdog cap is smaller.
    for key in ("gyro-s", "q2_0"):
        assert strata_launch_options(key, context_tokens=32768, memory_limit_gb=8)
    policy = strata_memory_policy(32)
    assert policy["placement"] == "upstream_auto_gpu_ram_ssd"
    assert policy["requires_process_tree_watchdog"] is True
    assert policy["hard_rss_cap"] is False
    assert strata_memory_policy()["requires_process_tree_watchdog"] is False
