"""The two explicit, inference-only Strata model choices offered by Studio.

Pure data and validation: importing settings must not import a neural runtime.
Sources are the official Strata v0.2.8 catalog (setup.py at commit
870aedd9e1ebe1c91f94da93b59af17729f6c5c0) and the linked publisher model cards.
Download sizes are exact bytes; preparation and memory figures are estimates.
The upstream launch plan and running engine remain authoritative for a machine.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
from typing import Any


STRATA_RELEASE = "v0.2.8"
STRATA_SOURCE_COMMIT = "870aedd9e1ebe1c91f94da93b59af17729f6c5c0"
STRATA_KV_MODES = ("fp16", "int8", "k8v4", "q4_0")
STRATA_DEFAULT_CONTEXT = 32768

_COMMON = {
    "backend_family": "strata",
    "controller_runtime": "strata",
    "model_type": "text",
    "input_modalities": ["text"],
    "output_modalities": ["text"],
    "ternary": False,
    "training_supported": False,
    "supports_training": False,
    "inference_only": True,
    "training_label": "Inference only; no weight updates or LoRA training",
    "raw_params": 125_000_000_000,
    "ngram_params": 51_200_000_000,
    "parameter_count_scope": "Approximately 125B language-model parameters plus 51B n-gram embeddings; draft disabled",
    "base_params": "125B MoE + 51B n-gram",
    "max_context": 262144,
    "default_context": STRATA_DEFAULT_CONTEXT,
    "strata_kv_mode": "fp16",
    "est_speed": "Measured on your hardware",
    "storage_headroom_gb": 10.0,
    "model_license": "Qwen Community License 1.0",
    "source_kind": "strata_builtin",
    "strata_release": STRATA_RELEASE,
    "strata_source_commit": STRATA_SOURCE_COMMIT,
}

_PRESETS = {
    "strata_gyro_s": {
        **deepcopy(_COMMON),
        "id": "strata_gyro_s",
        "model_key": "gyro-s",
        "name": "Qwen3.8-Flash-Next Gyro-S (Strata; inference only)",
        "short_name": "Strata Gyro-S",
        "tag": "Strata · Gyro-S · inference only",
        "model_path": "strata://gyro-s",
        "native_strata_model": "Gyro-S",
        "strata_family": "gyro",
        "strata_api_model": "qwen3.8-flash-next-gyro-gyro-s",
        "repo_id": "agentionai/Qwen3.8-Flash-Next-Gyro-GGUF",
        "repo_revision": "1ada08422e404aca51ab9df665f089f34c419c6b",
        "precision": "Gyro APR trellis quantization (nonternary)",
        "vram": "GPU + RAM + SSD tiers; upstream plan determines fit",
        "download_bytes": 58_487_015_808,
        "download_gb": 58.487015808,
        "prepare_extra_gb_estimate": 25.8,
        "storage_gb_estimate": 58.487015808 + 25.8 + 10.0,
        "expert_arena_gb_estimate": 25.8,
        "storage_note": (
            "58.5 GB download, including the n-gram table. Low-RAM preparation can add about "
            "25.8 GB of expert files; allow another 10 GB for preparation. Draft and vision downloads are off."
        ),
        "memory_note": (
            "Uses Strata's GPU expert cache, RAM and SSD tiers. The 58.5 GB file need not fit "
            "in VRAM or RAM; speed depends on available memory and SSD reads."
        ),
        "model_card_url": "https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF",
        "strata_files": [{
            "path": "Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf",
            "bytes": 58_487_015_808,
            "sha256": "c6ea1b08ca312f99095b27588ba35976b53f948268e7f91e7761926aa01d9c87",
        }],
    },
    "strata_q2_0": {
        **deepcopy(_COMMON),
        "id": "strata_q2_0",
        "model_key": "q2_0",
        "name": "Qwen3.8-Flash-Next Q2_0 (Strata; inference only)",
        "short_name": "Strata Q2_0",
        "tag": "Strata · ISTA Q2_0 · inference only",
        "model_path": "strata://q2_0",
        "native_strata_model": "Q2_0",
        "strata_family": "qwen",
        "strata_api_model": "qwen3.8-flash-next-q2_0",
        "repo_id": "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
        "repo_revision": "ed59f92082b1e93c0e96d60a8b11aab089b52f09",
        "precision": "GSQ-RCO Q2_0 mixed quantization (2.40 bpw transformer; nonternary)",
        "vram": "GPU + RAM + SSD tiers; upstream plan determines fit",
        "download_bytes": 37_623_740_192 + 28_800_138_432,
        "download_gb": (37_623_740_192 + 28_800_138_432) / 1e9,
        "prepare_extra_gb_estimate": 40.0,
        "storage_gb_estimate": (37_623_740_192 + 28_800_138_432) / 1e9 + 40.0 + 10.0,
        "expert_arena_gb_estimate": 34.0,
        "storage_note": (
            "66.4 GB in two shards: 37.6 GB weights and 28.8 GB n-gram table. Low-RAM or AVX-512 "
            "preparation can add about 34–40 GB; allow another 10 GB. Draft and vision downloads are off."
        ),
        "memory_note": (
            "When the approximately 34 GB expert arena does not fit beside the OS, Strata maps "
            "experts from SSD or keeps a fitting complement in RAM. A small GPU can require more SSD reads."
        ),
        "model_card_url": "https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
        "strata_files": [
            {
                "path": "Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf",
                "bytes": 37_623_740_192,
                "sha256": "69820c02ec7d0b45ef2ebb19d6620299db749fe2aded7f39f93c6b88b199b720",
            },
            {
                "path": "Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf",
                "bytes": 28_800_138_432,
                "sha256": "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113",
            },
        ],
    },
}

# Consumers receive their own copies; registry mutations cannot broaden validation.
STRATA_PRESETS = deepcopy(_PRESETS)
_ALIASES = {
    alias: key
    for key, info in _PRESETS.items()
    for alias in (key, info["model_key"], info["model_path"])
}
_IDENTITY_FIELDS = (
    "id", "model_key", "model_path", "native_strata_model", "strata_family", "strata_api_model",
    "repo_id", "repo_revision", "backend_family", "controller_runtime", "model_type",
    "input_modalities", "ternary", "training_supported", "supports_training", "inference_only",
)


def get_strata_preset(model: str) -> dict[str, Any]:
    """Return a fresh registered preset by its Studio id, model key or URI."""
    if not isinstance(model, str) or model not in _ALIASES:
        raise ValueError("Choose the registered Strata Gyro-S or Q2_0 preset.")
    return deepcopy(_PRESETS[_ALIASES[model]])


def validate_strata_selection(selection: str | Mapping[str, Any]) -> dict[str, Any]:
    """Accept only the two registered identities; discard unrelated runtime extras.

    A model path/repository from an arbitrary custom GGUF cannot be relabelled as
    a supported Strata pack. Conflicting identifiers fail instead of selecting
    one opportunistically. A separately validated KV choice is retained.
    """
    if isinstance(selection, str):
        return get_strata_preset(selection)
    if not isinstance(selection, Mapping):
        raise ValueError("A registered Strata model selection is required.")
    keys = [selection[field] for field in ("id", "model_key", "model_path") if field in selection]
    if not keys:
        raise ValueError("Strata selection must name a registered id, model_key or model_path.")
    preset = get_strata_preset(keys[0])
    for field in _IDENTITY_FIELDS:
        if field in selection and selection[field] != preset[field]:
            raise ValueError(f"Strata {field} does not match the registered {preset['model_key']} preset.")
    if "strata_kv_mode" in selection:
        preset["strata_kv_mode"] = validate_strata_kv_mode(selection["strata_kv_mode"])
    return preset


def validate_strata_kv_mode(value: Any) -> str:
    if not isinstance(value, str) or value not in STRATA_KV_MODES:
        raise ValueError("Strata KV mode must be fp16, int8, k8v4 or q4_0.")
    return value


def strata_memory_policy(memory_limit_gb: float | None = None) -> dict[str, Any]:
    """Describe automatic tiering without inventing a process-memory guarantee.

    Upstream --memory-limit-mib describes container/system capacity, not Studio
    RSS. --resident-budget-gib caps only expert copies, omitting other runtime
    allocations. Neither is a safe translation of Studio's application cap.
    The managed adapter must monitor its owned process tree if that cap is on.
    """
    if memory_limit_gb is not None:
        if isinstance(memory_limit_gb, bool) or not isinstance(memory_limit_gb, (float, int)):
            raise ValueError("The Strata application memory limit must be a positive finite number or None.")
        try:
            memory_limit_gb = float(memory_limit_gb)
        except (ValueError, OverflowError):
            raise ValueError("The Strata application memory limit must be finite.") from None
        if not math.isfinite(memory_limit_gb) or memory_limit_gb <= 0:
            raise ValueError("The Strata application memory limit must be positive and finite.")
    return {
        "placement": "upstream_auto_gpu_ram_ssd",
        "expert_cache": "auto",
        "prefill": "auto",
        "application_limit_gb": memory_limit_gb,
        "requires_process_tree_watchdog": memory_limit_gb is not None,
        "hard_rss_cap": False,
        "note": "Strata chooses resident or mapped SSD experts; Studio's application cap needs its owned-process watchdog.",
    }


def strata_launch_options(
    selection: str | Mapping[str, Any], *, context_tokens: int,
    kv_mode: str | None = None, memory_limit_gb: float | None = None,
) -> list[str]:
    """Validated model/policy flags; the runtime adds private ownership and access.

    No shell text, model substitutions, arbitrary --set passthrough, or reduced
    context. The official planner chooses cache size, prefill and SSD placement.
    fp16 KV is the default; lower-precision KV requires an explicit selection.
    """
    info = validate_strata_selection(selection)
    if type(context_tokens) is not int or not 1024 <= context_tokens <= info["max_context"]:
        raise ValueError(f"Strata context must be an integer from 1024 to {info['max_context']} tokens.")
    kv_mode = validate_strata_kv_mode(info["strata_kv_mode"] if kv_mode is None else kv_mode)
    strata_memory_policy(memory_limit_gb)
    return [
        "--family", info["strata_family"], "--model", info["native_strata_model"],
        "--context", str(context_tokens), "--parallel", "1", "--vision", "off",
        "--kv", kv_mode, "--mtp", "off", "--drafting", "off",
        "--set", "spec=0", "--set", "STRATA_ROUTE_PRIOR=0",
        "--set", "STRATA_ROUTE_TAIL_SKIP=0", "--set", "STRATA_KVA=0",
    ]
