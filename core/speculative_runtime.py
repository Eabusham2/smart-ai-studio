"""Model-free contracts for the optional Bonsai speculative native runtime.

The hashes identify the published September 18 heads and their unchanged PQ2_0
target, not an arbitrary similarly named GGUF.  Nothing here downloads weights,
imports an inference library, or infers compatibility from a filename.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional


TARGET_REPOSITORY = "prism-ml/Ternary-Bonsai-2-27B-gguf"
TARGET_REVISION = "6ed5e12bf84b7a63069882c91dd9e9218647d17b"
TARGET_SHA256 = "3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1"
MTP_SHA256 = "3cb3f0056d2e34ee44245a64396004a21f8492573d6ce1266ec4b7222c131dd4"
DFLASH2_SHA256 = "9dd11c8adb910058faf9fb77b10d90c1c048a4f3c2887a890f592cbd882deb9a"
SOURCE_ARCHIVE_SHA256 = "8c0f589673b25574f27f013bb3278824384eb35eb984034a2445af3c437b9d05"
EXACTNESS_CONTRACT = "smartai-serial-target-v1"

# These are provenance links, not download actions.
SOURCES = {
    "mtp": "https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-MTP/blob/main/manifest.json",
    "dflash2": "https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-DFlash2/blob/main/manifest.json",
    "runtime": "https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-DFlash2/blob/main/runtime/manifest.json",
    "exactness_inspiration": "https://github.com/ashhart/TensorFold/tree/cb2ebf0540f42604e2759b2ddef497861e928248",
}


def normalize_speculative_mode(mode: str) -> str:
    value = str(mode or "none").strip().lower()
    value = {"off": "none", "pld": "lookup", "dflash": "dflash2"}.get(value, value)
    if value not in {"none", "auto", "lookup", "mtp", "dflash2"}:
        raise ValueError("Speculative mode must be Off, Auto, Lookup, MTP, or DFlash2")
    return value


def sha256_file(path: str, cancelled: Optional[Callable[[], bool]] = None) -> str:
    """Hash local bytes with bounded memory, without deserializing any tensors."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            if cancelled is not None and cancelled():
                raise RuntimeError("Native artifact verification cancelled")
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class SpeculativePlan:
    requested_mode: str
    active_mode: str
    target_sha256: str
    draft_sha256: Optional[str]
    max_draft_tokens: int
    runtime_contract: str = EXACTNESS_CONTRACT

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def plan_speculation(
    mode: str, *, target_sha256: str, draft_sha256: Optional[str] = None,
    max_draft_tokens: Optional[int] = None,
) -> SpeculativePlan:
    """Select only compatible *already selected* target/head artifacts.

    Auto chooses among installed artifacts; it never converts, downloads, swaps
    the target, or combines independent speculative algorithms.
    """
    requested = normalize_speculative_mode(mode)
    if target_sha256 not in {TARGET_SHA256, MTP_SHA256}:
        raise ValueError("This runtime requires the pinned original Bonsai 2 PQ2_0 target or its unchanged-base MTP export")
    active = requested
    if active == "auto":
        active = "mtp" if target_sha256 == MTP_SHA256 else ("dflash2" if draft_sha256 else "lookup")
    if active == "mtp":
        if target_sha256 != MTP_SHA256:
            raise ValueError("MTP requires selecting the combined PQ2_0-MTP-Q8_0 GGUF; the target is never replaced automatically")
        if draft_sha256:
            raise ValueError("MTP is embedded in the selected GGUF and does not use an external draft model")
    if active == "dflash2":
        if target_sha256 != TARGET_SHA256 or draft_sha256 != DFLASH2_SHA256:
            raise ValueError("DFlash2 requires the pinned original PQ2_0 target and its matching ProCreations Q8_0 draft")
    elif active not in {"mtp", "none"} and draft_sha256:
        raise ValueError("Lookup does not use an external neural draft model")
    default_tokens = {"none": 0, "lookup": 4, "mtp": 2, "dflash2": 3}[active]
    if max_draft_tokens is None:
        tokens = default_tokens
    else:
        if isinstance(max_draft_tokens, bool) or not isinstance(max_draft_tokens, int) or not 1 <= max_draft_tokens <= 64:
            raise ValueError("Draft token count must be an integer from 1 through 64")
        tokens = max_draft_tokens if active != "none" else 0
    return SpeculativePlan(requested, active, target_sha256, draft_sha256, tokens)


def validate_runtime_receipt(server_path: str, receipt_path: Optional[str] = None,
                             cancelled: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
    """Bind a locally built runtime to the pinned source and exact verifier patch.

    A normal ProCreations archive has target verification but no row-consistency
    guarantee.  Its help output alone must never satisfy the exactness gate.
    The build helper writes this receipt only after applying our native patch.
    """
    server = Path(server_path).expanduser().resolve(strict=True)
    receipt = Path(receipt_path or (str(server) + ".smartai.json"))
    if not server.is_file():
        raise ValueError("The explicitly selected llama-server executable is missing")
    if receipt.stat().st_size > 64 * 1024:
        raise ValueError("Native runtime receipt exceeds the allowed size")
    with receipt.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("Unsupported native runtime receipt")
    if data.get("source_archive_sha256") != SOURCE_ARCHIVE_SHA256:
        raise ValueError("The runtime was not built from the pinned Bonsai MTP/DFlash2 source")
    if data.get("exactness_contract") != EXACTNESS_CONTRACT:
        raise ValueError("The runtime lacks the required serial target verification contract")
    patcher = Path(__file__).with_name("native_exactness.py")
    if data.get("patcher_sha256") != sha256_file(str(patcher), cancelled):
        raise ValueError("The runtime was built with a different exactness patch; rebuild it with this Studio version")
    if data.get("binary_sha256") != sha256_file(str(server), cancelled):
        raise ValueError("Native executable bytes changed since its runtime receipt was written")
    return data


def speculative_arguments(plan: SpeculativePlan, draft_path: Optional[str] = None) -> list[str]:
    """Native arguments; sampling and context settings are deliberately absent."""
    spec_type = {"none": "none", "lookup": "ngram-simple", "mtp": "draft-mtp", "dflash2": "draft-dflash"}[plan.active_mode]
    args = ["--spec-type", spec_type]
    if plan.active_mode != "none":
        args += ["--spec-draft-n-max", str(plan.max_draft_tokens), "--spec-draft-p-min", "0"]
    if plan.active_mode == "dflash2":
        if not draft_path:
            raise ValueError("The verified DFlash2 draft path is required")
        args += ["--model-draft", str(draft_path)]
    return args
