"""Pinned, auditable native target-verification patch; never loads neural weights.

TensorFold's row decoders keep target arithmetic independent of verification
width. This independently written conservative port applies that invariant at
llama.cpp's microbatch boundary: one output row per target microbatch, while
ordinary prompt prefill and the separate noncausal drafter retain batching.
Target sampling and recurrent checkpoint rollback remain in the pinned native
runtime. Wider row-invariant GPU kernels are not claimed by this patch.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


CONTRACT = "smartai-serial-target-v1"
SOURCE_ARCHIVE_SHA256 = "8c0f589673b25574f27f013bb3278824384eb35eb984034a2445af3c437b9d05"
SOURCE_SHA256 = {
    "include/llama.h": "97b83e4fccf5196997cf0c0ad01311a9ac1945b5a3432c0c159a8a723a4e6fc2",
    "src/llama-context.h": "f0b65f44de53263b78a8779b9ee79ade212ce2e15514f24dc578e2500c1ebd6f",
    "src/llama-context.cpp": "c707962bf66ae0b7c9c34f28ab6822de74d907a07eb15726e262abffef229d94",
    "tools/server/server-context.cpp": "a79cf9e1d4a8d7c1f0ee608aa781628db403e8f59e25e731f997d0952d230e47",
    "common/common.cpp": "745220d782c0fc771300346a379a0320829855083b084b792dff19e23db85b68",
}

MICROBATCH_HEADER = '''// Smart AI Studio: independently written row-invariance fallback.
// Inspired by TensorFold cb2ebf0540f42604e2759b2ddef497861e928248.
#pragma once
#include <cstdint>

// Only the target is opted in. Draft contexts can require noncausal batching.
// A normal prefill requests only its final logits and retains its batch size.
constexpr uint32_t smartai_target_microbatch(
        bool target, uint32_t tokens, uint32_t outputs, uint32_t configured) {
    return target && tokens > 0 && outputs == tokens ? 1u : configured;
}
'''


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _replace_once(source: str, before: str, after: str) -> str:
    if source.count(before) != 1:
        raise ValueError("The pinned native patch anchor does not match exactly once")
    return source.replace(before, after, 1)


def patched_sources(source_root: str | Path) -> dict[str, str]:
    """Validate every preimage before returning any edited source; no writes."""
    root = Path(source_root)
    sources = {}
    for name, expected in SOURCE_SHA256.items():
        path = root / name
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Native source mismatch: {name}; use the pinned source archive")
        sources[name] = path.read_text(encoding="utf-8")

    name = "include/llama.h"
    sources[name] = _replace_once(sources[name],
        "    // Set the number of threads used for decoding\n",
        "    // Smart AI Studio addition: target-only, row-invariant verification.\n"
        "    LLAMA_API void llama_smartai_serial_target_v1(struct llama_context * ctx);\n\n"
        "    // Set the number of threads used for decoding\n")

    name = "src/llama-context.h"
    sources[name] = _replace_once(sources[name],
        "    void synchronize();\n",
        "    void synchronize();\n\n"
        "    // Smart AI Studio addition; never enabled on draft/MTP contexts.\n"
        "    void enable_smartai_serial_target() { smartai_serial_target = true; }\n"
        "    bool smartai_serial_target = false;\n")

    # The native recurrent splitter keeps its bounded rollback tail in one
    # microbatch. Disable that optimization in this dedicated build so a
    # one-row target uses the existing full-checkpoint restore/replay path.
    # The draft path already specifies zero in the pinned source.
    name = "common/common.cpp"
    sources[name] = _replace_once(sources[name],
        "    cparams.n_rs_seq          = params.speculative.need_n_rs_seq();",
        "    // Smart AI Studio: full recurrent checkpoints preserve one-row verification.\n"
        "    cparams.n_rs_seq          = 0;")

    name = "src/llama-context.cpp"
    sources[name] = _replace_once(sources[name], '#include "llama-context.h"\n',
        '#include "llama-context.h"\n#include "smartai-target-microbatch.h"\n')
    sources[name] = _replace_once(sources[name],
        "        mctx = memory->init_batch(*balloc, cparams.n_ubatch, output_all);",
        "        // Smart AI Studio: keep each verification row on the serial\n"
        "        // target graph. This does not change the separate draft context.\n"
        "        const uint32_t target_ubatch = smartai_target_microbatch(\n"
        "                smartai_serial_target, n_tokens_all, n_outputs_all, cparams.n_ubatch);\n"
        "        mctx = memory->init_batch(*balloc, target_ubatch, output_all);")
    sources[name] = _replace_once(sources[name],
        "int32_t llama_decode(\n",
        "// Smart AI Studio target-only verification contract: " + CONTRACT + ".\n"
        "void llama_smartai_serial_target_v1(llama_context * ctx) {\n"
        "    if (ctx != nullptr) ctx->enable_smartai_serial_target();\n"
        "}\n\nint32_t llama_decode(\n")

    name = "tools/server/server-context.cpp"
    sources[name] = _replace_once(sources[name],
        "        vocab = llama_model_get_vocab(model_tgt);\n",
        "        // Smart AI Studio addition. The exact baseline and speculative\n"
        "        // paths share one target context and identical one-row graphs.\n"
        "        // Reject cross-stream batching; do not opt the draft context in.\n"
        "        if (params_base.n_parallel != 1) {\n"
        "            SRV_ERR(\"%s\", \"smartai-serial-target-v1 requires --parallel 1\\n\");\n"
        "            return false;\n"
        "        }\n"
        "        llama_smartai_serial_target_v1(ctx_tgt);\n"
        "        SRV_INF(\"%s\", \"smartai-serial-target-v1: target rows are serial; draft batching retained\\n\");\n\n"
        "        vocab = llama_model_get_vocab(model_tgt);\n")
    sources["src/smartai-target-microbatch.h"] = MICROBATCH_HEADER
    return sources


def apply_native_patch(source_root: str | Path) -> dict[str, str]:
    """Apply to a disposable extracted tree after validating every preimage."""
    root = Path(source_root)
    sources = patched_sources(root)
    for name, source in sources.items():
        (root / name).write_text(source, encoding="utf-8", newline="\n")
    return {name: sha256_file(root / name) for name in sources}
