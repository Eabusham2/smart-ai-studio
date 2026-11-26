# Lossless execution speedups

Studio includes nine execution optimizations, with capability checks at the point where each optimization is used. Open **Speedups…** in the top bar to configure them. The selected target, full prompt/history, cache precision, temperature, top-p, and reasoning-temperature ladder remain authoritative.

The tests use synthetic tokenizers, cache arrays, HTTP responses, process handles, tiny synthetic neural fixtures, and an integer-only C++ fixture. CI also checks the patched native server on Linux CPU and macOS Metal and launches the pinned Bonsai 2.5 publisher executable with `--help`. **These checks do not load pretrained model weights and do not establish measured target-model output parity or a speed gain on a particular GPU.** Unsupported configurations report incompatibility; throughput depends on workload, cache hits, memory, and draft acceptance.

## Bonsai 2.5 defaults

The default main and CRACK presets are now [Bonsai 2.5 with MTP v2](bonsai25-models.md).
MLX attaches the verified publisher head; GGUF uses the combined PTQ1_0-MTP
artifact and a checksum-pinned publisher executable downloaded on first load.
**auto** selects MTP, **mtp** selects it explicitly, and **none** turns it off.
The multimodal processor/projector stays available for desktop media requests.

MLX MTP pauses exact prompt-state reuse, shared branch prefill, and persistent
prompt snapshots because the existing snapshots do not include draft/recurrent
state. Exact tokenizer memoization and supported lifecycle/memory controls remain
active. Turning MTP off restores the compatible ordinary cache paths. The nine
controls below are capabilities, not nine optimizations stacked on every model.

The original Bonsai 2 PQ2_0 path retains its separate strict patched verifier.
Its lookup/DFlash heads and target-row arithmetic contract do not apply to the
Bonsai 2.5 publisher runtime. New MTP output is target-verified; no claim of
bit-identical sampling across those two runtimes or a fixed speed increase is
made. Incompatible old draft/server choices produce an explicit error.

## What is included

| # | Optimization | Where it applies | Exactness and fallback |
|---|---|---|---|
| 1 | Exact prefix KV/state reuse | Supported MLX-LM native caches; native mlx-vlm APC for repeated identical prepared text prompts | Only complete saved state at the original prefill boundaries is reused. The final prompt token remains for normal next-token logits. Unknown layouts recompute. |
| 2 | Shared prompt prefill across Pro branches | The same supported MLX paths | Each branch receives isolated state from the prompt checkpoint, never a sibling branch's generated suffix or mutable cache. The existing temperature ladder and extra fixed branch are retained. |
| 3 | Tokenizer memoization | MLX text tokenization | Cache keys include tokenizer identity and configuration revision. Exact token tuples are stored; callers receive independent lists. No normalization or approximate matching. |
| 4 | Elimination of redundant cache resets | MLX request lifecycle | Completed request-owned state is released once; empty replacement caches are not repeatedly created and reset. Model changes, training, failure, cancellation and memory pressure still invalidate affected reusable state. |
| 5 | Persistent prompt state | Opt-in, supported native MLX-LM cache classes | Local safetensors plus checksummed JSON, exact artifact/tokenizer/adapter/runtime identity, atomic publication and bounded eviction. Unsupported or corrupt state is a cache miss. |
| 6 | GPU runtime warmup | Opt-in MLX GPU, at idle request boundaries | A tiny deterministic GPU primitive under the existing runtime lock. No model invocation, token generation or random-state consumption; no background loop. |
| 7 | Prompt lookup speculation | Explicit patched native runtime, supported Bonsai GGUF | The native ngram-simple proposer suggests tokens; the original target sampler must reproduce every accepted token. |
| 8 | Multi-token prediction (MTP) | Bonsai 2.5 MLX sidecar and combined MTP GGUF; original Bonsai 2 PQ2_0-MTP-Q8_0 on its separate runtime | Uses the actual compatible head and target verification. The selected target with drafting Off is the comparison baseline; runtime-specific exactness boundaries differ. |
| 9 | DFlash2 | Explicit original Bonsai PQ2_0 target plus its matching Q8_0 draft | Uses the real noncausal draft-dflash protocol. The target verifier uses the exact row policy described below; the separate draft remains batched. |

The first four are enabled by default when supported, subject to the MLX MTP cache restriction above. Persistent caching and GPU warmup are opt-in. Drafting defaults to **auto**; a saved **none** preference is preserved. Lookup, MTP and DFlash2 are alternative modes. Auto chooses a method supported by the selected model and runtime.

The older strict native drafting path is **text only** and rejects a vision projector. Its pinned PQ2_0/head files are not interchangeable with the new Bonsai 2.5 PTQ1_0/CRACK artifacts or arbitrary renamed files. The new 2.5 native path separately supports the pinned official projector.

## Original Bonsai 2 TensorFold-inspired target verification

TensorFold's frozen Python implementation is the source inspiration: commit [`cb2ebf0540f42604e2759b2ddef497861e928248`](https://github.com/ashhart/TensorFold/tree/cb2ebf0540f42604e2759b2ddef497861e928248), particularly its row-exact Qwen kernels, exact sampling, and [ternary Bonsai recipe](https://github.com/ashhart/TensorFold/blob/cb2ebf0540f42604e2759b2ddef497861e928248/docs/recipes/ternary-bonsai-2.md). Draft acceptance alone does not solve batch-dependent floating-point changes. TensorFold addresses arithmetic order, attention reductions, recurrent state and sampling together.

This port applies that principle conservatively to the pinned Prism/llama.cpp source. `core/native_exactness.py` is independently written; it does not transplant TensorFold's GPU kernels or claim their wide-batch performance.

The mandatory `smartai-serial-target-v1` build contract is:

1. **Identical target row arithmetic.** A target verification call requesting logits for every proposed token is split into one-token microbatches. The target therefore uses the same per-position graph shape as serial target decoding. Ordinary prompt prefill retains its normal chunking, and the separate noncausal DFlash drafter retains its batching.
2. **One owned generation slot.** The patched server rejects parallel slots. Cross-request batching cannot change the target graph shape.
3. **Exact target sampling.** The pinned verifier samples the target in position order and accepts a draft only when its token equals the target token. Target temperature, top-p, top-k, penalties, history and optional seed are preserved. Draft confidence does not override a target mismatch.
4. **Correct rollback.** The patch disables the bounded recurrent rollback tail (`n_rs_seq = 0`), which otherwise requires multiple trailing rows in one microbatch. The native full checkpoint path restores recurrent/KV state and the cloned sampler, including RNG, grammar and history, then replays the accepted prefix. Rejected speculative work cannot become the next request's state.
5. **Verified local build provenance.** Source archive, each edited source preimage, current patcher implementation and the compiled executable are bound by SHA-256 checks. Unpatched, stale or mismatched executables are rejected before a model is started.

The verifier may amortize host-side work, but it serializes target computation. **It can be slower than drafting Off**, especially with low acceptance or expensive checkpoints. Wider row-invariant GPU kernels would require a separate implementation and hardware validation. There is no synthetic or assumed speed multiplier in telemetry.

Exactness here is scoped to **the same selected target, patched runtime, hardware, context/prefill schedule, sampling settings and seed**, with drafting Off versus enabled. It does not promise byte-identical output across MLX and GGUF, different target exports, backends, device kernels or nondeterministic hardware configurations. Model-free tests cannot establish those comparisons.

## Prompt reuse and persistence

MLX-LM reuse recognizes exact native `KVCache`, `QuantizedKVCache`, and whole unbatched `ArraysCache` state. Existing quantized cache bytes are preserved without requantization; this feature does not introduce a lower-precision cache. TurboQuant, rotating/sliding, custom and unknown state layouts keep their existing full-prefill behavior. Cache-copy failures and invalid offsets also recompute.

A reusable prefix must match both exact token IDs and an original prefill checkpoint boundary. This avoids changing recurrent chunk execution when a new prompt merely shares some text with an old one. Retained state is bounded by the engine's token budget, at most two RAM entries and 256 MiB. The tokenizer memo is separately bounded to 128 entries, 65,536 token IDs and 8 MiB of accounted data.

For mlx-vlm, native APC schema 4 and the audited native KV/quantized-KV/Arrays layouts are required. Only the final prefill checkpoint for an **identical prepared text prompt and chunk schedule** is retained, at most one entry/256 MiB. Images, media changes, arbitrary extensions, unknown APC schemas and unsupported cache types use normal generation. No VLM state is written to disk by this feature.

Persistent MLX-LM storage defaults to 256 MiB and eight entries. It is off until selected in Speedups. The local cache contains prompt token IDs and model-derived state; choose its folder accordingly. The format uses safetensors and JSON, never pickle or metadata-selected imports. Checksums cover both payload and manifest. Full local weight/adapter content hashes bind snapshots to the exact loaded artifacts, with file-stability checks to prevent relabeling an old loaded model after a file is edited. First-use hashing has an I/O cost.

Model/tokenizer replacement, adapter changes, in-place training, unload and pressure boundaries invalidate reusable state. In-place training disables disk reuse for the active mutated model. A tokenizer implementation with private mutable configuration should call `invalidate_prompt_reuse()` or increment `_smartai_revision`; hidden mutations cannot be inferred safely.

GPU warmup runs a small deterministic primitive after idle when the runtime is available. It does not keep idle clocks high continuously, precompile model kernels, reserve a model or promise lower latency. Busy/missing-GPU/pressure/failure cases skip it.

## Using the original Bonsai 2 patched native runtime

Bonsai 2.5 uses its managed publisher executable instead; see the
[new runtime instructions](bonsai25-models.md#runtime-and-speedups). The steps
below preserve the original exactness-verified PQ2_0 implementation.

### Released builds

The rolling release includes separate `SmartAI-native-cpu-Linux-X64.zip` and `SmartAI-native-metal-macOS-ARM64.zip` assets when produced on those runner architectures. Names record the actual OS and architecture. Each contains the server, its `.smartai.json` receipt, this guide and upstream license. Use the asset matching your OS/architecture. Linux CPU builds use the pinned source's portable x86 instruction configuration; they are not a promise of support for every old CPU or Linux libc.

Extract the runtime locally. If the executable permission was lost, set it on `llama-server`. In **Speedups…**, choose the executable, select a native mode, and select the matching local draft file only for DFlash2. The receipt is discovered beside the server unless an explicit path is entered. Import/select the exact compatible target GGUF through the existing model picker, then load it. Studio never downloads, converts or switches targets as a consequence of enabling this setting.

Changing the server, target head configuration or draft length requires a model reload. Clearing the server path returns subsequent GGUF loads to the normal engine. Keeping the server selected with mode **none** gives the same patched target baseline without drafting.

The native server does not expose full-vocabulary model entropy. Set **Native branches** explicitly (default 1; 8 and 16 are also available). This count applies with drafting Off and on, and Pro keeps its existing temperature ladder plus the extra fixed 0.65 branch. Native automatic entropy routing is unavailable; it is not replaced by a prompt heuristic. Direct `solve()` callers must supply `force_branch_count` for this runtime. Native targets are inference only; automatic adapter consolidation is disabled and complete history is retained until the exact context limit, where overflow is reported.

### Build from reviewed source without running a model

Install a C++ toolchain and CMake. Obtain only this immutable **source** archive (about 36 MB):

```text
https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-DFlash2/resolve/4cfb6ad03268fed0f60ca96c1a659c0b1c77e50b/runtime/prism-dflash2-source.tar.gz
SHA256: 8c0f589673b25574f27f013bb3278824384eb35eb984034a2445af3c437b9d05
```

From the Studio source/package directory:

```bash
python tools/prepare_speculative_runtime.py \
  --source-archive /path/to/prism-dflash2-source.tar.gz \
  --output-dir /path/to/new-native-build \
  --backend metal --jobs 2
```

Use `--backend cpu` for the CPU build. CUDA/HIP toolchains are selectable but are not covered by the native compile matrix in this change. `--patch-only` validates and patches source without building or emitting a usable runtime receipt. `--output-dir` must be new; existing directories are never overwritten.

The helper downloads nothing and never starts the resulting server. It validates all source before editing, rejects unsafe archive entries, compiles only `llama-server`, and emits `build-smartai/bin/llama-server.smartai.json` (or beside the platform-specific `.exe`). OpenSSL and OpenMP dependencies are disabled for the local server; embedded Metal source is retained. Keep the binary and receipt together. A future patch implementation change requires a fresh matching build.

### Compatible artifact identities

Compatibility is checked by complete file contents, not filename. These are the supported published artifacts:

| Role | SHA-256 |
|---|---|
| Original PQ2_0 target | `3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1` |
| Combined PQ2_0-MTP-Q8_0 target | `3cb3f0056d2e34ee44245a64396004a21f8492573d6ce1266ec4b7222c131dd4` |
| Matching DFlash2 Q8_0 draft | `9dd11c8adb910058faf9fb77b10d90c1c048a4f3c2887a890f592cbd882deb9a` |

Publisher documentation: [original target](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/tree/6ed5e12bf84b7a63069882c91dd9e9218647d17b), [MTP](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-MTP), [DFlash2](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-DFlash2). Published warnings about ordinary batched verification are why the mandatory target-row patch exists. A matching hash proves artifact identity; it is not a substitute for a hardware output-equivalence test.

## Validation and release gate

The focused Python release gate runs with `SMARTAI_TEST_OFFLINE=1`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and `RUN_REAL_WEIGHTS_EVAL=0`. It covers tokenization identity, checkpoint boundaries, branch isolation, state corruption, exact context handling, cancellation before and during requests, owned-server teardown/retry, genuine native usage counters, preference persistence and GUI busy guards. External DNS and network connections are blocked inside the test process; synthetic localhost services remain available. The full six-lane suite remains available with `workflow_dispatch` → `full_tests`. A separate gate builds or verifies reused patched native artifacts on Linux CPU and macOS Metal without starting a model server.

The Python tests include small randomly initialized neural fixtures: a tiny GPT-2 for CPU/disk placement, a synthetic learning network, and synthetic LoRA layers. These quick fixtures remain enabled; the tests do not load or run the target Bonsai or other pretrained models. Their results do not establish target-model throughput or numerical equivalence.

Release packaging reuses compatible successful native build artifacts, validates their receipts against the packaged patch code, and verifies every uploaded runtime/application asset before moving the rolling `latest` tag. All application releases are model-free: no preset weights are downloaded or bundled by CI. Publisher-runtime smoke checks download only the small executable archive and invoke help; model weights are fetched only when selected in the app.

No real-model throughput, GPU kernel execution or numerical equivalence benchmark is claimed by this change. Unavailable acceptance/speed/entropy counters stay unavailable in the app. Cancellation cannot silently restart a request, and failed native shutdown retains the owner and model-file registration for a visible retry.
