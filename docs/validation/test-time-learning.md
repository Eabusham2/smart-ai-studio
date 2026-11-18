# Online learning validation record

This record covers the expanded online workflows, packed Bonsai adapters, scoped memory, document bank and streamed transcript work on `feature/persistent-context-learning`, following `fe5ee40b4cf60b9cb9460b91bed00b774d33db4d`. Prepared architectures and offline meta-training remain excluded. Native backend coverage below distinguishes real small-model tensor tests from deployment protocol fixtures and full checkpoint measurements.

## Completed runs

| Run | Result | Measured duration |
| --- | --- | --- |
| Final unit suite after sequential Best Work and compatibility repairs, with bitsandbytes present; two GUI modules run separately | **1,467 passed, 2 skipped, 58 subtests passed** | 179.58 s |
| Final real Tk tests: sequential passes, controls, original handoffs, both Sleep sources, scope changes and archive paging | **11 passed** | 3.97 s |
| Previous expanded unit suite with bitsandbytes present | **1,435 passed, 2 skipped, 58 subtests passed** | 145.36 s |
| Previous real Tk tests: original handoff, paging/learning panel and unloaded-model Sleep dialog | **7 passed** | 2.37 s |
| Dedicated native-learning and release-gate follow-up, including real MLX CPU and Torch | **141 passed, 5 subtests passed** | 35.85 s |
| Existing offline live integration suite | **18 passed, 1 skipped** | 503.68 s |
| Final task/reference-policy follow-up | **80 passed** | 30.96 s |
| Final query geometry and device follow-up | **36 passed** | 19.19 s |
| Final MLX position-state isolation follow-up | **26 passed** | 4.36 s |
| Root routing/release/Windows-argument follow-up | **26 passed** | 4.16 s |
| Optional bitsandbytes CPU precision and native-parent follow-up | **26 passed** | 19.06 s |
| Cross-platform checkpoint/export flush and learning follow-up, with bitsandbytes present | **106 passed** | 30.11 s |
| Isolated Python reward worker, inherited limits, transient/sustained resident-memory excess, timeout and child reaping | **52 passed** | 17.33 s |
| Original direct Torch and Learn behavior preservation | **45 passed** | — |
| Public direct CUDA precision policy and existing Transformers placement | **29 passed** | 19.35 s |
| Compilation, CLI help and `git diff --check` | Passed | — |

Focused development runs overlap the broad suites and are not added to these totals. Follow-up focused checks cover final sampling-configuration and split-query geometry corrections. The separate GitHub workflow runs the six existing platform/Python lanes and mandatory native MLX CPU and Metal lanes, retains native compiler/artifact gates, and prevents a feature-branch test dispatch from publishing a release.

The full 27B Apple Silicon/Metal module is opt-in. CPU tests execute actual tensors, forward passes, gradients and checkpoint reloads on small locally constructed models; no model download or external search is required. The required CI MLX lanes assert the real device and reject skipped packed-model tests.

The first expanded broad pass found an unloaded-model Sleep panel regression and an outdated release-dependency assertion. Both were fixed before the green rerun. Independent review additionally fixed eager archive formatting, bounded recall cache keys/values, compressed-document indexing, transient recall-failure caching, duplicate GUI user turns and exact parent-generation checks. Existing entropy/Pro routing, temperature policy, tool handoff, 80%→60% consolidation, explicit legacy adapters, lossless runtime changes and native artifact provenance remain covered.

### Corrections found by platform CI and preservation review

Hosted branch CI found three production platform issues. CPU bitsandbytes could select an NF4 packing kernel that rejected a small GQA projection. The GGUF trainer now prefers resident CPU/MPS precision when it fits and verifies actual model projection geometry, quantization, prefill and input gradients before an optional NF4 route; supported CUDA QLoRA remains available. Windows rejected `fsync` on read-only artifact handles. Checkpoint and export staging now use writable non-truncating handles, with failures still preventing publication. macOS rejected small address-space/data limits before the Python reward task started. Its checked worker now has a separate supervisor that enforces a resident-memory budget through live RSS checks and mandatory per-worker kernel peak validation. Transient allocations can exceed a polling interval but cannot earn a reward when their recorded peak exceeds the budget. Linux retains its hard address-space cap, and inherited resource limits remain in force. The native CPU/Metal lanes also test excess allocations, timeout handling and child cleanup.

The older Python 3.10 dependency stack also exposed test assumptions: an undersized recurrent head was invalid for the MLX-LM 0.31.3 Metal kernel, and an independent Torch reference scorer constructed CPU IDs for an MPS model. Packed fixtures now use a supported 64-wide recurrent head, and the independent scorer follows the actual embedding/output devices. Deterministic CPU retention tests explicitly select CPU throughout reload; other tests retain their normal device selection. The original gradient, fixed-weight, recall and persistence assertions remain in force.

Review against the original default branch found that migrating direct Torch execution through the shared controller had lost real-logit entropy, caller/search `top_p`, continuation-only branch decoding and the original direct CUDA BF16 policy. These contracts are restored and tested through the public loader, including a cold project-scope switch and return. The dtype tests simulate CUDA availability to verify the loader's policy; they do not run CUDA tensor operations. Imported Transformers models retain their checkpoint dtype. Prism GGUF token counts again use actual worker tokenizer IDs instead of falling back to a character estimate. The MLX load receipt uses the same capability check as the learning panel, including read-only document scopes. Learn again trains the entire verified synthesized note, with its matching source quotation kept separately for validation. Evidence from an earlier note cannot replace a later note. Attachment previews skip leading whitespace, retain file boundaries and preserve mixed-folder media reporting without eagerly loading the full source.

### Sequential Best Work and additional compatibility repairs

Best Work now defaults to two sequential passes around the existing router, temperature ladder and selector. Tests execute two real CPU Transformers decoder calls, compare the second call's exact token input with the first sampled draft, and confirm that only the final sample is displayed. An installed Pro/GGUF fixture routes nine candidates followed by one refinement and excludes losing drafts from the next prompt. Tool calls execute only after the final selected pass; a real tool result starts another stage. Tests also retain the one-pass mode, cancellation and cleanup, duplicate-action protection, structured attachments, complete-context errors and actual all-pass token accounting. The real Tk test drives this pipeline from the GUI worker and exercises the pass control alongside the original branch inspector.

The additional preservation review restored the original verified-episode Sleep queue as a separate GUI choice from current-chat consolidation. Its production test performs a small real PEFT/Fisher update, checkpoints the selected scope, and acknowledges the SQLite rows after validation. Tests cover empty chat, concurrent consumers, uninterrupted chat writes, crash recovery after publication, source changes, validation failure and cancellation. Queue locking also covers the original standalone daemon. The older queue has no source-model identity; the panel explicitly names the selected target instead of assigning one retrospectively.

Standalone GGUF/BitNet `train(data)` again publishes to its historical deployment path; tests repeat real tiny parent-model updates and inject publication failure to check deployment, PEFT-factor and optimizer rollback. Conversion itself is simulated in these fixtures. Non-MLX completion-only defaults, explicit full-sequence supervision, explicit MLX `save_path` exports, partial Fisher/reference subsets, and live public Torch model/tokenizer handles are covered. Explicit online standalone parent resolution is tested with simulated downloads; managed transactions retain local lineage checks. Learn preserves the earlier 16,000-character attachment evidence budget, replacement decoding for legacy text attachments, and continued processing of readable lessons after a malformed file. Direct ingestion remains strict.

The comparison with `d3cead010685a04c95e97b2469c07fa3de627156` removes no tracked files. Four previous test names changed to reflect real factor-axis validation, explicit legacy imports and managed transactional state; their relevant behavior is tested under the current interfaces. Internals were refactored, so this is a behavioral compatibility claim supported by tests, not a claim that every older source line is unchanged.

The preceding branch revision `b345eb860c3ce926dc662bb58efd12cfd3f098c4` passed [all required CI jobs](https://github.com/Eabusham2/smart-ai-studio/actions/runs/44235167538), including 133 tests on native MLX CPU and 133 on native MLX Metal and both native C++ jobs. The sequential-pass and additional compatibility changes require a fresh CI run on their own commit before promotion.

## Environment

- Linux x86_64; no CUDA device and no Apple Metal runtime.
- Python 3.12.14, PyTorch 2.15.0+cpu, Transformers 5.23.0, PEFT 0.21.3, pytest 9.2.0.
- MLX 0.32.4 with the official CPU backend and MLX-LM 0.32.2. Local MLX tests use real native autodiff and serialization.
- GUI streaming checks use a separate Python 3.13 environment with threaded Tcl/Tk 8.6.16. The bundled Tcl 9 runtime in the CPU environment cannot perform the same cross-thread GUI handoff tests correctly; that environment issue was reproduced before these changes.
- Offline guards: `SMARTAI_TEST_OFFLINE=1`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `RUN_REAL_WEIGHTS_EVAL=0`.
- CPU test runs limit OpenMP/OpenBLAS threads to one. GUI tests use a private authenticated dummy display.

The suite reports existing Pydantic/pytest configuration deprecations and Transformers/PEFT compatibility warnings. These are not target-model quality measurements.

## What the tests establish

**Query adaptation.** Real CPU Llama, Mistral, Qwen2 and Qwen3 models produce matching fixed-cache span losses and full causal-forward losses. A hook verifies one context prefill and short adaptation spans. Non-query parameters remain byte-identical. Managed tests verify selective query optimizer-state resume, rejected/cancelled candidate rollback, checkpoint receipts and cold-reload scoring parity.

**Policy updates.** Small real Torch models raise rewarded output probability and lower rejected output probability. Uniform reward groups leave parameters unchanged. A workflow test samples real CPU model outputs, evaluates them, performs a PEFT policy update, measures positive rewarded-versus-rejected likelihood contrast and reloads the committed checkpoint. Native bridges pass the objective to their verified parent trainers and exercise deployment/rollback contracts.

**Discovery and task workflows.** Tests check entropic concentration, PUCT, configurable budgets and schedules, ARC reversible transforms/inverse voting, joint agent-trajectory learning and fresh synthesis. Real Torch and MLX runs sample exact action IDs, compare frozen initial-policy likelihoods with independent references, update actual parameters and restore views after faults. Audit sidecars are integrity-checked. These are raw model likelihoods of the exact actions; temperature-sampled proposals are reported separately. Uniform groups, worse relative preferences and floating-point noise cannot certify an update. The Python verifier handles large case programs through standard input while preserving its timeout/resource checks.

**Bonsai/CRACK contracts.** Real MLX CPU tests use packed uint32 matmuls, signed Hadamard rotation and hybrid Qwen3.5 decoder blocks for the two runtime layouts. They compare frozen-prefix query replay with a full forward pass, check query-only changes and frozen query gates, execute down-projection updates, verify bit-identical base tensors, and compare hot/cold activation. Existing head-only generations still restore with their original layout. These tests do not load either full 27B checkpoint.

**Persistence and isolation.** Tests cover source/tokenizer identity, model/session/project/document separation, corruption, stale parent refusal, cancellation before and after publication, pinned old document generations, independent request copies, optimizer and EWC restoration, request cleanup, streaming exclusion, and restoration failure. A real PEFT test reloads an older pinned generation, reproduces its behavior and continues learning from a scoped copy.

**Streaming and retention.** Tests visit a source directory containing 134 files, check Unicode and whitespace, resume/cancel across segments, verify bounded indexed cursors, reject fabricated evidence, keep unanswered queries out of few-shot targets and reject old-fact recall regressions. A real 96-token-context GPT-2/PEFT fixture learns a source-free "slate" answer, retains it after a later 200-token source update, and reproduces it after reload. Exact passage digests and independent grounded questions gate retirement; a valid update without complete generated recall keeps the original text. Explicit document blocks preserve surrounding instructions, while fenced/inline literal markup stays ordinary user text.

**Document memory routing.** Tests exercise full-source postings, compressed DOCX extraction, terms across UTF-8 read boundaries, source mutation/rollback, generation pinning, model gating, multi-memory synthesis, bounded cache invalidation, cancellation restoration and external parent-generation changes. Ordinary branch/tool/sampling arguments remain intact and original source text never enters the recall question or synthesized prompt.

**GUI and archives.** Real Tk tests verify that older messages are paged without being deleted, a new turn appears once after viewing an earlier page, and model work in the learning panel returns through the UI queue. Complete transcript export retains text omitted from a long-message preview. The existing Best Work pipeline still displays and executes only the selected tool handoff, with its original router decisions and thinking sections.

## Reproduce

With project dependencies and a working threaded Tk display:

```bash
SMARTAI_TEST_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
RUN_REAL_WEIGHTS_EVAL=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m pytest tests/unit -q
```

The existing multipass live fixtures legitimately take several minutes:

```bash
SMARTAI_TEST_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
RUN_REAL_WEIGHTS_EVAL=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m pytest tests/live -q --timeout=600
```

## Hardware work not executed here

The real Bonsai 2 and Bonsai 2 CRACK 27B checkpoints, Apple Silicon/Metal, native GGUF conversion/runtime artifacts, and BitNet quantization toolchain were unavailable on this host. Consequently this change has **no measured claim** about those models' final recall accuracy, speedup, peak Metal/GPU memory, or full native deployment quality. The runnable target evaluator and explicit backend capability checks remain part of the delivery. See [the Apple Silicon evaluation instructions](../parametric-memory.md#measure-recall-on-apple-silicon).
