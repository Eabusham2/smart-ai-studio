# Smart AI Studio — Complete Project Documentation

**Repository:** `Eabusham2/smart-ai-studio`  
**Primary development branch:** `fix/real-benchmarks-final-32k`  
**Canonical historical base:** `master` at `06390e5357a07f80a8089ac28fc16c75461a48a6`  
**Latest implementation head documented:** `622481b918a5bc80f0860d9729444184fbb5934b`  
**Documentation consolidation:** 2026-10-04

This is the single canonical project document for the current Smart AI Studio feature line. It preserves the useful project architecture, behavior, invariants, runtime call chains, fixes, verification boundaries, packaging state, and operational requirements in one place.

## 1. Project purpose and operating rules

Smart AI Studio is a cross-backend AI application and evaluation/training system built around real model execution, real benchmark sources, explicit Learn and RSI flows, transactional model updates, and a canonical evaluation runner shared by CLI and desktop-app Eval.

Project rules that remain authoritative:

- preserve newer working behavior and make surgical changes rather than broad rewrites;
- do not replace an entire current file just because an older version has one useful behavior;
- port only the useful isolated delta from older/diverged work;
- hidden benchmark answers, rewards, expected outputs, and verifier results must not be fed into model reasoning;
- a successful function return is not proof that learning occurred;
- learning success requires real parameter/artifact change and persistence appropriate to the backend;
- failure/cancellation must not leave partially committed learned state;
- source/contracts and real target-hardware execution are different evidence levels;
- do not claim a workflow, build, release, benchmark run, or backend is green unless the corresponding evidence actually exists.

## 2. Current branch / implementation state

The implementation line through `622481b9` contains 151 linear feature commits above the canonical master baseline, with no feature-line merge commits. Later commits reorganize documentation only.

The implementation evolved through these major phases:

1. bounded training and memory safety;
2. cross-platform desktop Eval integration;
3. answer-blind RSI and isolated learned state;
4. transactional MLX/GGUF/BitNet/controller training;
5. real benchmark-source and strict-scoring hardening;
6. current-head audit/reconciliation and bounded Phase 3B repair;
7. standalone Bonsai 2 / Prism runtime alignment;
8. packaged Bonsai 2 model bundling for macOS;
9. persistent Eval-console restoration and canonical-runner contract locking.

The standalone 4K Eval target is **Bonsai 2**, not the short-lived historical Qwen target.

## 3. Post-audit implementation additions (commits 139–151)

After the older audit snapshot, current intended implementation added:

- packaged Bonsai model snapshot resolution;
- recognition of bundled release models as already installed;
- use of bundled Bonsai 2 MLX snapshots when available;
- streamed model-bundled macOS release support;
- `build_bonsai2_bundled_zip.py`;
- packaging of both Bonsai 2 MLX models in the macOS release path;
- a bundled-release contract;
- repaired streamed release workflow;
- packaged standalone 4K Eval using the bundled Bonsai 2 model when present;
- saved Eval-console restoration on window reopen;
- streaming continuation after restoration without duplicate rereads;
- a contract that the app Eval subprocess imports the canonical `master_4000_eval_suite.py` rather than maintaining a stale copied evaluator.

## 3. Current intended behavior

### 3.1 Chat / Pro temperature policy

- Normal chat N=1: **T=0.65**.
- Eval single-pass: **T=0.55**.
- Pro N>1: **0.20 → 0.95**, gamma **1.35**, plus the extra fixed **T=0.65** candidate.
- Older greedy and older 0.88-max policies are historical and must not silently replace these values.

### 3.2 Context / KV policy

- The top app has exactly one **Context Limit** control.
- One Context budget means **packed prompt/history + generated output**.
- The same context budget is now applied across MLX, GGUF/Prism, BitNet, and controller text backends.
- Eval and normal chat refuse silent truncation when the packed prompt already exceeds the available physical context.
- Current eval RSI/Phase-4 memory handling keeps **full-precision KV** and reduces prefill size on Metal OOM rather than switching to the old 4-bit/sliding-KV branch.
- Old lossy H2O/context-dropping behavior remains retired.
- App TurboQuant remains a separate accepted approximate cache path where supported; it is not used as a hidden eval fallback.

### 3.3 Main app top controls

Exactly one of each:

- **Eval** (text models only)
- **Context Limit**
- **Mem Limit** checkbox + GB value

The Eval page has its own eval-process memory watcher; it does **not** duplicate the normal app Context/Mem Limit controls.

### 3.4 Canonical Eval GUI

The desktop Eval UI is a launcher/monitor around the existing canonical runner:

`Eval button → eval/app_eval_runner.py → master_4000_eval_suite.py → Master4000EvaluationEngine.run_full_suite()`

It is **not** a second rewritten benchmark suite.

Features:

- text models only;
- dedicated Toplevel Eval window;
- live raw output console;
- elapsed/state/RAM/peak RAM/CPU telemetry;
- optional process-tree memory watcher with user-selected GB threshold;
- Pause/Resume at safe boundaries;
- confirmed Cancel;
- cancellation during model load;
- process-tree cleanup on app close;
- isolated eval checkpoints/DB/learned state while native runtimes/HF cache remain shared;
- standalone app packaging includes the canonical eval suite and required dependencies.

### 3.5 Cross-platform Eval backend behavior

Apple MLX keeps the existing optimized MLX path.

App-launched non-MLX Eval uses the production app backend:

- GGUF / Prism GGUF
- BitNet
- Transformers/controller

The bridge is dormant for normal CLI runs.

The canonical suite's module import graph was hardened so a Windows/Linux machine can import it without having `mlx_lm` installed before the cross-platform bridge takes control.

Windows SWE patch verification uses Git's unified-diff parser when POSIX `patch` is unavailable.

### 3.6 Memory accounting

The Eval memory watcher and training telemetry now use:

- macOS: process **physical footprint** when libproc exposes it, so MLX/Metal unified-memory pressure is counted;
- other platforms: process RSS fallback;
- Eval UI/runner: total of root process + native child processes.

This replaced misleading whole-system-used-RAM and plain-RSS-only paths.

### 3.7 Learning / Phase 3

#### MLX

- real `value_and_grad` + AdamW;
- LoRA-only trainables;
- q/v/down/linear-attention LoRA unfreeze is restricted to `lora_a/lora_b`;
- Fisher/EWC remains real;
- bounded Phase-3 backprop uses local target windows to avoid retaining one giant graph;
- completion-only targets;
- nonzero parameter drift required;
- atomic adapter persistence;
- live LoRA tensors and persisted adapter roll back on failure/cancellation.

#### GGUF / Prism GGUF

- quantized base remains frozen;
- real PEFT/QLoRA sidecar is trained against a declared compatible parent;
- backward + optimizer step + nonzero drift;
- training model is released before LoRA→GGUF conversion;
- learned GGUF adapter is hot-reloaded;
- rollback backups survive until successful return;
- cancellation is treated as a transaction failure.

Prism remains process-isolated and keeps its specialized runtime/projector path.

#### BitNet

- official/real bitnet.cpp runtime path;
- BF16 parent → PEFT update → merge → BitNet I2_S conversion;
- nonzero drift required;
- giant training models are released before conversion;
- rebuilt learned GGUF is reloaded;
- rollback on failure/cancellation;
- no synthetic BitLinear “learning success.”

#### Transformers/controller

- language projection LoRA only;
- completion-only masking;
- real AdamW/backward;
- nonzero drift;
- transactional directory persistence;
- live tensor rollback + persisted adapter rollback on failure/cancellation.

### 3.8 Learn / RSI / DB semantics

- Learn and RSI use the active real trainable backend.
- Zero parameter change is failure.
- RSI successful persistent memory remains in `rsi_self_memories` as the self-generated trace only.
- Benchmark question, expected answer, reward, PASS/FAIL and verifier metadata do not belong in the persistent RSI self-memory sample.
- Answer-blind deterministic verifier feedback can guide RSI round 2, but hidden expected answers remain reward-only.
- Non-MLX Phase 3 proves the persisted artifact before marking memory rows consolidated.
- Learn + RSI consolidation markers are committed in one SQLite transaction.

### 3.9 Phase-3 RAM / telemetry

- Fisher drops each anchor's transient graph.
- Phase 3 uses bounded local windows.
- transient gradient/cache references are released continuously.
- cumulative progress reports **global current/all targets**, not per-row resets.
- percent, TPS and ETA use the same global denominator.
- telemetry no longer carries fabricated fallback TPS/speculative rates.
- process RAM is measured, not unrelated whole-system used RAM.

### 3.10 Media learning

Native differentiable media training:

- requires real trainable parameters/loss/persistence;
- performs backward/optimizer updates;
- verifies finite loss/weights;
- requires measurable parameter delta;
- restores the pre-update tensors on failure/cancellation.

External media trainers:

- must persist an adapter/checkpoint;
- saved artifacts must pass backend verification;
- LoRA artifacts must expose a provable **nonzero learned output/update factor**;
- “file exists” is not accepted as proof that weights changed;
- unsupported/unverifiable formats fail closed rather than reporting `weights_updated=True`.

Media generation support does not automatically imply media training/RSI support.

## 10. Exact end-to-end call-chain trace

This audit traced the actual current call graph, not only file names or contract strings.

### 10.1 Model selection and load

`app_gui.SmartAIChatbotApp._on_toggle_load_unload/_on_switch_model_tab`
→ `ProReasoningEngine.load_model(..., model_info=target_info)`
→ metadata/runtime resolution in `core.controller_runtime.resolve_controller_runtime`
→ one real active backend:

- MLX / MLX-VLM
- Prism GGUF / generic GGUF
- BitNet
- Transformers/controller

`ProReasoningEngine.active_backend`, `active_model_path` and `active_input_modalities` are set before generation. The same loaded backend is assigned to `awake_consolidator.engine`. Media generators remain outside the text Pro engine.

### 10.2 User message → chat/Pro

`app_gui._on_send_message`
→ `_process_message_thread`.

The thread first handles explicit media commands and explicit `/learn`. Ordinary text goes through:

`self.multimodal.stream_solve(full_msg, history, cancel_event)`
→ `MediaController._stream_solve`
→ `self.app.engine.stream_solve(...)`
→ the existing `ProReasoningEngine`.

The media wrapper therefore does not replace the text planner/model. It lets the same Pro response either remain ordinary text or emit an exact `<media_call>{...}</media_call>` tool request, executes the local media tool, feeds the result back as untrusted data, and returns to the same Pro engine for the next/final response.

### 10.3 Pro routing, Context and streaming

Install order in `core/__init__.py` is intentional:

1. chat temperature policy;
2. awake-learning wrapper;
3. full-context hardening;
4. Pro runtime hardening.

The awake wrapper captured `stream_solve/solve`, but calls `self._format_prompt_with_history` dynamically. The later full-context hardening therefore owns the active history packer and removes the older free-RAM-based silent history truncation.

Current runtime chain:

`stream_solve_with_awake_learning`
→ selected single total Context budget
→ optional awake consolidation
→ entropy router
→ N=1 at T=0.65 or historical Pro N=8/N=16
→ remaining Context = packed prompt/history + generated output
→ active backend stream/branch generation.

If the packed prompt already exceeds Context, generation fails closed rather than dropping conversation history.

### 10.4 Awake consolidation

At the 80% Context watermark:

`_apply_awake_learning`
→ choose oldest complete user/assistant chunk
→ `AwakeOnlineConsolidator.consolidate_chunk_sync`
→ real active backend `train_mini_batch`
→ nonzero measured parameter drift + persisted artifact
→ only then remove that old dialogue chunk from active textual history.

If training fails, the dialogue remains in history. Every backend uses transaction/rollback semantics and shared training-memory cleanup.

### 10.5 Explicit Learn → real update → RSI

`app_gui._process_message_thread`
→ `AutonomousLearner.run_learning_session`.

For text/structured sources:

1. real source extraction/search;
2. active model synthesis;
3. verbatim evidence verification;
4. `consolidate_parameters` on the same active trainable backend;
5. require nonzero drift/persistence;
6. `recursive_self_improve` on the already-updated model;
7. source-grounded verification;
8. `consolidate_rsi_parameters` for the self-generated revision;
9. require another real nonzero update.

The historical `_require_live_mlx` name is now only a compatibility alias to `_require_live_trainable_backend`; Learn is not MLX-only.

### 10.6 Multimodal controller and media pipeline

The selected text model carries `input_modalities`. `ProReasoningEngine.supports_media_input/review_media_input` delegates to the actual loaded controller:

- MLX VLM language/vision runtime;
- Prism GGUF + real mmproj;
- compatible controller backend;
- BitNet correctly reports no media input.

`MediaController._can_perceive` requires both declared modality permission and a working backend perception hook.

Media Pro/RSI:

`MediaController.call("media_pro"/"media_rsi")`
→ generate candidate media
→ real controller review when supported
→ numeric self-grade required for RSI
→ winner becomes a real training sample
→ `MediaLearningService.learn`.

If the controller cannot actually ingest that modality, media RSI returns `unsupported`; it does not pretend to see/hear the artifact.

Explicit media Learn remains available independently when the selected generator has a real trainable backend.

### 10.7 Media parameter updates

Native media training:

real trainable module
→ real differentiable loss
→ backward/AdamW
→ finite-weight checks
→ measured nonzero tensor delta
→ persisted adapter
→ rollback live tensors + uncommitted directory on failure/cancellation.

External upstream trainers:

upstream trainer process
→ persisted adapter/checkpoint
→ backend save verification
→ proof of a nonzero learned LoRA output/update factor
→ only then `weights_updated=True`.

An artifact merely existing is not accepted as learning proof.

### 10.8 Canonical Eval GUI → same 4,014 runner

`Eval` top button
→ `core.gui_eval_panel`
→ isolated `eval.app_eval_runner` subprocess
→ `import master_4000_eval_suite as suite`
→ `Master4000EvaluationEngine.run_full_suite()`.

The GUI is a launcher/monitor around the same canonical suite, not a rewritten benchmark.

For app-launched non-MLX runs, `app_cross_platform_bridge` swaps only model/runtime plumbing. Dataset, scoring, prompts, checkpoints and stage ownership remain the canonical suite's.

### 10.9 Eval stage flow

The final `phase4_pro_rsi.run_full_rsi` orchestration is:

1. **Phase 1 Baseline** — real published/project split evaluation, single-pass T=0.55 path.
2. **Phase 2 Learn/MCTS** — supplied LearningFacts + semantic dialogue graph + bounded DSL MCTS teaching.
3. **RSI** — Phase-1 misses only, two answer-blind self-improvement rounds, hidden verifier used only after candidate selection.
4. **RSI persistence** — successful training samples persist only the self-generated trace in `rsi_self_memories`.
5. **Phase 3** — Learn + RSI parameter consolidation; MLX uses bounded completion-only graphs/Fisher/EWC/OGP; non-MLX uses the production backend trainer.
6. **Phase 3B Conversation Teach** — independent facts through the same production awake trainer.
7. **Learning retention test** — same updated logical model.
8. **Phase 4 Post-Consolidation** — Pro retests Phase-1 misses only; already-passed Phase-1 items are carried forward.
9. **Final conversation recall/report** — independent recall checks and final report.
10. Optional **DeepSWE** remains separate/opt-in and uses the flagship harness rather than being mislabeled as ordinary SWE-bench.

Resume paths require trustworthy boolean checkpoint results and the persisted learned artifact. RSI resume now reconstructs only question-free self traces and does not re-create legacy prompt/reward training rows.

### 10.10 Failure/crash paths traced

The source/call-graph audit explicitly checked:

- model-load failures;
- context overflow;
- missing native runtime/dependency;
- GUI cancellation and process-tree termination;
- Eval memory-limit cancellation;
- training exception/cancellation during MLX/GGUF/BitNet/controller updates;
- adapter/rebuilt-model persistence and rollback;
- GGUF/BitNet conversion memory overlap;
- media native/external trainer failure/cancellation;
- Phase-3 DB commit ordering;
- interrupted RSI resume;
- non-MLX Eval import before MLX is installed;
- Windows SWE patch application.

The code now fails closed or rolls back for these paths. This is source/call-graph verification; it is **not** a claim that every native backend/hardware combination was physically executed in this audit environment.

## 6. Standalone 4K Eval and desktop App Eval

There are two related entry contexts:

- **Standalone 4K Eval:** currently targets Bonsai 2 through the Prism-aware runtime.
- **Desktop App Eval:** uses the active production text backend where supported, while still invoking the canonical evaluation suite.

Desktop Eval is a launcher/monitor, not a second evaluator. The canonical path is:

`Eval UI → python -m eval.app_eval_runner → import master_4000_eval_suite as suite → Master4000EvaluationEngine.run_full_suite()`

App Eval supports isolated checkpoint/DB/learned-state locations so evaluation learning does not casually corrupt normal interactive app state.

The live Eval console is written to `live_console.log`. Reopening the Eval window restores the latest saved console, positions the live tail after the restored content, and continues appending new output without duplicating the restored text.

## 7. Benchmark policy and datasets

The ordinary suite uses real published benchmark data and one normal **32K total Context budget**.

Current benchmark direction includes:

- OlympiadBench;
- SuperGPQA;
- MMLU-Pro;
- HumanEval;
- SWE-bench Verified as the normal software-engineering benchmark;
- optional true DeepSWE as a separate flagship long-horizon agent workload.

DeepSWE is not a relabeling of SWE-bench. It remains separately opt-in and can require substantially larger context and external agent/Docker prerequisites.

Scoring is fail-closed and strict. Numeric substring acceptance and loose multiple-choice letter matching are not acceptable scoring strategies.

Hidden benchmark answers/rewards/verifier outcomes are used only after a final candidate is produced.

## 8. Learn, RSI, and consolidation semantics

**Learn** and **RSI** are intentionally different:

- Learn consumes explicit user/training information.
- RSI is self-improvement from the model's own generated traces.

Persistent RSI self-memory stores the **self-generated reasoning trace only**. Benchmark question, expected answer, reward, PASS/FAIL, verifier metadata, and hidden labels are not persisted as the RSI training sample.

Phase 3 is the real consolidation/update stage. Learn and RSI consolidation markers are committed atomically only after the corresponding update/persistence requirements are satisfied.

A real local run previously established a genuine Phase-3 update with `trained=38` and `Δtrainable=12.78552211`. A subsequent Phase 3B failure exposed unsupported MLX `CustomKernel` VJP. The fix changed Phase 3B to a bounded compatible adapter-training route rather than weakening the real-update assertion.

## 9. Backend training model

### MLX

- LoRA/adapter trainables;
- real gradient path + AdamW;
- completion-only training targets;
- Fisher/EWC protection;
- bounded windows for memory safety;
- nonzero trainable-delta proof;
- atomic persistence;
- rollback/cancellation restoration;
- base model is not accidentally unfrozen wholesale.

### GGUF / Prism GGUF

- quantized deployment base remains frozen;
- trainable lineage/parent metadata is required;
- real PEFT/QLoRA update occurs against a compatible parent;
- learned adapter is converted/persisted for the deployment runtime;
- the learned artifact is hot-reloaded;
- persisted-artifact evidence is required before non-MLX Phase 3 is committed;
- backups survive until the success path fully returns;
- failure/cancellation rolls back.

### BitNet

- real Microsoft `bitnet.cpp` inference path;
- compatible BF16/high-precision parent lineage is required for learning;
- PEFT update → merge → real BitNet `I2_S` deployment rebuild;
- rebuilt artifact is reloaded;
- failure/cancellation rolls back;
- missing prerequisites fail closed.

### Controller / custom trainable runtime

- actual backward + optimizer behavior;
- transactional adapter persistence;
- nonzero-drift requirement;
- rollback of both live tensors and persisted adapter state on failure.

## 10. Context, KV, and memory policy

- one total Context budget means packed prompt/history + generated output;
- the same concept applies across MLX, GGUF/Prism, BitNet, and controller text backends;
- current Eval RSI/Phase-4 uses full-precision KV;
- old forced 4-bit/sliding-KV fallbacks are retired;
- old lossy H2O/context dropping is retired;
- Metal OOM handling may reduce prefill chunking, but may not silently change correctness policy;
- app TurboQuant is separate and explicit where supported;
- macOS memory telemetry uses process physical footprint when available;
- other platforms use process RSS fallback;
- Eval process-tree monitoring includes native child processes;
- training cleanup drops transient graphs/models before heavyweight conversion or rebuild steps.

## 11. Temperature / generation policy

Current intended text-generation policy:

- normal chat N=1: **T=0.65**;
- Eval single-pass: **T=0.55**;
- Pro N>1: **0.20 → 0.95**, gamma **1.35**, plus a fixed **T=0.65** candidate.

Older greedy Eval wording and the older 0.88 maximum are historical and must not silently replace the current policy.

## 12. Packaging and bundled Bonsai 2

Packaged macOS releases can carry local Bonsai 2 snapshots. Source checkouts may still resolve/fetch models normally, but packaged builds can resolve the bundled snapshots and treat them as installed.

The packaged standalone 4K Eval uses the bundled Bonsai 2 snapshot when available.

Relevant implementation areas include:

- `build_bonsai2_bundled_zip.py`;
- `build_app.py`;
- `config/paths.py`;
- macOS release workflow;
- bundled-release contract coverage.

This packaging behavior is current and should not be reverted merely because it landed after the older implementation audit.

## 13. Media tools and media learning

The desktop app's existing text/Pro engine can request catalog image, video and audio generation through bounded media tools. This is an app integration, not a change to the 4K evaluator or text model trainer.

## Commands

- `/media image <prompt>`, `/media video <prompt>`, `/media audio <prompt>` use the existing media generators.
- `/learn media MODEL_ID SOURCE` trains the selected media model through a real architecture-matched adapter/trainer. SOURCE may be a local file/folder, structured manifest, direct URL/page, or bounded search query. Existing `/learn <topic>` stays on the text learning path.
- `/rsi media MODEL_ID <prompt>` is available only when the active text controller can directly ingest that media type. It generates alternatives, lets that same controller inspect/grade the actual output, then trains the selected media model from the chosen self-generated sample.

All text controllers can generate image/video/audio and run explicit media Learn. Media RSI is capability-gated: image input unlocks image RSI; audio input unlocks audio RSI; video input unlocks image+video+audio RSI; image+audio also unlocks all three. Text-only controllers cannot grade/RSI media they cannot ingest.

## Actual support and limits

Generation delegates to the existing image/video/audio engines. It inherits their model compatibility and installed dependency requirements; a catalog label alone is not proof a checkpoint can load.

Media Learn resolves trainers by architecture/base checkpoint rather than catalog slot. Current trainer families include SDXL, FLUX.2/Z-Image, Wan/LTX/CogVideoX paths, Stable Audio 3, and generic registered trainer extensions. Quantized/MLX/GGUF derivatives resolve back to a differentiable base checkpoint when metadata permits. If no real trainer can be resolved, Learn returns `unsupported` rather than faking a weight update.

Media review/RSI uses the active text controller's own declared and working media-input path. A controller that cannot ingest the requested modality can still generate and explicitly train that media model, but it cannot claim to inspect or self-grade the artifact.

The memory scheduler leaves text resident when estimated RAM/VRAM headroom allows. Otherwise it pauses at a completed text-generation boundary, saves and verifies learned MLX LoRA tensors, runs media, and restores the same text model and tensor snapshot. GGUF restores the same model/adapter configuration. If there is still insufficient headroom, it aborts rather than truncating context or lowering precision. Estimates are not hard allocation guarantees.

Weight updates are serialized, require finite gradients and a measurable parameter change, and publish a checkpoint only after saved tensors are verified. Failed updates roll parameters back. A successful update does not prove improved quality. Existing EWC calculation is reused only when the backend supplies genuine Fisher/anchor data; the basic SDXL adapter does not claim it has those statistics.

## Verification

Contract/unit coverage locks tool routing, capability-gated RSI, exact text-model pause/resume identity, local/online Learn sources, trainer resolution, cancellation/error recovery, and verified adapter persistence. These tests do not substitute for running every large media model/trainer end-to-end on real hardware.

## 4. Concrete defects found and fixed during this full audit

The following were real current-branch defects discovered while walking the code, not hypothetical concerns:

1. **GGUF Phase-3B tokenizer assignment** — conversation-teach unconditionally wrote `backend.tokenizer`; GGUF exposes a read-only tokenizer facade. Fixed by binding only when objects actually differ.
2. **macOS Eval RAM undercounting** — watcher used RSS only. Fixed with `proc_pid_rusage(...).ri_phys_footprint` when available.
3. **MLX q/v LoRA trainability leak** — q/v projections still used unrestricted `.unfreeze()`. Fixed to `lora_a/lora_b` only.
4. **MLX training transactionality** — added nonzero-drift fail-closed behavior, pre-update snapshots, atomic safetensors persistence and rollback.
5. **Controller PEFT transactionality** — added live tensor rollback and persisted adapter rollback.
6. **Single Context semantics** — non-MLX chat paths still used old fixed `max_new_tokens`. Fixed so one Context budget constrains prompt/history + output for every text backend.
7. **Eval report syntax defect** — a literal escaped `\n` existed inside Python source after the target-model label patch. Replaced with valid Python.
8. **Cross-platform legacy engine surface** — app Eval adapter was missing the LIF controller required by the canonical suite. Added the real controller.
9. **Legacy fabricated telemetry** — removed fallback `12.0 t/s`, `15.0 t/s`, `42.5%` speculative rate and synthetic `[Offline: ...]` benchmark output.
10. **Windows SWE patch verification** — old path assumed POSIX `patch`. Added Git fallback on Windows/no-`patch` systems.
11. **Non-MLX import blocker** — `live_generation_stream.py` and `rsi_generation_memory_hardening.py` imported MLX-LM unconditionally. Guarded imports so the canonical suite can load without MLX.
12. **Non-MLX Phase-3 commit order** — rows could be marked consolidated before persisted artifact proof. Reordered.
13. **Learn/RSI table atomicity** — Learn and RSI markers were separate DB writes. Combined into one SQLite transaction.
14. **Cancellation rollback** — GGUF/BitNet callers and conversion trainers caught `Exception`, not `KeyboardInterrupt`. Changed transaction handlers to cover cancellation too.
15. **Rollback backup lifetime** — controller/GGUF/BitNet removed backups before the successful function return. Backups now survive until the success path is complete.
16. **Cross-platform Eval token counting** — GGUF could fall back to a rough character estimate despite native llama.cpp tokenization. Native `encode`/ `tokenize` / backend `count_tokens` are tried first.
17. **Eval memory-limit interrupt** — child watcher self-signaled with `os.kill(SIGINT)`; switched to Python main-thread interrupt for portable cancellation.
18. **Media external trainer proof** — a nonempty checkpoint could be reported as learning. Now a nonzero learned LoRA update factor must be demonstrable.
19. **Native media cancellation** — pre-update tensors are restored and uncommitted adapter directories removed on cancellation.
20. **Failure-path cleanup** — app Eval non-MLX Phase 3 now releases backend training memory even if training throws.
21. **MLX Phase-3 integrity contract** — bounded MLX Phase 3 updated real LoRA weights but did not return `real_trainable_delta_l2`, so the integrity wrapper could falsely reject a successful update. Fixed by snapshotting only LoRA trainables, measuring real L2 drift after training, failing on zero drift, and returning the measured value.
22. **Audit-test regressions** — the newly added audit contract itself contained one invalid quoted string and one stale assertion that referenced the pre-atomic Phase-3 marker path. Both were corrected before final documentation.
23. **RSI resume persistence semantics** — the resume layer still reconstructed successful RSI items through legacy `episodic_interactions` rows containing prompt/reward/session metadata, and base RSI would clear the new question-free inbox on restart. Resume now queries/restores only exact `rsi_self_memories.trace` values, never persists the benchmark question/reward/PASS state as training data, and preserves reconstructed self-traces across the resumed base-RSI startup clear.

## 15. Reconciled / intentionally retired behavior

The following decisions remain intentional. Older/diverged implementations are not authoritative when they conflict with these policies.

The audit did **not** merge diverged branches wholesale.

### Historical path

Useful parts already ported/current:

- LoRA-only trainability;
- Fisher graph cleanup;
- bounded Phase 3;
- daemon RAM cleanup.

Rejected stale behavior:

- forced 4-bit/quantized KV policy.

### Historical path

Useful parts retained:

- answer-blind deterministic verifier feedback;
- same-model checks every recursive round;
- earlier release of unused branch strings.

Rejected:

- 4-bit/sliding KV fallback;
- old persistence semantics.

### Historical path

No useful behavior beyond the superseded quantized-KV memory strategy.

### Historical path

Kept only the useful synchronize-before-pressure-cache-release detail.

Did not restore unconditional per-branch purging as the normal policy.

### Historical path

Current HLE logic is newer:

- treats the token after `T = ZFC +` as arbitrary opaque literal text;
- first completed `Con(...)` is final;
- stronger anti-repeat/anti-recheck wording.

Rejected the old digit-only assumption and fixed 128-token cap.

### Historical path

Superseded by the current real benchmark runtime plus later schema/fetch fixes.

### Historical path

Useful architecture modules already exist in the current tree, including:

- adaptive hyperparameters;
- empirical Fisher/EWC;
- GRPO;
- MCTS discovery;
- in-process MCP;
- knowledge graph;
- round-robin LoRA;
- SmartKV;
- dual/MoE buffers.

Safe diagnostic/lifecycle pieces were reconciled additively:

- grammar-guided drafter diagnostics;
- LIF entropy helpers;
- projected-daemon lifecycle/status telemetry.

Rejected staging regressions:

- obsolete app/model registry rewrites;
- lossy H2O/context dropping;
- speculative decoding activation without proven exact equivalence;
- old MLX-only wrapper architecture;
- stale quantized-KV assumptions.

## 2. Current history integrity

At the frozen implementation head, the branch was **134 commits ahead / 0 behind master**. The audited post-master history is linear: the reviewed feature commits are single-parent commits, not unexplained merge commits. Earlier known guard-file removals are coordination cleanup only and do not remove implementation.

The latest model choice is intentional:
- app `model_1`: **Bonsai 2 27B Ternary Multimodal**
- app `model_2`: **Bonsai 2 27B CRACK**
- standalone 4K eval: **Bonsai 2 27B Ternary MLX**
- commits 125–127 temporarily restored Qwen3.8; commits 128–130 intentionally superseded that and restored Bonsai 2. Those commits remain in history.

## 3. End-to-end code trace

### Model selection and Hugging Face import

The app starts on `model_1`; existing model choices remain present.

Custom HF text-model admission reads Hub card/config/file metadata, requires explicit ternary/1.58-bit/trit proof, infers modalities and runtime family, chooses exact native GGUF/BitNet artifacts when uniquely provable, captures mmproj/projector metadata, and records a declared trainable parent lineage when present. Runtime loading consumes that persisted metadata instead of guessing from a display name.

### Model load → message → Pro

`ProReasoningEngine.load_model(..., model_info=...)` routes by resolved runtime metadata:
- Bonsai/other Apple MLX → `MLXReasoningBackend`;
- Prism GGUF → `PrismGGUFReasoningBackend`;
- ordinary GGUF → `GGUFReasoningBackend`;
- verified BitNet → `BitNetCppReasoningBackend`;
- other controller families → `UniversalControllerBackend`.

Normal chat N=1 remains T=0.65. Pro N>1 remains 0.20→0.95, gamma 1.35, plus the dedicated T=0.65 branch. The single Context budget continues to mean packed history/prompt plus generated output.

### Multimodal Bonsai MLX

Bonsai-2 MLX is not treated as a plain stock MLX checkpoint. The runtime family resolves to the bundled repository VLM loader. The bundled loader installs the Hadamard-aware packed layers into the model's real `language_model`.

Learn/RSI obtains that inner language module through `get_training_model()`, reuses the existing real MLX LoRA/Fisher/AdamW update code, and raw-adapter persistence restores tensors into the same language module on reload.

### Consolidation / Learn / RSI

Awake consolidation and explicit Learn select the active real trainable backend instead of using an MLX-only gate.

- **MLX:** LoRA-only trainables, real gradients/AdamW, EWC/Fisher, nonzero drift, transactional adapter save/rollback.
- **GGUF/Prism:** frozen quantized base + real PEFT/QLoRA sidecar on a declared compatible parent, nonzero drift, official LoRA→GGUF conversion, hot reload, rollback.
- **BitNet:** official bitnet.cpp deployment runtime; when a verified BF16 training lineage exists, BF16 parent → PEFT update → merge → I2_S rebuild → hot reload, with rollback. If that lineage/toolchain cannot be proven, training fails closed rather than reporting fake learning.
- **Transformers/controller:** language-projection LoRA, completion-only labels, real backward/AdamW, nonzero drift, transactional persistence.

RSI persistent self-memory remains question/reward/PASS-free: successful self-generated traces are stored separately and only hidden verification gates eligibility.

### Media / multimodal RSI

Media generation and media training are separate capabilities.

`MediaLearningService` resolves a compatible trainer by runtime/model metadata. A media update only reports success after a real nonzero trainable delta (or externally persisted nonzero LoRA factor) is proven and persisted; cancellation/failure rolls back.

`media_rsi` is genuine only when the active text controller can directly perceive the target modality. It:
1. generates candidate media,
2. has the active multimodal controller inspect/score the actual artifact,
3. refuses blind RSI if no real numeric self-grade exists,
4. selects the best valid candidate,
5. invokes the real media-learning backend,
6. reports `weights_updated` only from that backend.

If the loaded controller cannot ingest that modality, media generation and explicit media Learn remain available but media RSI correctly returns unsupported.

### Standalone 4K eval

The standalone 4K target remains Bonsai-2 by user decision. A current-head bug was found during this audit: simply changing `EngineSettings.mlx_model_path` to Bonsai-2 left the standalone engine using plain `mlx_lm.load()`, while this Prism checkpoint requires its bundled Hadamard-aware runtime.

Commit `72950772...` fixes that **without rewriting the eval pipeline**:
- it reuses the app's `MLXReasoningBackend`;
- loads Bonsai-2 through its bundled runtime;
- exposes the correctly loaded real language module/tokenizer to the existing benchmark/training stages.

The canonical stage flow remains the existing suite: real Phase 1 → Learn/RSI → bounded Phase 3/3B → Phase 4 retest.

## 4. Concrete current-head fixes made during this continuation

1. **Focused audit import isolation** — the metadata contract imported `core.model_policy` through package `core/__init__.py`, accidentally requiring NumPy even though the contract only tests pure metadata logic. It now loads that source module directly.
2. **Standalone Bonsai runtime correctness** — switched the already-selected Bonsai-2 standalone eval from plain MLX loading to the proper bundled Prism runtime, while preserving the existing eval stages.
3. **Eval target contract wording** — the test now checks the exact Bonsai repo ID used by the report rather than a stale display label.
4. **BitNet audit assertion** — the test now verifies the real platform paths `build/bin/llama-server` and `build/bin/Release/llama-server.exe` instead of searching for a nonexistent exact string literal.

No model choices were removed, and the Bonsai-2 main/default decision was not reverted.

## 7. Validation status — what is and is not proven

### Source/call-graph verified in this audit

- complete sequential branch history was read oldest→newest;
- final `master..feature` diff inspected;
- canonical Eval wrapper/install order inspected;
- canonical Eval startup imports checked for non-MLX portability;
- active backend routing and training call graph inspected;
- DB/persistence ordering inspected;
- cancellation/rollback paths inspected;
- top-bar control uniqueness checked;
- no fabricated eval fallback speed/spec telemetry remains;
- focused full-branch source-contract sweep passed after correcting two bad test assumptions; the concurrent MLX real-delta commit was then checked against the integrity layer before documentation sign-off.

### Not claimed as executed in this audit environment

This environment could not clone GitHub into the local shell, so this audit does **not** claim:

- a local `py_compile` over the entire repository;
- a local pytest run;
- full CI;
- real Apple MLX heavyweight training;
- real Prism native library execution;
- real bitnet.cpp rebuild/reload on Windows/Linux;
- CUDA controller PEFT training;
- every external media trainer;
- Docker/Pier flagship DeepSWE;
- release packaging/publishing.

Those are runtime/hardware validation tasks, not source-implementation gaps. They must be reported honestly if/when executed.

## 5. Verification evidence

The focused audit run for implementation head `e2163c6b...` completed successfully:
- **228 / 228 tracked Python files syntax-compiled**
- **58 / 58 focused source contracts passed**
- exact source archive captured
- exact master source archive captured
- full binary-safe `master-to-feature.diff` captured
- complete reverse chronological-to-sequential commit patch ledger captured
- numstat ledger captured

A static unfinished-code scan of current `core/`, `eval/`, `consolidation/`, `app_gui.py`, `run_studio_complete.py`, and `master_4000_eval_suite.py` found **no TODO, FIXME, or NotImplementedError markers**. Remaining `pass` statements are exception/fallback/lifecycle handling, not unimplemented method placeholders in the scanned paths.

This remains a source/contract audit, not a claim that every heavyweight native backend was physically executed on every hardware target. Full CI, real Apple 27B training, Prism native GGUF libraries, bitnet.cpp rebuilds, CUDA training, all external media trainers, Docker/Pier DeepSWE, and release publishing require their real target environments and are not fabricated here.

## 8. Remaining external prerequisites, not hidden “unfinished code”

Some paths are intentionally fail-closed when their required native/upstream dependency is absent:

- Prism specialized llama.cpp/mtmd libraries and projector;
- Microsoft bitnet.cpp / converter toolchain;
- BF16 parent model for trainable BitNet/GGUF lineage;
- Docker + official SWE-bench package for SWE-bench Verified scoring;
- Pier/mini-swe-agent/Docker for optional flagship DeepSWE;
- Diffusers/accelerate, MFLUX, Stable Audio, AI Toolkit, etc. for particular media trainers.

The code must not report these as successful when the dependency is missing.

## 20. Evidence levels and verification boundary

Use these evidence levels when describing project status:

- **Implemented/source-present:** code exists and the call graph can be inspected.
- **Contract-tested:** focused tests/assertions verify source/runtime contracts.
- **Native-tested:** the real backend/model/hardware path actually executed.
- **Published:** the exact release/ref/artifact state was verified.

Source-level proof can establish call-chain structure, rollback logic, persistence conditions, metadata routing, parameter-delta checks, imports, and exact Git history.

It does **not** automatically prove every Apple MLX 27B job, CUDA configuration, Prism library combination, BitNet rebuild, media trainer, Docker/Pier DeepSWE execution, package launch, or release asset on every supported machine.

## 21. Editing / maintenance rules

When changing this branch:

1. read the current branch head first;
2. treat current changes as intentional unless a concrete bug is identified;
3. prefer narrow deltas over subsystem/file replacement;
4. when an older version has one better detail, port only that detail;
5. preserve answer blindness, Learn/RSI separation, and transactional persistence;
6. do not silently change the standalone Eval target;
7. do not use an old successful audit as proof of a later head;
8. do not claim learning unless real parameters/artifacts changed;
9. do not claim release/build/CI success without current evidence;
10. keep the app Eval tied to the canonical runner.

## 22. Project status summary

The feature branch is not a master rewrite. It is an additive/surgical hardening line whose major goals are:

- eliminate the known Phase-3/Fisher/training RAM blowups;
- make learning real and transactional across supported text backends;
- preserve question-free RSI self-memory semantics;
- keep telemetry measured rather than fabricated;
- expose the same canonical 4,014 runner through a dedicated text-model Eval GUI;
- make that Eval path import/run through non-MLX production backends;
- keep one Context and one main-app memory control;
- preserve newer prompt/scoring/stage behavior;
- reconcile only the better parts of diverged branches.

No known useful branch-only behavior was intentionally discarded during this audit. Old behavior was excluded only where the current branch is newer/safer or where the old behavior conflicts with explicit requirements.

