# Smart AI Studio — Full Feature-Branch Audit and Reconciliation

**Repository:** `Eabusham2/smart-ai-studio`  
**Authoritative branch:** `fix/real-benchmarks-final-32k`  
**Historical baseline:** `master` at `06390e5357a07f80a8089ac28fc16c75461a48a6`  
**Implementation head audited before documentation commits:** `ce26510b50c06fdc37cfbe2181e05918ba6e9a1e`  
**Audit date:** 2026-09-19

## 1. Audit rule

The feature branch is authoritative. `master` is only a historical comparison baseline.

This audit does **not** revert a newer feature-branch implementation merely because `master` is simpler. It uses these rules:

1. Preserve working/newer feature-branch behavior.
2. Prefer surgical layering over file/subsystem rewrites.
3. If old/diverged branches contain a genuinely better isolated behavior, port only that behavior.
4. Reject stale policy regressions, especially quantized/lossy eval KV, context dropping, fabricated telemetry, fake learning success, mock/offline benchmark output, and MLX-only assumptions in cross-platform paths.
5. Preserve the canonical 4,014 stage/scoring/checkpoint/prompt flow unless a concrete bug requires a narrow fix.
6. Distinguish **source/call-graph verification** from **real hardware/native-runtime execution**. They are not interchangeable.
7. Full CI was not run because the standing project rule is not to run full CI unless explicitly requested.

## 2. Final branch-vs-master shape

At the final code-audit checkpoint before these documentation commits:

- the feature branch was **ahead of master and 0 behind**;
- the implementation diff was concentrated in RAM/training safety, Eval GUI/cross-platform plumbing, Phase-3/RSI hardening, packaging, and narrow diagnostic helpers;
- `app_gui.py` itself was only incrementally modified rather than replaced;
- the only branch-file removals were two obsolete session/branch guard documents:
  - `ACTIVE_SESSION_GUARD_2026-09-18_GGUF_ONLY.md`
  - `BRANCH_GUARD_GGUF_LEARNING_20260918.txt`

No useful implementation was removed by those deletions.

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

## 5. Diverged/rogue branch reconciliation

The audit did **not** merge diverged branches wholesale.

### `code-low-think-hardening`

Useful parts already ported/current:

- LoRA-only trainability;
- Fisher graph cleanup;
- bounded Phase 3;
- daemon RAM cleanup.

Rejected stale behavior:

- forced 4-bit/quantized KV policy.

### `fix/rsi-memory-safe-resume-v2`

Useful parts retained:

- answer-blind deterministic verifier feedback;
- same-model checks every recursive round;
- earlier release of unused branch strings.

Rejected:

- 4-bit/sliding KV fallback;
- old persistence semantics.

### `fix/rsi-resume-compact-telemetry`

No useful behavior beyond the superseded quantized-KV memory strategy.

### `fix/metal-branch-memory`

Kept only the useful synchronize-before-pressure-cache-release detail.

Did not restore unconditional per-branch purging as the normal policy.

### `fix/hle-i0-loop`

Current HLE logic is newer:

- treats the token after `T = ZFC +` as arbitrary opaque literal text;
- first completed `Con(...)` is final;
- stronger anti-repeat/anti-recheck wording.

Rejected the old digit-only assumption and fixed 128-token cap.

### `fix/real-4k-benchmarks-32k`

Superseded by the current real benchmark runtime plus later schema/fetch fixes.

### `arch-merge-staging`

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

## 6. Gemini-history reconciliation

The historical Gemini log is useful as a record of desired architecture and experiments, **not as proof of completion**.

Claims later contradicted or not supported by reliable execution evidence include:

- “everything is built” / all checks passing / packaged in `dist/`;
- repeated 86/86, 94/94, 112/112 and 123/123 completion claims;
- 100% across all benchmark splits;
- DeepSWE/SWE-bench results presented as if they proved the final DeepSWE flagship workload;
- release binaries claimed uploaded and master claimed synchronized without durable verification;
- contradictory TPS explanations and changing “hardware ceiling” claims;
- verification scripts that printed a success count instead of actually checking requirements.

The current branch explicitly removes fake fallback telemetry and requires source/runtime proof where possible.

See `docs/GEMINI_POSTMORTEM_2026-09-19.md` for the detailed postmortem.

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

## 8. Remaining external prerequisites, not hidden “unfinished code”

Some paths are intentionally fail-closed when their required native/upstream dependency is absent:

- Prism specialized llama.cpp/mtmd libraries and projector;
- Microsoft bitnet.cpp / converter toolchain;
- BF16 parent model for trainable BitNet/GGUF lineage;
- Docker + official SWE-bench package for SWE-bench Verified scoring;
- Pier/mini-swe-agent/Docker for optional flagship DeepSWE;
- Diffusers/accelerate, MFLUX, Stable Audio, AI Toolkit, etc. for particular media trainers.

The code must not report these as successful when the dependency is missing.

## 9. Sequential commit ledger

The implementation was audited in chronological order. Documentation commits are intentionally excluded from this implementation ledger.

The final two implementation commits landed concurrently during documentation and were re-audited before sign-off:
- `f267e44bd1` fixes the audit test to match the atomic DB contract.
- `ce26510b50` returns the real MLX Phase-3 LoRA delta required by the stage-integrity layer.

| # | Commit | Phase | Change | Audit disposition |
|---:|---|---|---|---|
| 1 | `a9ba9c00c3` | RAM / bounded Phase 3 | Keep EWC rollback compatible with bounded Phase 3 | Retained; later transaction/cancel hardening extends it |
| 2 | `35bd3884bb` | RAM / bounded Phase 3 | Port bounded Phase 3 without losing current Learn RSI semantics | Retained; later transaction/cancel hardening extends it |
| 3 | `d8f0b719b5` | RAM / bounded Phase 3 | Add backend-neutral training memory guard | Retained; later transaction/cancel hardening extends it |
| 4 | `86526c674a` | RAM / bounded Phase 3 | Guard app awake consolidation memory on every backend | Retained; later transaction/cancel hardening extends it |
| 5 | `22234eafbe` | RAM / bounded Phase 3 | Guard app Learn and RSI memory across active backends | Retained; later transaction/cancel hardening extends it |
| 6 | `3e49d834fd` | RAM / bounded Phase 3 | Release GGUF QLoRA training memory deterministically | Retained; later transaction/cancel hardening extends it |
| 7 | `0ce83994e8` | RAM / bounded Phase 3 | Release BitNet rebuild training memory deterministically | Retained; later transaction/cancel hardening extends it |
| 8 | `6914ddbee3` | RAM / bounded Phase 3 | Release controller PEFT training memory after updates | Retained; later transaction/cancel hardening extends it |
| 9 | `f162b3bcb1` | RAM / bounded Phase 3 | Fix Learn RSI EWC and RAM telemetry accuracy | Retained; later transaction/cancel hardening extends it |
| 10 | `1829b2625a` | RAM / bounded Phase 3 | Drop GGUF training graph before adapter conversion | Retained; later transaction/cancel hardening extends it |
| 11 | `9e40426508` | RAM / bounded Phase 3 | Drop BitNet training models before I2_S conversion | Retained; later transaction/cancel hardening extends it |
| 12 | `7a063763e0` | RAM / bounded Phase 3 | Avoid duplicate adapter tensors during awake consolidation | Retained; later transaction/cancel hardening extends it |
| 13 | `6c9b6fd0ae` | RAM / bounded Phase 3 | Make Phase 3 telemetry cumulative across all targets | Retained; later transaction/cancel hardening extends it |
| 14 | `915f371a27` | RAM / bounded Phase 3 | Drop final GGUF training references before conversion | Retained; later transaction/cancel hardening extends it |
| 15 | `ebb59dde3b` | RAM / bounded Phase 3 | Drop final BitNet training references before conversion | Retained; later transaction/cancel hardening extends it |
| 16 | `4a0d3d9fc7` | RAM / bounded Phase 3 | Drop final controller training references after updates | Retained; later transaction/cancel hardening extends it |
| 17 | `ab72296d7e` | RAM / bounded Phase 3 | Keep cumulative Phase 3 telemetry bounded in memory | Retained; later transaction/cancel hardening extends it |
| 18 | `c118cef33d` | RAM / bounded Phase 3 | Add app eval cross-platform runtime bridge | Retained; later transaction/cancel hardening extends it |
| 19 | `7883826d47` | RAM / bounded Phase 3 | Wire app eval bridge into canonical suite | Retained; later transaction/cancel hardening extends it |
| 20 | `3c6e335093` | App Eval / cross-platform foundation | Keep MLX legacy wrapper off non-MLX app evals | Retained; later audit closes portability/failure edges |
| 21 | `bb782976f7` | App Eval / cross-platform foundation | Add isolated app eval subprocess runner | Retained; later audit closes portability/failure edges |
| 22 | `52f93dcc6c` | App Eval / cross-platform foundation | Add text-model Eval window to desktop app | Retained; later audit closes portability/failure edges |
| 23 | `5df88c336a` | App Eval / cross-platform foundation | Wire Eval window into desktop app | Retained; later audit closes portability/failure edges |
| 24 | `621220ec93` | App Eval / cross-platform foundation | Ship eval suite in standalone app bundles | Retained; later audit closes portability/failure edges |
| 25 | `9450f879f2` | App Eval / cross-platform foundation | Add standalone eval runtime dependencies | Retained; later audit closes portability/failure edges |
| 26 | `1790718ab9` | App Eval / cross-platform foundation | Declare cross-platform app eval dependencies | Retained; later audit closes portability/failure edges |
| 27 | `7e1ebd485a` | App Eval / cross-platform foundation | Switch standalone eval default to Bonsai 2 | Retained; later audit closes portability/failure edges |
| 28 | `dee1cab77c` | App Eval / cross-platform foundation | Report actual Bonsai 2 or selected app eval model | Retained; later audit closes portability/failure edges |
| 29 | `2f6b3993ba` | App Eval / cross-platform foundation | Fix app eval dataset and RSI memory wiring | Retained; later audit closes portability/failure edges |
| 30 | `55095ca10f` | App Eval / cross-platform foundation | Align Phase 3 integrity with current memory queue schema | Retained; later audit closes portability/failure edges |
| 31 | `75f4319c6a` | App Eval / cross-platform foundation | Make non-MLX eval pause between Pro branches | Retained; later audit closes portability/failure edges |
| 32 | `df8e3402a5` | App Eval / cross-platform foundation | Prevent orphaned eval workers on app close | Retained; later audit closes portability/failure edges |
| 33 | `cf73e9f56e` | App Eval / cross-platform foundation | Enforce app eval RAM limit inside runner | Retained; later audit closes portability/failure edges |
| 34 | `2db156fa41` | App Eval / cross-platform foundation | Make Eval window output truly live | Retained; later audit closes portability/failure edges |
| 35 | `1ea2eb8feb` | App Eval / cross-platform foundation | Clamp app eval to actual backend context | Retained; later audit closes portability/failure edges |
| 36 | `58b4579bd8` | App Eval / cross-platform foundation | Use active backend adapter path throughout app eval | Retained; later audit closes portability/failure edges |
| 37 | `abd9ff4eb1` | App Eval / cross-platform foundation | Allow isolated app-eval GGUF adapter state | Retained; later audit closes portability/failure edges |
| 38 | `4e71c953bc` | App Eval / cross-platform foundation | Allow isolated app-eval BitNet learned state | Retained; later audit closes portability/failure edges |
| 39 | `bc52e23a81` | App Eval / cross-platform foundation | Allow isolated app-eval controller adapter state | Retained; later audit closes portability/failure edges |
| 40 | `7408f1e104` | App Eval / cross-platform foundation | Separate app eval learned state from shared runtimes | Retained; later audit closes portability/failure edges |
| 41 | `155ea77a38` | App Eval / cross-platform foundation | Isolate MLX app-eval adapter state | Retained; later audit closes portability/failure edges |
| 42 | `e2dcdcb197` | App Eval / cross-platform foundation | Lock desktop Eval integration contracts | Retained; later audit closes portability/failure edges |
| 43 | `fae826327b` | App Eval / cross-platform foundation | Allow verified app backends through eval orchestrator | Retained; later audit closes portability/failure edges |
| 44 | `255771ba49` | App Eval / cross-platform foundation | Make top Eval button label explicit | Retained; later audit closes portability/failure edges |
| 45 | `d892c2dc1a` | Eval UX / safety / contracts | Label top context control explicitly | Retained |
| 46 | `5ec5b7ed2f` | Eval UX / safety / contracts | Fail app eval closed on GGUF runtime errors | Retained |
| 47 | `2292c09170` | Eval UX / safety / contracts | Lock Eval run settings while active | Retained |
| 48 | `78b3659da2` | Eval UX / safety / contracts | Install Context Limit and Eval controls explicitly at app startup | Retained |
| 49 | `e543c6641f` | Eval UX / safety / contracts | Keep top Context control compatible with current chat arguments | Retained |
| 50 | `103fc7a26f` | Eval UX / safety / contracts | Make Eval cancel safe during model load | Retained |
| 51 | `d6fec5e7aa` | Eval UX / safety / contracts | Lock Eval Context and Mem Limit top-bar contracts | Retained |
| 52 | `e2d83ea48c` | Eval UX / safety / contracts | Keep app model controls exclusive during Eval | Retained |
| 53 | `601dbbf689` | Eval UX / safety / contracts | Keep conversation teach backend identity accurate | Retained |
| 54 | `94d577e6fe` | Eval UX / safety / contracts | Verify Learn and RSI consolidation in their own tables | Retained |
| 55 | `d360d5d755` | Eval UX / safety / contracts | Lock single Context and Mem Limit controls | Retained |
| 56 | `1ee980fe67` | Eval UX / safety / contracts | Port answer-blind RSI feedback without old KV regressions | Retained |
| 57 | `50a4f356ab` | Eval UX / safety / contracts | Synchronize MLX work before pressure cache release | Retained |
| 58 | `40990c6184` | Eval UX / safety / contracts | Preserve grammar-guided drafter diagnostics | Retained |
| 59 | `3c06189f56` | Diverged-branch reconciliation | Preserve LIF diagnostic entropy helpers | Retained only as additive/safe reconciliation |
| 60 | `8a40007193` | Diverged-branch reconciliation | Preserve OGP daemon lifecycle telemetry | Retained only as additive/safe reconciliation |
| 61 | `a6e147e8f8` | Diverged-branch reconciliation | Lock best-of-diverged-branches reconciliation | Retained only as additive/safe reconciliation |
| 62 | `2c51d357c3` | Diverged-branch reconciliation | Remove BRANCH_GUARD_GGUF_LEARNING_20260918.txt | Intentional cleanup |
| 63 | `90b6946c66` | Diverged-branch reconciliation | Remove ACTIVE_SESSION_GUARD_2026-09-18_GGUF_ONLY.md | Intentional cleanup |
| 64 | `9796f421dd` | Diverged-branch reconciliation | Keep Phase 3B compatible with read-only GGUF tokenizer facade | Retained only as additive/safe reconciliation |
| 65 | `b7da23ecec` | Guard-file cleanup | Measure macOS training/eval memory by physical footprint | Audit correction / retained |
| 66 | `1661aa5949` | Guard-file cleanup | Use macOS physical footprint in Eval UI memory telemetry | Audit correction / retained |
| 67 | `358e22904e` | Full-audit corrective hardening | Enforce Eval RAM limit with macOS physical footprint | Audit correction / retained |
| 68 | `8a53447c1e` | Full-audit corrective hardening | Clean app Eval Phase 3 memory on failure | Audit correction / retained |
| 69 | `04fb103f3e` | Full-audit corrective hardening | Restrict all MLX LoRA trainables to adapter tensors | Audit correction / retained |
| 70 | `7aef6c6370` | Full-audit corrective hardening | Make MLX LoRA training transactional | Audit correction / retained |
| 71 | `890b6e01c8` | Full-audit corrective hardening | Make controller PEFT training transactional | Audit correction / retained |
| 72 | `e34da4f1c5` | Full-audit corrective hardening | Apply one Context budget across all text backends | Audit correction / retained |
| 73 | `39e11b35ac` | Full-audit corrective hardening | Close MLX adapter persistence cancellation window | Audit correction / retained |
| 74 | `b362044a6f` | Full-audit corrective hardening | Fix eval report target-model syntax | Audit correction / retained |
| 75 | `55da13b154` | Full-audit corrective hardening | Complete app Eval legacy engine surface | Audit correction / retained |
| 76 | `4454687413` | Full-audit corrective hardening | Report actual eval process memory | Audit correction / retained |
| 77 | `c6a3dc6e0a` | Full-audit corrective hardening | Remove legacy fabricated eval telemetry fallbacks | Audit correction / retained |
| 78 | `d17252a65d` | Full-audit corrective hardening | Make SWE patch verification cross-platform | Audit correction / retained |
| 79 | `52cacb5a1e` | Full-audit corrective hardening | Make live eval stream import-safe off MLX | Audit correction / retained |
| 80 | `3450925244` | Full-audit corrective hardening | Make RSI memory module import-safe off MLX | Audit correction / retained |
| 81 | `b9e114b0e7` | Full-audit corrective hardening | Commit non-MLX Phase 3 only after persisted artifact proof | Audit correction / retained |
| 82 | `b6b1a9282f` | Full-audit corrective hardening | Roll back GGUF training on cancellation | Audit correction / retained |
| 83 | `d485a66b55` | Full-audit corrective hardening | Roll back BitNet training on cancellation | Audit correction / retained |
| 84 | `cb7209f506` | Full-audit corrective hardening | Restore GGUF PEFT state on cancellation | Audit correction / retained |
| 85 | `5ea3712b99` | Full-audit corrective hardening | Restore BitNet PEFT state on cancellation | Audit correction / retained |
| 86 | `d0f2cd9b97` | Full-audit corrective hardening | Require proven nonzero media LoRA updates | Audit correction / retained |
| 87 | `0cefd44394` | Full-audit corrective hardening | Roll back native media training on cancellation | Audit correction / retained |
| 88 | `7bafce3e7c` | Full-audit corrective hardening | Lock full branch audit regressions | Audit correction / retained |
| 89 | `1c9b25b89a` | Full-audit corrective hardening | Use native tokenizers for cross-platform Eval context accounting | Audit correction / retained |
| 90 | `04c437b3a1` | Full-audit corrective hardening | Commit Learn and RSI Phase 3 markers atomically | Audit correction / retained |
| 91 | `cbace46be4` | Full-audit corrective hardening | Keep controller adapter backup until successful return | Audit correction / retained |
| 92 | `0c6b4672e1` | Full-audit corrective hardening | Keep GGUF rollback backups until successful return | Audit correction / retained |
| 93 | `84d7867b93` | Full-audit corrective hardening | Keep BitNet rollback backup until successful return | Audit correction / retained |
| 94 | `9990af4ab0` | Full-audit corrective hardening | Make Eval memory-limit interrupt cross-platform | Audit correction / retained |
| 95 | `39919e0a4f` | Full-audit corrective hardening | Extend full branch audit contracts | Audit correction / retained |
| 96 | `c7deb15a85` | Full-audit corrective hardening | Correct full branch audit contract invariants | Audit correction / retained |
| 97 | `f267e44bd1` | Full-audit corrective hardening | Fix full branch audit test regressions | Audit correction / retained |
| 98 | `ce26510b50` | Full-audit corrective hardening | Return real MLX Phase 3 parameter drift | Audit correction / retained |

## 10. Final conclusion

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
