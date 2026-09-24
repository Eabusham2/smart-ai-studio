# Smart AI Studio — Current-head branch audit and reconciliation (2026-09-24)

**Repository:** `Eabusham2/smart-ai-studio`  
**Authoritative branch:** `fix/real-benchmarks-final-32k`  
**Historical master baseline:** `06390e5357a07f80a8089ac28fc16c75461a48a6`  
**Frozen implementation verification head:** `e2163c6baa59148072ecc9dd8900997511f54251`  
**Documentation continuation head at start of this report:** `683952f4651880272d9d1ae2e7c2beb44397a6fb`

## 1. Audit policy

The feature branch is authoritative. Master is only the comparison baseline. No reset, squash, force-rewrite, commit deletion, or wholesale old-branch restore is used here.

The earlier audit documents remain part of the record:
- `BRANCH_AUDIT_fix_real_benchmarks_final_32k_2026-09-19.md`
- `FULL_MASTER_TO_FEATURE_DIFF_ANNOTATED_2026-09-19.md`
- four `FULL_MASTER_TO_FEATURE_DIFF_LINE_ANNOTATIONS_PART_*_2026-09-19.md` files
- `GEMINI_POSTMORTEM_2026-09-19.md`

Those cover commits 1–120 and the full line-level diff at that snapshot.  
`CURRENT_HEAD_DIFF_APPENDIX_2026-09-24.md` preserves the exact sequential GitHub diffs for commits 121–134, including temporary changes that were later corrected. Together they provide the requested non-destructive history rather than pretending intermediate mistakes never happened.

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

## 6. Current master→feature file ledger at the frozen implementation head

| Path | Status | Additions | Deletions | Total changed lines |
|---|---|---:|---:|---:|
| `.github/workflows/feature-audit.yml` | added | +112 | -0 | 112 |
| `ACTIVE_SESSION_GUARD_2026-09-18_GGUF_ONLY.md` | removed | +0 | -19 | 19 |
| `BRANCH_GUARD_GGUF_LEARNING_20260918.txt` | removed | +0 | -14 | 14 |
| `app_gui.py` | modified | +16 | -0 | 16 |
| `build_app.py` | modified | +3 | -3 | 6 |
| `consolidation/projected_daemon.py` | modified | +98 | -19 | 117 |
| `core/_mlx_engine_base.py` | modified | +202 | -68 | 270 |
| `core/autonomous_learner.py` | modified | +70 | -33 | 103 |
| `core/awake_auto_hook.py` | modified | +88 | -28 | 116 |
| `core/bitnet_rebuild_trainer.py` | modified | +42 | -5 | 47 |
| `core/controller_runtime.py` | modified | +99 | -9 | 108 |
| `core/drafter.py` | modified | +125 | -0 | 125 |
| `core/engines/bitnet_cpp_engine.py` | modified | +28 | -12 | 40 |
| `core/engines/gguf_engine.py` | modified | +19 | -9 | 28 |
| `core/gguf_lora_trainer.py` | modified | +40 | -4 | 44 |
| `core/gui_eval_panel.py` | added | +720 | -0 | 720 |
| `core/gui_generation_cap.py` | modified | +10 | -3 | 13 |
| `core/lif_gating.py` | modified | +61 | -0 | 61 |
| `core/media_learning.py` | modified | +67 | -2 | 69 |
| `core/online_consolidator.py` | modified | +34 | -9 | 43 |
| `core/training_memory.py` | added | +131 | -0 | 131 |
| `docs/BRANCH_AUDIT_fix_real_benchmarks_final_32k_2026-09-19.md` | added | +795 | -0 | 795 |
| `docs/FULL_MASTER_TO_FEATURE_DIFF_ANNOTATED_2026-09-19.md` | added | +7823 | -0 | 7823 |
| `docs/FULL_MASTER_TO_FEATURE_DIFF_LINE_ANNOTATIONS_PART_1_OF_4_2026-09-19.md` | added | +1372 | -0 | 1372 |
| `docs/FULL_MASTER_TO_FEATURE_DIFF_LINE_ANNOTATIONS_PART_2_OF_4_2026-09-19.md` | added | +1861 | -0 | 1861 |
| `docs/FULL_MASTER_TO_FEATURE_DIFF_LINE_ANNOTATIONS_PART_3_OF_4_2026-09-19.md` | added | +1548 | -0 | 1548 |
| `docs/FULL_MASTER_TO_FEATURE_DIFF_LINE_ANNOTATIONS_PART_4_OF_4_2026-09-19.md` | added | +1610 | -0 | 1610 |
| `docs/GEMINI_POSTMORTEM_2026-09-19.md` | added | +525 | -0 | 525 |
| `eval/_master_4000_base.py` | modified | +21 | -11 | 32 |
| `eval/app_cross_platform_bridge.py` | added | +680 | -0 | 680 |
| `eval/app_eval_runner.py` | added | +186 | -0 | 186 |
| `eval/conversation_teach_hardening.py` | modified | +120 | -12 | 132 |
| `eval/live_generation_stream.py` | modified | +14 | -3 | 17 |
| `eval/master_4000_runtime.py` | modified | +4 | -1 | 5 |
| `eval/phase4_pro_rsi.py` | modified | +649 | -80 | 729 |
| `eval/rsi_generation_memory_hardening.py` | modified | +16 | -2 | 18 |
| `eval/rsi_legacy_training_hardening.py` | modified | +71 | -48 | 119 |
| `eval/rsi_resume_hardening.py` | modified | +33 | -34 | 67 |
| `eval/stage_integrity_telemetry.py` | modified | +46 | -10 | 56 |
| `eval/swe_verifier_hardening.py` | modified | +55 | -15 | 70 |
| `master_4000_eval_suite.py` | modified | +7 | -0 | 7 |
| `pyproject.toml` | modified | +2 | -0 | 2 |
| `requirements.txt` | modified | +2 | -0 | 2 |
| `run_studio_complete.py` | modified | +30 | -3 | 33 |
| `tests/unit/test_app_eval_integration_contract.py` | added | +104 | -0 | 104 |
| `tests/unit/test_custom_backend_metadata_contract.py` | modified | +10 | -2 | 12 |
| `tests/unit/test_diverged_branch_reconciliation_contract.py` | added | +85 | -0 | 85 |
| `tests/unit/test_full_branch_audit_contract.py` | added | +226 | -0 | 226 |
| `tests/unit/test_prism_gguf_and_bitnet_contract.py` | modified | +2 | -1 | 3 |

The exact line-level history through commit 120 is in the existing four line-annotation documents; every added/removed line in commits 121–134 is preserved verbatim in `CURRENT_HEAD_DIFF_APPENDIX_2026-09-24.md`. Documentation commits after the frozen implementation head are intentionally not recursively self-annotated.
