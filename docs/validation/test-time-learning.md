# Online learning validation record

This record covers the online-method, scoped-memory, document-bank and bounded-transcript additions on `feature/persistent-context-learning`, following commit `8ec3dc8c786dc4bc8d9f55308225768c38940b73`.

## Completed runs

| Run | Result | Measured duration |
| --- | --- | --- |
| Unit suite, excluding the two GUI modules run separately | **1,327 passed, 6 skipped, 58 subtests passed** | 100.65 s |
| Real Tk tests: existing Best Work rendering/steer handoff and new paging/learning-panel tests | **5 passed** | 1.55 s |
| Final lifecycle, CLI/paging, Pro routing and media follow-up after the changed-source restoration guard | **20 passed** | 2.44 s |
| Existing offline `tests/live` suite | **18 passed, 1 skipped** | 499.57 s |
| Compilation, CLI help and `git diff --check` | Passed | — |

The follow-up run overlaps the broad unit suite. Its additional test confirms that changing the model identity during a reload cannot be reported as a successful restoration of the previous learned model.

The live-suite skip is the opt-in 27B Apple Silicon/Metal module. Native MLX tests also skip when MLX is unavailable. The CPU tests execute actual tensors, forward passes, gradients and checkpoint reloads on small locally constructed models; no model download or external search is required for these test runs.

The first broad pass found one legacy retirement fixture that did not supply the evidence now required by the stronger recall gate, and three Pro-routing failures caused by a new wrapper hiding its callable signature. The fixture now provides recallable synthetic facts, and the streaming wrapper preserves the original signature and callbacks. The completed rerun above passed those regressions.

## Environment

- Linux x86_64; no CUDA device and no Apple Metal runtime.
- Python 3.12.14, PyTorch 2.15.0+cpu, Transformers 5.23.0, PEFT 0.21.3, pytest 9.2.0.
- GUI streaming checks use a separate Python 3.13 environment with threaded Tcl/Tk 8.6.16. The bundled Tcl 9 runtime in the CPU environment cannot perform the same cross-thread GUI handoff tests correctly; that environment issue was reproduced before these changes.
- Offline guards: `SMARTAI_TEST_OFFLINE=1`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `RUN_REAL_WEIGHTS_EVAL=0`.
- CPU test runs limit OpenMP/OpenBLAS threads to one. GUI tests use a private authenticated dummy display.

The suite reports existing Pydantic/pytest configuration deprecations and Transformers/PEFT compatibility warnings. These are not target-model quality measurements.

## What the tests establish

**Query adaptation.** Real CPU Llama, Mistral, Qwen2 and Qwen3 models produce matching fixed-cache span losses and full causal-forward losses. A hook verifies one context prefill and short adaptation spans. Non-query parameters remain byte-identical. Managed tests verify selective query optimizer-state resume, rejected/cancelled candidate rollback, checkpoint receipts and cold-reload scoring parity.

**Policy updates.** Small real Torch models raise rewarded output probability and lower rejected output probability. Uniform reward groups leave parameters unchanged. A workflow test samples real CPU model outputs, evaluates them, performs a PEFT policy update, measures positive rewarded-versus-rejected likelihood contrast and reloads the committed checkpoint. Native bridges pass the objective to their verified parent trainers and exercise deployment/rollback contracts.

**Bonsai/CRACK contracts.** CPU protocol fixtures exercise both publisher MLX and JANG packed-head layouts. The adapted head changes while the packed base stays unchanged. Unsupported query-only updates fail explicitly. Separate conditional tests exercise native MLX differentiation when that runtime is installed.

**Persistence and isolation.** Tests cover source/tokenizer identity, model/session/project/document separation, corruption, stale parent refusal, cancellation before and after publication, pinned old document generations, independent request copies, optimizer and EWC restoration, request cleanup, streaming exclusion, and restoration failure. A real PEFT test reloads an older pinned generation, reproduces its behavior and continues learning from a scoped copy.

**Streaming and retention.** Tests visit a source directory containing 134 files, check Unicode and line boundaries, resume/cancel, verify bounded indexed cursors, reject fabricated evidence, keep unanswered queries out of few-shot targets, enforce strict consensus majority, and reject old-fact recall regressions. A valid weight update without complete generated recall keeps the original chat turns active.

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
