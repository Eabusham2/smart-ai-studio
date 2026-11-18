# Original and managed implementations

Studio restores the original algorithms from commit `d3cead010685a04c95e97b2469c07fa3de627156`, before PR #17, as working implementations alongside the persistent-learning features. **The default is `legacy` learning and one Best Work pass.** Selecting a new feature no longer requires replacing the original implementation.

## Choose the behavior

Open **Memory DB → Learning & saved memory…**. Under **Learning implementation**, choose **legacy** for the original Awake, Learn, RSI and Sleep behavior, or **managed** for scoped persistent context learning. Click **Apply implementation**. A resident model reloads through its normal loader; an unsuccessful or cancelled change attempts to restore its previous model, scope and checkpoint. The selection is saved for the desktop's next start.

The learning implementation does not change the selected Best Work pass count. **Branches → Best Work passes → Apply passes** selects one original pass or two or more sequential refinement passes. The original branch router, verifier, tool authorization and tool-call limits remain in use. For managed archived chat, even a single pass uses the preparation needed to preserve committed retirement receipts.

## What was restored

| Area | Original implementation | Managed or additional implementation retained |
| --- | --- | --- |
| MLX updates | Whole-example ChatML next-token loss, a fresh AdamW optimizer for each batch, the original EWC matching-key rule, original attention/MLP adapter selection and equal weighting of complete Fisher anchors. | Bounded windows, selected projections, resumable optimizers, validation, query/reward objectives and compatible packed-model adapters. |
| Torch, GGUF and BitNet updates | The original completion-masked per-row loops, optimizer resets, snapshot/L2 helpers, formatting and standalone trainer deployment paths. | Transactional learning, verification, context streaming, scoped checkpoints and rollback. |
| Direct CUDA loading | The original tokenizer/model load with BF16, automatic placement, `eval()` and direct model/tokenizer handles when the original CUDA route is selected in legacy mode. | Managed Transformers controller loading and the newer CPU/MPS fallback support. |
| Awake context handling | Original 80% trigger/60% target algorithm and its original consolidation service, including background MLX work and synchronous hard-capacity consolidation. | Recall-gated retirement, durable receipts, archive projection, streaming and recovery across eligible exchanges. |
| Learn and RSI | Original evidence matching, reward acceptance, source extraction limits and direct backend training loops. | Full-source streaming, selected evidence and transactional training. |
| Sleep | The original `SleepConsolidationDaemon` and verified-episode cycle. | Active-backend managed Sleep for verified episodes or completed chat exchanges. |
| Best Work | The original single-pass streaming/tool handoff loop. | Explicit sequential refinement, per-pass measurements and completed-draft fallback when a later refinement cannot fit. |
| Chat display | Original complete in-memory transcript in legacy mode. | Persistent archive and bounded paging in managed mode. Switching implementations retains each mode's current transcript. |
| Export | Original **Export Chat to Markdown** tool action using `chat_history_export.md`. | **Export complete transcript as…** in the chat context menu retains the full archived-transcript export and destination picker. |
| Learning telemetry | Original learned-parameter/EWC status remains visible. | Measured adapter-update status appears alongside it. |

The original public `AwakeOnlineConsolidator` and `AutonomousLearner` constructors select the original implementations. Their managed counterparts are exported as `ManagedAwakeOnlineConsolidator` and `ManagedAutonomousLearner`. The factories `make_online_consolidator(..., implementation=...)` and `make_autonomous_learner(..., implementation=...)` select explicitly. Mode-aware backend methods accept `learning_mode="legacy"` or `learning_mode="managed"`. Managed-only arguments require the managed algorithm; incompatible explicit combinations raise an error.

`tools/online_learning.py` and `tools/evaluate_persistent_memory.py` explicitly select managed mode because they exercise the newer online tasks and durable-memory protocol. Existing supported direct backend calls and standalone trainers remain available.

New desktop Eval runs record the selected learning implementation in their run configuration. Existing Eval configurations without that field continue using their original managed checkpoint layout, so they can resume. Both implementations keep Eval weights inside the run directory and do not read or write the desktop's learned weights.

## Saved state stays separate

An explicitly selected legacy backend stores its managed-format durability records under a separate `legacy-implementation` directory. Managed mode retains its existing state root. Model identity, scope, checksums and immutable generations still protect both modes. Neither implementation overwrites the other one's committed checkpoint. MLX external exports also reject destinations inside either implementation's state and serialize writes to a shared external destination.

Changing modes does not transfer learned facts automatically between the two adapters. Switching back restores that mode's own state. An explicitly selected old checkpoint remains an import source; it is not silently applied to another model or overwritten as part of the switch.

The restored algorithms retain correctness protections needed by the combined application: real capability checks, MLX owner-thread execution, cancellation, rollback, checkpoint integrity and read-only pinned scopes. The legacy Sleep panel reports the actual remaining queue or an unavailable count instead of inventing a count. These protections do not replace the original training objectives or Sleep daemon.

Packed Bonsai 2 and CRACK did not have a working generic pre-PR #17 adapter layout. Their compatible packed implementation remains available rather than pretending the old generic layout can train those tensors. Strata remains inference-only; GGUF/Prism/BitNet training still depends on a compatible parent and deployment toolchain.

## Verification

The tests compare restored MLX parameter updates and Fisher values with the original equations, and compare native trainer outputs with the original per-row algorithms. Additional tests cover original public entry points, original single-pass behavior, managed refinement, both exports, both transcript views, original Sleep dispatch, scope/mode transitions, rollback, cancellation and protection against cross-mode checkpoint overwrites.

The normal CI matrix continues to run on Linux, macOS and Windows. Both native MLX CPU and Metal lanes run the existing small packed Bonsai/CRACK tests and the new original-algorithm parity tests; these required cases cannot be skipped. Passing those tests does not establish full 27B checkpoint recall, latency or memory performance. Those measurements require the separate full-model evaluator on suitable hardware.

Both implementations still operate within a finite model context. Managed mode can retain learned information beyond the active token window after successful validation and checkpoint publication. It cannot guarantee unlimited exact recall or that every possible request will fit.
