# Hybrid learning

**Hybrid learning and Pro branches are enabled by default.** Hybrid is a third implementation alongside `legacy` (the original algorithms) and `managed` (persistent context learning). An existing explicit selection saved in the desktop database is respected; a new installation starts with `hybrid`.

## What Hybrid combines

Hybrid uses the original per-row update behavior for compatible ordinary lessons inside the persistent learning transaction. The transaction measures the candidate, checks retention, saves an immutable model-bound checkpoint and records which context may be retired. The model performs one update procedure for each submitted batch.

The precise numerical path depends on the backend and request. MLX ordinary examples use the original ChatML objective and fresh AdamW updates, with bounded accumulation when a source row must be split. Native and Transformers completion-only lessons can use the original completion-masked per-row optimizer. Explicit masks, weights, prompt-wide supervision and long rows retain the handling required to learn the intended targets. Query-only adaptation and reward-based objectives use their specialized update paths. Checkpoint metadata identifies the selected algorithm.

Before choosing the original native update, Hybrid checks that its conditioning tokens agree with the prefix used by the learning and recall path. When those formats differ, including an untemplated causal model or an incompatible multi-turn template, Hybrid uses the bounded managed update with the same acceptance checks.

Original and managed choices keep their existing algorithms. Their public constructors and explicit backend selections remain available. Hybrid adds its own orchestration and state; it does not convert an existing original or managed checkpoint in place.

| Choice | Learning and memory behavior | Saved state |
| --- | --- | --- |
| `legacy` | Original Awake, Learn, RSI and standalone Sleep daemon; original optimizer behavior. | Existing original implementation namespace and in-memory chat behavior. |
| `managed` | Existing persistent context learning, validation, streaming and online objectives. | Existing managed checkpoints, scoped memories and archived chat. |
| `hybrid` | Original update behavior combined with persistent validation, saved memory and context retirement; specialized updates when the task requires them. | Independent hybrid checkpoints, scoped memories and archived chat. |

## Select an implementation

Open **Memory DB → Learning & saved memory…**, choose `legacy`, `managed` or `hybrid`, and click **Apply implementation**. Changing a resident model reloads it through the existing loader and restores the selected implementation's own state. A failed or cancelled change attempts to restore the previous runtime. Switching back resumes that implementation's saved memory; learned facts are not automatically copied between implementations.

Managed and hybrid also keep separate saved project/session selections. The panel refreshes the selected scope after switching, and startup restores the selected implementation's own preference. Adapter exports reject destinations inside existing immutable checkpoint generations, including a different state directory or a backend with no bound store. Referring to the owning backend's current checkpoint publishes a successor without overwriting that generation.

Hybrid supports **model**, **session**, **project** and temporary learning scopes, document memories and the online learning tasks exposed by the loaded backend. Its **Sleep EWC** control uses the current model and its verified-episode or completed-chat transaction. Selecting `legacy` continues to open the original standalone Sleep daemon.

The setting is `learning_implementation="hybrid"`, or `LEARNING_IMPLEMENTATION=hybrid` in an environment configuration. Explicit legacy or managed settings still select those implementations. Existing Eval run configurations without an implementation field retain their managed checkpoint layout so they can resume; newly launched desktop Eval runs record the selected implementation.

The online-learning CLI and the full Bonsai/CRACK persistent-memory evaluator accept `--learning-implementation hybrid`. Their existing command-line default remains `managed`, and saved evaluator jobs without the field keep that original selection. This allows either persistent implementation to be measured explicitly without changing an existing evaluation run.

## Pro branches

The **Pro branches** checkbox starts checked. The existing entropy router chooses the candidate count; enabling Pro does not force every prompt to use the largest branch count. The selected candidate continues through the existing verifier and authorized tool handoff.

**Best Work passes** remains an independent setting with a default of **one**. Select two or more under **Branches → Best Work passes** for sequential refinement. Turning off **Pro branches** retains ordinary streaming.

## Memory and backend limits

Hybrid can keep successfully learned information after its source leaves the active prompt. Source retirement still requires the validation and durable publication checks. A failed update keeps the source available rather than claiming it was learned. This does not make physical context capacity or exact recall unlimited.

MLX, Transformers and eligible GGUF, Prism and BitNet learning paths use their existing runtime and conversion requirements. Packed Bonsai 2 and CRACK keep compatible packed-model training paths. Strata and other inference-only runtimes remain usable for inference and report unsupported learning operations. Small real-tensor tests exercise update behavior and rollback; full 27B quality and memory measurements require suitable target hardware.
