# Persistent parametric memory

On a supported text model, Studio can learn from an old exchange, save the updated adapter, and omit that exchange from subsequent model input. The full exchange stays visible in the chat. Later answers can draw on the loaded parameters with no lookup of the retired source text. A parameter update is lossy, so a retired exchange can still be recalled incorrectly.

## What happens to a chat

1. Studio counts the packed history and current prompt against the selected Context budget. An eligible backend starts awake consolidation at **80%** of that budget. It selects old, complete user/assistant exchanges and keeps the most recent complete exchange. System instructions, tool exchanges and unfinished turns stay in text.
2. The trainer makes three passes over the selected dialogue using bounded, overlapping causal-token windows. The one-token overlap supplies the preceding token for the next prediction. Both sides of an ordinary exchange contribute training targets, including a fact supplied by the user and acknowledged only briefly by the assistant.
3. The service scores short continuations drawn from the source, a held-out foundation example and up to two earlier rehearsal examples. It runs the same scoring on the candidate. Acceptance requires finite scores, improvement in aggregate source loss and no source or retention loss increase greater than 0.10 nats per target token. This is a teacher-forced continuation measurement. Answer generation from a fresh question is evaluated separately.
4. The trainer checks finite, nonzero parameter change and stages the checkpoint. One atomic pointer change publishes the parameter generation and its retirement receipt together. Only then does the active prompt shrink. The service can consolidate another batch until the active prompt approaches **60%** of Context. If the remaining input still exceeds the physical window, generation reports the capacity error.

**Sleep** starts one batch on the loaded backend, including recent complete exchanges when eligible. It can run below the awake watermark. **Learn** uses the shared transaction after source verification. **RSI** trains its self-generated revision. Its training row contains the revision and a generic instruction. The source question, expected answer and verification verdict stay out of that row. Cancellation or rejection before publication keeps the previous checkpoint current and the uncommitted exchanges available as input. [Awake hook](../core/awake_auto_hook.py) · [Consolidation service](../core/online_consolidator.py) · [Training windows](../core/learning_data.py) · [Learn and RSI](../core/autonomous_learner.py)

## Backend capabilities

| Loaded text runtime | How the candidate reaches inference | Conditions for committing an update |
| --- | --- | --- |
| Native MLX language models | LoRA updates compatible language projections on the loaded model. MLX model operations run on its owner thread. | Compatible adapter layout, finite gradients and drift, candidate scoring, and checked MLX checkpoint serialization. |
| Packed Bonsai 2 and Bonsai 2 CRACK on MLX | A fresh adapter targets the language output head. The packed base supplies its existing logits and the low-rank branch learns a correction. Existing supported adapters retain their verified layout. | A supported packed loader, valid language head dimensions and matching checkpoint tensors. The packed backbone stays frozen. |
| Transformers controllers on CPU or CUDA | PEFT updates compatible language projections. The updated in-process model scores the candidate, and the PEFT checkpoint is bound to the selected base. | A causal language module, supported projection targets, compatible installed training dependencies and validated persistence. |
| GGUF with llama.cpp or supported Prism runtimes | A verified compatible Hugging Face parent receives the PEFT update. The bridge converts the adapter to GGUF, loads it into the selected inference runtime, and scores the deployed candidate. | Matching parent lineage and tokenizer evidence, training resources, converter, native adapter loading and completion scoring. Native speculative servers additionally need a verified adapter API. |
| BitNet with bitnet.cpp | Train the verified BF16 parent, merge the adapter, rebuild I2_S and score the rebuilt model after loading it. | Verified parent lineage, a working conversion/quantization toolchain and a scoreable served deployment. |
| Strata | Strata serves the complete active text context using its native context limit. | The current integration lacks a verified parameter-update and adapter-loading path. Awake and Sleep retain chat text on this runtime. |

Each loader checks training capability for the selected model and device. The native bridges verify the deployed result because conversion and quantization can change what an adapter learned. Unsupported backends continue serving within their context limits. [MLX adapter selection](../core/mlx_learning.py) · [Transformers trainer](../core/torch_context_learning.py) · [Native deployment transaction](../core/native_context_learning.py) · [Training lineage](../core/training_lineage.py) · [Strata backend](../core/engines/strata_engine.py)

## Retention, storage and resume

EWC uses a diagonal Fisher estimate from foundation examples and penalizes movement of important adapter parameters relative to their pre-update values. Ordinary consolidation also rehearses one foundation example and up to two earlier continuations. The model-bound rehearsal pool stores up to eight bounded cue/continuation pairs. Rehearsal and validation use those pairs during learning. Ordinary generation receives the active transcript and the loaded weights.

Checkpoint generations reside under `learned_memory/<identity-digest>/` in the Studio data directory. The identity covers the selected base and adapter format, with verified source weights or revision and tokenizer/configuration evidence as available. The store keeps the current and preceding committed generations. Stage files can temporarily require more disk, especially while rebuilding a BitNet deployment. The adapter parameter count stays fixed for a given layout. Transcript text and historical retirement receipts grow with use. The receipt index holds turn IDs on disk and checks membership in batches. The GUI also holds its visible archive in RAM while the tab is open, so bounded model context does not make the whole application's memory or storage constant. [State store](../core/learning_state.py) · [Transcript archive and retirement receipts](../core/context_memory.py)

Reopening Studio restores the visible text transcript for its model tab. Loading the matching checkpoint restores the corresponding active view. Memory Explorer shows the generation ID. **Clear Chat** and **New Chat** clear that tab’s transcript while preserving learned weights. API callers can use a distinct `ChatArchive(db_path, session_name)` for each resumable conversation. A caller that resubmits plain lists can give its consolidator a stable `session_id` to identify that session across processes.

When Studio discovers an older shared MLX adapter without a recorded base, it lists the file in **Memory Explorer**. Load its original base model, open Memory Explorer, choose **Import MLX adapter…** and select the raw `.safetensors` file. Studio reloads that base, validates and attaches the selected tensors, and preserves the source file. The next accepted learning update saves a model-bound generation. A shape match cannot establish which base produced an unlabelled adapter. [MLX checkpoint validation](../core/mlx_adapter_persistence.py) · [Memory Explorer](../app_gui.py)

## Measure recall on Apple Silicon

Use a Mac with working MLX and the selected Bonsai 2 and CRACK snapshots. From the repository root, run the following command with the Studio Python environment active. Each run creates its own directory under the chosen output path and uses isolated model state.

```bash
python tools/evaluate_persistent_memory.py --model both --cycles 3 --output-dir ./eval_results/persistent_memory
```

For local snapshots outside the model cache, supply `--bonsai-path /path/to/bonsai` and `--crack-path /path/to/crack`. Downloads require `--allow-download`. The script writes `job.json`, per-stage records and `report.json`. It starts separate processes for baseline, update and reload measurements, asks for generated answers with empty source history, and retests earlier associations alongside unrelated controls.

By default, passing requires at least 75% final normalized exact-match fact accuracy, improvement over baseline, retention of previously correct facts, matching post-update and post-reload fact responses, and retention of initially correct controls. The implementation tests ran on Linux. The two 27B Apple Silicon runs are pending, so their recall rates and resource requirements under this evaluation remain unmeasured. [Fresh-process evaluation script](../tools/evaluate_persistent_memory.py)

### Implementation verification

The regression tests cover complete token coverage, model and tokenizer isolation, checkpoint corruption, cancellation on both sides of atomic publication, rejected-candidate rollback, restored optimizer state, context retirement on subsequent GUI turns, and transcript restart. The real CPU tests construct a small local GPT-2/PEFT model and execute the production update and validation service. They check improved source-free answer likelihood, held-out retention, and cold-reload score parity. These tests establish the update and persistence mechanics; they do not establish 27B model recall quality. [State tests](../tests/unit/test_learning_state.py) · [CPU and native bridge tests](../tests/unit/test_native_context_learning.py) · [Context and GUI tests](../tests/unit/test_persistent_context_memory.py)

Packed MLX protocol fixtures exercise the adapter forward path and numerical update, and separate Apple-only tests exercise MLX differentiation. Native deployment transaction fixtures verify conversion, activation and rollback contracts. Full Bonsai/CRACK Metal inference, real native GGUF deployment and BitNet quantizer execution require the corresponding hardware and runtime integration tests. [MLX tests](../tests/unit/test_mlx_context_memory.py) · [Persistence tests](../tests/unit/test_mlx_adapter_persistence.py)

## Research scope

TTT-E2E meta-trains models to learn from context through next-token updates. Titans trains a dedicated long-term neural memory module. Studio applies periodic low-rank updates to existing inference checkpoints. The supported checkpoint retains its original architecture and finite attention window. Adapter capacity and the retention checks are finite, so longer chat histories carry no guarantee of perfect recall. [TTT-E2E paper](https://arxiv.org/abs/2512.23675) · [Titans paper](https://research.google/pubs/titans-learning-to-memorize-at-test-time/)
