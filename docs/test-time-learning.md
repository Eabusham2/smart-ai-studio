# Online learning and scoped parameter memory

Studio exposes several explicit learning workflows through **Memory Explorer → Learning & saved memory**, the `ProReasoningEngine.run_online_learning()` API, and `tools/online_learning.py`. These workflows update supported adapters during use. They do not require running a separate pretraining or meta-training job.

This implementation adds mechanisms from several research directions. It is not a reproduction of every published training recipe or benchmark. The [research coverage table](ttt-research-coverage.md) records the distinction for each paper discussed. The [persistent-memory guide](parametric-memory.md) describes ordinary Awake/Sleep consolidation.

## What is implemented

| Workflow | Implemented behavior |
| --- | --- |
| Stream a document or source directory | Read bounded chunks, train every selected chunk through the validated update transaction, and save a resumable cursor. No fixed first-16K-character learning cutoff. |
| Select useful evidence | Ask the active model for exact source spans relevant to a question; reject invented quotations; adapt on the selected spans. |
| Elicit knowledge | Generate source-grounded question/answer pairs, require verbatim supporting evidence, and study answers the source-free model did not already produce. |
| Few-shot adaptation | Form leave-one-out training tasks from 2–12 independent demonstrations, vary their ordering, adapt, and vote among bounded answer generations. |
| Query-only adaptation | Prefill a fixed context once, keep each layer's K/V immutable during adaptation, and update only query-side adapters on short causal spans. |
| Learn from rewards | Sample candidate answers, assign verifier or strict-majority rewards, and optimize a signed group-relative objective with a clipped importance ratio. |
| Consolidate candidate solutions | Select verified or majority-supported sampled traces and train on those traces, then retain the resulting scoped adapter. This is supervised trace consolidation, separate from the reward objective. |
| Retrieve a parameter memory | Train independent document adapters, index bounded descriptive metadata, pin a validated generation, select a matching entry, and answer with that adapter loaded. The source document is not inserted into the answer prompt. |

Online updates use the existing model's native chat formatting, finite-drift checks, candidate validation, checkpoint integrity checks, and rollback. EWC and bounded replay guard earlier behavior. A fixed amount of adaptive state does not guarantee indefinite retention of every fact.

## Select the memory lifetime

| Scope | Behavior |
| --- | --- |
| `model` | The existing default model-wide memory. Old checkpoint directories remain compatible. |
| `session` | Separate persistent state for a named conversation/session. Reuse the same key to resume it. |
| `project` | Separate persistent state and document catalog for a named project. |
| `request` | A temporary copy of the currently loaded, verified checkpoint. It includes optimizer and retention state. Restore the previous model and delete only this request's state when the operation ends. |
| `document` | An independent document's learned state. A catalog selection activates an immutable, read-only generation. |

Few-shot, query adaptation, reward learning, and candidate consolidation default to a temporary request. Document ingestion defaults to the currently selected persistent scope. API callers can explicitly choose a persistent scope for task adaptation. Temporary scopes cannot nest. A failed restoration is reported, and the surviving request checkpoint is retained for recovery.

Changing scope reloads the same selected model through its ordinary loader. This preserves runtime metadata and validates model/tokenizer identity. Inference and scope transitions share a lifecycle lock, including the complete lifetime of streaming generators. Explicit legacy adapters remain confined to the model-wide migration path; they must first become a validated checkpoint before being copied into a request.

In the GUI, choose **model**, **session**, or **project** in the learning panel. A named scope requires a key. **Apply memory** reloads the model and opens that scope's transcript. **Clear Chat** clears the selected transcript; it does not erase its learned adapter.

## Read a complete source

Create a request file:

```json
{
  "method": "ingest",
  "source": "/absolute/path/to/project",
  "strategy": "stream",
  "chunk_chars": 2048,
  "max_chunks": 8,
  "steps": 1,
  "learning_rate": 0.0001
}
```

`max_chunks` bounds each invocation. A `paused` result means more source remains. Run the same request again with the same model and scope to resume. Omitting `max_chunks` processes the source to completion, still in bounded chunks. Source revision, traversal order, strategy and training settings identify the job. A changed source starts a different job; changing a source during a pass is reported. Cancellation preserves completed, committed chunks and does not advance over an uncommitted chunk.

`strategy: "evidence"` additionally requires a `question`. It learns selected quotations rather than every source character. `strategy: "elicitation"` studies grounded question/answer pairs. The result reports scanned source characters, selected characters, trained chunks, cursor progress, and completion status. Selecting nothing is distinct from learning the entire source.

For Bonsai 2:

```bash
python tools/online_learning.py \
  --backend mlx --target bonsai2 \
  --model-path /models/bonsai2 \
  --scope project --scope-key my-project \
  --request request.json --output result.json
```

For CRACK, use `--target bonsai2-crack --model-path /models/bonsai2-crack`. Both require the matching publisher runtime on Apple Silicon. The runner uses existing local weights and does not initiate a model download. `--state-dir /path/to/isolated-state` is useful for experiments that should have separate application state.

For an ordinary local Hugging Face causal model, select `--backend cpu`, `cuda`, `mps`, or `torch`. A GGUF or BitNet bridge also needs exact training-parent and converter metadata; pass it through `--model-info metadata.json`. Missing support produces an error with a reason instead of a pretend update.

## Query adaptation and examples

```json
{
  "method": "query_adapt",
  "context": "The document to study ...",
  "query": "Which project uses the copper setting?",
  "max_context_tokens": 2048,
  "max_window_tokens": 64,
  "steps": 2
}
```

This implementation accepts at most **4,096 context tokens**, **512 tokens per adaptation span**, and **8 update steps**. It rejects an oversized context rather than silently truncating it. Supported families are the compatible full-attention, split-query Llama, Mistral, Qwen2 and Qwen3 implementations in Transformers or MLX-LM. Fused projections, unsupported caches, and packed Bonsai/JANG kernels do not qualify. The full context stays available during query adaptation and is included in the subsequent answer. This workflow improves use of context; it does not replace a growing context with constant-size memory.

```json
{
  "method": "few_shot",
  "demonstrations": [
    {"prompt": "ab", "completion": "ba"},
    {"prompt": "cat", "completion": "tac"},
    {"prompt": "door", "completion": "rood"}
  ],
  "query": "desk",
  "votes": 3,
  "steps": 1
}
```

The unanswered query must differ from each demonstration input. No query answer is supplied during training. The vote report includes ties. Ordering augmentation is semantics-preserving for independent examples; this implementation does not automatically invent geometry-specific transformations for ARC puzzles or perform the paper's prior fine-tuning.

## Learn from checked or consensus rewards

```json
{
  "method": "reinforce",
  "prompt": "Write a Python function square(x) that returns x squared.",
  "mode": "verifier",
  "verifier": {
    "kind": "python_tests",
    "function": "square",
    "cases": [
      {"args": [3], "expected": 9},
      {"args": [-4], "expected": 16}
    ],
    "timeout_seconds": 2,
    "memory_mb": 128
  },
  "rollouts": 4,
  "rounds": 2
}
```

Python verification accepts a restricted pure-function subset and JSON input/output cases. It enforces time and memory limits. It does not execute arbitrary assertion programs from a request. An optional `baseline_ms` produces a continuous timing reward only after correctness checks pass; local timings include noise and are not a hardware benchmark. A built-in `{"kind":"exact","expected":"..."}` verifier checks an exact training-task response. Python API callers can supply their own bounded `callable(text) -> reward` returning a finite value in `[0,1]`.

Verifier targets/cases evaluate sampled answers; they are not inserted into the model's rollout prompt. They are training supervision, so a meaningful quality evaluation still needs separate held-out questions or tests.

For consensus learning, set `mode` to `consensus` and omit the verifier. A strict majority is required. Ties, all-wrong/all-right reward groups, and empty usable rollouts produce no policy update. A consensus label is agreement, not proof of correctness. Limits are 2–12 rollouts per round, up to 8 rounds and 8 update steps, with bounded generated lengths.

The reward objective can improve the relative likelihood of rewarded actions while reducing that of rejected actions. Candidate acceptance measures that contrast separately from ordinary source NLL. Native GGUF/BitNet routes optimize the verified parent and check the converted serving model; they do not backpropagate through native serving kernels. Text-based rollout routes reconstruct and validate their training prefix; exact sampling-time token/logprob export is not available on every native server.

## Save and retrieve document parameters

```json
{
  "method": "bank_ingest",
  "key": "rendering-guide",
  "title": "Rendering conventions",
  "source": "/absolute/path/rendering.md",
  "keywords": ["renderer", "shader", "materials"],
  "strategy": "stream",
  "max_chunks": 8
}
```

Run until `completed`. Only a completed study with at least one accepted update can be registered. The registration verifies the source digest, model identity and exact generation. A paused document retains its progress but is not yet selected as a completed memory.

Use `{"method":"bank_list"}` to list entries and `{"method":"bank_select","query":"shader conventions","top_k":3}` to inspect lexical selection. `{"method":"bank_answer","query":"How should I name shader parameters?"}` loads the top matching pinned adapter for that answer and restores the prior model afterward. Selection uses a capped lexical index of metadata and source text; it can miss relevant memories. It selects one adapter rather than merging unrelated adapters. The answer result records which entry/generation was used.

`bank_export` accepts `key` and a new `destination` ZIP path. `bank_delete` removes the selected catalog entry and its pin; it does not delete the underlying document's current training checkpoint. Pinned generations survive normal checkpoint collection until their catalog entry is removed or replaced. Disk storage grows with the number of document memories.

## Backend support and verification boundaries

| Serving path | Context/evidence/few-shot updates | Reward objective | Fixed-KV query objective |
| --- | --- | --- | --- |
| Compatible MLX-LM | Supported adapters | Supported adapters | Supported full-attention split-query families |
| Bonsai 2 publisher MLX / CRACK JANG | Frozen packed base plus output-head adapter | Output-head adapter | Unavailable: no verified packed query-gradient path |
| Compatible Transformers on CPU/CUDA/MPS | PEFT, subject to actual device/model support | PEFT | Supported full-attention split-query families |
| llama.cpp GGUF / Prism GGUF / verified Prism speculative API | Compatible local HF parent, converter, deployed candidate scoring | Same verified parent bridge | Unavailable through the native serving KV cache |
| bitnet.cpp | Compatible parent, merge, rebuild I2_S, deployed candidate scoring | Same verified rebuild bridge | Unavailable through the native serving KV cache |
| Pinned Strata / legacy unsupported native routes | Unavailable | Unavailable | Unavailable |
| Other inference-only controllers and image/audio/video generators | These text-learning methods are unavailable | Unavailable | Unavailable |

Use `--capabilities` or the learning panel's **Check model support** for the currently loaded selection. API methods are shared across backends, but actual gradient/deployment support is checked at runtime. Unsupported routes preserve their existing generation behavior and original context limits.

Tests execute real small CPU model updates for query and reward objectives, reload checkpoints, exercise scoped/temporary state, and validate native bridge transactions. Packed MLX protocol fixtures execute on Linux; native MLX tests require Apple Silicon. The target Bonsai/CRACK fresh-process recall evaluator is `tools/evaluate_persistent_memory.py`. Full target-model recall, throughput and Metal memory usage require those hardware runs and are not inferred from small-model tests.

Results report complete operation wall time, sampled process RSS, update metrics, and checkpoint/selection identifiers. Process RSS excludes separately owned server processes and is distinct from GPU or Metal peak allocation. Parameter updates can make an answer worse; rollback and finite retention tests reduce that risk but do not establish unlimited reliable memory.
