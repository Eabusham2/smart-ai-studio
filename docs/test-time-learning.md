# Online learning and scoped parameter memory

Studio exposes several explicit learning workflows through **Memory Explorer → Learning & saved memory**, the `ProReasoningEngine.run_online_learning()` API, and `tools/online_learning.py`. These workflows update supported adapters during use. They do not require running a separate pretraining or meta-training job.

This implementation adds mechanisms from several research directions. It is not a reproduction of every published training recipe or benchmark. The [research coverage table](ttt-research-coverage.md) records the distinction for each paper discussed. The [persistent-memory guide](parametric-memory.md) describes ordinary Awake/Sleep consolidation.

## What is implemented

| Workflow | Implemented behavior |
| --- | --- |
| Stream a document or source directory | Read bounded chunks, train every selected chunk through the validated update transaction, and save a resumable cursor. No fixed first-16K-character learning cutoff. |
| Select useful evidence | Ask the active model for exact source spans relevant to a question; reject invented quotations; adapt on the selected spans. |
| Elicit knowledge | Generate source-grounded question/answer pairs, require verbatim supporting evidence, and study answers the source-free model did not already produce. |
| Few-shot adaptation | Form leave-one-out training tasks from independent demonstrations, vary their ordering, adapt, and vote. ARC grids additionally use reversible geometry/color transforms and inverse-transformed votes. |
| Query-only adaptation | Prefill a fixed context once, keep each layer's K/V immutable during adaptation, and update only query-side adapters on short causal spans. |
| Learn from rewards | Sample candidate answers, assign verifier or strict-majority rewards, and optimize a signed group-relative objective with a clipped importance ratio. |
| Discover a better solution | Select reusable candidate states with PUCT, sample improvements, and train with adaptive entropic advantages that emphasize the best verified outcomes. |
| Consolidate candidate solutions | Select verified or majority-supported traces, including supplied agent trajectories; learn individual and peer-conditioned completions; generate a fresh answer after an accepted update. |
| Retrieve a parameter memory | Train independent document adapters, index the complete source incrementally, pin validated generations, and ask each selected model whether its memory is useful. The ordinary solver synthesizes the accepted recalled answers. Original documents are not inserted into the answer prompt. |

Online updates use the existing model's native chat formatting, finite-drift checks, candidate validation, checkpoint integrity checks, and rollback. EWC and bounded replay guard earlier behavior. A fixed amount of adaptive state does not guarantee indefinite retention of every fact.

## Select the memory lifetime

| Scope | Behavior |
| --- | --- |
| `model` | The existing default model-wide memory. Old checkpoint directories remain compatible. |
| `session` | Separate persistent state for a named conversation/session. Reuse the same key to resume it. |
| `project` | Separate persistent state and document catalog for a named project. |
| `request` | A temporary copy of the currently loaded, verified checkpoint. It includes optimizer and retention state. Restore the previous model and delete only this request's state when the operation ends. |
| `document` | An independent document's learned state. A catalog selection activates an immutable, read-only generation. |

All learning workflows default to the currently selected persistent scope. Select **Temporary practice**, or pass `"temporary": true`, to fork a verified request checkpoint and restore the previous state afterwards. API callers can also select a persistent `scope` explicitly. Temporary scopes cannot nest. A failed restoration is reported, and the surviving request checkpoint is retained for recovery.

Changing scope uses an authenticated same-model MLX adapter switch when supported, with a cold reload fallback. Both routes validate the model/tokenizer identity. Changing adapters invalidates generation caches; keys and values from another adapted model are never reused. Inference and scope transitions share a lifecycle lock, including the complete lifetime of streaming generators. Explicit legacy adapters remain confined to the model-wide migration path; they must first become a validated checkpoint before being copied into a request.

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

There are no fixed 4,096-context, 512-span or eight-step ceilings. `max_context_tokens`, when supplied, is an explicit caller budget; the model's actual context remains a bound and oversized input is rejected without truncation. Supported full-attention split-query families include compatible Llama, Mistral, Qwen2 and Qwen3 decoders. Packed Bonsai/JANG Qwen3.5 layouts can adapt the terminal full-attention query while retaining their frozen hybrid prefix; replay is checked against the native forward result. Native GGUF/BitNet routes use the verified parent model's full-KV objective and validate the converted serving model. The full context remains available for the subsequent answer, so query adaptation does not supply constant-memory context compression.

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

The unanswered query must differ from each demonstration input. No query answer is supplied during training. The vote report includes ties. `arc_augment: "auto"` recognizes rectangular grids of integers 0–9, including `input`/`output` demonstrations. It applies eight square symmetries with two reversible palettes and inverse-transforms predictions before voting. `arc_augment: false` disables those transforms. Invalid grid predictions are reported separately. This is online adaptation; the paper's prior fine-tuning remains excluded.

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

Python verification accepts a restricted pure-function subset and JSON input/output cases. Each checked worker has a wall-time budget. Linux applies a hard address-space limit before reading and compiling the candidate. On macOS, `memory_mb` bounds the worker's resident memory. A separate process checks live RSS every 10 ms and terminates a worker observed over budget. The verifier also checks the kernel's peak RSS for that worker before awarding a reward, so an allocation freed between checks can still disqualify the result. Inherited OS resource limits continue to apply. It does not execute arbitrary assertion programs from a request. An optional `baseline_ms` produces a continuous timing reward only after correctness checks pass; local timings include noise and are not a hardware benchmark. A built-in `{"kind":"exact","expected":"..."}` verifier checks an exact training-task response. Python API callers can supply their own bounded `callable(text) -> reward` returning a finite value in `[0,1]`.

Verifier targets/cases evaluate sampled answers; they are not inserted into the model's rollout prompt. They are training supervision, so a meaningful quality evaluation still needs separate held-out questions or tests.

For consensus learning, set `mode` to `consensus` and omit the verifier. A strict majority is required. Ties, all-wrong/all-right reward groups, and empty usable rollouts produce no policy update. A consensus label is agreement, not proof of correctness. Rollouts, rounds, update steps and generated lengths have configurable positive budgets, without the earlier fixed maxima. The serving context and available memory still constrain each operation.

The reward objective can improve the relative likelihood of rewarded actions while reducing that of rejected actions. Candidate acceptance measures that contrast separately from ordinary source NLL. Native GGUF/BitNet routes optimize the verified parent and check the converted serving model; they do not backpropagate through native serving kernels. Text-based rollout routes reconstruct and validate their training prefix; exact sampling-time token/logprob export is not available on every native server.

### Discovery with entropic rewards and candidate reuse

```json
{
  "method": "discover",
  "prompt": "Write a Python function sum_squares(values) that returns the sum of squares.",
  "verifier": {
    "kind": "python_tests",
    "function": "sum_squares",
    "cases": [
      {"args": [[1, 2, 3]], "expected": 14},
      {"args": [[]], "expected": 0}
    ]
  },
  "rounds": 3,
  "rollouts": 4,
  "max_tokens": 128,
  "archive_size": 16
}
```

`discover` uses a separate adaptive entropic weighting rule. It adjusts the reward tilt by bisection under a KL concentration budget, then forms leave-one-out advantages. The default concentration target is `ln(2)`. Uniform rewards produce no gradient update; unreachable concentration targets, ties and finite bounds are handled explicitly. Each informative newly sampled group receives one update rather than repeated off-policy training on archived completions.

The search archive stores candidate solutions, rewards and lineage. Its PUCT score combines the best observed direct-child reward with a rank-based exploration prior. Expanding a candidate increments visitation counts along its ancestry. `children_per_expansion` defaults to two and can be increased; archive storage stays within the configured capacity while preserving initial seeds. Optional `seed` accepts an initial candidate or a list. A `schedule` list can override rollouts, steps, generated tokens, temperature, learning rate, exploration and entropic concentration by round. The output records parent selections, schedules, update metrics, the best result and the archive.

For Python discovery, an omitted `baseline_ms` uses **1 ms as a reward normalization unit**, not as a claim about an existing implementation's speed. Correct candidates receive `1 / (1 + median_cases_ms)`; invalid candidates receive zero. This supplies a continuous reward without asking the user to benchmark a seed first.

Compatible Torch and MLX discovery routes capture the actual initial adapter policy and export exact sampled token IDs and log probabilities, including EOS. The default reference KL coefficient is 0.1 when that support is available; set `reference_kl_coef` explicitly to change it or zero to disable it. The frozen reference is restored after scoring, including errors and cancellation. Exact sampled actions also drive reward validation and have a digest-checked checkpoint audit record. Native text-only deployments report unavailable reference scoring; explicitly requesting it fails instead of manufacturing log probabilities. The entropic concentration budget and initial-policy KL penalty are separate controls. This remains a local TTT-Discover variant, without the paper's distributed rollout infrastructure or benchmark configuration. [Method and equations](https://arxiv.org/html/2601.16175v1)

`consolidate_candidates` can take `trajectories` containing strings or `{ "agent": "name", "messages": [...] }` records. It learns accepted individual and peer-conditioned completions. After an accepted update, `synthesize: true` (the default) generates a new answer; the original `best_sample` remains available. This does not add TTT-Consolidate's offline meta-training.

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

Use `{"method":"bank_list"}` to list entries and `{"method":"bank_select","query":"shader conventions","top_k":3}` to inspect lexical selection. The index covers the entire source, using incremental decoding and SQLite postings. `bank_answer` defaults to `routing: "model"`: it selects candidates, temporarily loads each pinned generation, asks for relevant recalled facts, restores the original model, and synthesizes an answer from up to `top_k` accepted memories. The candidate and generation budgets are configurable. Ordinary `solve` and `stream_solve` also use this path when the selected scope owns document memories; set `engine.memory_auto_recall = False` to disable it. Existing explicit `solve_with_memory(..., routing="lexical")` remains available.

The result identifies accepted/rejected candidates and exact generations. Recalled-answer caching is bounded and keyed by model, scope, immutable generation and query. This cache avoids repeated recall work; it is not shared KV across different adapters. Model gating and lexical candidate ranking can both miss relevant information. No document source text is loaded into the answer prompt.

`bank_export` accepts `key` and a new `destination` ZIP path. `bank_delete` removes the selected catalog entry and its pin; it does not delete the underlying document's current training checkpoint. Pinned generations survive normal checkpoint collection until their catalog entry is removed or replaced. Disk storage grows with the number of document memories.

## Backend support and verification boundaries

| Serving path | Context/evidence/few-shot updates | Reward objective | Fixed-KV query objective |
| --- | --- | --- | --- |
| Compatible MLX-LM | Supported adapters | Supported adapters | Supported full-attention split-query families |
| Bonsai 2 publisher MLX / CRACK JANG | Terminal query/down-projection adapters on supported hybrid layouts; existing head-only checkpoints stay compatible | Compatible packed adapters | Terminal full-attention replay with a frozen hybrid prefix |
| Compatible Transformers on CPU/CUDA/MPS | PEFT, subject to actual device/model support | PEFT | Supported full-attention split-query families |
| llama.cpp GGUF / Prism GGUF / verified Prism speculative API | Compatible local HF parent, converter, deployed candidate scoring | Same verified parent bridge | Compatible verified parent's KV cache, then deployed candidate scoring |
| bitnet.cpp | Compatible parent, merge, rebuild I2_S, deployed candidate scoring | Same verified rebuild bridge | Compatible verified parent's KV cache, then rebuilt candidate scoring |
| Pinned Strata / legacy unsupported native routes | Unavailable | Unavailable | Unavailable |
| Other inference-only controllers and image/audio/video generators | These text-learning methods are unavailable | Unavailable | Unavailable |

Use `--capabilities` or the learning panel's **Check model support** for the currently loaded selection. API methods are shared across backends, but actual gradient/deployment support is checked at runtime. Unsupported routes preserve their existing generation behavior and original context limits.

Tests execute real small Torch and MLX CPU model updates, packed uint32 matmuls, hybrid decoder replay, query/down-projection gradients, frozen-policy scoring, reloads, scoped memory and native bridge transactions. Required CI lanes exercise MLX on CPU and Metal separately, alongside the existing six Python/platform lanes. The target Bonsai/CRACK fresh-process recall evaluator is `tools/evaluate_persistent_memory.py`. Full target-model recall, throughput and Metal memory usage require full checkpoint runs and are not inferred from small-model tests.

Results report complete operation wall time, sampled process RSS, update metrics, and checkpoint/selection identifiers. Process RSS excludes separately owned server processes and is distinct from GPU or Metal peak allocation. Parameter updates can make an answer worse; rollback and finite retention tests reduce that risk but do not establish unlimited reliable memory.
