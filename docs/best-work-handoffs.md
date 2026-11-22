# Best Work handoffs, output limits, and release packaging

## Choosing the number of passes

With **Pro branches** checked, Studio requests **two sequential inference passes
inside every branch per stage** by default. Each branch first produces an original
draft, then reviews and revises it. With three or more passes, each refinement
receives that branch's latest answer. A pass is a complete model generation, not
one token-decoding step. Weights and adapters are unchanged during the candidate,
refinement, and selection cycle; existing learning operations remain separate.
The progress card shows pass and branch progress. Ordinary chat keeps its existing
streaming path.

In the **Branches** dialog, set **Best Work passes** to a positive whole number
and choose **Apply passes** for the next Best Work request. Set the value to
**1** to skip refinement and the original-versus-revision comparison. Pro still
selects among any branches requested by the existing router. A single branch
with one pass keeps its live streaming path. To set the startup default, use the
`BEST_WORK_PASSES` environment variable. Python callers can set
`settings.best_work_passes` or override it for one request with
`stream_best_work(controller, prompt, passes=1)`. The media controller exposes
the same override as `stream_solve(..., best_work_enabled=True, best_work_passes=1)`.
Multi-pass Best Work requires per-branch selector support; external adapters that
only stream text can use `passes=1`, and requesting more passes fails explicitly.

## Routing, selection, and tools

At the start of each stage, the existing Pro activation and entropy routing rules
choose the branch budget. The original 1/8/16 routing, convex temperature ladder,
fixed-temperature extra candidate on multi-branch rounds, Pro control, and native
runtime branch-budget exceptions remain in place. Refinement stays inside each
chosen branch and does not recursively start another Pro search. A real tool
result starts a new stage and reroutes using the updated conversation, as before.

Every branch completes its Best Work passes before the cross-branch tournament.
Within a branch, the model compares the original answer against the final
completed revision in a temporary isolated inference context. Only those two
candidates compete; intermediate revisions are not extra entries. The model
returns an existing candidate ID, and Studio validates that ID and retains the
selected answer exactly. It does not merge or rewrite the candidates.

The branch winners then compete in pairs until one overall winner remains.
An unpaired candidate advances to the next round. Each match uses the same loaded
model in a separate temporary context, with the original request and relevant
history. No hidden expected answer, verifier result, or reward tells the model
which candidate to choose. Invalid selection output is retried once, then
reported as an error; it never silently means "choose the first branch."

Each candidate ends at normal model completion or at its first complete,
well-formed top-level tool invocation. The decoder closes its iterator at that
boundary. Fenced code, inline examples, and thinking text remain ordinary
candidate content. A refinement receives the original thinking from its branch
plus its latest answer as quoted draft data. Revision thinking and superseded
answers are omitted from the next refinement. The original answer is retained
separately for the within-branch comparison. A proposed call in a draft or a
judging output has no tool result and grants no authorization.

Only the overall winner is displayed in chat. The continued conversation keeps
the original thinking from that winning branch plus the selected original or
revised answer. Judging instructions, selection reasoning, revision thinking,
and losing candidates do not enter the continued chat, learned memory, or
reusable inference context. Temporary inference caches are discarded or restored
at the fork boundary; the forks share model weights rather than copying them.

If the selection requests a tool, Studio displays the text and invocation, checks
cancellation and authorization against the original user request, and dispatches
only the overall winner's permitted action.
Studio adds the actual result to the conversation and starts a fresh stage
with the configured number of passes in each newly routed branch. A final answer
ends the request without an additional continuation generation.

Studio records an attempted action before execution and blocks an identical
repeat. Stop closes the active generation. A steer waits for its worker to
finish cleanup before routing the new request. The chat renders the selected
response once and preserves the order of thinking sections, invocations, and
actual results.

## Context and telemetry

Chat and Pro use the remaining **Context** window for generation until EOS,
a tool boundary, cancellation, or the model's context limit. A refinement must
fit the complete request, its active history, and the quoted prior draft with
room for generation. Each pass may use the entire remaining window. If a
refinement cannot fit, that branch stops refining and the progress card identifies
the skipped pass. If a revision has already completed, the original competes
against that last completed revision. If none has completed, the original is
that branch's only candidate. A comparison must fit both complete candidates and
the task context; insufficient capacity reports an explicit error. Candidates are
never silently truncated to make a comparison fit, and capacity failure does not
choose a default winner.

The desktop and direct `stream_best_work` entry points both prepare a stored
chat archive against the active model checkpoint before copying turns into a
pass. Verified retirement keeps older exchanges in the archive for display or
export while freeing space in the model's working history.

When the first request in a stage exceeds the active window after eligible
history consolidation, Studio reports the context error. Drafts and comparisons
remain private until selection completes, so a failed comparison cannot splice
partial competing answers into the displayed response.

Usage includes actual decoder invocations across every pass, candidate,
comparison, and tool stage. Token totals come from sampled IDs or runtime-reported
counters. Missing counts appear as unavailable. Displaying a selected branch adds no tokens.
Decode timing excludes prefill and tool waits. Wall throughput includes elapsed
request time. Strata's authenticated status endpoint supplies generated-token
counters when its stream omits them.

The completed context indicator shows the **Last invocation**, which may be a
comparison. It measures that invocation's actual prompt and generated tokens,
not the saved conversation. Continuing the chat rebuilds context from the
selected answer and original thinking; discarded comparison text is not retained.

`last_best_work_metadata` retains the detailed Pro branch audit and records
per-branch pass results and tournament matches. The final branch entries contain
each branch's selected candidate. `passes_requested` is the configured target
for each branch. At stage completion, `passes_completed` records the number
completed by the winning branch; the selected answer can still be its original.
`total_passes_completed` counts completed candidate passes across all branches
and stages, excluding judging calls. `pass_summaries` contains the original
generation batch and each individual refinement.

The `best_work` record identifies the `model_tournament` method, each branch's
`selected_pass` and `selected_version`, and the candidate IDs chosen in `matches`.
Its `comparison_summaries` records judging usage without retaining the model's
judging text. `pass_usage` aggregates observed invocations and token counts across
the request, including comparisons. A skipped refinement records
`completed=False`, `skip_reason="context_limit"`, and `selected_pass` in its
summary. Displaying an earlier answer does not count its tokens again.

## Release formats

The native build matrix produces Windows x86-64, Linux x86-64, and macOS arm64
packages. Portable packages contain a checksum-pinned Python runtime and installed
Python dependencies. Setup packages contain Python and automatically install the
platform's tested dependency versions into a user-owned directory on first launch.
Source checkouts can still use their own Python environment.

Portable launch does not run pip. Setup requires internet for dependency download.
OS graphics drivers and a supported desktop environment remain host prerequisites.
Every native package is relocated to a path containing spaces and must pass a
model-free GUI/database smoke test. Setup must also install into a clean private
environment and pass both its first and second launch. The macOS disk image is
mounted read-only and the contained .app is launched before a success receipt is
written. The
publisher verifies commit IDs, sizes, hashes, and all required packages before
moving the rolling release tag. Large archives have verified reassembly helpers.

## Validation scope

Automated tests exercise original and revised winners, tournament advancement,
odd branch counts, invalid selections, context isolation, tool boundaries,
cancellation, and context errors with controlled streams and local HTTP fixtures.
Routing and temperature regressions remain covered separately. These simulations
prove how generated outputs are routed and retained; they do not prove that a
model always chooses the objectively best answer. Packaging smoke tests initialize
the GUI without loading a model. Answer quality and throughput benchmarking
require a separate pretrained-model evaluation.
