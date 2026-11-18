# Best Work handoffs, output limits, and release packaging

## Choosing the number of passes

With **Best Work** checked, Studio requests **two sequential inference passes
per stage** by default. The first pass produces a draft. The second receives the
request, its conversation history, and the first pass's selected draft. It
reviews that draft and generates the complete response shown in chat. With three
or more passes, each refinement receives the immediately preceding winner. The
progress card shows the current pass and branch progress while the draft is being
generated. Ordinary chat keeps its existing streaming path.

In the **Branches** dialog, set **Best Work passes** to a positive whole number
and choose **Apply passes** for the next Best Work request. Set the value to
**1** for the original single-pass behavior. To set the startup default, use the
`BEST_WORK_PASSES` environment variable. Python callers can set
`settings.best_work_passes` or override it for one request with
`stream_best_work(controller, prompt, passes=1)`. The media controller exposes
the same override as `stream_solve(..., best_work_enabled=True, best_work_passes=1)`.

## Routing, selection, and tools

Each pass uses the existing entropy router and verifier or consensus selector.
A pass can use a single stream or search multiple branches with the calibrated
convex temperature ladder and its fixed-temperature extra candidate. Native
servers without full-vocabulary entropy use their configured runtime branch
budget. A refinement reroutes using its updated context, so successive passes
can have different branch counts.

Each candidate ends at normal model completion or at its first complete,
well-formed top-level tool invocation. The decoder closes its iterator at that
boundary. Fenced code, inline examples, and thinking text remain ordinary
candidate content. Intermediate winners enter the next pass as quoted draft
data. A proposed call in a draft has no tool result and grants no authorization.

The final pass streams the selected response to chat. If a refinement cannot
fit, the most recent completed winner becomes the final selection for that
stage. If the selection requests a tool, Studio displays the text and
invocation, checks cancellation and
authorization against the original user request, and dispatches the permitted action.
Studio adds the actual result to the conversation and starts a fresh stage
with the configured number of passes. For example, with two passes, a response
requiring one tool result normally takes four passes across two stages. A final
answer ends the request.

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
completed draft fills it, Studio returns that draft in full and the progress
card identifies the skipped refinement. A draft with a selected tool call
follows the same authorization and handoff rules.

The desktop and direct `stream_best_work` entry points both prepare a stored
chat archive against the active model checkpoint before copying turns into a
pass. Verified retirement keeps older exchanges in the archive for display or
export while freeing space in the model's working history.

When the first request in a stage exceeds the active window after eligible
history consolidation, Studio reports the context error. The same applies if a
refinement has already displayed text when a capacity failure occurs. This
keeps the original request intact and avoids splicing two responses together.

Usage includes actual decoder invocations across every pass, candidate, and tool
stage. Token totals come from sampled IDs or runtime-reported counters. Missing
counts appear as unavailable. Displaying a selected branch adds no tokens.
Decode timing excludes prefill and tool waits. Wall throughput includes elapsed
request time. Strata's authenticated status endpoint supplies generated-token
counters when its stream omits them.

`last_best_work_metadata` retains the detailed Pro branch audit. It records the
per-stage pass target in `passes_requested`, progress for the current stage in
`passes_completed`, and the request-wide total in `total_passes_completed`.
`pass_summaries` reports individual passes and `pass_usage` aggregates observed
invocations and token counts across the request. A skipped refinement records
`completed=False`, `skip_reason="context_limit"`, and `selected_pass` in its
summary. Replaying the earlier winner leaves its original token count intact.

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

Automated tests exercise routing, winner selection, tool boundaries, cancellation,
and context errors with controlled streams and local HTTP fixtures. A small local
CPU Transformers model also generates two actual samples. The test checks the
second decoder input against the selected first sample and checks the displayed
answer against the second sample. Packaging smoke tests initialize the GUI
without loading a model. Answer quality and throughput benchmarking require a
separate pretrained-model evaluation.
