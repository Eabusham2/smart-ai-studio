# Best Work handoffs, output limits, and release packaging

## Preserved behavior

The existing `EntropyRouter`, verifier/consensus selection, convex temperature
ladder, and fixed-temperature extra candidate remain authoritative. The desktop
checkbox only enables or disables that existing Pro path. There is no new
keyword-based complexity selector or GUI branch-count dropdown. Native servers
without full-vocabulary entropy retain their existing runtime-specific budget;
Studio does not invent a confidence value for them.

## Candidate and execution lifecycle

Each candidate ends at normal model completion or at its first complete,
well-formed top-level tool invocation. It does not execute the tool or generate
an imagined tool result. Fenced code, inline examples, and thinking text are not
executable calls. Candidate generation closes its owned iterator at the boundary.

The existing selector chooses the winner. Its text and invocation are displayed
before dispatch, and only its selected action executes. Actual tool output is
then returned in the conversation. The original routing method reevaluates the
new context and can choose either a normal stream or another Pro round. A full
answer without a tool call finishes without an extra generation request.
Cancellation is checked before dispatch. A steer waits for the prior owned worker
to finish cleanup before starting the new context-aware routing pass; its pending
callback cannot start a duplicate request. The handoff tracks attempted operations
before execution so a repeated identical action cannot be automatically retried.

The chat has a Best Work progress card when Pro actually runs, including after a
tool result. The selected response is rendered once, rather than retaining both
raw stream text and the formatted copy. Multiple thinking sections remain in order
around the invocation and actual result, with each section rendered as a thinking
control rather than leaving later raw tags in the response.

## Output and telemetry

Chat and Pro use the entire remaining **Context** window. The legacy
`max_new_tokens` setting is no longer a separate output ceiling; EOS, the real
context boundary, a completed tool invocation, and the user's Stop control still
end generation. Prompts are not silently truncated to make room.

Telemetry observes sampled token IDs or runtime-reported counters. UI chunks,
word counts, and replaying the selected branch are not counted as inference.
All observed candidates contribute to aggregate work; decode timing excludes
prefill/tool waits, while wall throughput is a separate measure. Missing runtime
counters are reported unavailable, never fabricated. Strata's authenticated
status endpoint supplies real generated-token counters when its stream does not.

## Release formats

The native build matrix produces Windows x86-64, Linux x86-64, and macOS arm64
packages. Portable packages contain a checksum-pinned Python runtime and installed
Python dependencies. Setup packages contain Python and automatically install the
platform's tested dependency versions into a user-owned directory on first launch.
Source checkouts can still use their own Python environment.

Portable launch does not run pip. Setup requires internet for dependency download.
OS graphics drivers and a supported desktop environment remain host prerequisites.
Every native package is relocated to a path containing spaces and must pass a
model-free GUI/database smoke test before a success receipt is written. The
publisher verifies commit IDs, sizes, hashes, and all required packages before
moving the rolling release tag. Large archives have verified reassembly helpers.

## Validation scope

Automated tests use synthetic token streams, local HTTP fixtures, and small random
networks where applicable. Packaging initializes the GUI without loading a model.
These checks do not measure pretrained-model parity or throughput and are not a
claim that real model inference has been benchmarked.
