# Strata-inspired automatic speed tuning

This branch adds native execution tuning to Smart AI Studio. It follows the
measurement and rollback principles in the open-source
[Niko1221/Strata](https://github.com/Niko1221/Strata) project while keeping Studio's
model loaders, chat, memory, evaluation, and learning paths.

## Use it

**Auto speed** is enabled by default. Load an already selected model normally.
On a supported runtime, Studio either restores a matching measured profile or
runs a bounded comparison in the model-loading worker. Progress appears beside
the speed controls. Model switching uses the same background loader so the UI
and Cancel button remain responsive.

**Retune** repeats the comparison on the loaded model. **Cancel** stops at a safe
native token/run boundary and keeps the previously working execution settings.
Turning **Auto speed** off restores the original native execution defaults.
The checkbox preference is merged into the existing app preferences, preserving
saved memory limits and other settings.

The default measurement budget is **90 seconds**. This is cooperative: a single
native prefill or token evaluation already in progress must finish before Python
can observe cancellation or expiry. A slower model can therefore return
"time budget reached" without obtaining a new profile. This does not prevent
normal inference. Developers may set `SMARTAI_SPEED_TUNE_SECONDS` between 15 and
600, or set `SMARTAI_AUTO_SPEED=0` for an initial default of off. A saved desktop
checkbox preference takes precedence over that initial environment default.

## What is tuned

| Runtime | Eligible setting | Boundaries |
| --- | --- | --- |
| Native MLX-LM text | Prompt prefill chunk size | Uses the app's existing prompt-cache factory. The existing memory-sensitive prefill policy caps active overrides. |
| Supported MLX-VLM text paths, including the Bonsai loaders | Prompt prefill chunk size when the installed runtime exposes a verifiable native control | Uses the existing model, processor, prompt formatter, and native VLM cache path. Vision weights remain loaded; the comparison measures text serving. |
| Standard GGUF / llama.cpp | Decode threads and prompt-processing threads | Changes one thread setting at a time using the native live setter; retains GPU placement, context, and cache types. |
| Prism GGUF / Bonsai private kernels | The same native thread settings | Measurements run inside the actual isolated Prism worker. Progress and cancellation cross the worker boundary. |
| BitNet server, Transformers, image/audio/video generation, or incompatible library versions | Existing native configuration retained | The UI reports unsupported tuning rather than treating a connected or loaded model as measured. |

Tuning does not switch to another model, download a different quantization,
change a model's context allowance, change the user's sampling settings, or enable
speculative decoding. Existing policies remain distinct: normal MLX-LM chat uses
its existing TurboQuant-preferred cache factory, while evaluation/RSI keeps its
existing full-precision paths. The VLM path retains its own native cache behavior.
It measures the library's unchanged prefill default by omitting the override. If
there is too little headroom to measure that baseline, tuning is deferred; the
baseline is never silently reduced to make a comparison pass.

The GGUF comparison does not reload large models to sweep GPU-layer placement.
It does not claim to implement Strata's custom expert-cache kernels or reproduce
its published throughput on arbitrary models.

## How a setting wins

Each comparison uses the same synthetic inventory/programming prompt, a fixed
small output allowance, and greedy sampling solely for the measurement. The
prompt and generated tokens never enter chat history, the memory database,
evaluation datasets, or learning.

1. Warm the runtime once, then measure two baseline runs to estimate A/A noise.
2. Change one execution setting and run interleaved baseline/candidate/baseline
   measurements. Rotate the order of candidates between rounds.
3. Measure first-token/prompt time separately from subsequent decoding. Compare
   the geometric mean of prompt and decode throughput.
4. Require identical generated-token hashes and token counts, a repeatable win
   in both comparison rounds, and a gain exceeding the largest of **3%**, twice
   measured baseline noise, or twice measured local drift.
5. Reject a candidate if either throughput metric falls by more than **2%**.
6. Apply and save the winner only after the comparison completes. Errors,
   cancellation, memory pressure, and incomplete runs preserve working settings.

"Improved" means the best repeatably faster configuration **among the tested
candidates for this short workload**. "Baseline retained" means no candidate
met the acceptance rules. Unsupported, failed, or timed-out comparisons are
reported separately. A short deterministic comparison cannot prove identical
answers for every future prompt or guarantee a universal speedup. Longer-context
workloads and other active applications can behave differently.

## Memory, lifecycle, and saved profiles

Before and during comparisons, Studio checks live free memory and the enabled
process-memory limit. It retains at least the larger of 1 GiB or 5% of system
memory as free headroom and 256 MiB below a specified process cap. The existing
memory watchdog remains in control. No native library is initialized merely to
discover hardware for a profile key.

Inference, learning, model unloading, and tuning share runtime locks. The UI
also blocks conflicting chat, media, sleep, and evaluation operations while a
comparison owns the model. App closing requests cancellation and waits through
UI callbacks. A failed optional tuner does not turn a successfully loaded model
into a failed load.

Profiles live in `speed_profiles.json` in Studio's existing portable data
directory. They are keyed by model/config/weight-file identity, runtime versions
and native library identity where available, hardware identity, context, adapter,
and projector or cache policy. Weight files are identified from resolved paths
and file metadata; the tuner does not hash multi-gigabyte weights at startup.
Free RAM is checked live instead of being a permanent identity field.

Writes use atomic replacement and locking. The cache accepts only bounded
execution settings, retains at most 64 profiles, and ignores malformed, stale,
or incompatible records. The adapter revalidates that a saved setting belongs to
the allowed candidates before applying it. A failed new comparison does not
overwrite a previously accepted saved profile.

MLX calibration builds a private synthetic cache and preserves the app cache.
GGUF/Prism reset their disposable native prefix cache to avoid measuring a prefix
cache hit; Studio supplies the complete history-packed prompt on the next real
request, so saved conversation history remains intact. No large KV snapshot is
copied merely to run a benchmark.

## Research that informed the design

The inspected Strata revision is
[`870aedd9e1ebe1c91f94da93b59af17729f6c5c0`](https://github.com/Niko1221/Strata/tree/870aedd9e1ebe1c91f94da93b59af17729f6c5c0).
These are upstream source and first-party user reports, not speed measurements
of Smart AI Studio:

- [`strata_control/autotune.py`](https://github.com/Niko1221/Strata/blob/870aedd9e1ebe1c91f94da93b59af17729f6c5c0/strata_control/autotune.py)
  starts from the current configuration, interleaves controls, measures noise and
  drift, guards resources, and keeps incomplete results from being applied.
- [`strata_control/tune_knobs.py`](https://github.com/Niko1221/Strata/blob/870aedd9e1ebe1c91f94da93b59af17729f6c5c0/strata_control/tune_knobs.py)
  distinguishes execution controls from quality-changing or nonrepeatable
  controls and encodes hardware-specific exclusions.
- [Issue #4402](https://github.com/Niko1221/Strata/issues/4402) records reports of
  long-context repetition and tool-call failures associated with faster routing
  defaults. The maintainer reverted the routing-prior default. This motivates
  preserving model-quality choices and measuring useful work, not chasing a
  token/s figure through approximate routing.
- [Issue #1337](https://github.com/Niko1221/Strata/issues/1337) describes a failed
  calibration candidate losing earlier work. Studio retains prior working
  settings and records failed/incomplete outcomes separately.
- [Issue #2301](https://github.com/Niko1221/Strata/issues/2301) highlights tuning
  without the configured vision encoder loaded. Studio benchmarks the actual
  loaded runtime with its existing adapter/projector/vision memory consumers.
- [Issue #3613](https://github.com/Niko1221/Strata/issues/3613) concerns saved
  calibration and missing packaged helpers. Studio's platform builders already
  copy the entire `core` tree, including the new tuning modules.
- [Strata's Mac guide](https://github.com/Niko1221/Strata/blob/870aedd9e1ebe1c91f94da93b59af17729f6c5c0/docs/MAC.md)
  describes a distinct Metal engine and large model packs; its documented Mac
  measurements do not establish a speed benefit for Studio's existing 16 GiB
  MLX setup. No automatic engine/model substitution is made here.

Strata is MIT-licensed; model weights and some bundled components have separate
licenses. This implementation uses independent native adapters and does not
vendor Strata source, model weights, fonts, or experimental data.

## Verification and limits

Behavioral tests exercise measured selection, warmup/noise handling, exact-token
rejection, corruption and stale cache handling, native setting propagation,
Prism progress/cancellation, memory checks, busy locks, preference preservation,
and rollback. Existing prompt, sampling, cache, learning, model-routing, and GUI
contracts remain part of regression verification.

Tests using deterministic fake native streams verify control logic; they are not
hardware performance results. Actual speed measurements require the user's
installed engine and model on the target machine. Platform builds and live
Metal/CUDA/Prism inference must be verified separately from a source/unit-test
result.
