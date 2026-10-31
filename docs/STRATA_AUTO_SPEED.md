# Strata integration and native automatic speed tuning

Smart AI Studio offers two explicit models served by the actual open-source
[Niko1221/Strata](https://github.com/Niko1221/Strata) runtime, alongside measured
execution tuning for existing native MLX and llama.cpp models. Strata manages
large model experts across GPU memory, RAM and SSD. Native auto speed adjusts
execution settings on the already loaded native model. The existing default
model is unchanged; selecting a Strata preset is an explicit choice.

The managed integration is pinned to **Strata v0.2.8**, whose bundled Terminal
Control version is **0.1.54**. Its source was inspected at
[`870aedd9e1ebe1c91f94da93b59af17729f6c5c0`](https://github.com/Niko1221/Strata/tree/870aedd9e1ebe1c91f94da93b59af17729f6c5c0).
Model identities and quality settings are a narrow registry in
[`config/strata_models.py`](../config/strata_models.py); arbitrary custom GGUFs
cannot acquire Strata support merely by changing their backend label.

## The two Strata models

Both choices are text-only inference presets for Qwen3.8-Flash-Next: approximately
125 billion language-model parameters plus a separate 51 billion parameter
n-gram embedding table. They are much larger than Studio's existing small Qwen
preset. The n-gram table is accessed from disk a few rows at a time; its file size
is not a requirement to keep the whole table resident in RAM. These quantizations
are **nonternary**, including Gyro-S despite the `TQ1_0` suffix in its filename.
The publisher cards and upstream model guide describe their quality tradeoffs.

| Studio choice | Official family / model | Publisher and pinned revision | Model download | Additional preparation estimate | Free SSD planning estimate |
| --- | --- | --- | --- | --- | --- |
| **Strata Gyro-S** (`strata_gyro_s`) | `gyro` / `Gyro-S` | [agentionai/Qwen3.8-Flash-Next-Gyro-GGUF](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF/tree/1ada08422e404aca51ab9df665f089f34c419c6b) | 58,487,015,808 bytes (**58.5 GB**), one file including n-gram data | About **25.8 GB** for an extracted expert arena where needed | About **94.3 GB**, including 10 GB preparation headroom |
| **Strata Q2_0** (`strata_q2_0`) | `qwen` / `Q2_0` | [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/ed59f92082b1e93c0e96d60a8b11aab089b52f09) | 66,423,878,624 bytes (**66.4 GB**), two shards | About **34–40 GB**, depending on low-RAM and CPU preparation | About **116.4 GB**, using the larger estimate and 10 GB headroom |

Storage figures in this table use decimal GB. They exclude the Terminal, native
engine, logs and other application data. Existing complete downloads can be
reused. Preparation requirements vary by backend and hardware; Strata's plan and
disk-space checks are authoritative, rather than these rounded estimates.
Download sizes and digests come from the pinned upstream
[`setup.py` catalog](https://github.com/Niko1221/Strata/blob/870aedd9e1ebe1c91f94da93b59af17729f6c5c0/setup.py).

Gyro-S uses agentionai's APR trellis quantization and is the smaller choice, with
a quality tradeoff. Q2_0 uses ISTA-DASLab's GSQ-RCO mixed quantization, advertised
as 2.40 bits per weight for the transformer portion. Neither label describes the
precision of every tensor or changes Studio's existing ternary models. See the
[Gyro publisher card](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF),
[ISTA publisher card](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF),
and [upstream model guide](https://github.com/Niko1221/Strata/blob/870aedd9e1ebe1c91f94da93b59af17729f6c5c0/docs/MODELS.md).

These presets expose **inference only**. They do not support persistent weight
updates, LoRA training or a pretend successful sleep-training step. Conversation
history and ordinary inference remain available; training-capable native
backends retain their own learning behavior.

## Load, cancel and close

Select **Strata Gyro-S** or **Strata Q2_0** in the desktop model selector and start
loading. The loading worker performs these steps:

1. Install the pinned official Terminal privately under Studio's portable data
   directory. Check the archive's exact SHA-256 and byte count, then use the
   checksum-verified bundle's official verifier to validate the release feed
   against a public Ed25519 key pinned in Studio. Publish the installation only
   after verification succeeds.
2. Ask the official launch planner to inspect the computer and obtain a
   compatible engine. A missing engine fails before starting a new large model
   download. Studio does not install GPU drivers or build toolchains.
3. Download and prepare the selected, registered pack using the official launch
   runner. Model preparation can take substantially longer than a normal cached
   model load; progress reports the current stage.
4. Start one owned profile on a loopback address with a generated API key and
   verify that the expected model is loaded. Studio submits complete structured
   conversations to the local `/v1/chat/completions` endpoint.

The Terminal, engines and model data are kept beneath `runtime/strata/` in
Studio's portable data directory. Each serving session gets private runtime,
profile and tuning-state directories. Studio does not change shell profiles,
take over an existing global Strata service, or issue a global stop command.
The ownership mechanism uses upstream's `runtime start --profile …
--die-with-pid …`; stopping or closing Studio targets only its own session.
The official lifecycle contract is documented in
[Strata Terminal](https://github.com/Niko1221/Strata/blob/870aedd9e1ebe1c91f94da93b59af17729f6c5c0/docs/TERMINAL.md).

**Cancel** stops a load or preparation attempt and keeps downloaded model data
and completed preparation files. A subsequent load lets upstream verify what is
usable and resume the remaining work. Stopping a generated response closes the
active request; model unloading or app closing stops the owned service. Native
work already in progress can take time to exit, and cleanup may wait for that
process to finish. No fixed first-load or cancellation time is promised.

## Platforms and memory

The relevant hardware case is a supported **8 GB GPU, 32 GB system RAM and a fast
SSD**. Strata can keep hot experts in GPU memory, use a fitting RAM complement,
and read other experts from SSD. Consequently a 58–66 GB download is not rejected
just because it exceeds either VRAM or system RAM. The exact tier layout depends
on the engine, free memory, context and selected pack. Automatic cache sizing and
prompt processing remain upstream's responsibility. See the
[official low-memory implementation](https://github.com/Niko1221/Strata/blob/870aedd9e1ebe1c91f94da93b59af17729f6c5c0/setup.py)
and [engine details](https://github.com/Niko1221/Strata/blob/870aedd9e1ebe1c91f94da93b59af17729f6c5c0/docs/DETAILS.md).

The [community RTX 5060 Laptop report](https://www.reddit.com/r/LocalLLaMA/comments/1xft2ae/have_8_gb_of_vram_and_32_gb_of_ram_you_can_run/)
describes that class of machine. It is evidence of an upstream user run, not a
benchmark of this integration. GPU model, drivers, SSD latency, free RAM,
quantization, drafting, KV precision and routing options all affect the result.
Studio's conservative quality settings differ from some fast community setups.
The 8 GB / 32 GB combination does not establish a universal minimum or speed.

| Platform | Managed Terminal package | Important engine qualification |
| --- | --- | --- |
| Windows x86-64 | Official `windows-x86_64` package | A supported NVIDIA or AMD engine and working driver are required; capacity alone does not establish compatibility. |
| Linux x86-64 | Official `linux-x86_64` package | Upstream supports specified NVIDIA and AMD GPUs. Intel Arc has a Linux engine, but model preparation has additional requirements that Studio does not install. |
| Apple Silicon | Official `macos-arm64` package | **macOS 15 or newer** and the Metal engine. Unified memory is one pool. Intel Macs are unsupported. |
| Linux ARM64, including DGX Spark / GB10 | Unavailable for these two presets | Upstream's ARM engine supports other packs. **Q2_0 and Gyro require x86 CPU kernels on Linux**, so Studio rejects them before installation or model download. Apple Silicon uses a separate Metal path. |

Consult the pinned upstream [installation guide](https://github.com/Niko1221/Strata/blob/870aedd9e1ebe1c91f94da93b59af17729f6c5c0/docs/INSTALL.md),
[Mac guide](https://github.com/Niko1221/Strata/blob/870aedd9e1ebe1c91f94da93b59af17729f6c5c0/docs/MAC.md),
[Intel Arc guide](https://github.com/Niko1221/Strata/blob/870aedd9e1ebe1c91f94da93b59af17729f6c5c0/docs/INTEL_ARC.md),
and [DGX Spark guide](https://github.com/Niko1221/Strata/blob/870aedd9e1ebe1c91f94da93b59af17729f6c5c0/docs/DGX_SPARK_LINUX.md)
for GPU-specific limits. The Mac guide reports SSD-streaming tests on 32–34 GB
machines and recommends 64 GB for Q2_0 held in memory. It does not establish that
a 16 GB Mac suitable for Studio's existing MLX models can run these packs well.
An internal fast SSD is the documented Mac streaming path.

Strata presets default to **32,768 tokens** of context. Larger context requires
an explicit setting and enough runtime memory; Studio does not silently reduce
the requested context or change the model to make a load succeed. The pack's
registered maximum is 262,144 tokens, not a fit guarantee on every machine.
To change Context for a loaded Strata model, unload it, apply the new value, and
load it again. Editing the field while it is loaded keeps the existing value and
shows that instruction. The server uses its native tokenizer to account for the
complete prompt and output; Studio does not subtract its conservative display
estimate from the output allowance. A genuinely oversized prompt is rejected by
the server with its context error, without a truncated retry.

Studio's application memory cap includes its own process and the tracked Strata
process tree. It is monitored, not a hard allocation ceiling: transient usage
can exceed it between checks, and GPU allocation and memory-mapped file behavior
vary by OS. Upstream's `--memory-limit-mib` describes system/container capacity,
while `--resident-budget-gib` limits only a subset of expert storage. Neither is
misrepresented as an application-wide RSS limit. Close other large workloads
when assessing whether a selected pack fits.

## Strata quality and automatic placement policy

The managed launch explicitly requests the following settings:

| Setting | Studio's default | Purpose |
| --- | --- | --- |
| KV cache | `fp16` | Avoid automatic KV quantization. Other registered Strata KV modes require an explicit selection on a supported engine; the pinned Metal engine accepts fp16 only. |
| MTP layer | `off` | Do not download/load the draft layer as an implicit speed choice. |
| Drafting | `off` | Do not substitute a speculative drafter. |
| Speculative tokens | `spec=0` | Disable speculative token generation. |
| Cached-expert routing prior | `STRATA_ROUTE_PRIOR=0` | Do not prefer experts merely because they are already cached. |
| Route tail skipping | `STRATA_ROUTE_TAIL_SKIP=0` | Do not omit low-ranked expert work to gain speed. |
| Approximate KV attention | `STRATA_KVA=0` | Keep this approximate attention shortcut off. |
| Vision | `off` | These are text-only presets; no vision projector download. |
| Concurrent requests | `1` | Keep one interactive request active in the owned model. |
| Expert cache and prefill | Upstream automatic configuration | Let the real engine fit execution to the computer and SSD tier. |

The policy preserves the chosen quantized model; it does not make a lossy model
pack equivalent to full-precision weights. It also does not promise bitwise
identical results between GPU architectures. Upstream explains its approximate
speed controls in [engine details](https://github.com/Niko1221/Strata/blob/870aedd9e1ebe1c91f94da93b59af17729f6c5c0/docs/DETAILS.md).
[Issue #4402](https://github.com/Niko1221/Strata/issues/4402) and the v0.2.8 notes
record why cached-expert routing became opt-in again after long-session quality
reports. An approximate routing option is therefore not quietly re-enabled to
match a published tokens-per-second figure.

For Strata, **Check placement** reports the loaded native configuration. It
does **not** run upstream `tune auto`, compare candidates, or claim a measured
gain. The upstream quick and thorough tuners can take many minutes and reload
engines; they are not wired into Studio's short native calibration control in
this version. Turning **Auto speed** off does not disable Strata's necessary
memory placement or remove its SSD tier.

## Native Auto speed: use it

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

## Native settings that are measured

| Runtime | Eligible setting | Boundaries |
| --- | --- | --- |
| Native MLX-LM text | Prompt prefill chunk size | Uses the app's existing prompt-cache factory. The existing memory-sensitive prefill policy caps active overrides. |
| Supported MLX-VLM text paths, including the Bonsai loaders | Prompt prefill chunk size when the installed runtime exposes a verifiable native control | Uses the existing model, processor, prompt formatter, and native VLM cache path. Vision weights remain loaded; the comparison measures text serving. |
| Standard GGUF / llama.cpp | Decode threads and prompt-processing threads | Changes one thread setting at a time using the native live setter; retains GPU placement, context, and cache types. |
| Prism GGUF / Bonsai private kernels | The same native thread settings | Measurements run inside the actual isolated Prism worker. Progress and cancellation cross the worker boundary. |
| Managed Strata Gyro-S / Q2_0 | No short native calibration | The real Strata engine manages placement; Check placement reports configuration without a measured gain. |
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

## How a native setting wins

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

## Native calibration memory, lifecycle, and saved profiles

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

## Research behind native calibration

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

## Release provenance and licenses

The managed runtime uses the official
[v0.2.8 release assets](https://github.com/Niko1221/Strata/releases/tag/v0.2.8).
Terminal archive checksums are pinned in
[`core/strata_runtime.py`](../core/strata_runtime.py); the official signed release
feed must authorize the selected platform package. Engine acquisition remains
inside the official Terminal with the pinned release feed. Model repository
revisions, file sizes and SHA-256 identities are recorded in the catalog and
checked by upstream preparation. These integrity checks establish which files
were selected; they do not establish model quality or performance.

Strata's source is [MIT-licensed](https://github.com/Niko1221/Strata/blob/870aedd9e1ebe1c91f94da93b59af17729f6c5c0/LICENSE).
Its runtime components and the publisher model weights have their own applicable
licenses. The selected weights reference the Qwen Community License; consult
their linked model cards. Studio's existing license applies to Studio and does
not replace those upstream terms. Studio's distribution does not embed the large
model weights: the explicit Strata selection downloads the official components
and chosen pack separately.

## Verification and limits

Behavioral tests exercise registered Strata identities and quality flags,
release verification and private lifecycle handling, full conversation transport,
SSE framing, model identity checks and request cancellation. Local HTTP fixtures
and fake process/native streams allow these contracts to be checked without
downloading weights or claiming that a fixture is a running large model.

Native calibration tests exercise measured selection, warmup/noise handling, exact-token
rejection, corruption and stale cache handling, native setting propagation,
Prism progress/cancellation, memory checks, busy locks, preference preservation,
and rollback. Existing prompt, sampling, cache, learning, model-routing, and GUI
contracts remain part of regression verification.

These checks are not hardware performance results. This implementation has not
been benchmarked here with the full Gyro-S or Q2_0 pack on an 8 GB GPU / 32 GB RAM
machine. Actual speed measurements require the installed engine and model on the
target computer. Platform builds and live Metal/CUDA/HIP/Prism inference must be
verified separately from a source/unit-test result. A successful startup check
establishes that a request can run, not sustained throughput, long-context answer
quality or support for an untested hardware combination.
