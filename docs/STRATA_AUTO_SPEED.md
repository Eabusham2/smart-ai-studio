# Chosen-model placement, Strata, and native automatic speed tuning

Smart AI Studio keeps the model and quantization you choose. You can import a
local model or fetch an exact Hugging Face artifact, then use the compatible
runtime's memory placement. Large files are not rejected solely because their
total size exceeds graphics memory. Whether they can execute depends on the
architecture, tensor format, working memory, context and backend.

The managed [Strata](https://github.com/Niko1221/Strata) engine is one path for
supported large MoE GGUF models. Other compatible GGUFs use native GPU/CPU
placement and memory-mapped weights; supported Transformers checkpoints can use
Accelerate CPU/disk dispatch. MLX checkpoints retain their existing MLX loader.
**Gyro-S and Q2_0 are optional examples, not the scope of the feature or a
replacement for your chosen model.** The existing default model is unchanged.

Memory placement and measured **Auto speed** are separate. Placement makes a
compatible model usable within estimated resource budgets. Auto speed measures
a few execution settings on the loaded model. Neither promises the fastest
possible configuration, a fixed token rate, or support for every custom runtime.

## Import the model you chose

Use the custom model import or **Fetch Hugging Face Custom** dialog. Choose a
local directory/file or enter a Hugging Face repository ID. If a source contains
multiple GGUF quantizations, select the exact **GGUF file**; Studio does not rank
them and pick one for you. Selecting any shard selects its complete matching
shard group, without fetching another quantization.

The **Runtime** choice has three modes:

- **Automatic** uses the inspected format and supported architecture to choose
  a candidate backend. Known compatible Strata encodings can use managed Strata;
  other artifacts retain their native route. Real preflight and loading still
  determine compatibility.
- **Native** keeps the existing compatible GGUF/Prism, MLX or Transformers route.
- **Strata** requests the managed engine for the selected GGUF. A model format,
  architecture or preparation path that it cannot handle produces a clear error.

Admission examines bounded GGUF metadata and tensor directories, or checkpoint
configuration, weight indexes and SafeTensors headers. It does not infer ternary
precision from a filename or repository name. A successful structural check is
not an inference test. A config alone, incomplete shard group or known custom
format requiring an unavailable loader is not accepted as a usable model.

Hugging Face selection is pinned to the repository's immutable commit. The
fetcher downloads the chosen weight group and required metadata, then performs
local inspection before registration. Supported repository loaders retain their
same-revision Python/kernel sources and declared assets. Media pipeline
components retain their own indexes and required weight files. Local model paths, selected filenames,
source revision and observed precision are retained. A failed load does not
switch to Gyro, Q2, a different quantization or a smaller context. Implementation:
[`core/model_admission.py`](../core/model_admission.py),
[`core/model_import_entry.py`](../core/model_import_entry.py), and
[`config/strata_models.py`](../config/strata_models.py).

## Placement by format

| Selected artifact | Placement behavior | Practical boundary |
| --- | --- | --- |
| Native-compatible GGUF, including dense and MoE models | Inspect the complete selected shard group, estimate layer/KV/workspace memory, and place the largest estimated fitting layer count on the GPU. Keep remaining weights on CPU with `mmap` where supported. | OS paging can let the file exceed resident RAM, but it is not Strata's selective expert streaming. The required working set still must fit; heavy paging can be very slow. |
| Prism-compatible GGUF | The same planner operates inside the isolated native worker, with the GUI process included in application memory accounting. | The private runtime must support both the actual tensor encoding and model architecture. A PTQ label alone cannot establish that. |
| Strata-compatible local GGUF | The pinned official importer/planner validates the selected files, then its native engine chooses supported expert caches and RAM/SSD tiers. | This is a supported architecture/encoding path, not a general loader for arbitrary GGUF, MLX or Transformers weights. |
| Supported MLX checkpoint | Retain its actual MLX-LM, MLX-VLM or existing repository-specific loader and memory policy. Where exposed, Auto speed measures prompt chunk size. | No general expert-to-SSD loader is added to stock MLX. A format needing a different publisher runtime requires that integration. |
| Supported Transformers checkpoint | Pass live GPU/CPU budgets and a private disk-offload directory to Accelerate's automatic dispatch. Keep the selected checkpoint precision. Report the actual dispatch map. | Offloaded execution is inference only. Compatibility, disk space and sufficient execution memory are still required; disk dispatch can be slow. |
| Existing image, audio and video models | Preserve the existing media route and capability checks. | The text-only Strata route does not claim their modalities. |

Native GGUF planning reserves memory for the unchanged context, full-precision
KV cache, logits, execution workspaces and any projector. Apple unified memory
is counted once. A recognized allocation failure permits one more conservative
GPU-placement attempt; a format/architecture error is not treated as an OOM.
Adapters that require resident weights disable the `mmap` assumption. The
planner chooses placement before loading; short thread calibration does not
reload the model repeatedly to sweep GPU layers. See
[`core/gguf_placement.py`](../core/gguf_placement.py).

Transformers dispatch reserves OS/runtime headroom and an estimated KV budget.
It does not enable a smaller quantization to fit the checkpoint. Weight updates
are disabled when CPU/disk dispatch is active; a saved adapter is not silently
omitted if it cannot be restored. These are planning budgets rather than hard
allocation ceilings. See
[`core/transformers_placement.py`](../core/transformers_placement.py),
[`core/controller_runtime.py`](../core/controller_runtime.py), and
[Accelerate's big-model inference documentation](https://huggingface.co/docs/accelerate/usage_guides/big_modeling).

## Qwen3.8-Flash-Next ternary on a Mac

The exact published artifact matters. Qwen3.8-Flash-Next is an architecture; its
GGUF, MLX, APR trellis and custom ternary exports are not interchangeable. The
managed Strata release is pinned to **v0.2.8 / Terminal Control 0.1.54**, inspected
at [`870aedd9e1ebe1c91f94da93b59af17729f6c5c0`](https://github.com/Niko1221/Strata/tree/870aedd9e1ebe1c91f94da93b59af17729f6c5c0).
Its Metal expert kernels support Gyro's trellis formats; this does not establish
support for actual `TQ1_0`, `TQ2_0` or `PTQ1_0` ternary tensors.

| Observed format | Meaning and routing |
| --- | --- |
| GGUF types **34 / 35**, `TQ1_0` / `TQ2_0` | True ggml ternary formats. Keep a compatible native GGUF route; do not promote these to pinned Strata merely because the architecture is Qwen. |
| GGUF type **143**, `PTQ1_0` | A distinct packed ternary format. Admission selects the private Prism route; that runtime must still support the exact model. It is not supported by the pinned Strata expert kernels. |
| GGUF types **144 / 145 / 146**, `TQ2_T` / `TQK6` / `TQK7` | APR trellis formats used by Gyro. These are **nonternary**, even when a published filename ends in `TQ1_0`. Supported layouts can be inspected by Strata. |
| Custom MLX ternary layouts | Require the publisher's matching loader and kernels. A generic `mlx` tag or an affine default in `config.json` does not make the custom expert representation loadable by stock MLX. |

These distinctions come from the pinned
[GGUF type reader](https://github.com/Niko1221/Strata/blob/870aedd9e1ebe1c91f94da93b59af17729f6c5c0/tools/gguf_reader.py),
[Metal expert capability tables](https://github.com/Niko1221/Strata/blob/870aedd9e1ebe1c91f94da93b59af17729f6c5c0/metal/src/engine/iq_host.cpp), and
[Metal quantization kernels](https://github.com/Niko1221/Strata/blob/870aedd9e1ebe1c91f94da93b59af17729f6c5c0/metal/kernels/quant.metal).
Mixed models containing an unsupported ternary tensor type are also excluded
from automatic Strata promotion; checking only a whole-model ternary label
would miss that case.

Two concrete publisher examples illustrate the current boundary:

- **Jakevin/Qwen3.8-Flash-Next-ternary-mlx**, inspected at
  [`e5f8a6bbb1546dab7f3cd0d6e1d243535e1049ba`](https://huggingface.co/Jakevin/Qwen3.8-Flash-Next-ternary-mlx/tree/e5f8a6bbb1546dab7f3cd0d6e1d243535e1049ba),
  stores custom per-layer binary codes plus a skeleton. Its
  [publisher runtime](https://huggingface.co/Jakevin/Qwen3.8-Flash-Next-ternary-mlx/blob/e5f8a6bbb1546dab7f3cd0d6e1d243535e1049ba/runtime.py)
  needs oMLX's Qwen4 patch, `mlx-ternary` kernels, and additional tensors from a
  local original FP8 checkpoint. It is a mixed ternary/affine model, not a normal
  self-contained checkpoint for `mlx_lm.load`. Studio has not added this custom
  loader or downloaded the missing FP8 source for it.
- **fuutott/Qwen3.8-Flash-Next-MLX-t5-imatrix-q3down-ple8**, inspected at
  [`15ec9e857b55f19cfadf53e9bf9da7e793175612`](https://huggingface.co/fuutott/Qwen3.8-Flash-Next-MLX-t5-imatrix-q3down-ple8/tree/15ec9e857b55f19cfadf53e9bf9da7e793175612),
  declares custom `omlx_t5` packed experts. Its
  [publisher instructions](https://huggingface.co/fuutott/Qwen3.8-Flash-Next-MLX-t5-imatrix-q3down-ple8/blob/15ec9e857b55f19cfadf53e9bf9da7e793175612/README.md)
  require the Qwen48 oMLX fork and native Bonsai Metal kernel, full Xcode and
  macOS 15+, with the model-specific verification tool. Stock oMLX, MLX-LM and
  MLX-VLM are explicitly unsupported by that publisher. Studio rejects this
  marker with the required-runtime explanation before downloading its weights.

Those custom runtimes are separate engineering work. Selecting a true ternary
model does not authorize substituting a trellis or Q2 model. No MacBook memory
size or token rate is promised for an unsupported artifact.

## Example Strata presets

**Strata Gyro-S** and **Strata Q2_0** remain convenient optional examples of
Qwen3.8-Flash-Next: approximately 125 billion language-model parameters plus a
separate 51 billion parameter n-gram table. The n-gram file is read in small
pieces rather than requiring the entire table resident. Both choices are
**nonternary, text-only and inference only**. Training-capable native backends
retain their own learning behavior.

| Example | Official family / model | Publisher and pinned revision | Model download | Additional preparation estimate | Free SSD planning estimate |
| --- | --- | --- | --- | --- | --- |
| **Strata Gyro-S** (`strata_gyro_s`) | `gyro` / `Gyro-S` | [agentionai/Qwen3.8-Flash-Next-Gyro-GGUF](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF/tree/1ada08422e404aca51ab9df665f089f34c419c6b) | 58,487,015,808 bytes (**58.5 GB**), one file | About **25.8 GB** where an extracted expert arena is needed | About **94.3 GB**, including 10 GB preparation headroom |
| **Strata Q2_0** (`strata_q2_0`) | `qwen` / `Q2_0` | [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/ed59f92082b1e93c0e96d60a8b11aab089b52f09) | 66,423,878,624 bytes (**66.4 GB**), two shards | About **34–40 GB**, depending on preparation | About **116.4 GB**, using the larger estimate and 10 GB headroom |

Storage figures use decimal GB and exclude the Terminal, native engine, logs
and other app data. Existing complete downloads can be reused. Actual disk
requirements depend on the selected model/backend; the official preparation
checks determine them. Custom imports do not inherit these two presets' sizes.
Download sizes and digests come from the pinned
[`setup.py` catalog](https://github.com/Niko1221/Strata/blob/870aedd9e1ebe1c91f94da93b59af17729f6c5c0/setup.py).

Gyro-S uses APR trellis quantization. Q2_0 uses ISTA-DASLab's GSQ-RCO mixed
quantization, advertised as 2.40 bits per weight for the transformer portion.
Neither label describes every tensor's precision. See the
[upstream model guide](https://github.com/Niko1221/Strata/blob/870aedd9e1ebe1c91f94da93b59af17729f6c5c0/docs/MODELS.md)
and the linked publisher cards for their respective quality tradeoffs.

## Managed Strata load, cancel and close

For a selected local Strata model, Studio records the complete exact GGUF set
and verifies it again before preparation and serving. The official importer
checks architecture, tokenizer, tensor layout and quantization. It can identify
a compatible same-layout build without requiring a particular publisher name;
this remains an unverified engine candidate until actual loading succeeds.
Studio preserves its warnings and never replaces a custom fine-tune with the
catalog model used as a preparation recipe. See the upstream
[`models.plan_import` implementation](https://github.com/Niko1221/Strata/blob/870aedd9e1ebe1c91f94da93b59af17729f6c5c0/strata_control/models.py).

The local path currently covers supported native-GGUF preparation recipes for
Qwen3.8-Flash-Next and supported DeepSeek V4.1-Flash file sets. Recipes that would
replace the chosen file with a separate ready-made pack are refused. Qwen3.6's
separate own-path engine is not integrated into this managed local path; native
backends retain their own compatibility checks. Local DeepSeek sets must include
the exact required companions, such as Engram or metadata files where that
recipe requires them. Missing dependencies are reported rather than silently
starting a large additional model download.

The loading worker then:

1. Installs the pinned official Terminal privately. It checks exact archive
   SHA-256 and byte count, then uses that verified bundle's official verifier to
   validate the signed release feed against Studio's pinned Ed25519 public key.
2. Asks the official planner for the compatible engine. A missing engine fails
   before a new built-in model download. Drivers and compiler toolchains are not
   installed by this integration.
3. Prepares the exact selected local files. For an explicitly selected built-in
   example, the official runner downloads and verifies that preset's pinned
   files. Local-source preparation is prevented from downloading replacement
   model weights.
4. Starts one owned profile on loopback with a generated API key, then verifies
   the expected served model identity. Complete structured conversations go to
   its local `/v1/chat/completions` endpoint.

Managed files live beneath `runtime/strata/` in Studio's portable data directory.
Custom prepared data and discovery/settings roots are isolated by the source's
absolute paths and file identity, so two fine-tunes using the same recipe cannot
reuse each other's prepared experts. A source change requires re-import. Session
runtime/profile directories are private, and `runtime start --profile …
--die-with-pid …` ties the owned service to Studio's lifetime. An existing global
Strata service is not stopped or reconfigured. See
[`core/strata_runtime.py`](../core/strata_runtime.py) and upstream
[Terminal lifecycle documentation](https://github.com/Niko1221/Strata/blob/870aedd9e1ebe1c91f94da93b59af17729f6c5c0/docs/TERMINAL.md).

**Cancel** stops a load/preparation attempt while retaining downloaded source
files and completed preparation files. A later attempt lets upstream verify
and reuse usable files. Stopping a response closes the active request; unloading
or closing Studio stops its owned service. A native operation in progress can
take time to exit, so no fixed first-load or cancellation time is promised.

## Platforms and memory

A supported **8 GB GPU + 32 GB RAM + fast SSD** is a relevant Strata configuration
for some large MoE packs: hot experts can reside on the GPU, a fitting complement
can stay in RAM, and other expert data can be read from SSD. A 58–66 GB download
therefore need not fit wholly in VRAM or RAM. That property depends on the exact
native engine and pack; it cannot be applied to an arbitrary dense checkpoint or
a custom MLX layout. See upstream
[engine details](https://github.com/Niko1221/Strata/blob/870aedd9e1ebe1c91f94da93b59af17729f6c5c0/docs/DETAILS.md).

The [community RTX 5060 Laptop report](https://www.reddit.com/r/LocalLLaMA/comments/1xft2ae/have_8_gb_of_vram_and_32_gb_of_ram_you_can_run/)
is an upstream user run, not a benchmark of this integration. Drivers, free RAM,
SSD latency, context and quality settings all affect the result. Studio's
conservative defaults differ from some fast community configurations. Neither
that hardware combination nor the model's file size establishes a universal
minimum, a guaranteed fit or a token rate.

For native GGUF on Apple Silicon, the source dependency manifests include stock
`llama-cpp-python`. Packaged Mac builds carry both dependency manifests and an
explicit **Install_Dependencies.command**, which creates a user environment and
requests a Metal build. It needs native Python with Tkinter and Xcode Command
Line Tools. In the multipart model bundle, open **SmartAI.app → Show Package
Contents → Contents/Resources/app** to find it; the regular package also exposes
it beside the app. Launchers use that same environment. Launching the app does
not automatically install packages or models. This native binding is distinct
from the separately managed Strata engine and the private Prism runtime.

| Platform | Managed integration | Important qualification |
| --- | --- | --- |
| Windows x86-64 | Official `windows-x86_64` Terminal | Requires a compatible native engine and driver for the selected model. |
| Linux x86-64 | Official `linux-x86_64` Terminal | NVIDIA/AMD compatibility is model and GPU specific. Intel Arc preparation can require tools Studio does not install. |
| Apple Silicon | Official `macos-arm64` Terminal, **macOS 15+** | Separate Metal engines, unified memory counted once, and encoding-specific kernels. Qwen/Gyro support does not imply support for true ternary or custom MLX formats. |
| Intel Mac / Linux ARM64 | No managed installation in this version | Upstream's support for some other ARM packs does not establish this integration's support. Linux Gyro/Q2_0 specifically require x86 CPU kernels. |

Consult the pinned [installation guide](https://github.com/Niko1221/Strata/blob/870aedd9e1ebe1c91f94da93b59af17729f6c5c0/docs/INSTALL.md),
[Mac guide](https://github.com/Niko1221/Strata/blob/870aedd9e1ebe1c91f94da93b59af17729f6c5c0/docs/MAC.md), and
[DeepSeek Mac guide](https://github.com/Niko1221/Strata/blob/870aedd9e1ebe1c91f94da93b59af17729f6c5c0/docs/DEEPSEEK_MAC.md).
Mac SSD-streaming reports for Qwen use larger-memory machines than a 16 GB
MacBook; DeepSeek's Mac guide recommends at least 64 GB and reports testing on
128 GB. These are upstream cases, not Studio hardware validation.

The two example presets start at **32,768 tokens** of context. Custom entries use
their inspected context limit and selected default; local DeepSeek defaults to
8,192. Studio does not reduce a requested context to make a load pass. Unload a
Strata model before applying a different Context, then reload it. Its server
accounts for the complete prompt with its native tokenizer; a genuinely
oversized prompt produces the server's context error without a truncated retry.

The application memory cap includes Studio and the tracked Strata process tree.
It is monitored, not a hard allocation ceiling. Transient allocations, native
workspaces and mapped pages vary by OS. Upstream capacity and expert-residency
budgets are not application-wide RSS guarantees. A fit estimate can become
outdated when other workloads consume memory.

## Strata quality and automatic placement policy

| Setting | Managed policy |
| --- | --- |
| KV cache | Explicit `fp16` on the Qwen Flash-Next path; alternate modes require an explicit supported selection. Metal rejects unsupported modes. Family-specific engines keep their native full-precision policy instead of receiving another engine's incompatible flags. |
| MTP / drafting / speculative tokens | Off. Qwen uses `spec=0`; DeepSeek explicitly uses `dspark none`. |
| Cached-expert routing prior | `STRATA_ROUTE_PRIOR=0`; no cache-biased approximate routing. |
| Route tail skipping | `STRATA_ROUTE_TAIL_SKIP=0`; no expert-tail shortcut. |
| Approximate KV attention | `STRATA_KVA=0`. |
| DeepSeek active experts | Full quality: six experts, not the lower-expert speed mode. |
| Vision / concurrency | Text-only managed sessions, one request at a time. Existing multimodal native models keep their own routes. |
| Expert cache and prefill | Official engine automatic sizing and supported SSD placement. |

This preserves the selected quantized artifact; it does not make lossy weights
equivalent to full-precision weights or promise bitwise identity between GPUs.
Upstream [engine details](https://github.com/Niko1221/Strata/blob/870aedd9e1ebe1c91f94da93b59af17729f6c5c0/docs/DETAILS.md)
and [issue #4402](https://github.com/Niko1221/Strata/issues/4402) explain why
approximate routing defaults matter to long-session quality.

**Check placement** reports the loaded Strata configuration. It does not run
upstream `tune auto`, compare candidate settings or claim a measured gain. Those
quick/thorough tuners can take many minutes and reload engines. Turning **Auto
speed** off does not disable the memory placement needed to load the model.

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
| Managed Strata, including supported custom GGUFs | No short native calibration | The real Strata engine manages placement; Check placement reports configuration without a measured gain. |
| BitNet server, Transformers, image/audio/video generation, or incompatible library versions | Existing native configuration retained | The UI reports unsupported tuning rather than treating a connected or loaded model as measured. |

Tuning does not switch to another model, download a different quantization,
change a model's context allowance, change the user's sampling settings, or enable
speculative decoding. Existing policies remain distinct: normal MLX-LM chat uses
its existing TurboQuant-preferred cache factory, while evaluation/RSI keeps its
existing full-precision paths. The VLM path retains its own native cache behavior.
It measures the library's unchanged prefill default by omitting the override. If
there is too little headroom to measure that baseline, tuning is deferred; the
baseline is never silently reduced to make a comparison pass.

The GGUF comparison operates after native memory placement. It does not reload
large models to sweep GPU layers, implement Strata's custom expert-cache
kernels, or reproduce its published throughput on arbitrary models.

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
inside the official Terminal with the pinned release feed. Built-in examples'
model revisions, file sizes and SHA-256 identities are recorded in the catalog
and checked by upstream preparation. Imported Hugging Face sources retain their
immutable revision; local Strata sources retain paths and file metadata and are
checked for changes before use. Local identity checks do not hash every byte of
an arbitrary large model. None of these checks establishes model quality or
performance.

Strata's source is [MIT-licensed](https://github.com/Niko1221/Strata/blob/870aedd9e1ebe1c91f94da93b59af17729f6c5c0/LICENSE).
Its runtime components and the publisher model weights have their own applicable
licenses. The two Qwen example presets reference the Qwen Community License;
other chosen artifacts retain their publisher's terms. Studio's existing
license applies to Studio and does not replace those terms. Studio's
distribution does not embed large model weights. A built-in Strata selection
downloads its chosen pack separately; a custom local selection uses the files
already selected by the user.

## Verification and limits

Behavioral tests exercise exact custom artifact/shard selection, immutable
downloads, companion assets, source changes, cancelled GUI callbacks, model
switch/removal races, registered Strata identities and quality flags,
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
been benchmarked here with full-size custom Qwen/DeepSeek weights, true ternary
Qwen on a MacBook, or the Gyro-S/Q2_0 examples on an 8 GB GPU / 32 GB RAM machine. Actual speed measurements require the installed engine and model on the
target computer. Platform builds and live Metal/CUDA/HIP/Prism inference must be
verified separately from a source/unit-test result. A successful startup check
establishes that a request can run, not sustained throughput, long-context answer
quality or support for an untested hardware combination.
