# Local media presets

All 15 original built-in model slots remain in the catalog. Install downloads
the selected model and its required companions; Load prepares its compatible
runtime. Image/video weights are allocated for generation and released after
the request. Install, Cancel, Load and Unload share the same primary button.

## Restored presets

Sizes below are decimal GB of selected model files, including companions.
Runtime dependencies and temporary generation files need additional disk space.

| Preset | Selected backend and components | Model download |
| --- | --- | ---: |
| Z-Image Turbo NSFW v2 | Original Q8 GGUF transformer, Qwen3-4B Q4 encoder and Z-Image VAE; bundled native GGUF runtime | 10.1 GB |
| Qwen Image Edit Rapid v9 | Q4 GGUF transformer, publisher's Q4 abliterated Qwen2.5-VL encoder, Q8 vision projector and Qwen VAE; bundled native GGUF runtime | 19.1 GB |
| Ideogram Instant | The original preset's **RealVisXL V5.0** local SDXL FP16 pipeline | 6.9 GB |
| Wan 2.2 Remix v2 | Original T2V Q4 high/low-noise expert pair, Q4 UMT5 encoder and Wan VAE; bundled native GGUF runtime | 23.2 GB |
| LTX-Video 2.5 | Publisher's Q4 MLX distilled pipeline, text encoder, connector, audio/video VAEs, vocoder and upscaler | 28.8 GB |
| MiniMax-H3 + AfterMidnight | Publisher's Q4 MLX transformer, original Ref2VA text encoder and audio/video VAEs, plus AfterMidnight v1.2 rank64 LoRA | 104.2 GB |

**Ideogram Instant is a preset name, not an integration with the Ideogram
service or proprietary weights.** It retains the same RealVisXL repository
selected by the original app. Its two catalog entries reuse the same cache;
they do not download duplicate copies. The old 4-step and universal 16 GB claims
are not retained.

The two MLX video presets require an **Apple Silicon Mac**. For the selected
LTX pack/default settings, the working-memory estimate is 20 GiB and a 32 GB
Mac is recommended. MiniMax's working-memory estimate is 64 GiB and a 96 GB
Mac is recommended. These estimates account for the separate text encoder and
decoders; they are not measurements of a full pretrained generation in CI.
The runtime checks available memory before allocating these models. LTX's
publisher repository also requires Hugging Face access approval.

## Image editing, progress and saved output

Select **Qwen Image Edit**, install/load it, attach a local image and send an
editing instruction. The reference image is passed to the native pipeline as
an image; its binary bytes are never flattened into the text prompt. The
`edit_image_rapid` tool accepts `prompt` and a local `image_path`.

Generation reports actual sampling steps. Wan can show a latent-projection
preview and LTX can show decoded intermediate frames; these are labelled as
previews. The final image or video must decode successfully before completion
is reported. Videos are MP4 files with playable frames, and LTX/MiniMax preserve
their generated audio. Use the large preview, expand/play controls, **Save As**,
or **Open original**. The embedded video preview is silent; Open original uses
the system player for audio.

Stopping generation cancels the owned worker/process group and keeps existing
output files intact. Runtime setup uses private environments for conflicting
MLX dependencies, preserving the main chat environment.

## Installation and cache recovery

Each built-in selection records immutable repository revisions and exact
filenames in [the media catalog](../config/media_models.py) and
[component inventories](../config/media_components.py). The app downloads one
selected quantization/version and checks every companion. A transformer on its
own cannot mark a multi-component model installed.

The installer and restart checks use the same selected files and revisions.
Completed cache files are reused even when the Hub cache has no `refs/main`.
If a selected file is missing or zero bytes, retry repairs only that file;
completed model files retain their bytes and timestamps. Interrupted Xet/CAS
transfers use the shared bounded HTTP recovery path.

The GGUF executable is built from pinned
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp/tree/52491bd9f1fc15281791723bdff13baab7fd6819)
sources during native packaging and ships inside both package formats. Users
do not need a compiler. The LTX and MiniMax private workers use the pinned
[LTX MLX publisher](https://github.com/dgrauet/ltx-2-mlx/tree/e5db9dd6935d3a2ff00108853d3ccc0f919154f6)
and [MiniMax MLX publisher](https://github.com/PipeNetwork/minimax-h3-mlx/tree/b2f7e4d2b7861cefe68b75e4b59ab81cc4e7c318)
implementations. Generation workers use already selected local files offline.

The bundled GGUF runtime uses Metal on Mac. This release builds its Linux
runtime for CPU to retain Ubuntu 22 compatibility; Windows uses Vulkan when a
compatible SDK was available at build time, otherwise CPU. Other existing
Diffusers pipelines retain their usual device selection.

Release checks exercise native executable startup, publisher APIs, tiny native
operators, LoRA application and media codecs. They do not download the large
pretrained models or certify generation quality across every GPU/RAM setup.
