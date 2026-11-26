# Bonsai 2.5 models and MTP

The main Bonsai and CRACK tabs select Bonsai 2.5 27B. The selected artifact is
platform-specific: Apple Silicon uses the MLX pack, while Windows/Linux use the
combined PTQ1_0-MTP GGUF. Imported models keep their own backend and identity.

## Pinned sources

| Tab and format | Publisher repository | Revision | MTP artifact |
|---|---|---|---|
| Bonsai MLX | [prism-ml/Ternary-Bonsai-2.5-27B-mlx-2bit](https://huggingface.co/prism-ml/Ternary-Bonsai-2.5-27B-mlx-2bit/tree/aa432fd0e17612f6684e728d4d3c42b6656bcabb) | `aa432fd0e17612f6684e728d4d3c42b6656bcabb` | `mtp/mtp-v2-head.safetensors` |
| Bonsai GGUF | [prism-ml/Ternary-Bonsai-2.5-27B-GGUF](https://huggingface.co/prism-ml/Ternary-Bonsai-2.5-27B-GGUF/tree/28a7093cc25bc151cc2b284c5570f12ce343fdcf) | `28a7093cc25bc151cc2b284c5570f12ce343fdcf` | `Ternary-Bonsai-2.5-27B-PTQ1_0-mtp.gguf` |
| CRACK MLX | [barozp/Ternary-Bonsai-2.5-27B-Uncensored-MLX-2bit](https://huggingface.co/barozp/Ternary-Bonsai-2.5-27B-Uncensored-MLX-2bit/tree/0a50dabd13f96ad36394921a03b073dec5b1ab30) | `0a50dabd13f96ad36394921a03b073dec5b1ab30` | Original `mtp/mtp-v2-head.safetensors` |
| CRACK GGUF | [kelsey1337/Ternary-Bonsai-2.5-27B-Uncensored-MTP-GGUF](https://huggingface.co/kelsey1337/Ternary-Bonsai-2.5-27B-Uncensored-MTP-GGUF/tree/ad67f3e0770aefcdac514848ee3595d7edcdf7ad) | `ad67f3e0770aefcdac514848ee3595d7edcdf7ad` | `Ternary-Bonsai-2.5-27B-Uncensored-PTQ1_0-mtp.gguf` |

The uncensored packs contain standalone AtomicChat-derived ternary weights.
Their publishers retain the official MTP v2 head; they are not adapters that
require the old Bonsai 2 CRACK checkpoint. This establishes the intended artifact
and runtime compatibility, not equal quality or measured speed on your hardware.

The catalog in [`config/bonsai_models.py`](../config/bonsai_models.py) stores exact
revisions and SHA-256 identities. Downloads include the MLX runtime files,
processor and MTP head, or the exact combined GGUF plus the official vision
projector. The MLX head is checked before attaching it; native target/projector
files are checked before starting the server. Renaming another GGUF does not
make it compatible.

## Runtime and speedups

**MLX:** the publisher's `vision_artifact` loader preserves the multimodal model
and processor. Its `mtp_artifact` loader attaches the verified head to that same
resident target. Studio uses `draft_kind="mtp"`; the default draft block is two
positions (one proposal plus the target's bonus token). MLX work, including the
split mlx-vlm generation modules, stays on the existing owner thread.

MTP is enabled by **auto** or **mtp** in Speedups. Selecting **none** turns MTP off
and keeps the selected target and ordinary generation. Prompt-state reuse, shared
branch prefill and persistent prompt snapshots pause while MLX MTP is active:
those snapshots do not contain the draft/recurrent state needed to resume it
correctly. Tokenizer memoization and supported memory/prefill controls remain.

The dependency set pins MLX 0.32.0, MLX-LM 0.31.3, MLX-VLM 0.6.3 and
Transformers 5.5.0, with compatible MLX-Audio 0.4.6 and TurboQuant 0.28.1.
The older `dealignai/Bonsai-2-27B-CRACK-Ternary-JANG` loader still exists for
compatible vMLX installations, but vMLX is no longer installed in the default
package: its required `mlx-vlm<0.6` conflicts with the new MTP runtime. Keep a
separate older installation for that legacy checkpoint; the new CRACK preset
does not require vMLX. The updater retains old model files rather than deleting
or converting them.

**GGUF:** the combined PTQ1_0-MTP file uses the publisher's
[Prism runtime release `prism-b12220-41c86f7`](https://github.com/PrismML-Eng/llama.cpp/releases/tag/prism-b12220-41c86f7).
Studio downloads the matching executable archive on first load, verifies its
size and SHA-256, then caches it. It does not compile a runtime or fetch another
model for each request. Apple runtimes use Metal; automatic Windows/Linux
selection uses CUDA when detected, otherwise CPU. Advanced users can choose a
published variant with `SMARTAI_BONSAI_NATIVE_VARIANT` or a verified cached server
with `SMARTAI_BONSAI_SERVER`.

The native backend retains the projector for desktop image/video requests,
exact tokenizer/context accounting, cancellation ownership and the existing
Pro/Best Work workflow. It does not expose a full target vocabulary entropy;
**Native branches** remains the explicit branch-count control. Native inference
does not train the GGUF weights.

Bonsai 2.5 accepts **auto**, **mtp** and **none**. The older Bonsai 2 DFlash2 head
and prompt-lookup configuration belong to the separate original PQ2_0 runtime;
they cannot be combined with the new 2.5 packs. The older runtime and nine
speedup controls remain available for their supported artifacts. Read
[speedup boundaries](ternary-speedups.md) before comparing results. Neither
model-free CI nor publisher acceptance figures demonstrate a measured speedup
or byte-identical output for your machine.

## Download and context

CI/release packages contain no model weights. Select the desired tab and choose
**Download** in Studio, then **Load**. Existing user model downloads are kept by
the updater; the new presets request their own revisions when needed.

Model files, runtime allocations, draft state and the context cache all consume
memory. Artifact sizes are not a peak-RAM promise. Set Context to fit the machine;
for native GGUF, unload and reload after changing the desired context so the
server is allocated with that value. The advertised model maximum is 262,144,
while the ordinary preset starts at 32,768 before hardware constraints.

For Codex, Claude Code and Hermes, load the model and use
[API Server](local-model-api.md). The API has text and client-tool support; the
desktop's image/video workflow remains separate.
