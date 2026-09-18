# Text-controlled local media

The desktop app's existing text/Pro engine can request catalog image, video and audio generation through bounded media tools. This is an app integration, not a change to the 4K evaluator or text model trainer.

## Commands

- `/media image <prompt>`, `/media video <prompt>`, `/media audio <prompt>` use the existing media generators.
- `/learn media MODEL_ID SOURCE` trains the selected media model through a real architecture-matched adapter/trainer. SOURCE may be a local file/folder, structured manifest, direct URL/page, or bounded search query. Existing `/learn <topic>` stays on the text learning path.
- `/rsi media MODEL_ID <prompt>` is available only when the active text controller can directly ingest that media type. It generates alternatives, lets that same controller inspect/grade the actual output, then trains the selected media model from the chosen self-generated sample.

All text controllers can generate image/video/audio and run explicit media Learn. Media RSI is capability-gated: image input unlocks image RSI; audio input unlocks audio RSI; video input unlocks image+video+audio RSI; image+audio also unlocks all three. Text-only controllers cannot grade/RSI media they cannot ingest.

## Actual support and limits

Generation delegates to the existing image/video/audio engines. It inherits their model compatibility and installed dependency requirements; a catalog label alone is not proof a checkpoint can load.

Media Learn resolves trainers by architecture/base checkpoint rather than catalog slot. Current trainer families include SDXL, FLUX.2/Z-Image, Wan/LTX/CogVideoX paths, Stable Audio 3, and generic registered trainer extensions. Quantized/MLX/GGUF derivatives resolve back to a differentiable base checkpoint when metadata permits. If no real trainer can be resolved, Learn returns `unsupported` rather than faking a weight update.

Media review/RSI uses the active text controller's own declared and working media-input path. A controller that cannot ingest the requested modality can still generate and explicitly train that media model, but it cannot claim to inspect or self-grade the artifact.

The memory scheduler leaves text resident when estimated RAM/VRAM headroom allows. Otherwise it pauses at a completed text-generation boundary, saves and verifies learned MLX LoRA tensors, runs media, and restores the same text model and tensor snapshot. GGUF restores the same model/adapter configuration. If there is still insufficient headroom, it aborts rather than truncating context or lowering precision. Estimates are not hard allocation guarantees.

Weight updates are serialized, require finite gradients and a measurable parameter change, and publish a checkpoint only after saved tensors are verified. Failed updates roll parameters back. A successful update does not prove improved quality. Existing EWC calculation is reused only when the backend supplies genuine Fisher/anchor data; the basic SDXL adapter does not claim it has those statistics.

## Verification

Contract/unit coverage locks tool routing, capability-gated RSI, exact text-model pause/resume identity, local/online Learn sources, trainer resolution, cancellation/error recovery, and verified adapter persistence. These tests do not substitute for running every large media model/trainer end-to-end on real hardware.
