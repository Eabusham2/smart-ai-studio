# Text-controlled local media

The desktop app's existing text/Pro engine can request catalog image, video and audio generation through bounded media tools. This is an app integration, not a change to the 4K evaluator or text model trainer.

## Commands

- `/media image <prompt>`, `/media video <prompt>`, `/media audio <prompt>` use the existing media generators.
- `/learn media model_3 examples.jsonl` applies a supported media adapter update. Existing `/learn <topic>` stays on the text learning path.
- `/rsi media model_3 <prompt>` proposes alternatives with the existing text Pro solver, generates and measures actual artifacts, and updates a supported media adapter from the selected self-generated example. Alignment selection is not a correctness reward or proof of quality improvement.

A learning JSONL row is an object with `path` (relative to the manifest, inside the workspace) and `caption`. The interactive path allows one to eight examples. It updates LoRA adapters; it does not rewrite all base model weights.

## Actual support and limits

Generation delegates to the existing image/video/audio engines. It inherits their model compatibility and installed dependency requirements; a catalog label alone is not proof a checkpoint can load.

Native weight training is implemented for the existing RealVisXL / SDXL image preset (`model_3`). Other media backends, including the current video, audio and mflux presets, return `unsupported` for weight learning until their model-specific loss, save and checkpoint-verification adapter is registered through trusted Python code. They are not passed into the text QA trainer.

Image reviews use CLIP embeddings from actual pixels. Video review samples up to eight frames, not temporal/audio quality. Audio review uses CLAP on at most the first ten seconds. These are alignment proxies, not human-equivalent perception or guaranteed quality grades. Missing review dependencies/models produce no fabricated score.

The memory scheduler leaves text resident when estimated RAM/VRAM headroom allows. Otherwise it pauses at a completed text-generation boundary, saves and verifies learned MLX LoRA tensors, runs media, and restores the same text model and tensor snapshot. GGUF restores the same model/adapter configuration. If there is still insufficient headroom, it aborts rather than truncating context or lowering precision. Estimates are not hard allocation guarantees.

Weight updates are serialized, require finite gradients and a measurable parameter change, and publish a checkpoint only after saved tensors are verified. Failed updates roll parameters back. A successful update does not prove improved quality. Existing EWC calculation is reused only when the backend supplies genuine Fisher/anchor data; the basic SDXL adapter does not claim it has those statistics.

## Verification

Twenty lightweight control-flow tests cover routing, exact identity on resume, error recovery, cancellation, bounded calls, input paths and unsupported training. A separate CPU-only test of the exact updater method and persistence helpers demonstrated a real AdamW parameter change, tensor checkpoint equality, and rollback after failed verification, nonfinite loss and cancellation. The CPU fixture did not exercise SDXL, CLIP/CLAP, MLX, GGUF model inference or optional EWC calculations. No full local-model run or full CI was performed.
