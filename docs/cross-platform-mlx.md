# Cross-platform MLX checkpoint support (experimental branch)

MLX is a checkpoint format and a framework, not a requirement to use Metal.
This branch keeps the selected safetensors, quantization and model identity
unchanged. It does not download or substitute a GGUF.

- macOS ARM64: the existing native MLX engine, speed and training paths.
- Linux: install the optional `mlx-linux-cuda12`, `mlx-linux-cuda13` or
  `mlx-linux-cpu` extra in the Python environment used by Studio. Example:
  `python -m pip install 'mlx[cuda12]==0.32.0' 'mlx-lm==0.31.3'`.
  Match NVIDIA drivers and CUDA versions to the official MLX documentation.
- Windows: build/install a compatible TensorFold native executable following
  https://github.com/ashhart/TensorFold/blob/main/docs/WINDOWS.md. Set
  `SMARTAI_TENSORFOLD_EXECUTABLE` to its full path or put it on PATH.
  Linux can also select TensorFold via `SMARTAI_MLX_RUNTIME=tensorfold`.

Import a complete compatible MLX text checkpoint (local directory or Hugging
Face repository) through the existing model import UI and download it as
usual. For Linux, `SMARTAI_MLX_RUNTIME=native` forces the MLX library.
The default is auto. The existing platform-specific Bonsai presets and all
GGUF engines remain unchanged.

TensorFold is launched with a private loopback port, ephemeral authentication,
the selected local checkpoint, and a bounded context. It uses the checkpoint's
local tokenizer and streams raw completions, so the Studio desktop and external
API harness retain ownership of prompts. Only real server usage is reported.
TensorFold is inference-only in this branch; no Studio learning, adapter
deployment, multimedia, speculative controls or measured speedups are claimed.

Bonsai 2.5/CRACK publisher MLX packs need their custom Hadamard kernels, vision
runtime and MTP port. These remain unavailable on non-Mac systems rather than
silently loading the GGUF equivalent. TensorFold model and hardware support
varies; Windows and Linux real GPU execution have not been verified in Studio.
See https://ml-explore.github.io/mlx/build/html/install.html and
https://github.com/ashhart/TensorFold for upstream support boundaries.
