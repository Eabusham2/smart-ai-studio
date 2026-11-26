# Original MLX checkpoints on Windows and Linux

Studio loads the selected MLX safetensors and its matching tokenizer directly.
The native route uses the same Studio MLX engine and learning implementations
as macOS. It does not convert a checkpoint or substitute a GGUF.

## Choose the runtime

| Platform / hardware | Native MLX route | Optional TensorFold route |
| --- | --- | --- |
| Apple Silicon macOS | Existing Metal runtime and dependency pins | Not used by this adapter |
| Linux x64 / ARM64, CPU | MLX CPU | Only models and backends supported by the selected native binary |
| Linux with supported NVIDIA GPU | MLX CUDA 12 or CUDA 13 | Native CUDA, if advertised by the binary |
| Windows x64, CPU | Official MLX CPU wheels | Requires a separately installed native binary |
| Windows x64 with supported NVIDIA GPU | Official MLX CUDA 12 or CUDA 13 wheels | Requires a compatible native CUDA build |
| Windows ARM64 | Official MLX CPU wheels; use Python 3.11 or newer | Depends on the native build |
| AMD or Intel GPU | Native MLX uses the CPU; it has no ROCm or XPU backend | HIP/ROCm or XPU only when compiled into the selected binary; standard supported text models only |

Native MLX supports Studio's existing legacy, managed and hybrid learning
paths, adapters, Fisher/EWC updates, Best Work, and per-model chat history.
Compatibility still depends on the model architecture and available memory.
CPU execution has the same operations but is not a performance substitute
for an accelerated 27B deployment.

Automatic MLX selection means **native MLX**. A missing or broken native runtime
produces a setup error. Studio never silently changes a learning-capable MLX
model into an inference-only TensorFold model. The per-model Runtime choice
takes precedence over `SMARTAI_MLX_RUNTIME`.

## Install or repair native MLX

The normal Windows/Linux bootstrap installs the native runtime before marking
setup ready. For an existing environment, run these commands from the Studio
repository with the same Python interpreter Studio uses:

```sh
python tools/setup_mlx.py --backend auto
python tools/setup_mlx.py --check
```

To target a different Studio interpreter explicitly:

```sh
python tools/setup_mlx.py --python "/path/to/studio/python" --backend cpu
```

Choose an accelerated backend explicitly when needed:

```sh
python tools/setup_mlx.py --backend cuda12
python tools/setup_mlx.py --backend cuda13
```

`--dry-run` prints the package plan. `--check` runs a real packed matrix
multiply, Hadamard transform and autodiff check without installing packages or
downloading a model. Setup also verifies that MLX-LM and MLX-VLM import.
A failed probe leaves setup incomplete with the error visible.

The Windows/Linux stack is pinned to MLX **0.32.5**, MLX-LM **0.31.3** and MLX-VLM
**0.6.3**. Setup installs an explicit `mlx-cpu`, `mlx-cuda-12` or
`mlx-cuda-13` package. This matters on Windows because the MLX frontend's
CPU/CUDA extras use Linux-only dependency markers in this release.

For a new environment, automatic selection checks the first visible NVIDIA
device and its driver. It chooses CUDA 13 for a compatible GPU and driver
580 or later, CUDA 12 for a compatible GPU and driver 550.54.14 or later,
and CPU otherwise. Supported CUDA GPUs require compute capability 7.5 or
newer. Linux wheels require glibc 2.35 or newer. The execution probe, rather
than driver detection alone, decides whether setup succeeds. An already
installed single backend is retained by `auto`; use an explicit backend to
switch it. CUDA visibility specified by UUID or MIG identifier needs an
explicit backend choice.

Only one MLX backend distribution should be installed in an environment.
The setup command downloads a replacement backend before removing a
conflicting one, then reinstalls the selected native libraries and checks
them. `SMARTAI_MLX_BACKEND=cpu|cuda12|cuda13|auto` controls the bootstrap choice.

The project also exposes `mlx-linux-cpu`, `mlx-linux-cuda12`,
`mlx-linux-cuda13`, `mlx-windows-cpu`, `mlx-windows-cuda12` and
`mlx-windows-cuda13` pip extras. Use the setup command for backend switching
and diagnosis.

MLX-LM 0.31.3 imports the POSIX `resource` module in its loader. Studio's
Windows compatibility module adapts that one file-limit operation to the
Windows CRT during import. It does not install a process-wide fake
`resource` module or edit the installed MLX-LM package.

## Bonsai 2.5 and CRACK 2.5

Windows and Linux expose two explicit MLX model choices alongside their
existing GGUF choices:

- **Bonsai 2.5 27B Ternary Multimodal (MLX)**:
  `prism-ml/Ternary-Bonsai-2.5-27B-mlx-2bit`.
- **Bonsai 2.5 27B CRACK (Abliterated/Uncensored) (MLX)**:
  `barozp/Ternary-Bonsai-2.5-27B-Uncensored-MLX-2bit`.

Both select the existing pinned publisher loader, packed weights, vision
assets and MTP head. Importing a complete local snapshot or Hugging Face
repository through the model import dialog is also supported.

The publisher's packed projections use MLX Hadamard and quantized matrix
operations. The CPU/CUDA route keeps those packed tensors intact. Studio
guards the pinned MLX-VLM GatedDeltaNet verification helpers so CPU/CUDA
execution uses their portable array equations instead of invoking
Metal-only kernels. macOS keeps the Metal implementations.

These are the current Bonsai/CRACK **2.5** packs. Older JANG archives use a
separate Apple Silicon runtime and are not made portable by this change.

Bonsai and CRACK GPU acceleration on AMD/Intel is **not implemented**.
TensorFold v1.3.5's native runtime does not provide this publisher's Bonsai
model family. Its older Python Bonsai recipe is not a replacement native
backend. Select native CPU on those machines, or a supported NVIDIA CUDA
device for the native accelerated route.

## Optional TensorFold inference

TensorFold is an explicit option for a standard **text-only MLX-LM**
checkpoint. Install or build the native runtime separately, then set
`SMARTAI_TENSORFOLD_EXECUTABLE` to its absolute path, or put
`tensorfold-native` on `PATH`. Select **TensorFold** in the import dialog,
or use `SMARTAI_MLX_RUNTIME=tensorfold` for a model without an overriding
runtime choice.

The adapter follows the native CLI and protocol from TensorFold **v1.3.5**
(commit `02b324ae9767253d9911d753d5c835a5070eed1b`). It asks the executable
for its advertised capabilities. `SMARTAI_TENSORFOLD_BACKEND` can select a
supported backend. Upstream HIP uses the CLI value `rocm`; a dedicated XPU
binary uses its advertised default instead of an invented `--backend xpu`
flag. A release binary is not assumed to contain every GPU backend.
Windows requires a compatible native build; this branch does not download
an unverified Windows executable.

Studio owns the native process, private loopback port and ephemeral token.
It keeps the user's full prompt and history, checks native tokenization
against the checkpoint tokenizer, and submits exact token IDs to the raw
completion endpoint. The adapter reads real final usage and stop metadata.
It rejects unsupported inputs and options instead of ignoring them.
Cancellation closes the active transport, and unload waits for owned
resources to stop before the GUI switches models.

TensorFold does **not** expose Studio learning, LoRA/Fisher/EWC updates,
vision, Bonsai MTP, or measured token entropy through this adapter. The UI,
local API and speed controls preserve this inference-only boundary.

## Validation and remaining hardware checks

The branch contains focused tests for runtime selection, setup failure
handling, backend visibility, original checkpoint identity, Windows import
compatibility, TensorFold's native protocol and cancellation, GUI unload
ownership, API behavior, and chat-history preservation.

The MLX numerical tests exercise the actual CPU runtime with packed
2-bit projections, Hadamard transforms, decoder/cache operations, gradients,
Fisher accumulation, learning updates and adapter persistence. They do not
replace full-size hardware testing.

The dedicated Windows CPU workflow installs the pinned native runtime, runs
its arithmetic/import probe, and executes offline numerical and process
fixtures. It fails if any selected test is skipped; check its result on the
reviewed commit. No physical NVIDIA, AMD or Intel GPU was available for this
implementation session. A complete 27B Bonsai/CRACK generation, multimodal
run and MTP performance comparison must still be exercised on the intended
hardware. Do not infer GPU speedups or complete 27B acceptance from CPU checks.

The targeted checks can be run without downloading model weights:

```sh
python -m pytest -q tests/unit/test_native_mlx_setup.py tests/unit/test_cross_platform_mlx.py tests/unit/test_tensorfold_mlx_engine.py
python -m pytest -q tests/unit/test_mlx_packed_terminal.py tests/unit/test_mlx_learning_implementations.py
```

The final combined application regression run passed **478 tests and 16
subtests**. Ten tests were skipped in that lightweight environment because
optional Torch or MLX packages were absent. Separate real MLX CPU environments
passed the 17 packed/learning checks and 27 publisher/runtime/memory checks.
The TensorFold suite also passed 25 subprocess/socket protocol and lifecycle
tests; capability discovery was additionally checked against the official
checksum-verified Linux TensorFold v1.3.5 binary.

### Observed Linux CPU results

The following runs used MLX 0.32.5 and the original published artifacts:

- Full **mlx-community/SmolLM-135M-Instruct-4bit** at revision
  642e06afe3fab57fd6cc518637c471af0a569e1e loaded through
  MLXReasoningBackend and generated two native decode tokens, producing
  "of a" after "The capital". Peak resident memory was 215,576 KiB.
- The complete small published **malaiwah/qwen3-5-tiny-random-mlx4-g32-rtn-format**
  at revision b5c9c098555edb8253091f443711f7b3adae5fd8 passed chat and
  one managed training step. The step changed 18 adapter tensors containing
  6,912 trainable parameters, with L2 update norm 0.0262611, and saved the
  adapter. This is an execution check of a tiny random fixture, not a quality
  evaluation or a 27B training result.
- Original final-layer packed projections were range-read independently from
  each pinned Bonsai 2.5 and CRACK 2.5 repository. Their own runtime.Packed
  implementation evaluated eight rows of q_proj (5,120 input columns)
  and eight rows of down_proj (17,408 input columns). Against independent
  NumPy bit unpacking and signed Hadamard transforms, maximum absolute errors
  were 0.00041414 and 0.00071836, respectively.

The packed-publisher check is reproducible with an explicit download opt-in:

```sh
python tools/verify_publisher_mlx.py --download-samples --device cpu
```

Use --device cuda or --device metal on corresponding hardware. The
script pins the model revisions and executable source checksums and fetches
bounded weight ranges. It does not run during setup or CI and does not load
an entire 27B model. A SmolLM training attempt exceeded the 40-second check
limit; only the tiny Qwen training result above is a completed full-graph
training check from this session.

## Upstream references

- [MLX 0.32.5 frontend wheels](https://pypi.org/project/mlx/0.32.5/)
- [MLX 0.32.5 CPU backend wheels](https://pypi.org/project/mlx-cpu/0.32.5/)
- [MLX installation and CUDA requirements](https://ml-explore.github.io/mlx/build/html/install.html)
- [TensorFold v1.3.5 source](https://github.com/ashhart/TensorFold/tree/02b324ae9767253d9911d753d5c835a5070eed1b)
- [TensorFold Windows build instructions](https://github.com/ashhart/TensorFold/blob/02b324ae9767253d9911d753d5c835a5070eed1b/docs/WINDOWS.md)
