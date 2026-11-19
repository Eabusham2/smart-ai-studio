<div align="center">

> **Current defaults — Hybrid + Pro branches:** The third learning choice, **hybrid**, combines original update behavior with persistent memory, validation and context retirement. **Pro branches** starts enabled. The **legacy** and **managed** choices remain available with their separate saved states; Best Work still starts at one pass. This update supersedes the default selection described in the earlier restoration note below. See [Hybrid learning](docs/hybrid-learning.md).

> **Original implementations restored:** Studio now starts with the original learning algorithms and one-pass Best Work. The newer persistent-learning mode and multiple sequential passes remain available through **Learning & saved memory… → Learning implementation → managed** and **Branches → Best Work passes**. See [Original and managed implementations](docs/original-and-managed-learning.md) for the behavior and checkpoint separation. Historical descriptions below of automatic managed learning or two default passes apply only when those features are selected.

# 🧠 Smart AI Studio
### Autonomous 1.58-Bit Local AI & Multi-Agent Reasoning Desktop Engine

[![Version: 2.0.0](https://img.shields.io/badge/Release-v2.0.0-blue.svg)](https://github.com/Eabusham2/smart-ai-studio/releases)
[![CI & Multi-Platform Release](https://github.com/Eabusham2/smart-ai-studio/actions/workflows/ci-build-release.yml/badge.svg)](https://github.com/Eabusham2/smart-ai-studio/actions)
[![License: Commercial Source-Available](https://img.shields.io/badge/License-Commercial_Source--Available-red.svg)](LICENSE)
[![Platform: macOS | Windows | Linux](https://img.shields.io/badge/Platform-macOS%20%7C%20Windows%20%7C%20Linux-brightgreen.svg)](#-installation--quickstart)
[![Precision: 1.58-Bit Ternary BitLinear](https://img.shields.io/badge/Precision-1.58--Bit%20Ternary%20BitLinear-purple.svg)](#-model-matrix--architectures)
[![Context Window: Auto-Scaled up to 256K](https://img.shields.io/badge/Context%20Window-Dynamic%20up%20to%20256K-cyan.svg)](#-hardware-adaptive-context-scaling)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-yellow.svg)](https://www.python.org/)

**Smart AI Studio** is a high-performance, low-memory local AI reasoning and coding studio designed for Apple Silicon (native MLX) and multi-platform environments (CUDA / CPU). Featuring real-time token streaming, a ChatGPT-style side-by-side AI Canvas, conversation history sidebar, reinforcement learning with ground-truth verification (RLVR), interactive thinking dropdowns, and checkpoint-backed learning from chat.

---

[Documentation](#-project-documentation) • [Key Features](#-key-features) • [Model Architectures](#-model-matrix--architectures) • [Hardware Context Scaling](#-hardware-adaptive-context-scaling) • [Agent Tools & MCP](#-agent-tool-suite--mcp) • [Installation](#-installation--quickstart) • [Packaging](#-multi-platform-packaging) • [Licensing](#-commercial-licensing)

---

</div>

## 📚 Project Documentation

For detailed architecture, backend behavior, Eval, learning, media tools, packaging, and platform notes, see **[docs/PROJECT_DOCUMENTATION.md](docs/PROJECT_DOCUMENTATION.md)**.

Media commands documented there include:

```text
/media image <prompt>
/media video <prompt>
/media audio <prompt>
/learn media MODEL_ID SOURCE
/rsi media MODEL_ID <prompt>
```

The project document is the authoritative reference for implementation details; this README remains the quick-start/project overview.

### Lossless execution speedups

Open **Speedups…** for exact prompt-state reuse, shared branch prefill, tokenizer
memoization, opt-in persistent caching and GPU runtime warmup. Compatible native
Bonsai GGUF artifacts can also use prompt lookup, MTP or DFlash2 through the
patched, receipt-verified server. The TensorFold-inspired target-row policy keeps
verification arithmetic aligned with serial decoding and preserves sampler and
recurrent rollback state. Unsupported paths recompute or report incompatibility.

See **[the nine speedups and compatibility guide](docs/ternary-speedups.md)** for
defaults, exactness boundaries, released native runtimes and source-only build
instructions. The tests and native compile gates run without a model; they do
not claim measured hardware output parity or a throughput gain.

### Desktop controls, chat and model memory

The **Eval**, **Context Limit**, **Auto speed** and **Speedups…** controls are
installed when the desktop starts, including the packaged Mac launcher. **Eval**
is visible for normal Bonsai 2 before and after loading. Starting an evaluation
first unloads chat; a failed unload shows **Retry unload** and prevents a second
model from starting. A control that fails to initialize produces a visible
notice and a diagnostic in the app data folder's `logs/gui-extensions.log`.

Exact prompt reuse, shared branch prefill and tokenizer memoization stay enabled
on compatible backends when calibration reports insufficient headroom. Native
drafting defaults to **auto** for a verified compatible native runtime; an
explicitly saved **none** (off) choice is preserved. Enabling an optimization does not
claim a measured speed increase for every prompt.

Chats, episodic recall and learning queues follow their selected model. Each
learning implementation has its own episodic database and saved state. Earlier
unscoped records remain available under **Memory → Earlier shared history…** as
read-only history. Standalone Legacy Sleep requires an unloaded chat model and a
compatible Transformers training base; use Managed or Hybrid Sleep for the
bundled packed models' native training path.

Thinking sections expand while reasoning arrives, can be collapsed immediately,
and collapse once reasoning finishes. Their toggle remains available to reopen
them. Scrolling away from the bottom suspends automatic following until you
return to the bottom. A message sent after Stop waits for the previous worker's
cleanup before starting, preserving its cancellation and model ownership.
Theme changes update existing dialogs and chat widgets as
well as the main window. **AI RAM** reports the app and its owned workers in GiB;
MLX active tensors, peak and cache are shown separately because they overlap the
process footprint.

The **Context** meter uses the active invocation's actual input IDs or native
usage record, including system text, chat-template markers, tool messages and
processor-expanded input. It advances with observed generated tokens. Best Work
candidate costs remain separate from that invocation's context occupancy. A
backend that has not reported an exact count displays **unknown** until it can
measure one; old model/request counters are never reused as a live reading.

Python **Canvas → Run** preserves editor line numbers and reports both printed
output and the exception traceback. Printing before an error no longer hides the
reason it failed; long Python output keeps its final diagnostics visible in a
bounded preview. The existing interpreter, workspace and six-second execution
limit still apply.

MLX checks memory headroom before every request, including when optional tuning
or the user-selected RAM watchdog is off. It limits prefill batches and optional
cache copies, releases failed inference state, and retries a memory failure once
only if no response has been delivered. Recovery preserves the selected model
and full conversation. The background memory monitor cannot raise a stricter
request cache limit, and unchanged monitor ticks leave the allocator alone.
The entropy prepass requests only the final vocabulary projection on compatible
MLX models while still evaluating the full prompt; its routing formula stays the
same. A failed prepass releases its temporary tensors before generation starts.
A model and context still need to fit available memory;
repeated small native CPU/Metal regressions do not certify every 27B workload.

### Your chosen model: automatic placement and speed configuration

Import a local model or fetch an exact Hugging Face artifact. Studio preserves
**your model, selected files and quantization**, including nonternary models;
multiple GGUF choices require an explicit filename. Downloads are revision
pinned and complete shard groups are inspected before registration. Choose
**Automatic**, **Native** or **Strata** as the runtime preference.

Compatible large GGUFs can use native GPU/CPU placement and memory-mapped
weights. Supported Transformers checkpoints can dispatch across GPU, CPU and
disk for inference. Supported MoE GGUF layouts can use the actual
[Strata](https://github.com/Niko1221/Strata) engine's GPU/RAM/SSD expert placement.
**Gyro-S and Q2_0 are optional examples, not a model whitelist or an automatic
replacement for a model that fails to load.** The existing default is unchanged.

Managed Strata uses a private, verified **v0.2.8 / Terminal Control 0.1.54**,
checks the exact selected source with the official planner, prepares it in a
source-specific directory, and starts an authenticated local service owned by
Studio. Unsupported architecture/encoding combinations fail with an explanation.
A model larger than VRAM can run through a compatible offload path, but this does
not guarantee that every model fits RAM, runs quickly, or has a supported loader.

For a **Qwen3.8-Flash-Next ternary model on a Mac**, inspect the actual format.
Strata Metal's Gyro APR trellis support is **nonternary** and does not establish
support for true `TQ1_0`, `TQ2_0` or `PTQ1_0`. Jakevin's custom ternary export and
the Qwen48 T5 export require their publisher-specific MLX/oMLX runtimes, which
this integration does not add. Studio does not replace them with Gyro or Q2.
See the [exact format and Mac limits](docs/STRATA_AUTO_SPEED.md#qwen38-flash-next-ternary-on-a-mac).

An **8 GB GPU + 32 GB RAM + fast SSD** is a relevant configuration for some
supported Strata packs. The
[RTX 5060 Laptop report](https://www.reddit.com/r/LocalLLaMA/comments/1xft2ae/have_8_gb_of_vram_and_32_gb_of_ram_you_can_run/)
is an upstream community result, not a Studio benchmark or a speed guarantee.
The example downloads are **58.5 GB for Gyro-S** and **66.4 GB for Q2_0**, with
rough free-SSD planning estimates of **94.3 GB** and **116.4 GB** respectively,
plus runtime/app data. Custom models have their own preparation requirements.

**Auto speed** and **Retune** separately compare conservative execution settings
on supported loaded MLX and llama.cpp models, accepting repeatable gains only
with the same calibration tokens. Strata's **Check placement** reports its
configuration; it does not run the upstream quick/thorough tuner or claim a
measured gain. Strata and unsupported offloaded Transformers placements remain
inference only. Supported resident CPU, MPS and CUDA placements can use the
validated online learning workflows.

See **[Chosen-model placement, Strata, and native auto speed](docs/STRATA_AUTO_SPEED.md)**
for import behavior, exact sources, quality defaults, storage, cancellation,
platform limits and verification.

---

## 🏷️ Repository Tags & Topics

```
ai • llm • local-ai • mlx • apple-silicon • 1-58bit • bitlinear • bitnet • ternary-bonsai • qwen • dolphin-vision • reasoning-model • rlvr • reinforcement-learning • sleep-consolidation • ewc • code-generation • ai-canvas • desktop-app • tkinter • model-context-protocol • mcp
```

---

## 🌟 Key Features

* **⚡ Native 1.58-Bit BitLinear & Apple Silicon MLX Acceleration**:
  Executes massive **27.4B foundation matrices** on standard 16GB Apple Silicon Macs with **~5.8 GB VRAM footprint** and sub-second generation latencies.
* **💬 Real-Time Token Streaming**:
  Token-by-token real-time streaming directly into the Obsidian high-contrast desktop interface with live token-per-second (`tok/s`) telemetry.
* **📂 Conversation History Sidebar**:
  Collapsible left sidebar with `+ New Chat`, past conversation history indexing, and instant thread switching.
* **🎨 Side-by-Side ChatGPT-Style AI Canvas**:
  Split-pane workspace for editing, inspecting, formatting, sandboxing (`▶ Run`), and exporting documents and multi-language code generated by the AI.
* **🎯 Live Steer & Prompt Task Queue**:
  - **Live Steer (`⌘+Enter` / `Ctrl+Enter`)**: Steer and guide the model live while it is actively typing.
  - **Sequential Task Queue (`Enter`)**: Enqueue prompts while generation is running with automatic sequential runner.
* **💭 Interactive Thinking Dropdowns**:
  Collapsible reasoning pill cards (`[▶ 💭 Reasoning Process (153 tokens)]`) keeping chat clean while preserving complete chain-of-thought traces.
* **🧪 RLVR & Sandbox Verification**:
  Supports deterministic AST validation, unit-test execution, and resource-limited sandbox checks before eligible results are used by learning and memory workflows.
* **🛡️ System RAM & VRAM Memory Watchdog**:
  Continuous background telemetry monitoring host memory pressure and auto-unloading models when thresholds exceed 90% RAM to prevent system freezing.
* **📈 Dynamic Metric Scaling**:
  Human-readable dynamic units for learned synapses ($100 \rightarrow \text{K} \rightarrow \text{M} \rightarrow \text{B}$) and model parameters ($\text{M} \rightarrow \text{B} \rightarrow \text{T}$).

---

## Persistent chat memory

At 80% of Context, eligible text backends start learning from old, complete exchanges. Sleep offers the original verified-episode/EWC queue and a separate current-chat update on the loaded model. Parameter validation and generated recall checks gate retirement from model input; the complete transcript stays in a paged archive. Memory Explorer shows checkpoints, scoped memory, and online learning controls. [Backend support, storage and recall evaluation](docs/parametric-memory.md)

**Online learning** adds resumable document ingestion, evidence selection, few-shot practice, fixed-KV query adaptation on supported models, verifier/consensus reward updates, entropic discovery with PUCT candidate reuse, and a searchable bank of document adapters. Learning persists in the selected model, session or project scope by default. Select **Temporary practice** to use a request copy and restore the prior learned state afterwards. Hardware and runtime capabilities are checked before updates. [Workflows and CLI](docs/test-time-learning.md) · [Research coverage and exclusions](docs/ttt-research-coverage.md)

### New learning and Best Work controls

Start with **[Persistent learning and Best Work: a practical guide](docs/persistent-learning-update.md)**
for the features merged in [PR #17](https://github.com/Eabusham2/smart-ai-studio/pull/17).
The guide connects the desktop controls, reusable learned memory and online tasks.

| Goal | Where to start | Result |
| --- | --- | --- |
| Keep learning for one project | **Memory DB → Learning & saved memory…**; choose `project`, enter a key and select **Apply memory** | Opens a separate transcript and learned-state location for that project and model. Accepted updates create checkpoints; reuse the key to resume. |
| Check the selected model's learning support | Choose **Check model support**, then **Run learning task** | Reports the actual loaded backend's capabilities and missing prerequisites. |
| Study a long document or source directory | Use the panel's ingestion task, or the [online-learning CLI](docs/test-time-learning.md#read-a-complete-source) | Reads bounded segments, validates updates and saves resumable progress. |
| Build reusable document memories | Use the [document adapter bank](docs/test-time-learning.md#save-and-retrieve-document-parameters) | Pins learned document generations and recalls relevant answers during solving. |
| Practice without retaining that task's updates | Select **Temporary practice** | Uses a temporary request copy and restores the previous learned state afterwards. |
| Refine an answer through multiple passes | Enable **Pro branches**; open **Branches → Best Work passes → Apply passes** | Best Work defaults to two sequential passes. Set **1** for the original single-pass behavior. |

Each Best Work pass keeps the existing router and candidate selector. Only the final
selected answer can dispatch an authorized tool; its real result starts a new stage.
If a later refinement cannot fit before displaying output, the completed prior answer
is retained in full. The progress card identifies the skipped refinement, and
token/throughput figures reflect the actual generation.
See [Best Work, tool handoffs and telemetry](docs/best-work-handoffs.md).

The new workflows coexist with ordinary streaming, Learn, RSI and both **Sleep EWC**
sources: **Verified episodes (original Sleep/EWC)** and **Current chat exchanges**.
Older adapter formats have explicit restore/import paths, and evaluation runs keep
their learned checkpoints separate from desktop memory. [Compatibility and storage](docs/PROJECT_DOCUMENTATION.md#23-persistent-learning-and-sequential-best-work-pr-17)

**Memory beyond the working window:** accepted learning updates can remain in saved
parameters after verified source text leaves the active prompt. The transcript stays
archived. A rejected recall check, protected or unfinished content, an oversized
request, or unavailable training can still reach the model's finite context limit.
Strata remains inference-only; native GGUF/Prism/BitNet learning needs a compatible
parent and deployment toolchain. [Backend and recall conditions](docs/parametric-memory.md)

The merged revision passed all six platform/Python CI lanes and real MLX CPU and
Metal learning tests. These include smaller packed Bonsai/CRACK layouts; full 27B
recall, throughput and peak memory have not been measured in these runs.
[Completed validation and release record](docs/validation/test-time-learning.md#merged-revision-and-published-release-pr-17)
· [Downloads](https://github.com/Eabusham2/smart-ai-studio/releases/tag/latest)
· [Research coverage and offline-training exclusions](docs/ttt-research-coverage.md)

## 🤖 Model Matrix & Architectures

Smart AI Studio switches between its existing local model backends and the
optional, separately managed Strata engine:

| Model Tab | Parameter Scale | Quantization / Precision | VRAM Footprint | Primary Specialization |
| :--- | :---: | :---: | :---: | :--- |
| **✦ Ternary Bonsai** | `27.4B Base` | `1.58-Bit BitLinear (2-bit MLX)` | `~5.8 GB` | Complex reasoning, multi-step math, and software architecture |
| **⚡ Qwen 3.8 Flash Next** | `3.8B Base` | `1.58-Bit / 4-bit KV Cache` | `~1.8 GB` | High-throughput sub-second coding and concise query answering |
| **🔓 Dolphin Vision 2.9** | `7.0B Base` | `Abliterated Multimodal Vision` | `~4.8 GB` | Uncensored multimodal comprehension, image analysis, and vector art |
| **Strata Gyro-S** | `~125B MoE + 51B n-gram` | `Gyro APR trellis; nonternary` | `Automatic GPU / RAM / SSD placement` | Large text-model inference; no weight training |
| **Strata Q2_0** | `~125B MoE + 51B n-gram` | `ISTA GSQ-RCO Q2_0; nonternary` | `Automatic GPU / RAM / SSD placement` | Large text-model inference; no weight training |

The last two rows are examples. Imported models retain their own inspected
format, context and source identity. These large Qwen packs are separate from
the smaller existing Qwen preset; their publisher links and storage estimates
are in the [Strata guide](docs/STRATA_AUTO_SPEED.md#example-strata-presets).

---

## 📊 Hardware-Adaptive Context Scaling

The studio auto-detects physical host RAM via `core.platform` to maximize context retention without risking Out-of-Memory (OOM) paging:

| Host Physical RAM | Auto Context Ceiling | Profile Target |
| :--- | :---: | :--- |
| **$\ge$ 64 GB RAM** (M-Max / M-Ultra / Studio) | **262,144 Tokens (256K)** | Full repository context, book-length synthesis, large codebases |
| **$\ge$ 32 GB RAM** (M-Pro / 32GB GPUs) | **131,072 Tokens (128K)** | Multi-file refactoring, deep dependency mapping |
| **$\ge$ 15 GB RAM** (16GB Baseline) | **65,536 Tokens (64K)** | Standard multi-turn coding and continuous reasoning |
| **$<$ 15 GB RAM** (8GB Lean) | **32,768 Tokens (32K)** | Portable laptop execution with minimal memory usage |

The two Strata examples start at **32K context**; custom entries retain their
inspected limits and selected context. The table above is not a fit guarantee
for an imported model. More context reserves additional runtime memory and can
reduce available model residency. Placement does not silently shrink context.

---

## 🛠️ Agent Tool Suite & MCP

Smart AI Studio integrates 15+ built-in autonomous workspace tools and native **Model Context Protocol (MCP)** discovery:

* **🌐 Web Search & Crawling**: Real-time web retrieval via `web_search <query>`, `web_fetch <url>`, and `web_crawler <query>`.
* **💻 System Terminal**: Isolated command execution via `run_terminal <cmd>`.
* **📁 Filesystem I/O**: `read_file`, `write_file`, `edit_file`, `file_search`, and `list_dir` for direct codebase development.
* **📐 SymPy Math & Calculus**: Exact symbolic differentiation, integration, and algebraic equations.
* **🐍 Python Sandbox**: Resource-constrained execution sandbox with AST lint validation.
* **💾 Episodic Memory Database**: SQLite-backed interaction logging, semantic recall, and EWC synaptic parameter consolidation.
* **🎨 Bezier Vector Art & Image Generator**: Parametric cubic spline rendering and SVG vector generation.
* **🔌 Model Context Protocol (MCP)**: Dynamic tool discovery and execution across external MCP servers via `mcp_list_tools`.

---

## 🚀 Installation & Quickstart

### Prerequisites
* **macOS** (Apple Silicon M1/M2/M3/M4 recommended for native MLX acceleration), **Linux**, or **Windows**.
* **Python 3.10** or higher.

### 1. Clone Repository
```bash
git clone https://github.com/Eabusham2/smart-ai-studio.git
cd smart-ai-studio
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Native GGUF inference on Apple Silicon uses stock `llama-cpp-python` with Metal.
For the packaged Mac app, run **Install_Dependencies.command** once before using
that backend. The command is beside the app in the regular package; in the
multipart model bundle it is inside **SmartAI.app → Show Package Contents →
Contents/Resources/app**. It requires native Python 3.10+ with Tkinter and Xcode
Command Line Tools, installs into a user environment, and builds the Metal
binding. The app launcher uses that environment. It does not download models;
managed Strata installs its own separate verified runtime when selected.

### 3. Launch Desktop Studio (GUI)
```bash
python3 app_gui.py
```

### 4. Interactive CLI Mode
```bash
python3 main.py --live
```

### Optional: load a large model through Strata

Import your selected compatible GGUF (or fetch its exact Hugging Face file) and
choose **Automatic** or explicitly **Strata**. The official preflight checks
whether that artifact has a usable native preparation path. Alternatively,
select the optional **Strata Gyro-S** or **Strata Q2_0** example.

First load installs a verified private Terminal and compatible engine, then
prepares the selected files. Built-in examples download their pinned weights;
custom local preparation does not fetch replacement model weights. Follow the
loading progress. **Cancel** retains reusable downloads/preparation files, and
closing or unloading stops only Studio's owned service. No global Strata install
or manual local server is required.

Use the [platform and memory guide](docs/STRATA_AUTO_SPEED.md#platforms-and-memory)
for the exact backend requirements. Apple Silicon requires **macOS 15+** for
managed Strata. A Mac being able to run a small MLX model does not establish
memory fit or format support for a larger custom ternary checkpoint.

---

## 🧪 Testing

Run the repository test suite with:
```bash
python3 -m pytest tests/ -v
```

Test coverage and platform-specific verification may vary by branch and environment; see the project documentation and CI results for technical details.

---

## 📦 Multi-Platform Packaging

**Best Work** runs two sequential inference passes by default, refining the selected draft through the existing router and verifier. Only the final selection displays an answer or executes its selected tool. If refinement cannot fit, the completed prior draft is returned with a clear status. Set **Best Work passes** in **Branches…** to change the count; **1** keeps the original single-pass behavior.

Release packages and Best Work behavior are documented in
[Best Work handoffs and packaging](docs/best-work-handoffs.md). Portable packages
bundle Python and dependencies; setup packages install dependencies into a private
user directory on first launch. No pretrained model is executed by the package
smoke check.

**Apple Silicon full installer:** Download
[`Install-SmartAI-macOS-Full.command`](https://github.com/Eabusham2/smart-ai-studio/releases/download/latest/Install-SmartAI-macOS-Full.command)
and [its SHA256 receipt](https://github.com/Eabusham2/smart-ai-studio/releases/download/latest/Install-SmartAI-macOS-Full.command.sha256)
from the same release. Quit SmartAI, then run these commands in the download folder:

```bash
shasum -a 256 -c Install-SmartAI-macOS-Full.command.sha256 &&
  /bin/bash Install-SmartAI-macOS-Full.command
```

The helper verifies a commit-pinned kit, downloads and checks every model/archive
part, assembles the complete app with its embedded runtime and both Bonsai 2
models, installs `/Applications/SmartAI.app`, removes its own redundant downloads,
and opens the app. It preserves chat/learning data and keeps the previous app as
a dated backup. It requests an administrator password only when the Applications
folder requires one. Allow at least 55 GB free for a fresh download and extraction.
After a failed attempt, completed downloads remain available for retry. The
installer needs no Homebrew, Xcode or system Python. To install elsewhere, set
`SMARTAI_FULL_TARGET` to an absolute path ending in `/SmartAI.app`.

**Apple Silicon no-admin portable:** The release also provides
`SmartAI-macOS-arm64-NoAdmin-Portable-Kit.zip`. It assembles the existing
SHA256-verified split Mac ZIP (which includes bundled Python/dependencies and
both original Bonsai 2 MLX model snapshots) into `~/Applications` without
administrator rights, Homebrew, or Xcode. The first assembly downloads the
release parts; model weights are available offline afterward. See
[No-admin macOS portable instructions](docs/macos-noadmin-portable.md).


Generate standalone release distribution bundles for macOS, Windows, and Linux:
```bash
python3 build_app.py
```

Output packages created in `dist/`:
* **macOS**: `dist/SmartAI.app` and `dist/SmartAI-macOS-arm64.zip`
* **Windows**: `dist/SmartAI-Windows.zip` (with `SmartAI.bat` standalone launcher)
* **Linux**: `dist/SmartAI-Linux-x86_64.tar.gz`

---

## 📜 Commercial Licensing

Smart AI Studio is distributed under the **Commercial Source-Available & Business Use License (Version 1.0)**.

* **Personal, Educational & Research Use**: **FREE**. Individual developers, students, researchers, and hobbyists may inspect, run, test, and evaluate the software free of charge for non-commercial purposes.
* **Business, Commercial & Enterprise Use**: **MANDATORY PAID LICENSE REQUIRED**. Any use, deployment, integration, or internal utilization of this software by, on behalf of, or for the benefit of any company, business, enterprise, or revenue-generating entity requires a valid commercial license.
* **No Unauthorized Commercial Redistribution or SaaS Hosting**: You may not host, sublicense, resell, or distribute this software or derivative works as a commercial cloud service without explicit written permission.

For commercial licensing agreements and enterprise pricing, contact: `licensing@smart-ai-studio.local`.

---

<div align="center">
<sub>Copyright © 2026 Smart AI Studio Contributors. All Rights Reserved.</sub>
</div>
