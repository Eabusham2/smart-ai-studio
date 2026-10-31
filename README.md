<div align="center">

# 🧠 Smart AI Studio
### Autonomous 1.58-Bit Local AI & Multi-Agent Reasoning Desktop Engine

[![Version: 2.0.0](https://img.shields.io/badge/Release-v2.0.0-blue.svg)](https://github.com/Eabusham2/smart-ai-studio/releases)
[![CI & Multi-Platform Release](https://github.com/Eabusham2/smart-ai-studio/actions/workflows/ci-build-release.yml/badge.svg)](https://github.com/Eabusham2/smart-ai-studio/actions)
[![License: Commercial Source-Available](https://img.shields.io/badge/License-Commercial_Source--Available-red.svg)](LICENSE)
[![Platform: macOS | Windows | Linux](https://img.shields.io/badge/Platform-macOS%20%7C%20Windows%20%7C%20Linux-brightgreen.svg)](#-installation--quickstart)
[![Precision: 1.58-Bit Ternary BitLinear](https://img.shields.io/badge/Precision-1.58--Bit%20Ternary%20BitLinear-purple.svg)](#-model-matrix--architectures)
[![Context Window: Auto-Scaled up to 256K](https://img.shields.io/badge/Context%20Window-Dynamic%20up%20to%20256K-cyan.svg)](#-hardware-adaptive-context-scaling)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-yellow.svg)](https://www.python.org/)

**Smart AI Studio** is a high-performance, low-memory local AI reasoning and coding studio designed for Apple Silicon (native MLX) and multi-platform environments (CUDA / CPU). Featuring real-time token streaming, a ChatGPT-style side-by-side AI Canvas, conversation history sidebar, reinforcement learning with ground-truth verification (RLVR), interactive thinking dropdowns, and biological synaptic sleep consolidation.

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

### Strata models and automatic speed configuration

Choose **Strata Gyro-S** or **Strata Q2_0** to run the actual
[Strata](https://github.com/Niko1221/Strata) engine. These optional, text-only
Qwen3.8-Flash-Next presets use Strata's automatic GPU, RAM and SSD placement, so
the model files do not all have to fit in graphics memory or RAM. The existing
default model remains selected until you choose a Strata preset.

Studio installs a private, verified **Strata v0.2.8 / Terminal Control 0.1.54**,
prepares the selected model, and starts an authenticated local service whose
lifetime belongs to Studio. The model downloads are substantial: **58.5 GB for
Gyro-S** or **66.4 GB for Q2_0**, with additional preparation files. Allow about
**94.3 GB** or **116.4 GB** of free SSD space respectively, plus room for runtime
components and other application data. These are planning estimates; the
upstream preparation checks determine the actual requirement.

An **8 GB GPU + 32 GB RAM + SSD** is a relevant tiered-inference configuration,
not a promise of a particular speed or compatibility with every 8 GB GPU. The
[reported RTX 5060 Laptop case](https://www.reddit.com/r/LocalLLaMA/comments/1xft2ae/have_8_gb_of_vram_and_32_gb_of_ram_you_can_run/)
is a community result, not a Smart AI Studio benchmark. Studio explicitly uses
fp16 KV and disables drafting, speculative decoding and the listed approximate
routing shortcuts. These settings can perform differently from that report.
Both Strata presets are **nonternary and inference only**; learning remains
available on the existing backends that support weight updates.

The separate **Auto speed** and **Retune** controls compare conservative native
execution settings on supported MLX and llama.cpp models. They keep working
settings unless a candidate repeatedly clears a measured noise margin and
produces the same calibration tokens. For Strata, **Check placement** reports
its automatic configuration; it does not run the upstream quick/thorough tuner
or claim a measured speed gain.

See **[Strata integration and native auto speed: setup, behavior, and limits](docs/STRATA_AUTO_SPEED.md)**
for model identities, storage, platform support, cancellation, and verification.

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
* **💤 Biological Synaptic Sleep Consolidation (EWC-LoRA)**:
  Consolidates high-reward interaction traces during offline sleep cycles using Elastic Weight Consolidation (EWC) to prevent catastrophic forgetting.
* **🛡️ System RAM & VRAM Memory Watchdog**:
  Continuous background telemetry monitoring host memory pressure and auto-unloading models when thresholds exceed 90% RAM to prevent system freezing.
* **📈 Dynamic Metric Scaling**:
  Human-readable dynamic units for learned synapses ($100 \rightarrow \text{K} \rightarrow \text{M} \rightarrow \text{B}$) and model parameters ($\text{M} \rightarrow \text{B} \rightarrow \text{T}$).

---

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

Strata's parameter counts, quantization and file requirements are separate from
the smaller existing Qwen preset. Its registered choices and publisher links are
listed in the [Strata guide](docs/STRATA_AUTO_SPEED.md#the-two-strata-models).

---

## 📊 Hardware-Adaptive Context Scaling

The studio auto-detects physical host RAM via `core.platform` to maximize context retention without risking Out-of-Memory (OOM) paging:

| Host Physical RAM | Auto Context Ceiling | Profile Target |
| :--- | :---: | :--- |
| **$\ge$ 64 GB RAM** (M-Max / M-Ultra / Studio) | **262,144 Tokens (256K)** | Full repository context, book-length synthesis, large codebases |
| **$\ge$ 32 GB RAM** (M-Pro / 32GB GPUs) | **131,072 Tokens (128K)** | Multi-file refactoring, deep dependency mapping |
| **$\ge$ 15 GB RAM** (16GB Baseline) | **65,536 Tokens (64K)** | Standard multi-turn coding and continuous reasoning |
| **$<$ 15 GB RAM** (8GB Lean) | **32,768 Tokens (32K)** | Portable laptop execution with minimal memory usage |

The Strata presets start at **32K context** and use their own placement planner.
The table above is not a Strata memory-fit guarantee. Increasing context reserves
more runtime memory and can reduce the expert cache or require more SSD reads.

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

### 3. Launch Desktop Studio (GUI)
```bash
python3 app_gui.py
```

### 4. Interactive CLI Mode
```bash
python3 main.py --live
```

### Optional: load a Strata model

In the desktop model selector, choose **Strata Gyro-S** or **Strata Q2_0** and
start loading. The first load downloads the verified private Terminal, a
compatible native engine, and the selected model files, then prepares them on
disk. Follow the loading progress; **Cancel** stops the active operation while
keeping downloaded model data for a later attempt. No global Strata install or
manual local server is required.

Use a supported GPU and a fast SSD with the space described in the
[Strata setup guide](docs/STRATA_AUTO_SPEED.md#platforms-and-memory). Apple Silicon
requires **macOS 15 or newer** for this engine; a 16 GB MLX-capable Mac is not
therefore a verified fit for these much larger Strata packs.

---

## 🧪 Testing

Run the repository test suite with:
```bash
python3 -m pytest tests/ -v
```

Test coverage and platform-specific verification may vary by branch and environment; see the project documentation and CI results for technical details.

---

## 📦 Multi-Platform Packaging

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
