<div align="center">

# 🧠 Smart AI Studio
### Autonomous 1.58-Bit Local AI & Multi-Agent Reasoning Engine

[![CI & Release Pipeline](https://github.com/smart-ai-studio/smart-ai/actions/workflows/ci-build-release.yml/badge.svg)](https://github.com/smart-ai-studio/smart-ai/actions)
[![License: Commercial Source-Available](https://img.shields.io/badge/License-Commercial_Source--Available-blue.svg)](LICENSE)
[![Platform: macOS | Windows | Linux](https://img.shields.io/badge/Platform-macOS%20%7C%20Windows%20%7C%20Linux-brightgreen.svg)](#installation--quickstart)
[![Architecture: 1.58--Bit Ternary BitLinear](https://img.shields.io/badge/Precision-1.58--Bit%20Ternary-purple.svg)](#model-architectures)
[![Context: Up to 256K Dynamic](https://img.shields.io/badge/Context%20Window-Auto--Scaled%20up%20to%20256K-cyan.svg)](#hardware-adaptive-context-scaling)

*Ultra-low memory, native Apple Silicon MLX / PyTorch desktop studio with real-time token streaming, RLVR sandbox verification, interactive thinking dropdowns, and continuous EWC synaptic sleep consolidation.*

---

[Key Features](#key-features) • [Model Architectures](#model-architectures) • [Installation & Quickstart](#installation--quickstart) • [Agent Tools & MCP](#agent-tool-suite--mcp) • [CI/CD & Releases](#cicd--releases) • [Commercial Licensing](#commercial-licensing)

---

</div>

## 🌟 Key Features

* **⚡ Native 1.58-Bit BitLinear & MLX Acceleration**:
  Runs massive 27.4B foundation matrices on 16GB Apple Silicon Macs with **~5.8 GB VRAM footprint** and zero memory bloat.
* **🧠 Test-Time Pro Reasoning & Best-of-N Search**:
  Dynamic entropy-based compute routing ($H(x)$) scales test-time compute for complex coding and algorithmic challenges.
* **🧪 Ground-Truth RLVR Sandbox**:
  Deterministic AST code verification and subprocess sandbox with resource constraints ensuring 100% verified solutions before committing to memory.
* **💤 Biological Sleep Consolidation (EWC-LoRA)**:
  Replays verified interaction traces during offline sleep cycles, updating synaptic weights without catastrophic forgetting using Elastic Weight Consolidation (EWC).
* **💬 Real-Time Token Streaming**:
  Instantaneous token generation directly to the high-contrast Obsidian desktop interface.
* **💭 Interactive Thinking Dropdowns**:
  Collapsible reasoning traces (`[▶ 💭 Reasoning Process (153 tokens)]`) keeping chat clean and readable.
* **🎯 Steer & Prompt Task Queue**:
  One-click steering modes (`Balanced`, `Code Focus`, `Creative`, `Deep Math`, `Concise`) with sequential background task queue management.
* **📊 Hardware-Adaptive Context Scaling**:
  Dynamically autosets context capacity from **32K to 256K tokens** based on physical host RAM.

---

## 🤖 Model Architectures

Smart AI Studio provides unified, mutual-exclusion VRAM memory management across three specialized model backends:

| Model Tab | Base Parameters | Quantization / Precision | VRAM Allocated | Primary Capability |
| :--- | :---: | :---: | :---: | :--- |
| **✦ Ternary Bonsai** | `27.4B Base` | `1.58-Bit BitLinear (2-bit MLX)` | `~5.8 GB` | Advanced multi-step reasoning, synthesis, and deep logic |
| **⚡ Qwen 3.8 Flash Next** | `3.8B Base` | `1.58-Bit / 4-bit KV Cache` | `~1.8 GB` | High-throughput sub-second responses and quick coding |
| **🔓 Dolphin Vision 2.9** | `7.0B Base` | `Abliterated Multimodal Vision` | `~4.8 GB` | Uncensored vision, image comprehension, and vector art |

---

## 📈 Hardware-Adaptive Context Scaling

The studio automatically inspects host physical RAM via `core.platform` to maximize context window retention without risking out-of-memory (OOM) faults:

| Physical Host RAM | Auto-Calculated Context Budget | Typical Workload Profile |
| :--- | :---: | :--- |
| **$\ge$ 64 GB RAM** (M-Max / Studio) | **262,144 Tokens (256K)** | Massive multi-file codebase analysis & book-length synthesis |
| **$\ge$ 32 GB RAM** (M-Pro / 32GB GPUs) | **131,072 Tokens (128K)** | Full repository context & deep architectural refactors |
| **$\ge$ 15 GB RAM** (16GB Baseline) | **65,536 Tokens (64K)** | Extensive multi-turn chat sessions with zero truncation |
| **$<$ 15 GB RAM** (8GB Lean) | **32,768 Tokens (32K)** | Low-RAM portable laptop execution |

---

## 🛠️ Agent Tool Suite & MCP

Smart AI Studio includes 15+ built-in autonomous tools and native **Model Context Protocol (MCP)** discovery:

* **🌐 Web Search & Scraper**: Real-time web retrieval via `web_search <query>` and `web_fetch <url>`.
* **💻 System Terminal**: Local bash/sh command execution via `run_terminal <cmd>`.
* **📁 Filesystem I/O**: `read_file`, `write_file`, `edit_file`, and `list_dir` for direct workspace development.
* **📐 SymPy Math Solver**: Exact symbolic differentiation, integration, and equation solving.
* **🐍 Python Sandbox**: Isolated execution environment with AST lint validation.
* **💾 Memory Vault**: Persistent SQLite episodic storage and semantic memory retrieval.
* **🎨 Vector Canvas**: Parametric cubic Bezier spline rendering and SVG visual asset generation.
* **🔌 Model Context Protocol (MCP)**: Dynamic tool discovery and execution across external MCP server endpoints.

---

## 🚀 Installation & Quickstart

### Prerequisites
* **macOS** (Apple Silicon M1/M2/M3/M4 recommended for native MLX acceleration), **Linux**, or **Windows**.
* **Python 3.10** or higher.

### 1. Clone Repository
```bash
git clone https://github.com/your-org/smart-ai-studio.git
cd smart-ai-studio
```

### 2. Create Virtual Environment & Install Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Launch Desktop Studio
```bash
python3 app_gui.py
```

### 4. Interactive CLI Mode
```bash
python3 main.py --live
```

---

## 🧪 Testing

Execute the complete 80-test verification suite:
```bash
python3 -m pytest tests/ -v
```

---

## 📦 Multi-Platform Packaging

Generate standalone release bundles for macOS, Windows, and Linux:
```bash
python3 build_app.py
```
Output artifacts generated in `dist/`:
* **macOS**: `dist/SmartAI.app` & `dist/SmartAI-macOS-arm64.zip`
* **Windows**: `dist/SmartAI-Windows.zip` (with `SmartAI.bat` launcher)
* **Linux**: `dist/SmartAI-Linux-x86_64.tar.gz`

---

## 📜 Commercial Licensing

Smart AI Studio is released under the **Commercial Source-Available & Business Use License (Version 1.0)**.

* **Personal & Educational Use**: Free for individual, non-commercial evaluation, personal learning, and research.
* **Business & Enterprise Use**: **MANDATORY COMMERCIAL LICENSE REQUIRED**. Any use, deployment, or modification by or for a business, enterprise, company, or revenue-generating venture requires an active commercial licensing agreement.
* **No Commercial Redistribution**: Redistribution, hosting as a SaaS, or commercial resale of the software or derivative works without written authorization is strictly prohibited.

For commercial licensing and enterprise inquiries, contact: `licensing@smart-ai-studio.local` or open an inquiry in the repository.

---

<div align="center">
<sub>Built with 🧠 by the Smart AI Studio Engineering Team.</sub>
</div>
