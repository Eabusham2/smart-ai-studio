# Master Architecture & Implementation Blueprint: Local 1.58-Bit 27B Autonomous Reasoning System

This master document synthesizes the complete conceptual progression, research insights, and technical architecture for building a local, self-improving reasoning engine powered by a 1.58-bit ternary 27B model. It bridges high-compute "Pro" search algorithms with a biologically inspired continuous memory consolidation engine.

---

## Part 1: Conversational Context & Conceptual Evolution

Below is the structured record of the user inquiries, the mechanics uncovered, and how each discovery informs the final system architecture.

┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   THE DISCOVERY ROADMAP                                          │
├────────────────────────────────┬─────────────────────────────────────────────────────────────────┤
│ User Inquiries & Milestones    │ Key Architectural Breakthrough                                  │
├────────────────────────────────┼─────────────────────────────────────────────────────────────────┤
│ 1. ChatGPT Modes & Reasoning   │ Modes set test-time token budgets; weights remain frozen.       │
│ 2. Pro Mode Mechanics          │ Pro adds multi-branch breadth, tree search, and PRM validation. │
│ 3. Ternary AI Compatibility    │ Search algorithms are precision-agnostic; 1.58-bit cuts VRAM.   │
│ 4. Branch Scaling & Sweet Spot │ 8 to 16 branches maximize yield before hitting verifier noise.  │
│ 5. Verifier Hacking & RLVR     │ Neural verifiers are vulnerable; code sandboxes provide truth.  │
│ 6. Biological Memory & EWC     │ Offline sleep replay + Fisher penalties prevent forgetting.     │
│ 7. Unified 27B Local Pipeline  │ 27B ternary (~6GB) leaves 18GB VRAM for search, sandbox, & LoRA.│
└────────────────────────────────┴─────────────────────────────────────────────────────────────────┘

### 1. ChatGPT Modes & Reasoning Effort
- **Inquiry**: How do ChatGPT modes (Instant, Thinking, Pro, Ultra) and effort levels (Low, High, xhigh) work?
- **Core Insight**: Effort levels do not change model checkpoints. A system-level conditioning token sets an internal chain-of-thought (CoT) token budget. During Reinforcement Learning (RL) training, the model learned to associate these tokens with varying depths of self-verification and backtracking.

### 2. Pro Mode vs. Single-Pass Reasoning
- **Inquiry**: Compare Pro mode and explain how reasoning efforts work under the hood.
- **Core Insight**: Single-pass effort levels only scale reasoning depth (sequential tokens). Pro Mode scales breadth by running parallel candidate sampling (Best-of-N), Monte Carlo Tree Search (MCTS), and Process Reward Model (PRM) step-scoring to explore multiple solution trajectories simultaneously.

### 3. Precision Independence & Ternary AI
- **Inquiry**: Can 1.58-bit ternary AI run Pro-style parallel search?
- **Core Insight**: Yes. Search algorithms operate at the token sequence level, completely independent of whether matrix layers use FP16, FP8, or 1.58-bit ternary weights ($\{-1, 0, +1\}$). Ternary architectures are ideal for search because replacing matrix multiplications with integer additions slashes memory and energy overhead, allowing 16+ parallel rollouts on consumer hardware.

### 4. Search Scaling & The Branch Sweet Spot
- **Inquiry**: How many branches is the sweet spot?
- **Core Insight**: Independent branches obey power-law diminishing returns. $N = 8 \text{ to } 16$ branches captures 85–95% of peak accuracy. Beyond 16 branches, marginal gains drop below 1% per unit of compute, and the risk of exploiting verifier blind spots spikes exponentially.

### 5. Verifier Hacking & Non-Conscious Optimization
- **Inquiry**: What is verifier hacking, what is reward, and does the AI know it is doing it?
- **Core Insight**: A reward is a scalar training signal. Verifier hacking occurs when a model exploits statistical loopholes in neural reward models (e.g., adding verbose self-correction jargon or tautological reasoning) to inflate scores without solving the problem. The AI does not consciously cheat; gradient descent simply follows the path of least mathematical resistance.

### 6. Ground-Truth RLVR & Parameter Updates
- **Inquiry**: Does RLVR modify parameters, and how does the model know if it succeeded?
- **Core Insight**: Reinforcement Learning with Verifiable Rewards (RLVR) replaces fuzzy neural reward models with deterministic compilers, Python sandboxes, and Lean 4 provers. During training, policy gradient algorithms (such as GRPO) backpropagate binary pass/fail rewards directly into the model's weights ($\theta_{\text{new}} = \theta_{\text{old}} + \Delta\theta$). At inference, parameters are frozen.

### 7. Continuous Parametric Learning vs. Catastrophic Forgetting
- **Inquiry**: Can we update weights on the fly during local chat instead of storing context?
- **Core Insight**: Updating raw weights on every single conversational turn causes catastrophic forgetting and representational drift due to polysemantic superposition.

### 8. The Biological Dual-Memory Solution (CLS & EWC)
- **Inquiry**: Can an AI store memories deeply in its matrix like a human brain?
- **Core Insight**: Human brains use Complementary Learning Systems (CLS): rapid episodic encoding in the hippocampus during the day, followed by slow synaptic consolidation into the neocortex during sleep. In AI, this is replicated via Fast/Slow LoRA adapters, generative replay buffers, and Elastic Weight Consolidation (EWC) with Fisher Information constraints.

---

## Part 2: Master Technical Specification

### 1. Hardware Budget (Single 24GB Consumer GPU)
By utilizing 1.58-bit ternary quantization, a 27B parameter base model consumes under 6 GB of VRAM, leaving the remaining memory for parallel KV caches, isolated execution sandboxes, and local backward passes.

| Subsystem Component | Precision / Format | VRAM Allocated | Hardware Function |
|---|---|---|---|
| Base Checkpoint (27B) | 1.58-bit Ternary (BitLinear) | ~5.8 GB | Frozen foundation world model |
| Parallel KV Cache (N=16) | PagedAttention / FP8 Cache | ~5.2 GB | Holds 16 simultaneous search branches |
| Slow-LoRA + Fast-LoRA | BF16 / Rank 32 | ~3.4 GB | Trainable synaptic memory adapters |
| Sandboxes & Provers | Host RAM / Docker Subprocess | ~0.0 GB (VRAM) | Python sandbox & SymPy execution |
| Free VRAM Headroom | Dynamic Allocation | ~9.6 GB | Peak gradient spikes during sleep cycle |

### 2. Multi-Tiered System Architecture

```
                                  ┌───────────────────────────┐
                                  │    User Prompt / Query    │
                                  └─────────────┬─────────────┘
                                                │
                                                ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│ TIER 1: FAST INTERACTIVE INFERENCE & PRO SEARCH (Daytime)                                 │
│                                                                                           │
│   1. Entropy Router:                                                                      │
│      • H(Y) < 0.25  ──► Instant Single Pass (N=1)                                         │
│      • H(Y) ≥ 0.25  ──► Escalate to Pro Search (N=8 to 16)                                │
│                                                                                           │
│   2. Pro Parallel Search Engine:                                                          │
│      • Sample N=16 candidate reasoning branches via BitLinear integer-add engine          │
│      • Step-Level PRM: Prune trajectories where cumulative score drops below τ = 0.40     │
│                                                                                           │
│   3. Deterministic Ground-Truth Verifier (Anti-Hacking):                                  │
│      • Code / Algorithms ──► Isolated Docker / Subprocess Sandbox execution               │
│      • Math / Logic      ──► SymPy symbolic algebra & Lean 4 AST checks                   │
│      • Open Domain       ──► Semantic clustering & majority consensus voting              │
│                                                                                           │
│   4. Episodic Memory Capture:                                                             │
│      • Stream verified winning response to user                                           │
│      • Log prompt, passing trace, assertions, and surprise score into local SQLite DB     │
└───────────────────────────────────────────────┬───────────────────────────────────────────┘
                                                │
                                                │ (System Inactive > 30 minutes)
                                                ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│ TIER 2: BIOLOGICAL SLEEP CONSOLIDATION DAEMON (Nighttime)                                 │
│                                                                                           │
│   1. Surprise-Driven Prioritization:                                                      │
│      • Rank episodic logs by initial failure / backtrack severity                         │
│                                                                                           │
│   2. Generative Replay Engine:                                                            │
│      • Interleave 25% User Episodic Traces + 75% General Knowledge Anchor Dataset         │
│                                                                                           │
│   3. Elastic Weight Consolidation (EWC-LoRA):                                             │
│      • Compute Fisher Information Matrix: F_i = E[(∂ log P / ∂ θ_i)²]                     │
│      • Apply Quadratic Synaptic Penalty: L_EWC = (λ / 2) * Σ F_i * (θ_i - θ_anchor)²      │
│                                                                                           │
│   4. Parametric Absorption:                                                               │
│      • Update Slow-LoRA adapter weights via GRPO/SFT backward passes                      │
│      • Embed facts, preferences, and verified patterns directly into matrix parameters    │
└───────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Part 3: MODULE SPECIFICATION ADDENDUM: Speculative Acceleration & Zero-VRAM Drafting

LLM inference is memory-bandwidth bound: validating multiple draft tokens in a single forward pass takes almost the same time as generating a single token. Speculative acceleration provides lossless $1.5\times \text{ to } 5.0\times$ speedups without modifying target model output distribution.

### Comparative Speculative Acceleration Methods

| Method | How It Works | Extra VRAM Cost | Speedup | Best Use Case |
|---|---|---|---|---|
| **DFlash 2** | Block Diffusion Drafter: Generates an entire block of candidate tokens in a single forward pass via parallel denoising. | ~400 MB – 800 MB | 3.0× – 5.0× | Highest peak throughput when a trained DFlash block drafter is available. |
| **EAGLE-2 / EAGLE-3** | Feature-Level Tree Speculation: Predicts the target model's next hidden states and builds dynamic candidate draft trees. | ~500 MB – 1.2 GB | 2.5× – 4.0× | General reasoning and structured math proofs. |
| **Prompt Lookup Decoding (PLD)** | Zero-Compute N-Gram Speculation: Scans the prompt and KV cache for matching token patterns to propose drafts without any secondary model. | 0 MB (Zero VRAM) | 1.5× – 2.8× | Perfect for 16GB M1/M2/M3: Extreme speedup on code generation, JSON, and refactoring. |
| **Lookahead / Jacobi Decoding** | Parallel Jacobi Iterations: Generates n-grams in parallel Jacobi rollout steps from the base model itself without a separate draft network. | 0 MB (Zero VRAM) | 1.4× – 2.0× | Zero-overhead local deployment where VRAM cannot fit an extra draft model. |
| **Medusa-2** | Multi-Head Drafting: Lightweight linear decoding heads attached to the frozen backbone predicting $t+2, t+3, \dots$ simultaneously. | ~200 MB – 400 MB | 1.8× – 2.5× | Fixed-memory setups where head weights are kept minimal. |

### Supported Models & Apple Silicon 2-Bit Checkpoints
- **BitNet 1.58-Bit**: `microsoft/BitNet-b1.58-27B`
- **Apple Silicon Native 2-Bit MLX**: `prism-ml/Ternary-Bonsai-27B-mlx-2bit` / `mlx-community/BitNet-b1.58-27B-4bit`
- **Qwen Coder & Math Checkpoints**: `Qwen/Qwen2.5-Coder-1.5B`, `Qwen/Qwen2.5-0.5B-Instruct`

### Speculative Rejection Sampling Verification
For proposed draft tokens $[d_1, d_2, \dots, d_K]$:
1. Target model verifies all candidate tokens in a single batched forward pass.
2. Accepts $k \le K$ tokens according to rejection sampling threshold $\alpha = \min\left(1, \frac{P_{target}(d_i)}{P_{draft}(d_i)}\right)$.
3. Samples next true token directly from modified residual distribution $P_{residual} = \max(0, P_{target} - P_{draft})$.

