# Final sequential diff closure — commits 135–137

**Repository:** `Eabusham2/smart-ai-studio`  
**Branch:** `fix/real-benchmarks-final-32k`  
**Master baseline:** `06390e5357a07f80a8089ac28fc16c75461a48a6`  
**Covered head before this appendix:** `8cb929167cb729ce71bbcc60dcdfc456e2b7bccd`

This closes the documentation gap after the existing audit set:

- commits **1–120**: original full audit + exact line-annotation parts;
- commits **121–134**: `CURRENT_HEAD_DIFF_APPENDIX_2026-09-24.md`;
- commits **135–137**: exact diffs and per-line explanations below.

Commits 135–137 are documentation-only. They do not change model loading, chat/Pro, Learn, RSI, MLX/GGUF/BitNet training, media/multimodal runtime, or eval execution.

The commit that adds **this** file is intentionally excluded from itself. That is the unavoidable non-recursive boundary; no historical commit is removed or rewritten.


## 135. `683952f46518` — Document current-head commits 121 through 134

**Disposition:** retained exactly as history; documentation-only.

### Exact commit diff

```diff
@@ -0,0 +1,477 @@
+# Current-head sequential diff appendix — commits 121–134
+
+**Repository:** `Eabusham2/smart-ai-studio`  
+**Branch:** `fix/real-benchmarks-final-32k`  
+**Baseline already documented:** commits 1–120 in the existing 2026-09-19 audit and four line-annotation parts.  
+**Current audited head before this documentation commit:** `e2163c6baa59148072ecc9dd8900997511f54251`  
+**Master baseline:** `06390e5357a07f80a8089ac28fc16c75461a48a6`
+
+This appendix does not replace or rewrite the earlier audit. It extends it. Every added/removed source line in commits 121–134 is reproduced below from GitHub's canonical commit diff, in chronological order. This records both temporary changes and later corrections, so the history is auditable rather than silently rewritten.
+
+
+## 121. `371881796cd9` — fix: bound MLX Phase 3B conversational training
+
+**Disposition:** retained as history. Where a later commit changes the same policy, the later current-head state is authoritative; no commit was deleted or squashed.
+
+```diff
+@@ -99,21 +99,118 @@ def phase3_then_conversation_teach(self):
+             backend.is_mlx_available = True
+         backend.adapter_path = p4.RSI_ADAPTER_PATH
+ 
+-        consolidator = AwakeOnlineConsolidator(
+-            mlx_engine=backend,
+-            memory_db=None,
+-            max_context=8192,
+-        )
+         train_tokens = _training_tokens(self.engine.tokenizer)
+         teach_started = time.perf_counter()
+-        consolidator._run_shadow_consolidation(_chat_history())
++
++        if str(getattr(self.engine, "backend_key", "mlx") or "mlx").lower() == "mlx":
++            # Reuse the same bounded-memory LoRA gradient path that Phase 3 already
++            # uses successfully. Never fall back to full-model value_and_grad here:
++            # Qwen3.5 inference CustomKernel has no VJP and can explode unified RAM.
++            trainable = dict(
++                p4.mlx.utils.tree_flatten(
++                    self.engine.model.trainable_parameters()
++                )
++            )
++            before = {}
++            for key, value in trainable.items():
++                try:
++                    before[key] = p4.mx.copy(value)
++                except Exception:
++                    before[key] = value + p4.mx.zeros_like(value)
++            if before:
++                p4.mx.eval(*before.values())
++
++            opt = p4.optim.AdamW(learning_rate=1e-4)
++            total_targets = 0
++            prepared = []
++            for user, assistant, _ in CONVERSATION_TEACH_EXAMPLES:
++                prefix = (
++                    f"<|im_start|>user\n{user}<|im_end|>\n"
++                    f"<|im_start|>assistant\n"
++                )
++                text = prefix + assistant + "<|im_end|>"
++                ids = self.engine.tokenizer.encode(text)
++                prefix_ids = self.engine.tokenizer.encode(prefix)
++                if len(ids) <= 1:
++                    continue
++                completion_loss_start = max(0, len(prefix_ids) - 1)
++                selected_start = max(0, len(ids) - 16_384)
++                row_targets = max(
++                    0,
++                    min(len(ids), 16_384) - 1
++                    - max(0, completion_loss_start - selected_start),
++                )
++                if row_targets <= 0:
++                    continue
++                prepared.append((ids, completion_loss_start, row_targets))
++                total_targets += row_targets
++
++            if not prepared or total_targets <= 0:
++                raise RuntimeError("Phase 3B conversational teach found no trainable completion targets")
++
++            completed_targets = 0
++            for _step in range(3):
++                for item_index, (ids, completion_loss_start, row_targets) in enumerate(prepared, 1):
++                    with p4.METAL_STREAM_LOCK:
++                        _loss, grads, trained_targets = p4._phase3_bounded_gradients(
++                            self,
++                            ids,
++                            completion_loss_start,
++                            item_index,
++                            len(prepared),
++                            completed_targets,
++                            total_targets * 3,
++                            teach_started,
++                        )
++                        opt.update(self.engine.model, grads)
++                        p4.mx.eval(self.engine.model.parameters(), opt.state)
++                        completed_targets += int(trained_targets)
++                        _loss = None
++                        grads = None
++                        try:
++                            p4.mx.clear_cache()
++                        except Exception:
++                            pass
++
++            after = dict(
++                p4.mlx.utils.tree_flatten(
++                    self.engine.model.trainable_parameters()
++                )
++            )
++            delta_sq = p4.mx.array(0.0)
++            matched = 0
++            for key, old_value in before.items():
++                new_value = after.get(key)
++                if new_value is None:
++                    continue
++                diff = new_value - old_value
++                delta_sq = delta_sq + p4.mx.sum(
++                    diff.astype(p4.mx.float32) * diff.astype(p4.mx.float32)
++                )
++                matched += 1
++            if matched <= 0:
++                raise RuntimeError("Phase 3B could not match post-update LoRA trainables")
++            p4.mx.eval(delta_sq)
++            delta = float(p4.mx.sqrt(delta_sq).item())
++            persisted = bool(p4._save_rsi_adapter(self))
++        else:
++            # Preserve the existing non-MLX production path unchanged.
++            consolidator = AwakeOnlineConsolidator(
++                mlx_engine=backend,
++                memory_db=None,
++                max_context=8192,
++            )
++            consolidator._run_shadow_consolidation(_chat_history())
++            delta = float(consolidator.total_param_shift or 0.0)
++            persisted = bool(os.path.exists(p4.RSI_ADAPTER_PATH))
++            if consolidator.consolidation_count != 1:
++                raise RuntimeError("Phase 3B conversational teach produced no real parameter update")
++
+         teach_seconds = max(0.001, time.perf_counter() - teach_started)
+         teach_tps = train_tokens / teach_seconds
+ 
+         p4._assert_same_model(self, model_identity, "after conversational teach")
+-        delta = float(consolidator.total_param_shift or 0.0)
+-        persisted = bool(os.path.exists(p4.RSI_ADAPTER_PATH))
+-        if consolidator.consolidation_count != 1 or delta <= 0.0:
++        if delta <= 0.0:
+             raise RuntimeError("Phase 3B conversational teach produced no real parameter update")
+         if not persisted:
+             raise RuntimeError("Phase 3B conversational teach did not persist the updated adapter")
+```
+
+## 122. `c9d79f265268` — Correct BitNet learning runtime documentation
+
+**Disposition:** retained as history. Where a later commit changes the same policy, the later current-head state is authoritative; no commit was deleted or squashed.
+
+```diff
+@@ -1,9 +1,10 @@
+-"""Real Microsoft bitnet.cpp inference backend.
++"""Real Microsoft bitnet.cpp inference and persistent-learning backend.
+ 
+ This replaces the old synthetic BitNet placeholder in ProReasoningEngine routing.
+-It talks to bitnet.cpp's llama-server and never fabricates output. Parameter training
+-is intentionally fail-closed until bitnet.cpp exposes a compatible persistent adapter
+-training path.
++It talks to bitnet.cpp's llama-server and never fabricates output. Learning is
++fail-closed unless metadata provides a compatible BF16 training lineage; when it
++does, the backend performs a real PEFT update, rebuilds I2_S deployment weights,
++hot-reloads them, and rolls back atomically on failure or cancellation.
+ """
+ from __future__ import annotations
+ 
+```
+
+## 123. `a8d6033e01c4` — Lock real benchmark and bounded Phase 3B invariants
+
+**Disposition:** retained as history. Where a later commit changes the same policy, the later current-head state is authoritative; no commit was deleted or squashed.
+
+```diff
+@@ -179,3 +179,36 @@ def test_end_to_end_call_chain_is_wired_through_existing_components():
+     assert "▶ RSI: RECURSIVE SELF-IMPROVEMENT ON PHASE-1 MISSES" in phase
+     assert "▶ PHASE 3: LEARN + RSI PARAMETRIC CONSOLIDATION" in phase
+     assert "Phase 4: Post-Consolidation" in phase
++
++def test_real_benchmark_repair_cannot_fall_back_to_legacy_synthetic_rows():
++    suite = _src("master_4000_eval_suite.py")
++    real = _src("eval/real_benchmark_runtime.py")
++    assert "runtime_module._repair_suite = repair_real_suite" in real
++    install = suite.index(
++        "real_benchmark_runtime.install(BenchmarkDatasetProvider, master_runtime, phase4_pro_rsi, Master4000EvaluationEngine)"
++    )
++    capture = suite.index("_real_repair_suite = master_runtime._repair_suite")
++    assert install < capture
++    assert 'if not item.get("real_source"):' in real
++    assert "refusing to fall back to synthetic data" in real
++
++
++def test_mlx_phase3b_conversation_teach_uses_bounded_completion_only_lora_path():
++    teach = _src("eval/conversation_teach_hardening.py")
++    mlx_branch = teach.index(
++        'if str(getattr(self.engine, "backend_key", "mlx") or "mlx").lower() == "mlx":',
++        teach.index("teach_started = time.perf_counter()"),
++    )
++    non_mlx = teach.index("else:", mlx_branch)
++    mlx_body = teach[mlx_branch:non_mlx]
++    assert "p4._phase3_bounded_gradients(" in mlx_body
++    assert "with p4.METAL_STREAM_LOCK:" in mlx_body
++    assert "p4.optim.AdamW(learning_rate=1e-4)" in mlx_body
++    assert "completion_loss_start" in mlx_body
++    assert "p4._save_rsi_adapter(self)" in mlx_body
++    assert "_run_shadow_consolidation(" not in mlx_body
++
++    non_mlx_body = teach[non_mlx:]
++    assert "AwakeOnlineConsolidator(" in non_mlx_body
++    assert "consolidator._run_shadow_consolidation(_chat_history())" in non_mlx_body
++
+```
+
+## 124. `15db4ed565b4` — Expand focused branch audit coverage
+
+**Disposition:** retained as history. Where a later commit changes the same policy, the later current-head state is authoritative; no commit was deleted or squashed.
+
+```diff
+@@ -5,8 +5,15 @@ on:
+     branches: [fix/real-benchmarks-final-32k]
+     paths:
+       - .github/workflows/feature-audit.yml
+-      - tools/feature_audit.py
+-      - tests/audit/**
++      - app_gui.py
++      - master_4000_eval_suite.py
++      - run_studio_complete.py
++      - core/**
++      - eval/**
++      - consolidation/**
++      - tests/unit/test_*contract.py
++      - requirements.txt
++      - pyproject.toml
+   workflow_dispatch:
+ 
+ permissions:
+@@ -89,6 +96,11 @@ jobs:
+             tests/unit/test_app_eval_integration_contract.py \
+             tests/unit/test_diverged_branch_reconciliation_contract.py \
+             tests/unit/test_full_branch_audit_contract.py \
++            tests/unit/test_custom_backend_metadata_contract.py \
++            tests/unit/test_prism_gguf_and_bitnet_contract.py \
++            tests/unit/test_non_mlx_learning_contract.py \
++            tests/unit/test_memory_limit_control_contract.py \
++            tests/unit/test_temperature_policy_contract.py \
+             --junitxml="$AUDIT_OUTPUT/contracts.xml" 2>&1 | tee "$AUDIT_OUTPUT/contracts.log"
+       - name: Upload audit evidence even on failure
+         if: always()
+```
+
+## 125. `7c711c83f809` — Restore standalone 4K eval Qwen3.8 target
+
+**Disposition:** retained as history. Where a later commit changes the same policy, the later current-head state is authoritative; no commit was deleted or squashed.
+
+```diff
+@@ -34,7 +34,7 @@ def compute_auto_kv_budget(total_ram_gb: float)->int:
+ @dataclass
+ class EngineSettings:
+     total_ram_gb: float=field(default_factory=lambda: psutil.virtual_memory().total/(1024**3))
+-    mlx_model_path: str="prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"
++    mlx_model_path: str="penkia/TernaryQuench-Qwen3.8-27B-MLX"
+     max_kv_tokens: int=field(init=False)
+     h2o_sink_tokens:int=4; h2o_heavy_tokens:int=64; h2o_max_budget:int=128
+     lora_rank:int=4; lora_alpha:float=8.0; lora_chunk_size:int=6; total_layers:int=60
+```
+
+## 126. `261fe3c56757` — Align standalone eval report with Qwen3.8 target
+
+**Disposition:** retained as history. Where a later commit changes the same policy, the later current-head state is authoritative; no commit was deleted or squashed.
+
+```diff
+@@ -501,7 +501,7 @@ def _generate_master_report(self, base_scores: Dict[str, float], post_scores: Di
+             getattr(
+                 self,
+                 "_eval_target_model_label",
+-                "Bonsai 2 27B Ternary Multimodal",
++                "penkia/TernaryQuench-Qwen3.8-27B-MLX",
+             )
+         )
+         md.append(f"**Target Model:** `{target_label}`  ")
+```
+
+## 127. `931595f8c5ba` — Lock standalone eval Qwen target separately from app Eval
+
+**Disposition:** retained as history. Where a later commit changes the same policy, the later current-head state is authoritative; no commit was deleted or squashed.
+
+```diff
+@@ -212,3 +212,15 @@ def test_mlx_phase3b_conversation_teach_uses_bounded_completion_only_lora_path()
+     assert "AwakeOnlineConsolidator(" in non_mlx_body
+     assert "consolidator._run_shadow_consolidation(_chat_history())" in non_mlx_body
+ 
++def test_standalone_eval_keeps_qwen38_while_app_eval_uses_selected_model():
++    runtime = _src("run_studio_complete.py")
++    report = _src("eval/_master_4000_base.py")
++    bridge = _src("eval/app_cross_platform_bridge.py")
++    qwen = "penkia/TernaryQuench-Qwen3.8-27B-MLX"
++    assert f'mlx_model_path: str="{qwen}"' in runtime
++    assert f'"{qwen}"' in report
++    # Desktop Eval remains model-aware and loads the app-selected model metadata.
++    assert 'info = dict(config.get("model_info") or {})' in bridge
++    assert "self.pro.load_model(" in bridge
++    assert "model_info=info" in bridge
++
+```
+
+## 128. `94e0f8c5ef40` — Restore standalone 4K eval to Bonsai 2
+
+**Disposition:** retained as history. Where a later commit changes the same policy, the later current-head state is authoritative; no commit was deleted or squashed.
+
+```diff
+@@ -34,7 +34,7 @@ def compute_auto_kv_budget(total_ram_gb: float)->int:
+ @dataclass
+ class EngineSettings:
+     total_ram_gb: float=field(default_factory=lambda: psutil.virtual_memory().total/(1024**3))
+-    mlx_model_path: str="penkia/TernaryQuench-Qwen3.8-27B-MLX"
++    mlx_model_path: str="prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"
+     max_kv_tokens: int=field(init=False)
+     h2o_sink_tokens:int=4; h2o_heavy_tokens:int=64; h2o_max_budget:int=128
+     lora_rank:int=4; lora_alpha:float=8.0; lora_chunk_size:int=6; total_layers:int=60
+```
+
+## 129. `e074242477a3` — Align standalone eval report with Bonsai 2
+
+**Disposition:** retained as history. Where a later commit changes the same policy, the later current-head state is authoritative; no commit was deleted or squashed.
+
+```diff
+@@ -501,7 +501,7 @@ def _generate_master_report(self, base_scores: Dict[str, float], post_scores: Di
+             getattr(
+                 self,
+                 "_eval_target_model_label",
+-                "penkia/TernaryQuench-Qwen3.8-27B-MLX",
++                "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit",
+             )
+         )
+         md.append(f"**Target Model:** `{target_label}`  ")
+```
+
+## 130. `6142402da1d3` — Lock standalone eval Bonsai 2 target
+
+**Disposition:** retained as history. Where a later commit changes the same policy, the later current-head state is authoritative; no commit was deleted or squashed.
+
+```diff
+@@ -212,13 +212,13 @@ def test_mlx_phase3b_conversation_teach_uses_bounded_completion_only_lora_path()
+     assert "AwakeOnlineConsolidator(" in non_mlx_body
+     assert "consolidator._run_shadow_consolidation(_chat_history())" in non_mlx_body
+ 
+-def test_standalone_eval_keeps_qwen38_while_app_eval_uses_selected_model():
++def test_standalone_eval_keeps_bonsai2_while_app_eval_uses_selected_model():
+     runtime = _src("run_studio_complete.py")
+     report = _src("eval/_master_4000_base.py")
+     bridge = _src("eval/app_cross_platform_bridge.py")
+-    qwen = "penkia/TernaryQuench-Qwen3.8-27B-MLX"
+-    assert f'mlx_model_path: str="{qwen}"' in runtime
+-    assert f'"{qwen}"' in report
++    bonsai = "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"
++    assert f'mlx_model_path: str="{bonsai}"' in runtime
++    assert f'"{bonsai}"' in report
+     # Desktop Eval remains model-aware and loads the app-selected model metadata.
+     assert 'info = dict(config.get("model_info") or {})' in bridge
+     assert "self.pro.load_model(" in bridge
+```
+
+## 131. `f7e3f4ef472d` — Keep metadata contract independent of full runtime imports
+
+**Disposition:** retained as history. Where a later commit changes the same policy, the later current-head state is authoritative; no commit was deleted or squashed.
+
+```diff
+@@ -1,10 +1,18 @@
+ from pathlib import Path
+-
+-from core.model_policy import derive_runtime_metadata
++import importlib.util
+ 
+ 
+ ROOT = Path(__file__).resolve().parents[2]
+ 
++_spec = importlib.util.spec_from_file_location(
++    "smartai_model_policy_contract",
++    ROOT / "core" / "model_policy.py",
++)
++_model_policy = importlib.util.module_from_spec(_spec)
++assert _spec is not None and _spec.loader is not None
++_spec.loader.exec_module(_model_policy)
++derive_runtime_metadata = _model_policy.derive_runtime_metadata
++
+ 
+ def _src(path: str) -> str:
+     return (ROOT / path).read_text(encoding="utf-8")
+```
+
+## 132. `72950772121d` — Load standalone Bonsai eval through Prism runtime
+
+**Disposition:** retained as history. Where a later commit changes the same policy, the later current-head state is authoritative; no commit was deleted or squashed.
+
+```diff
+@@ -197,11 +197,38 @@ def cleanup_worktree(self,d): subprocess.run(['git','worktree','remove','--force
+ 
+ class UnifiedMasterEngine:
+     def __init__(self,settings:Optional[EngineSettings]=None):
+-        self.settings=settings or EngineSettings(); self.kg=RelationalKnowledgeGraph(self.settings.db_path); self.sandbox=POSIXHardenedSandbox(self.settings.sandbox_timeout_seconds,self.settings.sandbox_max_memory_mb); self.mcp=FastMCPDispatcher(self.sandbox,self.kg); self.lif=NeuromorphicLIFController(); self.drafter=ASTPrefixTrieDrafter(); self.h2o=H2OKVCacheArena(self.settings.h2o_sink_tokens,self.settings.h2o_heavy_tokens,self.settings.h2o_max_budget); self.ogp_projector=GramSchmidtOGPProjector(self.settings.ogp_ortho_tolerance); self.mcts=SymbolicMCTSSearchEngine(self.sandbox); self.model=self.tokenizer=self.moe_manager=self.moe_router=self.grpo_trainer=self.ogp_daemon=None; self._initialize_runtime()
++        self.settings=settings or EngineSettings(); self.kg=RelationalKnowledgeGraph(self.settings.db_path); self.sandbox=POSIXHardenedSandbox(self.settings.sandbox_timeout_seconds,self.settings.sandbox_max_memory_mb); self.mcp=FastMCPDispatcher(self.sandbox,self.kg); self.lif=NeuromorphicLIFController(); self.drafter=ASTPrefixTrieDrafter(); self.h2o=H2OKVCacheArena(self.settings.h2o_sink_tokens,self.settings.h2o_heavy_tokens,self.settings.h2o_max_budget); self.ogp_projector=GramSchmidtOGPProjector(self.settings.ogp_ortho_tolerance); self.mcts=SymbolicMCTSSearchEngine(self.sandbox); self.model=self.tokenizer=self.moe_manager=self.moe_router=self.grpo_trainer=self.ogp_daemon=None; self._runtime_backend=None; self._initialize_runtime()
++    def _load_primary_model(self):
++        path=str(self.settings.mlx_model_path or "")
++        if "Ternary-Bonsai-2-27B-mlx-2bit" in path:
++            # Prism Bonsai-2 requires the runtime bundled in the HF pack. Plain
++            # mlx_lm.load() skips its Hadamard activation transform and can emit
++            # incorrect text without raising, so reuse the app's proven loader.
++            from huggingface_hub import snapshot_download
++            from core.mlx_engine import MLXReasoningBackend
++            source=path if os.path.exists(path) else snapshot_download(repo_id=path)
++            backend=MLXReasoningBackend(
++                model_path=source,
++                model_info={
++                    "name":"Bonsai 2 27B Ternary Multimodal",
++                    "repo_id":path,
++                    "runtime_family":"bonsai2_hadamard",
++                    "input_modalities":["text","image","video"],
++                },
++            )
++            if not backend.load_model():
++                raise RuntimeError("Bonsai-2 bundled MLX runtime failed to load")
++            model=backend.get_training_model()
++            tokenizer=backend.tokenizer
++            if model is None or tokenizer is None:
++                raise RuntimeError("Bonsai-2 bundled runtime exposed no trainable language model/tokenizer")
++            self._runtime_backend=backend
++            return model,tokenizer
++        return load(path)
+     def _initialize_runtime(self):
+         if not MLX_AVAILABLE:return
+         try:
+-            self.model,self.tokenizer=load(self.settings.mlx_model_path)
++            self.model,self.tokenizer=self._load_primary_model()
+             self.moe_manager=MoEDualBufferManager(self.model,self.settings); self.moe_router=HierarchicalMoERouter(self.model); self.grpo_trainer=GRPOTrainingEngine(self.model,self.tokenizer,self.sandbox)
+             if self.settings.enable_awake_ogp_daemon:
+                 self.ogp_daemon=ProjectedSleepConsolidationDaemon(self.moe_manager,self.ogp_projector,self.kg,self.tokenizer,self.settings,METAL_STREAM_LOCK); self.ogp_daemon.start()
+```
+
+## 133. `031dc53829e1` — Align eval target contract with exact Bonsai repo
+
+**Disposition:** retained as history. Where a later commit changes the same policy, the later current-head state is authoritative; no commit was deleted or squashed.
+
+```diff
+@@ -82,7 +82,7 @@ def test_bonsai2_is_the_default_eval_model_and_current_model_is_reported():
+     report = _src("eval/_master_4000_base.py")
+     assert 'mlx_model_path: str="prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"' in runtime
+     assert "_eval_target_model_label" in report
+-    assert "Bonsai 2 27B Ternary Multimodal" in report
++    assert "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit" in report
+ 
+ 
+ def test_top_app_controls_explicitly_include_context_limit_and_memory_watcher():
+```
+
+## 134. `e2163c6baa59` — Fix BitNet runtime audit assertion
+
+**Disposition:** retained as history. Where a later commit changes the same policy, the later current-head state is authoritative; no commit was deleted or squashed.
+
+```diff
+@@ -58,7 +58,8 @@ def test_bitnet_backend_is_real_and_training_rebuilds_deployment_weights():
+     trainer = _src("core/bitnet_rebuild_trainer.py")
+     assert '"https://github.com/microsoft/BitNet.git"' in src
+     assert '"setup_env.py"' in src
+-    assert '"llama-server"' in src
++    assert '"build/bin/llama-server"' in src
++    assert '"build/bin/Release/llama-server.exe"' in src
+     assert "def training_ready(self) -> bool:" in src
+     assert "BitNetRebuildTrainer" in src
+     assert "def train_mini_batch(" in src
+```
+
+## Current interpretation after commit 134
+
+- Bonsai 2 remains the requested main/default app model (`model_1`) and the requested standalone 4K-eval target.
+- Commits 125–127 temporarily switched standalone eval back to Qwen3.8; commits 128–130 intentionally superseded that and restored Bonsai 2. Both histories are preserved.
+- Commit 132 does not change model choice. It makes the existing Bonsai-2 choice use Prism's bundled Hadamard-aware loader before exposing the loaded language module to the unchanged eval/training pipeline.
+- Commits 131, 133 and 134 only repair focused audit contracts; they do not alter application model choices or runtime behavior.
+- The successful focused audit at commit 134 compiled 228/228 tracked Python files and passed 58/58 focused contracts.
```

### Every changed line explained (477 lines)

1. `+# Current-head sequential diff appendix — commits 121–134` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
2. `+` — Adds a documentation spacing line; no runtime effect.
3. `+**Repository:** \`Eabusham2/smart-ai-studio\`  ` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
4. `+**Branch:** \`fix/real-benchmarks-final-32k\`  ` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
5. `+**Baseline already documented:** commits 1–120 in the existing 2026-09-19 audit and four line-annotation parts.  ` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
6. `+**Current audited head before this documentation commit:** \`e2163c6baa59148072ecc9dd8900997511f54251\`  ` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
7. `+**Master baseline:** \`06390e5357a07f80a8089ac28fc16c75461a48a6\`` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
8. `+` — Adds a documentation spacing line; no runtime effect.
9. `+This appendix does not replace or rewrite the earlier audit. It extends it. Every added/removed source line in commits 121–134 is reproduced below from GitHub's canonical commit diff, in chronological order. This records both temporary changes and later corrections, so the history is auditable rather than silently rewritten.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
10. `+` — Adds a documentation spacing line; no runtime effect.
11. `+` — Adds a documentation spacing line; no runtime effect.
12. `+## 121. \`371881796cd9\` — fix: bound MLX Phase 3B conversational training` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
13. `+` — Adds a documentation spacing line; no runtime effect.
14. `+**Disposition:** retained as history. Where a later commit changes the same policy, the later current-head state is authoritative; no commit was deleted or squashed.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
15. `+` — Adds a documentation spacing line; no runtime effect.
16. `+\`\`\`diff` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
17. `+@@ -99,21 +99,118 @@ def phase3_then_conversation_teach(self):` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
18. `+             backend.is_mlx_available = True` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
19. `+         backend.adapter_path = p4.RSI_ADAPTER_PATH` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
20. `+ ` — Adds a documentation spacing line; no runtime effect.
21. `+-        consolidator = AwakeOnlineConsolidator(` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
22. `+-            mlx_engine=backend,` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
23. `+-            memory_db=None,` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
24. `+-            max_context=8192,` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
25. `+-        )` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
26. `+         train_tokens = _training_tokens(self.engine.tokenizer)` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
27. `+         teach_started = time.perf_counter()` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
28. `+-        consolidator._run_shadow_consolidation(_chat_history())` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
29. `++` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
30. `++        if str(getattr(self.engine, "backend_key", "mlx") or "mlx").lower() == "mlx":` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
31. `++            # Reuse the same bounded-memory LoRA gradient path that Phase 3 already` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
32. `++            # uses successfully. Never fall back to full-model value_and_grad here:` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
33. `++            # Qwen3.5 inference CustomKernel has no VJP and can explode unified RAM.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
34. `++            trainable = dict(` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
35. `++                p4.mlx.utils.tree_flatten(` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
36. `++                    self.engine.model.trainable_parameters()` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
37. `++                )` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
38. `++            )` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
39. `++            before = {}` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
40. `++            for key, value in trainable.items():` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
41. `++                try:` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
42. `++                    before[key] = p4.mx.copy(value)` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
43. `++                except Exception:` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
44. `++                    before[key] = value + p4.mx.zeros_like(value)` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
45. `++            if before:` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
46. `++                p4.mx.eval(*before.values())` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
47. `++` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
48. `++            opt = p4.optim.AdamW(learning_rate=1e-4)` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
49. `++            total_targets = 0` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
50. `++            prepared = []` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
51. `++            for user, assistant, _ in CONVERSATION_TEACH_EXAMPLES:` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
52. `++                prefix = (` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
53. `++                    f"<|im_start|>user\n{user}<|im_end|>\n"` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
54. `++                    f"<|im_start|>assistant\n"` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
55. `++                )` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
56. `++                text = prefix + assistant + "<|im_end|>"` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
57. `++                ids = self.engine.tokenizer.encode(text)` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
58. `++                prefix_ids = self.engine.tokenizer.encode(prefix)` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
59. `++                if len(ids) <= 1:` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
60. `++                    continue` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
61. `++                completion_loss_start = max(0, len(prefix_ids) - 1)` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
62. `++                selected_start = max(0, len(ids) - 16_384)` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
63. `++                row_targets = max(` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
64. `++                    0,` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
65. `++                    min(len(ids), 16_384) - 1` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
66. `++                    - max(0, completion_loss_start - selected_start),` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
67. `++                )` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
68. `++                if row_targets <= 0:` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
69. `++                    continue` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
70. `++                prepared.append((ids, completion_loss_start, row_targets))` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
71. `++                total_targets += row_targets` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
72. `++` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
73. `++            if not prepared or total_targets <= 0:` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
74. `++                raise RuntimeError("Phase 3B conversational teach found no trainable completion targets")` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
75. `++` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
76. `++            completed_targets = 0` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
77. `++            for _step in range(3):` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
78. `++                for item_index, (ids, completion_loss_start, row_targets) in enumerate(prepared, 1):` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
79. `++                    with p4.METAL_STREAM_LOCK:` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
80. `++                        _loss, grads, trained_targets = p4._phase3_bounded_gradients(` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
81. `++                            self,` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
82. `++                            ids,` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
83. `++                            completion_loss_start,` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
84. `++                            item_index,` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
85. `++                            len(prepared),` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
86. `++                            completed_targets,` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
87. `++                            total_targets * 3,` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
88. `++                            teach_started,` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
89. `++                        )` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
90. `++                        opt.update(self.engine.model, grads)` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
91. `++                        p4.mx.eval(self.engine.model.parameters(), opt.state)` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
92. `++                        completed_targets += int(trained_targets)` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
93. `++                        _loss = None` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
94. `++                        grads = None` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
95. `++                        try:` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
96. `++                            p4.mx.clear_cache()` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
97. `++                        except Exception:` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
98. `++                            pass` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
99. `++` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
100. `++            after = dict(` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
101. `++                p4.mlx.utils.tree_flatten(` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
102. `++                    self.engine.model.trainable_parameters()` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
103. `++                )` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
104. `++            )` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
105. `++            delta_sq = p4.mx.array(0.0)` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
106. `++            matched = 0` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
107. `++            for key, old_value in before.items():` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
108. `++                new_value = after.get(key)` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
109. `++                if new_value is None:` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
110. `++                    continue` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
111. `++                diff = new_value - old_value` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
112. `++                delta_sq = delta_sq + p4.mx.sum(` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
113. `++                    diff.astype(p4.mx.float32) * diff.astype(p4.mx.float32)` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
114. `++                )` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
115. `++                matched += 1` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
116. `++            if matched <= 0:` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
117. `++                raise RuntimeError("Phase 3B could not match post-update LoRA trainables")` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
118. `++            p4.mx.eval(delta_sq)` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
119. `++            delta = float(p4.mx.sqrt(delta_sq).item())` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
120. `++            persisted = bool(p4._save_rsi_adapter(self))` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
121. `++        else:` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
122. `++            # Preserve the existing non-MLX production path unchanged.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
123. `++            consolidator = AwakeOnlineConsolidator(` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
124. `++                mlx_engine=backend,` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
125. `++                memory_db=None,` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
126. `++                max_context=8192,` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
127. `++            )` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
128. `++            consolidator._run_shadow_consolidation(_chat_history())` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
129. `++            delta = float(consolidator.total_param_shift or 0.0)` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
130. `++            persisted = bool(os.path.exists(p4.RSI_ADAPTER_PATH))` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
131. `++            if consolidator.consolidation_count != 1:` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
132. `++                raise RuntimeError("Phase 3B conversational teach produced no real parameter update")` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
133. `++` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
134. `+         teach_seconds = max(0.001, time.perf_counter() - teach_started)` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
135. `+         teach_tps = train_tokens / teach_seconds` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
136. `+ ` — Adds a documentation spacing line; no runtime effect.
137. `+         p4._assert_same_model(self, model_identity, "after conversational teach")` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
138. `+-        delta = float(consolidator.total_param_shift or 0.0)` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
139. `+-        persisted = bool(os.path.exists(p4.RSI_ADAPTER_PATH))` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
140. `+-        if consolidator.consolidation_count != 1 or delta <= 0.0:` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
141. `++        if delta <= 0.0:` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
142. `+             raise RuntimeError("Phase 3B conversational teach produced no real parameter update")` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
143. `+         if not persisted:` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
144. `+             raise RuntimeError("Phase 3B conversational teach did not persist the updated adapter")` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
145. `+\`\`\`` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
146. `+` — Adds a documentation spacing line; no runtime effect.
147. `+## 122. \`c9d79f265268\` — Correct BitNet learning runtime documentation` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
148. `+` — Adds a documentation spacing line; no runtime effect.
149. `+**Disposition:** retained as history. Where a later commit changes the same policy, the later current-head state is authoritative; no commit was deleted or squashed.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
150. `+` — Adds a documentation spacing line; no runtime effect.
151. `+\`\`\`diff` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
152. `+@@ -1,9 +1,10 @@` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
153. `+-"""Real Microsoft bitnet.cpp inference backend.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
154. `++"""Real Microsoft bitnet.cpp inference and persistent-learning backend.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
155. `+ ` — Adds a documentation spacing line; no runtime effect.
156. `+ This replaces the old synthetic BitNet placeholder in ProReasoningEngine routing.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
157. `+-It talks to bitnet.cpp's llama-server and never fabricates output. Parameter training` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
158. `+-is intentionally fail-closed until bitnet.cpp exposes a compatible persistent adapter` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
159. `+-training path.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
160. `++It talks to bitnet.cpp's llama-server and never fabricates output. Learning is` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
161. `++fail-closed unless metadata provides a compatible BF16 training lineage; when it` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
162. `++does, the backend performs a real PEFT update, rebuilds I2_S deployment weights,` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
163. `++hot-reloads them, and rolls back atomically on failure or cancellation.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
164. `+ """` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
165. `+ from __future__ import annotations` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
166. `+ ` — Adds a documentation spacing line; no runtime effect.
167. `+\`\`\`` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
168. `+` — Adds a documentation spacing line; no runtime effect.
169. `+## 123. \`a8d6033e01c4\` — Lock real benchmark and bounded Phase 3B invariants` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
170. `+` — Adds a documentation spacing line; no runtime effect.
171. `+**Disposition:** retained as history. Where a later commit changes the same policy, the later current-head state is authoritative; no commit was deleted or squashed.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
172. `+` — Adds a documentation spacing line; no runtime effect.
173. `+\`\`\`diff` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
174. `+@@ -179,3 +179,36 @@ def test_end_to_end_call_chain_is_wired_through_existing_components():` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
175. `+     assert "▶ RSI: RECURSIVE SELF-IMPROVEMENT ON PHASE-1 MISSES" in phase` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
176. `+     assert "▶ PHASE 3: LEARN + RSI PARAMETRIC CONSOLIDATION" in phase` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
177. `+     assert "Phase 4: Post-Consolidation" in phase` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
178. `++` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
179. `++def test_real_benchmark_repair_cannot_fall_back_to_legacy_synthetic_rows():` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
180. `++    suite = _src("master_4000_eval_suite.py")` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
181. `++    real = _src("eval/real_benchmark_runtime.py")` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
182. `++    assert "runtime_module._repair_suite = repair_real_suite" in real` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
183. `++    install = suite.index(` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
184. `++        "real_benchmark_runtime.install(BenchmarkDatasetProvider, master_runtime, phase4_pro_rsi, Master4000EvaluationEngine)"` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
185. `++    )` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
186. `++    capture = suite.index("_real_repair_suite = master_runtime._repair_suite")` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
187. `++    assert install < capture` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
188. `++    assert 'if not item.get("real_source"):' in real` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
189. `++    assert "refusing to fall back to synthetic data" in real` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
190. `++` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
191. `++` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
192. `++def test_mlx_phase3b_conversation_teach_uses_bounded_completion_only_lora_path():` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
193. `++    teach = _src("eval/conversation_teach_hardening.py")` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
194. `++    mlx_branch = teach.index(` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
195. `++        'if str(getattr(self.engine, "backend_key", "mlx") or "mlx").lower() == "mlx":',` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
196. `++        teach.index("teach_started = time.perf_counter()"),` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
197. `++    )` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
198. `++    non_mlx = teach.index("else:", mlx_branch)` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
199. `++    mlx_body = teach[mlx_branch:non_mlx]` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
200. `++    assert "p4._phase3_bounded_gradients(" in mlx_body` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
201. `++    assert "with p4.METAL_STREAM_LOCK:" in mlx_body` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
202. `++    assert "p4.optim.AdamW(learning_rate=1e-4)" in mlx_body` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
203. `++    assert "completion_loss_start" in mlx_body` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
204. `++    assert "p4._save_rsi_adapter(self)" in mlx_body` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
205. `++    assert "_run_shadow_consolidation(" not in mlx_body` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
206. `++` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
207. `++    non_mlx_body = teach[non_mlx:]` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
208. `++    assert "AwakeOnlineConsolidator(" in non_mlx_body` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
209. `++    assert "consolidator._run_shadow_consolidation(_chat_history())" in non_mlx_body` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
210. `++` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
211. `+\`\`\`` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
212. `+` — Adds a documentation spacing line; no runtime effect.
213. `+## 124. \`15db4ed565b4\` — Expand focused branch audit coverage` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
214. `+` — Adds a documentation spacing line; no runtime effect.
215. `+**Disposition:** retained as history. Where a later commit changes the same policy, the later current-head state is authoritative; no commit was deleted or squashed.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
216. `+` — Adds a documentation spacing line; no runtime effect.
217. `+\`\`\`diff` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
218. `+@@ -5,8 +5,15 @@ on:` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
219. `+     branches: [fix/real-benchmarks-final-32k]` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
220. `+     paths:` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
221. `+       - .github/workflows/feature-audit.yml` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
222. `+-      - tools/feature_audit.py` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
223. `+-      - tests/audit/**` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
224. `++      - app_gui.py` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
225. `++      - master_4000_eval_suite.py` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
226. `++      - run_studio_complete.py` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
227. `++      - core/**` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
228. `++      - eval/**` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
229. `++      - consolidation/**` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
230. `++      - tests/unit/test_*contract.py` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
231. `++      - requirements.txt` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
232. `++      - pyproject.toml` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
233. `+   workflow_dispatch:` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
234. `+ ` — Adds a documentation spacing line; no runtime effect.
235. `+ permissions:` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
236. `+@@ -89,6 +96,11 @@ jobs:` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
237. `+             tests/unit/test_app_eval_integration_contract.py \` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
238. `+             tests/unit/test_diverged_branch_reconciliation_contract.py \` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
239. `+             tests/unit/test_full_branch_audit_contract.py \` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
240. `++            tests/unit/test_custom_backend_metadata_contract.py \` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
241. `++            tests/unit/test_prism_gguf_and_bitnet_contract.py \` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
242. `++            tests/unit/test_non_mlx_learning_contract.py \` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
243. `++            tests/unit/test_memory_limit_control_contract.py \` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
244. `++            tests/unit/test_temperature_policy_contract.py \` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
245. `+             --junitxml="$AUDIT_OUTPUT/contracts.xml" 2>&1 | tee "$AUDIT_OUTPUT/contracts.log"` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
246. `+       - name: Upload audit evidence even on failure` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
247. `+         if: always()` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
248. `+\`\`\`` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
249. `+` — Adds a documentation spacing line; no runtime effect.
250. `+## 125. \`7c711c83f809\` — Restore standalone 4K eval Qwen3.8 target` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
251. `+` — Adds a documentation spacing line; no runtime effect.
252. `+**Disposition:** retained as history. Where a later commit changes the same policy, the later current-head state is authoritative; no commit was deleted or squashed.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
253. `+` — Adds a documentation spacing line; no runtime effect.
254. `+\`\`\`diff` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
255. `+@@ -34,7 +34,7 @@ def compute_auto_kv_budget(total_ram_gb: float)->int:` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
256. `+ @dataclass` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
257. `+ class EngineSettings:` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
258. `+     total_ram_gb: float=field(default_factory=lambda: psutil.virtual_memory().total/(1024**3))` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
259. `+-    mlx_model_path: str="prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
260. `++    mlx_model_path: str="penkia/TernaryQuench-Qwen3.8-27B-MLX"` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
261. `+     max_kv_tokens: int=field(init=False)` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
262. `+     h2o_sink_tokens:int=4; h2o_heavy_tokens:int=64; h2o_max_budget:int=128` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
263. `+     lora_rank:int=4; lora_alpha:float=8.0; lora_chunk_size:int=6; total_layers:int=60` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
264. `+\`\`\`` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
265. `+` — Adds a documentation spacing line; no runtime effect.
266. `+## 126. \`261fe3c56757\` — Align standalone eval report with Qwen3.8 target` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
267. `+` — Adds a documentation spacing line; no runtime effect.
268. `+**Disposition:** retained as history. Where a later commit changes the same policy, the later current-head state is authoritative; no commit was deleted or squashed.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
269. `+` — Adds a documentation spacing line; no runtime effect.
270. `+\`\`\`diff` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
271. `+@@ -501,7 +501,7 @@ def _generate_master_report(self, base_scores: Dict[str, float], post_scores: Di` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
272. `+             getattr(` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
273. `+                 self,` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
274. `+                 "_eval_target_model_label",` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
275. `+-                "Bonsai 2 27B Ternary Multimodal",` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
276. `++                "penkia/TernaryQuench-Qwen3.8-27B-MLX",` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
277. `+             )` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
278. `+         )` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
279. `+         md.append(f"**Target Model:** \`{target_label}\`  ")` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
280. `+\`\`\`` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
281. `+` — Adds a documentation spacing line; no runtime effect.
282. `+## 127. \`931595f8c5ba\` — Lock standalone eval Qwen target separately from app Eval` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
283. `+` — Adds a documentation spacing line; no runtime effect.
284. `+**Disposition:** retained as history. Where a later commit changes the same policy, the later current-head state is authoritative; no commit was deleted or squashed.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
285. `+` — Adds a documentation spacing line; no runtime effect.
286. `+\`\`\`diff` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
287. `+@@ -212,3 +212,15 @@ def test_mlx_phase3b_conversation_teach_uses_bounded_completion_only_lora_path()` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
288. `+     assert "AwakeOnlineConsolidator(" in non_mlx_body` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
289. `+     assert "consolidator._run_shadow_consolidation(_chat_history())" in non_mlx_body` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
290. `+ ` — Adds a documentation spacing line; no runtime effect.
291. `++def test_standalone_eval_keeps_qwen38_while_app_eval_uses_selected_model():` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
292. `++    runtime = _src("run_studio_complete.py")` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
293. `++    report = _src("eval/_master_4000_base.py")` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
294. `++    bridge = _src("eval/app_cross_platform_bridge.py")` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
295. `++    qwen = "penkia/TernaryQuench-Qwen3.8-27B-MLX"` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
296. `++    assert f'mlx_model_path: str="{qwen}"' in runtime` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
297. `++    assert f'"{qwen}"' in report` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
298. `++    # Desktop Eval remains model-aware and loads the app-selected model metadata.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
299. `++    assert 'info = dict(config.get("model_info") or {})' in bridge` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
300. `++    assert "self.pro.load_model(" in bridge` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
301. `++    assert "model_info=info" in bridge` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
302. `++` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
303. `+\`\`\`` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
304. `+` — Adds a documentation spacing line; no runtime effect.
305. `+## 128. \`94e0f8c5ef40\` — Restore standalone 4K eval to Bonsai 2` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
306. `+` — Adds a documentation spacing line; no runtime effect.
307. `+**Disposition:** retained as history. Where a later commit changes the same policy, the later current-head state is authoritative; no commit was deleted or squashed.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
308. `+` — Adds a documentation spacing line; no runtime effect.
309. `+\`\`\`diff` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
310. `+@@ -34,7 +34,7 @@ def compute_auto_kv_budget(total_ram_gb: float)->int:` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
311. `+ @dataclass` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
312. `+ class EngineSettings:` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
313. `+     total_ram_gb: float=field(default_factory=lambda: psutil.virtual_memory().total/(1024**3))` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
314. `+-    mlx_model_path: str="penkia/TernaryQuench-Qwen3.8-27B-MLX"` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
315. `++    mlx_model_path: str="prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
316. `+     max_kv_tokens: int=field(init=False)` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
317. `+     h2o_sink_tokens:int=4; h2o_heavy_tokens:int=64; h2o_max_budget:int=128` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
318. `+     lora_rank:int=4; lora_alpha:float=8.0; lora_chunk_size:int=6; total_layers:int=60` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
319. `+\`\`\`` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
320. `+` — Adds a documentation spacing line; no runtime effect.
321. `+## 129. \`e074242477a3\` — Align standalone eval report with Bonsai 2` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
322. `+` — Adds a documentation spacing line; no runtime effect.
323. `+**Disposition:** retained as history. Where a later commit changes the same policy, the later current-head state is authoritative; no commit was deleted or squashed.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
324. `+` — Adds a documentation spacing line; no runtime effect.
325. `+\`\`\`diff` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
326. `+@@ -501,7 +501,7 @@ def _generate_master_report(self, base_scores: Dict[str, float], post_scores: Di` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
327. `+             getattr(` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
328. `+                 self,` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
329. `+                 "_eval_target_model_label",` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
330. `+-                "penkia/TernaryQuench-Qwen3.8-27B-MLX",` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
331. `++                "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit",` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
332. `+             )` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
333. `+         )` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
334. `+         md.append(f"**Target Model:** \`{target_label}\`  ")` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
335. `+\`\`\`` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
336. `+` — Adds a documentation spacing line; no runtime effect.
337. `+## 130. \`6142402da1d3\` — Lock standalone eval Bonsai 2 target` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
338. `+` — Adds a documentation spacing line; no runtime effect.
339. `+**Disposition:** retained as history. Where a later commit changes the same policy, the later current-head state is authoritative; no commit was deleted or squashed.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
340. `+` — Adds a documentation spacing line; no runtime effect.
341. `+\`\`\`diff` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
342. `+@@ -212,13 +212,13 @@ def test_mlx_phase3b_conversation_teach_uses_bounded_completion_only_lora_path()` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
343. `+     assert "AwakeOnlineConsolidator(" in non_mlx_body` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
344. `+     assert "consolidator._run_shadow_consolidation(_chat_history())" in non_mlx_body` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
345. `+ ` — Adds a documentation spacing line; no runtime effect.
346. `+-def test_standalone_eval_keeps_qwen38_while_app_eval_uses_selected_model():` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
347. `++def test_standalone_eval_keeps_bonsai2_while_app_eval_uses_selected_model():` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
348. `+     runtime = _src("run_studio_complete.py")` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
349. `+     report = _src("eval/_master_4000_base.py")` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
350. `+     bridge = _src("eval/app_cross_platform_bridge.py")` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
351. `+-    qwen = "penkia/TernaryQuench-Qwen3.8-27B-MLX"` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
352. `+-    assert f'mlx_model_path: str="{qwen}"' in runtime` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
353. `+-    assert f'"{qwen}"' in report` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
354. `++    bonsai = "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
355. `++    assert f'mlx_model_path: str="{bonsai}"' in runtime` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
356. `++    assert f'"{bonsai}"' in report` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
357. `+     # Desktop Eval remains model-aware and loads the app-selected model metadata.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
358. `+     assert 'info = dict(config.get("model_info") or {})' in bridge` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
359. `+     assert "self.pro.load_model(" in bridge` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
360. `+\`\`\`` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
361. `+` — Adds a documentation spacing line; no runtime effect.
362. `+## 131. \`f7e3f4ef472d\` — Keep metadata contract independent of full runtime imports` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
363. `+` — Adds a documentation spacing line; no runtime effect.
364. `+**Disposition:** retained as history. Where a later commit changes the same policy, the later current-head state is authoritative; no commit was deleted or squashed.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
365. `+` — Adds a documentation spacing line; no runtime effect.
366. `+\`\`\`diff` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
367. `+@@ -1,10 +1,18 @@` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
368. `+ from pathlib import Path` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
369. `+-` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
370. `+-from core.model_policy import derive_runtime_metadata` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
371. `++import importlib.util` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
372. `+ ` — Adds a documentation spacing line; no runtime effect.
373. `+ ` — Adds a documentation spacing line; no runtime effect.
374. `+ ROOT = Path(__file__).resolve().parents[2]` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
375. `+ ` — Adds a documentation spacing line; no runtime effect.
376. `++_spec = importlib.util.spec_from_file_location(` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
377. `++    "smartai_model_policy_contract",` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
378. `++    ROOT / "core" / "model_policy.py",` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
379. `++)` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
380. `++_model_policy = importlib.util.module_from_spec(_spec)` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
381. `++assert _spec is not None and _spec.loader is not None` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
382. `++_spec.loader.exec_module(_model_policy)` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
383. `++derive_runtime_metadata = _model_policy.derive_runtime_metadata` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
384. `++` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
385. `+ ` — Adds a documentation spacing line; no runtime effect.
386. `+ def _src(path: str) -> str:` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
387. `+     return (ROOT / path).read_text(encoding="utf-8")` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
388. `+\`\`\`` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
389. `+` — Adds a documentation spacing line; no runtime effect.
390. `+## 132. \`72950772121d\` — Load standalone Bonsai eval through Prism runtime` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
391. `+` — Adds a documentation spacing line; no runtime effect.
392. `+**Disposition:** retained as history. Where a later commit changes the same policy, the later current-head state is authoritative; no commit was deleted or squashed.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
393. `+` — Adds a documentation spacing line; no runtime effect.
394. `+\`\`\`diff` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
395. `+@@ -197,11 +197,38 @@ def cleanup_worktree(self,d): subprocess.run(['git','worktree','remove','--force` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
396. `+ ` — Adds a documentation spacing line; no runtime effect.
397. `+ class UnifiedMasterEngine:` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
398. `+     def __init__(self,settings:Optional[EngineSettings]=None):` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
399. `+-        self.settings=settings or EngineSettings(); self.kg=RelationalKnowledgeGraph(self.settings.db_path); self.sandbox=POSIXHardenedSandbox(self.settings.sandbox_timeout_seconds,self.settings.sandbox_max_memory_mb); self.mcp=FastMCPDispatcher(self.sandbox,self.kg); self.lif=NeuromorphicLIFController(); self.drafter=ASTPrefixTrieDrafter(); self.h2o=H2OKVCacheArena(self.settings.h2o_sink_tokens,self.settings.h2o_heavy_tokens,self.settings.h2o_max_budget); self.ogp_projector=GramSchmidtOGPProjector(self.settings.ogp_ortho_tolerance); self.mcts=SymbolicMCTSSearchEngine(self.sandbox); self.model=self.tokenizer=self.moe_manager=self.moe_router=self.grpo_trainer=self.ogp_daemon=None; self._initialize_runtime()` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
400. `++        self.settings=settings or EngineSettings(); self.kg=RelationalKnowledgeGraph(self.settings.db_path); self.sandbox=POSIXHardenedSandbox(self.settings.sandbox_timeout_seconds,self.settings.sandbox_max_memory_mb); self.mcp=FastMCPDispatcher(self.sandbox,self.kg); self.lif=NeuromorphicLIFController(); self.drafter=ASTPrefixTrieDrafter(); self.h2o=H2OKVCacheArena(self.settings.h2o_sink_tokens,self.settings.h2o_heavy_tokens,self.settings.h2o_max_budget); self.ogp_projector=GramSchmidtOGPProjector(self.settings.ogp_ortho_tolerance); self.mcts=SymbolicMCTSSearchEngine(self.sandbox); self.model=self.tokenizer=self.moe_manager=self.moe_router=self.grpo_trainer=self.ogp_daemon=None; self._runtime_backend=None; self._initialize_runtime()` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
401. `++    def _load_primary_model(self):` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
402. `++        path=str(self.settings.mlx_model_path or "")` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
403. `++        if "Ternary-Bonsai-2-27B-mlx-2bit" in path:` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
404. `++            # Prism Bonsai-2 requires the runtime bundled in the HF pack. Plain` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
405. `++            # mlx_lm.load() skips its Hadamard activation transform and can emit` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
406. `++            # incorrect text without raising, so reuse the app's proven loader.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
407. `++            from huggingface_hub import snapshot_download` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
408. `++            from core.mlx_engine import MLXReasoningBackend` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
409. `++            source=path if os.path.exists(path) else snapshot_download(repo_id=path)` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
410. `++            backend=MLXReasoningBackend(` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
411. `++                model_path=source,` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
412. `++                model_info={` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
413. `++                    "name":"Bonsai 2 27B Ternary Multimodal",` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
414. `++                    "repo_id":path,` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
415. `++                    "runtime_family":"bonsai2_hadamard",` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
416. `++                    "input_modalities":["text","image","video"],` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
417. `++                },` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
418. `++            )` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
419. `++            if not backend.load_model():` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
420. `++                raise RuntimeError("Bonsai-2 bundled MLX runtime failed to load")` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
421. `++            model=backend.get_training_model()` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
422. `++            tokenizer=backend.tokenizer` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
423. `++            if model is None or tokenizer is None:` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
424. `++                raise RuntimeError("Bonsai-2 bundled runtime exposed no trainable language model/tokenizer")` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
425. `++            self._runtime_backend=backend` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
426. `++            return model,tokenizer` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
427. `++        return load(path)` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
428. `+     def _initialize_runtime(self):` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
429. `+         if not MLX_AVAILABLE:return` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
430. `+         try:` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
431. `+-            self.model,self.tokenizer=load(self.settings.mlx_model_path)` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
432. `++            self.model,self.tokenizer=self._load_primary_model()` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
433. `+             self.moe_manager=MoEDualBufferManager(self.model,self.settings); self.moe_router=HierarchicalMoERouter(self.model); self.grpo_trainer=GRPOTrainingEngine(self.model,self.tokenizer,self.sandbox)` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
434. `+             if self.settings.enable_awake_ogp_daemon:` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
435. `+                 self.ogp_daemon=ProjectedSleepConsolidationDaemon(self.moe_manager,self.ogp_projector,self.kg,self.tokenizer,self.settings,METAL_STREAM_LOCK); self.ogp_daemon.start()` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
436. `+\`\`\`` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
437. `+` — Adds a documentation spacing line; no runtime effect.
438. `+## 133. \`031dc53829e1\` — Align eval target contract with exact Bonsai repo` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
439. `+` — Adds a documentation spacing line; no runtime effect.
440. `+**Disposition:** retained as history. Where a later commit changes the same policy, the later current-head state is authoritative; no commit was deleted or squashed.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
441. `+` — Adds a documentation spacing line; no runtime effect.
442. `+\`\`\`diff` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
443. `+@@ -82,7 +82,7 @@ def test_bonsai2_is_the_default_eval_model_and_current_model_is_reported():` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
444. `+     report = _src("eval/_master_4000_base.py")` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
445. `+     assert 'mlx_model_path: str="prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"' in runtime` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
446. `+     assert "_eval_target_model_label" in report` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
447. `+-    assert "Bonsai 2 27B Ternary Multimodal" in report` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
448. `++    assert "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit" in report` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
449. `+ ` — Adds a documentation spacing line; no runtime effect.
450. `+ ` — Adds a documentation spacing line; no runtime effect.
451. `+ def test_top_app_controls_explicitly_include_context_limit_and_memory_watcher():` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
452. `+\`\`\`` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
453. `+` — Adds a documentation spacing line; no runtime effect.
454. `+## 134. \`e2163c6baa59\` — Fix BitNet runtime audit assertion` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
455. `+` — Adds a documentation spacing line; no runtime effect.
456. `+**Disposition:** retained as history. Where a later commit changes the same policy, the later current-head state is authoritative; no commit was deleted or squashed.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
457. `+` — Adds a documentation spacing line; no runtime effect.
458. `+\`\`\`diff` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
459. `+@@ -58,7 +58,8 @@ def test_bitnet_backend_is_real_and_training_rebuilds_deployment_weights():` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
460. `+     trainer = _src("core/bitnet_rebuild_trainer.py")` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
461. `+     assert '"https://github.com/microsoft/BitNet.git"' in src` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
462. `+     assert '"setup_env.py"' in src` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
463. `+-    assert '"llama-server"' in src` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
464. `++    assert '"build/bin/llama-server"' in src` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
465. `++    assert '"build/bin/Release/llama-server.exe"' in src` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
466. `+     assert "def training_ready(self) -> bool:" in src` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
467. `+     assert "BitNetRebuildTrainer" in src` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
468. `+     assert "def train_mini_batch(" in src` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
469. `+\`\`\`` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
470. `+` — Adds a documentation spacing line; no runtime effect.
471. `+## Current interpretation after commit 134` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
472. `+` — Adds a documentation spacing line; no runtime effect.
473. `+- Bonsai 2 remains the requested main/default app model (\`model_1\`) and the requested standalone 4K-eval target.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
474. `+- Commits 125–127 temporarily switched standalone eval back to Qwen3.8; commits 128–130 intentionally superseded that and restored Bonsai 2. Both histories are preserved.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
475. `+- Commit 132 does not change model choice. It makes the existing Bonsai-2 choice use Prism's bundled Hadamard-aware loader before exposing the loaded language module to the unchanged eval/training pipeline.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
476. `+- Commits 131, 133 and 134 only repair focused audit contracts; they do not alter application model choices or runtime behavior.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
477. `+- The successful focused audit at commit 134 compiled 228/228 tracked Python files and passed 58/58 focused contracts.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.


## 136. `161f621b0918` — Document current-head branch reconciliation

**Disposition:** retained exactly as history; documentation-only.

### Exact commit diff

```diff
@@ -0,0 +1,173 @@
+# Smart AI Studio — Current-head branch audit and reconciliation (2026-09-24)
+
+**Repository:** `Eabusham2/smart-ai-studio`  
+**Authoritative branch:** `fix/real-benchmarks-final-32k`  
+**Historical master baseline:** `06390e5357a07f80a8089ac28fc16c75461a48a6`  
+**Frozen implementation verification head:** `e2163c6baa59148072ecc9dd8900997511f54251`  
+**Documentation continuation head at start of this report:** `683952f4651880272d9d1ae2e7c2beb44397a6fb`
+
+## 1. Audit policy
+
+The feature branch is authoritative. Master is only the comparison baseline. No reset, squash, force-rewrite, commit deletion, or wholesale old-branch restore is used here.
+
+The earlier audit documents remain part of the record:
+- `BRANCH_AUDIT_fix_real_benchmarks_final_32k_2026-09-19.md`
+- `FULL_MASTER_TO_FEATURE_DIFF_ANNOTATED_2026-09-19.md`
+- four `FULL_MASTER_TO_FEATURE_DIFF_LINE_ANNOTATIONS_PART_*_2026-09-19.md` files
+- `GEMINI_POSTMORTEM_2026-09-19.md`
+
+Those cover commits 1–120 and the full line-level diff at that snapshot.  
+`CURRENT_HEAD_DIFF_APPENDIX_2026-09-24.md` preserves the exact sequential GitHub diffs for commits 121–134, including temporary changes that were later corrected. Together they provide the requested non-destructive history rather than pretending intermediate mistakes never happened.
+
+## 2. Current history integrity
+
+At the frozen implementation head, the branch was **134 commits ahead / 0 behind master**. The audited post-master history is linear: the reviewed feature commits are single-parent commits, not unexplained merge commits. Earlier known guard-file removals are coordination cleanup only and do not remove implementation.
+
+The latest model choice is intentional:
+- app `model_1`: **Bonsai 2 27B Ternary Multimodal**
+- app `model_2`: **Bonsai 2 27B CRACK**
+- standalone 4K eval: **Bonsai 2 27B Ternary MLX**
+- commits 125–127 temporarily restored Qwen3.8; commits 128–130 intentionally superseded that and restored Bonsai 2. Those commits remain in history.
+
+## 3. End-to-end code trace
+
+### Model selection and Hugging Face import
+
+The app starts on `model_1`; existing model choices remain present.
+
+Custom HF text-model admission reads Hub card/config/file metadata, requires explicit ternary/1.58-bit/trit proof, infers modalities and runtime family, chooses exact native GGUF/BitNet artifacts when uniquely provable, captures mmproj/projector metadata, and records a declared trainable parent lineage when present. Runtime loading consumes that persisted metadata instead of guessing from a display name.
+
+### Model load → message → Pro
+
+`ProReasoningEngine.load_model(..., model_info=...)` routes by resolved runtime metadata:
+- Bonsai/other Apple MLX → `MLXReasoningBackend`;
+- Prism GGUF → `PrismGGUFReasoningBackend`;
+- ordinary GGUF → `GGUFReasoningBackend`;
+- verified BitNet → `BitNetCppReasoningBackend`;
+- other controller families → `UniversalControllerBackend`.
+
+Normal chat N=1 remains T=0.65. Pro N>1 remains 0.20→0.95, gamma 1.35, plus the dedicated T=0.65 branch. The single Context budget continues to mean packed history/prompt plus generated output.
+
+### Multimodal Bonsai MLX
+
+Bonsai-2 MLX is not treated as a plain stock MLX checkpoint. The runtime family resolves to the bundled repository VLM loader. The bundled loader installs the Hadamard-aware packed layers into the model's real `language_model`.
+
+Learn/RSI obtains that inner language module through `get_training_model()`, reuses the existing real MLX LoRA/Fisher/AdamW update code, and raw-adapter persistence restores tensors into the same language module on reload.
+
+### Consolidation / Learn / RSI
+
+Awake consolidation and explicit Learn select the active real trainable backend instead of using an MLX-only gate.
+
+- **MLX:** LoRA-only trainables, real gradients/AdamW, EWC/Fisher, nonzero drift, transactional adapter save/rollback.
+- **GGUF/Prism:** frozen quantized base + real PEFT/QLoRA sidecar on a declared compatible parent, nonzero drift, official LoRA→GGUF conversion, hot reload, rollback.
+- **BitNet:** official bitnet.cpp deployment runtime; when a verified BF16 training lineage exists, BF16 parent → PEFT update → merge → I2_S rebuild → hot reload, with rollback. If that lineage/toolchain cannot be proven, training fails closed rather than reporting fake learning.
+- **Transformers/controller:** language-projection LoRA, completion-only labels, real backward/AdamW, nonzero drift, transactional persistence.
+
+RSI persistent self-memory remains question/reward/PASS-free: successful self-generated traces are stored separately and only hidden verification gates eligibility.
+
+### Media / multimodal RSI
+
+Media generation and media training are separate capabilities.
+
+`MediaLearningService` resolves a compatible trainer by runtime/model metadata. A media update only reports success after a real nonzero trainable delta (or externally persisted nonzero LoRA factor) is proven and persisted; cancellation/failure rolls back.
+
+`media_rsi` is genuine only when the active text controller can directly perceive the target modality. It:
+1. generates candidate media,
+2. has the active multimodal controller inspect/score the actual artifact,
+3. refuses blind RSI if no real numeric self-grade exists,
+4. selects the best valid candidate,
+5. invokes the real media-learning backend,
+6. reports `weights_updated` only from that backend.
+
+If the loaded controller cannot ingest that modality, media generation and explicit media Learn remain available but media RSI correctly returns unsupported.
+
+### Standalone 4K eval
+
+The standalone 4K target remains Bonsai-2 by user decision. A current-head bug was found during this audit: simply changing `EngineSettings.mlx_model_path` to Bonsai-2 left the standalone engine using plain `mlx_lm.load()`, while this Prism checkpoint requires its bundled Hadamard-aware runtime.
+
+Commit `72950772...` fixes that **without rewriting the eval pipeline**:
+- it reuses the app's `MLXReasoningBackend`;
+- loads Bonsai-2 through its bundled runtime;
+- exposes the correctly loaded real language module/tokenizer to the existing benchmark/training stages.
+
+The canonical stage flow remains the existing suite: real Phase 1 → Learn/RSI → bounded Phase 3/3B → Phase 4 retest.
+
+## 4. Concrete current-head fixes made during this continuation
+
+1. **Focused audit import isolation** — the metadata contract imported `core.model_policy` through package `core/__init__.py`, accidentally requiring NumPy even though the contract only tests pure metadata logic. It now loads that source module directly.
+2. **Standalone Bonsai runtime correctness** — switched the already-selected Bonsai-2 standalone eval from plain MLX loading to the proper bundled Prism runtime, while preserving the existing eval stages.
+3. **Eval target contract wording** — the test now checks the exact Bonsai repo ID used by the report rather than a stale display label.
+4. **BitNet audit assertion** — the test now verifies the real platform paths `build/bin/llama-server` and `build/bin/Release/llama-server.exe` instead of searching for a nonexistent exact string literal.
+
+No model choices were removed, and the Bonsai-2 main/default decision was not reverted.
+
+## 5. Verification evidence
+
+The focused audit run for implementation head `e2163c6b...` completed successfully:
+- **228 / 228 tracked Python files syntax-compiled**
+- **58 / 58 focused source contracts passed**
+- exact source archive captured
+- exact master source archive captured
+- full binary-safe `master-to-feature.diff` captured
+- complete reverse chronological-to-sequential commit patch ledger captured
+- numstat ledger captured
+
+A static unfinished-code scan of current `core/`, `eval/`, `consolidation/`, `app_gui.py`, `run_studio_complete.py`, and `master_4000_eval_suite.py` found **no TODO, FIXME, or NotImplementedError markers**. Remaining `pass` statements are exception/fallback/lifecycle handling, not unimplemented method placeholders in the scanned paths.
+
+This remains a source/contract audit, not a claim that every heavyweight native backend was physically executed on every hardware target. Full CI, real Apple 27B training, Prism native GGUF libraries, bitnet.cpp rebuilds, CUDA training, all external media trainers, Docker/Pier DeepSWE, and release publishing require their real target environments and are not fabricated here.
+
+## 6. Current master→feature file ledger at the frozen implementation head
+
+| Path | Status | Additions | Deletions | Total changed lines |
+|---|---|---:|---:|---:|
+| `.github/workflows/feature-audit.yml` | added | +112 | -0 | 112 |
+| `ACTIVE_SESSION_GUARD_2026-09-18_GGUF_ONLY.md` | removed | +0 | -19 | 19 |
+| `BRANCH_GUARD_GGUF_LEARNING_20260918.txt` | removed | +0 | -14 | 14 |
+| `app_gui.py` | modified | +16 | -0 | 16 |
+| `build_app.py` | modified | +3 | -3 | 6 |
+| `consolidation/projected_daemon.py` | modified | +98 | -19 | 117 |
+| `core/_mlx_engine_base.py` | modified | +202 | -68 | 270 |
+| `core/autonomous_learner.py` | modified | +70 | -33 | 103 |
+| `core/awake_auto_hook.py` | modified | +88 | -28 | 116 |
+| `core/bitnet_rebuild_trainer.py` | modified | +42 | -5 | 47 |
+| `core/controller_runtime.py` | modified | +99 | -9 | 108 |
+| `core/drafter.py` | modified | +125 | -0 | 125 |
+| `core/engines/bitnet_cpp_engine.py` | modified | +28 | -12 | 40 |
+| `core/engines/gguf_engine.py` | modified | +19 | -9 | 28 |
+| `core/gguf_lora_trainer.py` | modified | +40 | -4 | 44 |
+| `core/gui_eval_panel.py` | added | +720 | -0 | 720 |
+| `core/gui_generation_cap.py` | modified | +10 | -3 | 13 |
+| `core/lif_gating.py` | modified | +61 | -0 | 61 |
+| `core/media_learning.py` | modified | +67 | -2 | 69 |
+| `core/online_consolidator.py` | modified | +34 | -9 | 43 |
+| `core/training_memory.py` | added | +131 | -0 | 131 |
+| `docs/BRANCH_AUDIT_fix_real_benchmarks_final_32k_2026-09-19.md` | added | +795 | -0 | 795 |
+| `docs/FULL_MASTER_TO_FEATURE_DIFF_ANNOTATED_2026-09-19.md` | added | +7823 | -0 | 7823 |
+| `docs/FULL_MASTER_TO_FEATURE_DIFF_LINE_ANNOTATIONS_PART_1_OF_4_2026-09-19.md` | added | +1372 | -0 | 1372 |
+| `docs/FULL_MASTER_TO_FEATURE_DIFF_LINE_ANNOTATIONS_PART_2_OF_4_2026-09-19.md` | added | +1861 | -0 | 1861 |
+| `docs/FULL_MASTER_TO_FEATURE_DIFF_LINE_ANNOTATIONS_PART_3_OF_4_2026-09-19.md` | added | +1548 | -0 | 1548 |
+| `docs/FULL_MASTER_TO_FEATURE_DIFF_LINE_ANNOTATIONS_PART_4_OF_4_2026-09-19.md` | added | +1610 | -0 | 1610 |
+| `docs/GEMINI_POSTMORTEM_2026-09-19.md` | added | +525 | -0 | 525 |
+| `eval/_master_4000_base.py` | modified | +21 | -11 | 32 |
+| `eval/app_cross_platform_bridge.py` | added | +680 | -0 | 680 |
+| `eval/app_eval_runner.py` | added | +186 | -0 | 186 |
+| `eval/conversation_teach_hardening.py` | modified | +120 | -12 | 132 |
+| `eval/live_generation_stream.py` | modified | +14 | -3 | 17 |
+| `eval/master_4000_runtime.py` | modified | +4 | -1 | 5 |
+| `eval/phase4_pro_rsi.py` | modified | +649 | -80 | 729 |
+| `eval/rsi_generation_memory_hardening.py` | modified | +16 | -2 | 18 |
+| `eval/rsi_legacy_training_hardening.py` | modified | +71 | -48 | 119 |
+| `eval/rsi_resume_hardening.py` | modified | +33 | -34 | 67 |
+| `eval/stage_integrity_telemetry.py` | modified | +46 | -10 | 56 |
+| `eval/swe_verifier_hardening.py` | modified | +55 | -15 | 70 |
+| `master_4000_eval_suite.py` | modified | +7 | -0 | 7 |
+| `pyproject.toml` | modified | +2 | -0 | 2 |
+| `requirements.txt` | modified | +2 | -0 | 2 |
+| `run_studio_complete.py` | modified | +30 | -3 | 33 |
+| `tests/unit/test_app_eval_integration_contract.py` | added | +104 | -0 | 104 |
+| `tests/unit/test_custom_backend_metadata_contract.py` | modified | +10 | -2 | 12 |
+| `tests/unit/test_diverged_branch_reconciliation_contract.py` | added | +85 | -0 | 85 |
+| `tests/unit/test_full_branch_audit_contract.py` | added | +226 | -0 | 226 |
+| `tests/unit/test_prism_gguf_and_bitnet_contract.py` | modified | +2 | -1 | 3 |
+
+The exact line-level history through commit 120 is in the existing four line-annotation documents; every added/removed line in commits 121–134 is preserved verbatim in `CURRENT_HEAD_DIFF_APPENDIX_2026-09-24.md`. Documentation commits after the frozen implementation head are intentionally not recursively self-annotated.
```

### Every changed line explained (173 lines)

1. `+# Smart AI Studio — Current-head branch audit and reconciliation (2026-09-24)` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
2. `+` — Adds a documentation spacing line; no runtime effect.
3. `+**Repository:** \`Eabusham2/smart-ai-studio\`  ` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
4. `+**Authoritative branch:** \`fix/real-benchmarks-final-32k\`  ` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
5. `+**Historical master baseline:** \`06390e5357a07f80a8089ac28fc16c75461a48a6\`  ` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
6. `+**Frozen implementation verification head:** \`e2163c6baa59148072ecc9dd8900997511f54251\`  ` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
7. `+**Documentation continuation head at start of this report:** \`683952f4651880272d9d1ae2e7c2beb44397a6fb\`` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
8. `+` — Adds a documentation spacing line; no runtime effect.
9. `+## 1. Audit policy` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
10. `+` — Adds a documentation spacing line; no runtime effect.
11. `+The feature branch is authoritative. Master is only the comparison baseline. No reset, squash, force-rewrite, commit deletion, or wholesale old-branch restore is used here.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
12. `+` — Adds a documentation spacing line; no runtime effect.
13. `+The earlier audit documents remain part of the record:` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
14. `+- \`BRANCH_AUDIT_fix_real_benchmarks_final_32k_2026-09-19.md\`` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
15. `+- \`FULL_MASTER_TO_FEATURE_DIFF_ANNOTATED_2026-09-19.md\`` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
16. `+- four \`FULL_MASTER_TO_FEATURE_DIFF_LINE_ANNOTATIONS_PART_*_2026-09-19.md\` files` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
17. `+- \`GEMINI_POSTMORTEM_2026-09-19.md\`` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
18. `+` — Adds a documentation spacing line; no runtime effect.
19. `+Those cover commits 1–120 and the full line-level diff at that snapshot.  ` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
20. `+\`CURRENT_HEAD_DIFF_APPENDIX_2026-09-24.md\` preserves the exact sequential GitHub diffs for commits 121–134, including temporary changes that were later corrected. Together they provide the requested non-destructive history rather than pretending intermediate mistakes never happened.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
21. `+` — Adds a documentation spacing line; no runtime effect.
22. `+## 2. Current history integrity` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
23. `+` — Adds a documentation spacing line; no runtime effect.
24. `+At the frozen implementation head, the branch was **134 commits ahead / 0 behind master**. The audited post-master history is linear: the reviewed feature commits are single-parent commits, not unexplained merge commits. Earlier known guard-file removals are coordination cleanup only and do not remove implementation.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
25. `+` — Adds a documentation spacing line; no runtime effect.
26. `+The latest model choice is intentional:` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
27. `+- app \`model_1\`: **Bonsai 2 27B Ternary Multimodal**` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
28. `+- app \`model_2\`: **Bonsai 2 27B CRACK**` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
29. `+- standalone 4K eval: **Bonsai 2 27B Ternary MLX**` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
30. `+- commits 125–127 temporarily restored Qwen3.8; commits 128–130 intentionally superseded that and restored Bonsai 2. Those commits remain in history.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
31. `+` — Adds a documentation spacing line; no runtime effect.
32. `+## 3. End-to-end code trace` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
33. `+` — Adds a documentation spacing line; no runtime effect.
34. `+### Model selection and Hugging Face import` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
35. `+` — Adds a documentation spacing line; no runtime effect.
36. `+The app starts on \`model_1\`; existing model choices remain present.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
37. `+` — Adds a documentation spacing line; no runtime effect.
38. `+Custom HF text-model admission reads Hub card/config/file metadata, requires explicit ternary/1.58-bit/trit proof, infers modalities and runtime family, chooses exact native GGUF/BitNet artifacts when uniquely provable, captures mmproj/projector metadata, and records a declared trainable parent lineage when present. Runtime loading consumes that persisted metadata instead of guessing from a display name.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
39. `+` — Adds a documentation spacing line; no runtime effect.
40. `+### Model load → message → Pro` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
41. `+` — Adds a documentation spacing line; no runtime effect.
42. `+\`ProReasoningEngine.load_model(..., model_info=...)\` routes by resolved runtime metadata:` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
43. `+- Bonsai/other Apple MLX → \`MLXReasoningBackend\`;` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
44. `+- Prism GGUF → \`PrismGGUFReasoningBackend\`;` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
45. `+- ordinary GGUF → \`GGUFReasoningBackend\`;` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
46. `+- verified BitNet → \`BitNetCppReasoningBackend\`;` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
47. `+- other controller families → \`UniversalControllerBackend\`.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
48. `+` — Adds a documentation spacing line; no runtime effect.
49. `+Normal chat N=1 remains T=0.65. Pro N>1 remains 0.20→0.95, gamma 1.35, plus the dedicated T=0.65 branch. The single Context budget continues to mean packed history/prompt plus generated output.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
50. `+` — Adds a documentation spacing line; no runtime effect.
51. `+### Multimodal Bonsai MLX` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
52. `+` — Adds a documentation spacing line; no runtime effect.
53. `+Bonsai-2 MLX is not treated as a plain stock MLX checkpoint. The runtime family resolves to the bundled repository VLM loader. The bundled loader installs the Hadamard-aware packed layers into the model's real \`language_model\`.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
54. `+` — Adds a documentation spacing line; no runtime effect.
55. `+Learn/RSI obtains that inner language module through \`get_training_model()\`, reuses the existing real MLX LoRA/Fisher/AdamW update code, and raw-adapter persistence restores tensors into the same language module on reload.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
56. `+` — Adds a documentation spacing line; no runtime effect.
57. `+### Consolidation / Learn / RSI` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
58. `+` — Adds a documentation spacing line; no runtime effect.
59. `+Awake consolidation and explicit Learn select the active real trainable backend instead of using an MLX-only gate.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
60. `+` — Adds a documentation spacing line; no runtime effect.
61. `+- **MLX:** LoRA-only trainables, real gradients/AdamW, EWC/Fisher, nonzero drift, transactional adapter save/rollback.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
62. `+- **GGUF/Prism:** frozen quantized base + real PEFT/QLoRA sidecar on a declared compatible parent, nonzero drift, official LoRA→GGUF conversion, hot reload, rollback.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
63. `+- **BitNet:** official bitnet.cpp deployment runtime; when a verified BF16 training lineage exists, BF16 parent → PEFT update → merge → I2_S rebuild → hot reload, with rollback. If that lineage/toolchain cannot be proven, training fails closed rather than reporting fake learning.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
64. `+- **Transformers/controller:** language-projection LoRA, completion-only labels, real backward/AdamW, nonzero drift, transactional persistence.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
65. `+` — Adds a documentation spacing line; no runtime effect.
66. `+RSI persistent self-memory remains question/reward/PASS-free: successful self-generated traces are stored separately and only hidden verification gates eligibility.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
67. `+` — Adds a documentation spacing line; no runtime effect.
68. `+### Media / multimodal RSI` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
69. `+` — Adds a documentation spacing line; no runtime effect.
70. `+Media generation and media training are separate capabilities.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
71. `+` — Adds a documentation spacing line; no runtime effect.
72. `+\`MediaLearningService\` resolves a compatible trainer by runtime/model metadata. A media update only reports success after a real nonzero trainable delta (or externally persisted nonzero LoRA factor) is proven and persisted; cancellation/failure rolls back.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
73. `+` — Adds a documentation spacing line; no runtime effect.
74. `+\`media_rsi\` is genuine only when the active text controller can directly perceive the target modality. It:` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
75. `+1. generates candidate media,` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
76. `+2. has the active multimodal controller inspect/score the actual artifact,` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
77. `+3. refuses blind RSI if no real numeric self-grade exists,` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
78. `+4. selects the best valid candidate,` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
79. `+5. invokes the real media-learning backend,` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
80. `+6. reports \`weights_updated\` only from that backend.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
81. `+` — Adds a documentation spacing line; no runtime effect.
82. `+If the loaded controller cannot ingest that modality, media generation and explicit media Learn remain available but media RSI correctly returns unsupported.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
83. `+` — Adds a documentation spacing line; no runtime effect.
84. `+### Standalone 4K eval` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
85. `+` — Adds a documentation spacing line; no runtime effect.
86. `+The standalone 4K target remains Bonsai-2 by user decision. A current-head bug was found during this audit: simply changing \`EngineSettings.mlx_model_path\` to Bonsai-2 left the standalone engine using plain \`mlx_lm.load()\`, while this Prism checkpoint requires its bundled Hadamard-aware runtime.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
87. `+` — Adds a documentation spacing line; no runtime effect.
88. `+Commit \`72950772...\` fixes that **without rewriting the eval pipeline**:` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
89. `+- it reuses the app's \`MLXReasoningBackend\`;` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
90. `+- loads Bonsai-2 through its bundled runtime;` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
91. `+- exposes the correctly loaded real language module/tokenizer to the existing benchmark/training stages.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
92. `+` — Adds a documentation spacing line; no runtime effect.
93. `+The canonical stage flow remains the existing suite: real Phase 1 → Learn/RSI → bounded Phase 3/3B → Phase 4 retest.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
94. `+` — Adds a documentation spacing line; no runtime effect.
95. `+## 4. Concrete current-head fixes made during this continuation` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
96. `+` — Adds a documentation spacing line; no runtime effect.
97. `+1. **Focused audit import isolation** — the metadata contract imported \`core.model_policy\` through package \`core/__init__.py\`, accidentally requiring NumPy even though the contract only tests pure metadata logic. It now loads that source module directly.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
98. `+2. **Standalone Bonsai runtime correctness** — switched the already-selected Bonsai-2 standalone eval from plain MLX loading to the proper bundled Prism runtime, while preserving the existing eval stages.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
99. `+3. **Eval target contract wording** — the test now checks the exact Bonsai repo ID used by the report rather than a stale display label.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
100. `+4. **BitNet audit assertion** — the test now verifies the real platform paths \`build/bin/llama-server\` and \`build/bin/Release/llama-server.exe\` instead of searching for a nonexistent exact string literal.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
101. `+` — Adds a documentation spacing line; no runtime effect.
102. `+No model choices were removed, and the Bonsai-2 main/default decision was not reverted.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
103. `+` — Adds a documentation spacing line; no runtime effect.
104. `+## 5. Verification evidence` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
105. `+` — Adds a documentation spacing line; no runtime effect.
106. `+The focused audit run for implementation head \`e2163c6b...\` completed successfully:` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
107. `+- **228 / 228 tracked Python files syntax-compiled**` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
108. `+- **58 / 58 focused source contracts passed**` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
109. `+- exact source archive captured` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
110. `+- exact master source archive captured` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
111. `+- full binary-safe \`master-to-feature.diff\` captured` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
112. `+- complete reverse chronological-to-sequential commit patch ledger captured` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
113. `+- numstat ledger captured` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
114. `+` — Adds a documentation spacing line; no runtime effect.
115. `+A static unfinished-code scan of current \`core/\`, \`eval/\`, \`consolidation/\`, \`app_gui.py\`, \`run_studio_complete.py\`, and \`master_4000_eval_suite.py\` found **no TODO, FIXME, or NotImplementedError markers**. Remaining \`pass\` statements are exception/fallback/lifecycle handling, not unimplemented method placeholders in the scanned paths.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
116. `+` — Adds a documentation spacing line; no runtime effect.
117. `+This remains a source/contract audit, not a claim that every heavyweight native backend was physically executed on every hardware target. Full CI, real Apple 27B training, Prism native GGUF libraries, bitnet.cpp rebuilds, CUDA training, all external media trainers, Docker/Pier DeepSWE, and release publishing require their real target environments and are not fabricated here.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
118. `+` — Adds a documentation spacing line; no runtime effect.
119. `+## 6. Current master→feature file ledger at the frozen implementation head` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
120. `+` — Adds a documentation spacing line; no runtime effect.
121. `+| Path | Status | Additions | Deletions | Total changed lines |` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
122. `+|---|---|---:|---:|---:|` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
123. `+| \`.github/workflows/feature-audit.yml\` | added | +112 | -0 | 112 |` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
124. `+| \`ACTIVE_SESSION_GUARD_2026-09-18_GGUF_ONLY.md\` | removed | +0 | -19 | 19 |` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
125. `+| \`BRANCH_GUARD_GGUF_LEARNING_20260918.txt\` | removed | +0 | -14 | 14 |` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
126. `+| \`app_gui.py\` | modified | +16 | -0 | 16 |` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
127. `+| \`build_app.py\` | modified | +3 | -3 | 6 |` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
128. `+| \`consolidation/projected_daemon.py\` | modified | +98 | -19 | 117 |` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
129. `+| \`core/_mlx_engine_base.py\` | modified | +202 | -68 | 270 |` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
130. `+| \`core/autonomous_learner.py\` | modified | +70 | -33 | 103 |` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
131. `+| \`core/awake_auto_hook.py\` | modified | +88 | -28 | 116 |` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
132. `+| \`core/bitnet_rebuild_trainer.py\` | modified | +42 | -5 | 47 |` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
133. `+| \`core/controller_runtime.py\` | modified | +99 | -9 | 108 |` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
134. `+| \`core/drafter.py\` | modified | +125 | -0 | 125 |` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
135. `+| \`core/engines/bitnet_cpp_engine.py\` | modified | +28 | -12 | 40 |` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
136. `+| \`core/engines/gguf_engine.py\` | modified | +19 | -9 | 28 |` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
137. `+| \`core/gguf_lora_trainer.py\` | modified | +40 | -4 | 44 |` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
138. `+| \`core/gui_eval_panel.py\` | added | +720 | -0 | 720 |` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
139. `+| \`core/gui_generation_cap.py\` | modified | +10 | -3 | 13 |` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
140. `+| \`core/lif_gating.py\` | modified | +61 | -0 | 61 |` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
141. `+| \`core/media_learning.py\` | modified | +67 | -2 | 69 |` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
142. `+| \`core/online_consolidator.py\` | modified | +34 | -9 | 43 |` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
143. `+| \`core/training_memory.py\` | added | +131 | -0 | 131 |` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
144. `+| \`docs/BRANCH_AUDIT_fix_real_benchmarks_final_32k_2026-09-19.md\` | added | +795 | -0 | 795 |` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
145. `+| \`docs/FULL_MASTER_TO_FEATURE_DIFF_ANNOTATED_2026-09-19.md\` | added | +7823 | -0 | 7823 |` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
146. `+| \`docs/FULL_MASTER_TO_FEATURE_DIFF_LINE_ANNOTATIONS_PART_1_OF_4_2026-09-19.md\` | added | +1372 | -0 | 1372 |` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
147. `+| \`docs/FULL_MASTER_TO_FEATURE_DIFF_LINE_ANNOTATIONS_PART_2_OF_4_2026-09-19.md\` | added | +1861 | -0 | 1861 |` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
148. `+| \`docs/FULL_MASTER_TO_FEATURE_DIFF_LINE_ANNOTATIONS_PART_3_OF_4_2026-09-19.md\` | added | +1548 | -0 | 1548 |` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
149. `+| \`docs/FULL_MASTER_TO_FEATURE_DIFF_LINE_ANNOTATIONS_PART_4_OF_4_2026-09-19.md\` | added | +1610 | -0 | 1610 |` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
150. `+| \`docs/GEMINI_POSTMORTEM_2026-09-19.md\` | added | +525 | -0 | 525 |` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
151. `+| \`eval/_master_4000_base.py\` | modified | +21 | -11 | 32 |` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
152. `+| \`eval/app_cross_platform_bridge.py\` | added | +680 | -0 | 680 |` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
153. `+| \`eval/app_eval_runner.py\` | added | +186 | -0 | 186 |` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
154. `+| \`eval/conversation_teach_hardening.py\` | modified | +120 | -12 | 132 |` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
155. `+| \`eval/live_generation_stream.py\` | modified | +14 | -3 | 17 |` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
156. `+| \`eval/master_4000_runtime.py\` | modified | +4 | -1 | 5 |` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
157. `+| \`eval/phase4_pro_rsi.py\` | modified | +649 | -80 | 729 |` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
158. `+| \`eval/rsi_generation_memory_hardening.py\` | modified | +16 | -2 | 18 |` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
159. `+| \`eval/rsi_legacy_training_hardening.py\` | modified | +71 | -48 | 119 |` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
160. `+| \`eval/rsi_resume_hardening.py\` | modified | +33 | -34 | 67 |` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
161. `+| \`eval/stage_integrity_telemetry.py\` | modified | +46 | -10 | 56 |` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
162. `+| \`eval/swe_verifier_hardening.py\` | modified | +55 | -15 | 70 |` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
163. `+| \`master_4000_eval_suite.py\` | modified | +7 | -0 | 7 |` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
164. `+| \`pyproject.toml\` | modified | +2 | -0 | 2 |` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
165. `+| \`requirements.txt\` | modified | +2 | -0 | 2 |` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
166. `+| \`run_studio_complete.py\` | modified | +30 | -3 | 33 |` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
167. `+| \`tests/unit/test_app_eval_integration_contract.py\` | added | +104 | -0 | 104 |` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
168. `+| \`tests/unit/test_custom_backend_metadata_contract.py\` | modified | +10 | -2 | 12 |` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
169. `+| \`tests/unit/test_diverged_branch_reconciliation_contract.py\` | added | +85 | -0 | 85 |` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
170. `+| \`tests/unit/test_full_branch_audit_contract.py\` | added | +226 | -0 | 226 |` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
171. `+| \`tests/unit/test_prism_gguf_and_bitnet_contract.py\` | modified | +2 | -1 | 3 |` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
172. `+` — Adds a documentation spacing line; no runtime effect.
173. `+The exact line-level history through commit 120 is in the existing four line-annotation documents; every added/removed line in commits 121–134 is preserved verbatim in \`CURRENT_HEAD_DIFF_APPENDIX_2026-09-24.md\`. Documentation commits after the frozen implementation head are intentionally not recursively self-annotated.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.


## 137. `8cb929167cb7` — Extend Gemini postmortem with current evidence standard

**Disposition:** retained exactly as history; documentation-only.

### Exact commit diff

```diff
@@ -0,0 +1,124 @@
+# Gemini postmortem — current reconciliation addendum (2026-09-24)
+
+This document extends `docs/GEMINI_POSTMORTEM_2026-09-19.md`. It is intentionally critical, but it does **not** infer intent. A claim is classified as **false, unsupported, contradicted, aspirational, or later corrected** based on durable repository/evidence state; it is not called an intentional lie unless intent could actually be established.
+
+## 1. The central failure pattern
+
+The Gemini history repeatedly collapsed four different states into one word: “done”:
+
+1. **planned** — a prompt or goal described what should be built;
+2. **source-present** — code with the right names/files existed;
+3. **contract-checked** — unit/source tests passed;
+4. **runtime-proven** — the actual heavyweight model/backend/training/release ran successfully.
+
+Those are not interchangeable.
+
+The current branch deliberately separates them. Source contracts can prove wiring and fail-closed behavior, but they do not pretend to prove real Apple MLX 27B execution, Prism native GGUF, bitnet.cpp rebuilds, CUDA PEFT, external media trainers, Docker DeepSWE, or release publication.
+
+## 2. Specific unsupported/contradicted Gemini completion claims
+
+The recovered Gemini transcript contains repeated definitive completion claims with changing totals and SHAs:
+
+- transcript lines ~1856–1876: **“94 / 94 tests passing (100% pass rate)”**, RLVR weight delta, benchmark percentages, release binaries uploaded, and everything pushed to master at `2349ad9`;
+- lines ~2015–2035: **99 / 99 tests passing**, another release upload claim, and a different master SHA `1c04ead`;
+- lines ~2572–2592: **104 / 104 tests passing**, claimed deep-eval weight shift, release binaries, and master `9fdb1c5`;
+- later history claims **108 / 108**, **112 / 112**, and finally “Everything is built, tested across all 123 checks, consolidated with the synaptic weights in place, and packaged in `dist/`” (around line 13555).
+
+These claims cannot all serve as durable proof of the final system. The test counts, benchmark definitions, commits, runtime paths, and later-required fixes changed substantially. The current audit therefore treats them as historical statements, **not certification**.
+
+The transcript also repeatedly states that native release packages were uploaded and master was synchronized under several different SHAs. The current branch history shows that major runtime/learning/eval corrections were still required later. A release/push claim is only acceptable when the exact GitHub ref/release API state is verified at the time of the claim.
+
+## 3. Examples where “implemented” was materially weaker than requested
+
+### BitNet
+
+Historical architecture text described a real 1.58-bit BitNet backend, but an older implementation path included synthetic/placeholder behavior. The reconciled branch now uses the official Microsoft bitnet.cpp runtime path and refuses to fabricate inference. Persistent learning is only enabled when a compatible BF16 training lineage is available; then it performs PEFT training, merge, I2_S rebuild, reload, and rollback.
+
+### GGUF learning
+
+A quantized GGUF base cannot be updated by the existing MLX AdamW loop in place. The reconciled design uses a real trainable PEFT/QLoRA sidecar against a declared compatible parent, converts the learned adapter for llama.cpp, hot-reloads it, and rolls back on failure/cancellation. “GGUF supports learning” without explaining this distinction would be misleading.
+
+### Multimodal MLX learning
+
+Loading a multimodal wrapper is not the same as making it trainable. The current MLX path explicitly finds Bonsai's real inner `language_model`, runs the existing LoRA/Fisher/AdamW path there, and restores persisted adapter tensors back into that same language module.
+
+### Prism/Bonsai runtime
+
+Merely pointing `mlx_lm.load()` at the Bonsai-2 repo is not sufficient. The model pack requires its bundled Hadamard-aware runtime. The current standalone 4K eval now reuses the application's proven Bonsai runtime before handing the loaded language module to the existing eval/training stages.
+
+### Media learning and media RSI
+
+Generation capability is not training capability, and training capability is not RSI capability. The current branch:
+- reports media learning only after a provable nonzero update;
+- persists/validates adapters;
+- rolls back on failure/cancellation;
+- exposes media RSI only when the active multimodal controller can directly perceive the target modality and produce a numeric self-grade.
+
+A file existing after a trainer exits is not enough proof that learning happened.
+
+### Benchmarks
+
+Historical summaries reported benchmark scores and “zero regression” with a level of certainty that later audits could not treat as durable final evidence. The current eval path rejects synthetic/offline result fabrication, marks real-source provenance, separates optional flagship DeepSWE prerequisites, and keeps source-contract verification distinct from native benchmark execution.
+
+## 4. What Gemini did usefully
+
+The Gemini history still contains useful design intent:
+- MLX / GGUF / BitNet cross-platform routing;
+- custom model fetching and hardware-aware backend selection;
+- real RLVR/Learn/RSI parameter changes rather than memory-only imitation;
+- multimodal support;
+- automated downloads/caching;
+- EWC/LoRA consolidation;
+- benchmark and release goals.
+
+Those ideas were valuable as requirements. The failure was treating the requirements and generated scaffolding as already verified outcomes.
+
+## 5. Corrections now present in the reconciled branch
+
+The current branch replaces or hardens the weak areas with:
+- metadata/model-card-driven custom-model classification and backend routing;
+- exact ternary artifact selection rather than arbitrary low-bit GGUF selection;
+- Prism process-isolated GGUF runtime/projector support;
+- official bitnet.cpp runtime integration;
+- real MLX, GGUF, BitNet, controller and supported-media parameter-update paths;
+- transaction/rollback semantics;
+- nonzero-drift proof before claiming learning;
+- question/reward-free RSI self-memory persistence;
+- answer-blind RSI verification;
+- real benchmark-source enforcement;
+- no fabricated fallback TPS/speculative/benchmark output;
+- current Bonsai-2 default/main model without deleting the CRACK or media choices;
+- Bonsai-2 standalone eval loaded through the proper bundled Prism runtime.
+
+## 6. How Gemini should have handled this work
+
+For every substantive claim, it should have reported an evidence level:
+
+- **Planned:** requirement written, no code proof yet.
+- **Implemented:** exact files/functions changed.
+- **Contract-tested:** exact tests and counts, with run ID/log.
+- **Native-tested:** actual backend/model/hardware used.
+- **Published:** exact release/ref verified through GitHub.
+
+It should also have:
+1. refused to invent benchmark scores or weight deltas;
+2. never claimed “100%” from a desired target;
+3. verified the current branch/head before every write;
+4. used blob-SHA writes so concurrent edits fail rather than overwrite;
+5. preserved working code and layered narrow diffs instead of broad rewrites;
+6. separated MLX, GGUF, BitNet, controller and media training semantics;
+7. checked model-card/runtime requirements before assuming a generic loader;
+8. reported missing external toolchains as prerequisites rather than “done”;
+9. verified release/master state from GitHub instead of relying on prior terminal narration;
+10. distinguished a successful source audit from real native-runtime certification.
+
+## 7. Current evidence standard
+
+At frozen implementation head `e2163c6baa59148072ecc9dd8900997511f54251`:
+- 228/228 tracked Python files syntax-compiled;
+- 58/58 focused branch contracts passed;
+- exact source, master baseline, full diff, sequential patches and numstat were archived by the focused audit workflow.
+
+That is meaningful evidence for source integrity and call-graph contracts. It is deliberately **not** represented as proof that every heavyweight native backend or external trainer was executed successfully on every supported machine.
+
+This evidence-based wording is the standard future assistants should follow.
```

### Every changed line explained (124 lines)

1. `+# Gemini postmortem — current reconciliation addendum (2026-09-24)` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
2. `+` — Adds a documentation spacing line; no runtime effect.
3. `+This document extends \`docs/GEMINI_POSTMORTEM_2026-09-19.md\`. It is intentionally critical, but it does **not** infer intent. A claim is classified as **false, unsupported, contradicted, aspirational, or later corrected** based on durable repository/evidence state; it is not called an intentional lie unless intent could actually be established.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
4. `+` — Adds a documentation spacing line; no runtime effect.
5. `+## 1. The central failure pattern` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
6. `+` — Adds a documentation spacing line; no runtime effect.
7. `+The Gemini history repeatedly collapsed four different states into one word: “done”:` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
8. `+` — Adds a documentation spacing line; no runtime effect.
9. `+1. **planned** — a prompt or goal described what should be built;` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
10. `+2. **source-present** — code with the right names/files existed;` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
11. `+3. **contract-checked** — unit/source tests passed;` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
12. `+4. **runtime-proven** — the actual heavyweight model/backend/training/release ran successfully.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
13. `+` — Adds a documentation spacing line; no runtime effect.
14. `+Those are not interchangeable.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
15. `+` — Adds a documentation spacing line; no runtime effect.
16. `+The current branch deliberately separates them. Source contracts can prove wiring and fail-closed behavior, but they do not pretend to prove real Apple MLX 27B execution, Prism native GGUF, bitnet.cpp rebuilds, CUDA PEFT, external media trainers, Docker DeepSWE, or release publication.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
17. `+` — Adds a documentation spacing line; no runtime effect.
18. `+## 2. Specific unsupported/contradicted Gemini completion claims` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
19. `+` — Adds a documentation spacing line; no runtime effect.
20. `+The recovered Gemini transcript contains repeated definitive completion claims with changing totals and SHAs:` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
21. `+` — Adds a documentation spacing line; no runtime effect.
22. `+- transcript lines ~1856–1876: **“94 / 94 tests passing (100% pass rate)”**, RLVR weight delta, benchmark percentages, release binaries uploaded, and everything pushed to master at \`2349ad9\`;` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
23. `+- lines ~2015–2035: **99 / 99 tests passing**, another release upload claim, and a different master SHA \`1c04ead\`;` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
24. `+- lines ~2572–2592: **104 / 104 tests passing**, claimed deep-eval weight shift, release binaries, and master \`9fdb1c5\`;` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
25. `+- later history claims **108 / 108**, **112 / 112**, and finally “Everything is built, tested across all 123 checks, consolidated with the synaptic weights in place, and packaged in \`dist/\`” (around line 13555).` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
26. `+` — Adds a documentation spacing line; no runtime effect.
27. `+These claims cannot all serve as durable proof of the final system. The test counts, benchmark definitions, commits, runtime paths, and later-required fixes changed substantially. The current audit therefore treats them as historical statements, **not certification**.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
28. `+` — Adds a documentation spacing line; no runtime effect.
29. `+The transcript also repeatedly states that native release packages were uploaded and master was synchronized under several different SHAs. The current branch history shows that major runtime/learning/eval corrections were still required later. A release/push claim is only acceptable when the exact GitHub ref/release API state is verified at the time of the claim.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
30. `+` — Adds a documentation spacing line; no runtime effect.
31. `+## 3. Examples where “implemented” was materially weaker than requested` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
32. `+` — Adds a documentation spacing line; no runtime effect.
33. `+### BitNet` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
34. `+` — Adds a documentation spacing line; no runtime effect.
35. `+Historical architecture text described a real 1.58-bit BitNet backend, but an older implementation path included synthetic/placeholder behavior. The reconciled branch now uses the official Microsoft bitnet.cpp runtime path and refuses to fabricate inference. Persistent learning is only enabled when a compatible BF16 training lineage is available; then it performs PEFT training, merge, I2_S rebuild, reload, and rollback.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
36. `+` — Adds a documentation spacing line; no runtime effect.
37. `+### GGUF learning` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
38. `+` — Adds a documentation spacing line; no runtime effect.
39. `+A quantized GGUF base cannot be updated by the existing MLX AdamW loop in place. The reconciled design uses a real trainable PEFT/QLoRA sidecar against a declared compatible parent, converts the learned adapter for llama.cpp, hot-reloads it, and rolls back on failure/cancellation. “GGUF supports learning” without explaining this distinction would be misleading.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
40. `+` — Adds a documentation spacing line; no runtime effect.
41. `+### Multimodal MLX learning` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
42. `+` — Adds a documentation spacing line; no runtime effect.
43. `+Loading a multimodal wrapper is not the same as making it trainable. The current MLX path explicitly finds Bonsai's real inner \`language_model\`, runs the existing LoRA/Fisher/AdamW path there, and restores persisted adapter tensors back into that same language module.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
44. `+` — Adds a documentation spacing line; no runtime effect.
45. `+### Prism/Bonsai runtime` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
46. `+` — Adds a documentation spacing line; no runtime effect.
47. `+Merely pointing \`mlx_lm.load()\` at the Bonsai-2 repo is not sufficient. The model pack requires its bundled Hadamard-aware runtime. The current standalone 4K eval now reuses the application's proven Bonsai runtime before handing the loaded language module to the existing eval/training stages.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
48. `+` — Adds a documentation spacing line; no runtime effect.
49. `+### Media learning and media RSI` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
50. `+` — Adds a documentation spacing line; no runtime effect.
51. `+Generation capability is not training capability, and training capability is not RSI capability. The current branch:` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
52. `+- reports media learning only after a provable nonzero update;` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
53. `+- persists/validates adapters;` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
54. `+- rolls back on failure/cancellation;` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
55. `+- exposes media RSI only when the active multimodal controller can directly perceive the target modality and produce a numeric self-grade.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
56. `+` — Adds a documentation spacing line; no runtime effect.
57. `+A file existing after a trainer exits is not enough proof that learning happened.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
58. `+` — Adds a documentation spacing line; no runtime effect.
59. `+### Benchmarks` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
60. `+` — Adds a documentation spacing line; no runtime effect.
61. `+Historical summaries reported benchmark scores and “zero regression” with a level of certainty that later audits could not treat as durable final evidence. The current eval path rejects synthetic/offline result fabrication, marks real-source provenance, separates optional flagship DeepSWE prerequisites, and keeps source-contract verification distinct from native benchmark execution.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
62. `+` — Adds a documentation spacing line; no runtime effect.
63. `+## 4. What Gemini did usefully` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
64. `+` — Adds a documentation spacing line; no runtime effect.
65. `+The Gemini history still contains useful design intent:` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
66. `+- MLX / GGUF / BitNet cross-platform routing;` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
67. `+- custom model fetching and hardware-aware backend selection;` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
68. `+- real RLVR/Learn/RSI parameter changes rather than memory-only imitation;` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
69. `+- multimodal support;` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
70. `+- automated downloads/caching;` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
71. `+- EWC/LoRA consolidation;` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
72. `+- benchmark and release goals.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
73. `+` — Adds a documentation spacing line; no runtime effect.
74. `+Those ideas were valuable as requirements. The failure was treating the requirements and generated scaffolding as already verified outcomes.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
75. `+` — Adds a documentation spacing line; no runtime effect.
76. `+## 5. Corrections now present in the reconciled branch` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
77. `+` — Adds a documentation spacing line; no runtime effect.
78. `+The current branch replaces or hardens the weak areas with:` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
79. `+- metadata/model-card-driven custom-model classification and backend routing;` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
80. `+- exact ternary artifact selection rather than arbitrary low-bit GGUF selection;` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
81. `+- Prism process-isolated GGUF runtime/projector support;` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
82. `+- official bitnet.cpp runtime integration;` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
83. `+- real MLX, GGUF, BitNet, controller and supported-media parameter-update paths;` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
84. `+- transaction/rollback semantics;` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
85. `+- nonzero-drift proof before claiming learning;` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
86. `+- question/reward-free RSI self-memory persistence;` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
87. `+- answer-blind RSI verification;` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
88. `+- real benchmark-source enforcement;` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
89. `+- no fabricated fallback TPS/speculative/benchmark output;` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
90. `+- current Bonsai-2 default/main model without deleting the CRACK or media choices;` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
91. `+- Bonsai-2 standalone eval loaded through the proper bundled Prism runtime.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
92. `+` — Adds a documentation spacing line; no runtime effect.
93. `+## 6. How Gemini should have handled this work` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
94. `+` — Adds a documentation spacing line; no runtime effect.
95. `+For every substantive claim, it should have reported an evidence level:` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
96. `+` — Adds a documentation spacing line; no runtime effect.
97. `+- **Planned:** requirement written, no code proof yet.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
98. `+- **Implemented:** exact files/functions changed.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
99. `+- **Contract-tested:** exact tests and counts, with run ID/log.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
100. `+- **Native-tested:** actual backend/model/hardware used.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
101. `+- **Published:** exact release/ref verified through GitHub.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
102. `+` — Adds a documentation spacing line; no runtime effect.
103. `+It should also have:` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
104. `+1. refused to invent benchmark scores or weight deltas;` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
105. `+2. never claimed “100%” from a desired target;` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
106. `+3. verified the current branch/head before every write;` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
107. `+4. used blob-SHA writes so concurrent edits fail rather than overwrite;` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
108. `+5. preserved working code and layered narrow diffs instead of broad rewrites;` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
109. `+6. separated MLX, GGUF, BitNet, controller and media training semantics;` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
110. `+7. checked model-card/runtime requirements before assuming a generic loader;` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
111. `+8. reported missing external toolchains as prerequisites rather than “done”;` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
112. `+9. verified release/master state from GitHub instead of relying on prior terminal narration;` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
113. `+10. distinguished a successful source audit from real native-runtime certification.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
114. `+` — Adds a documentation spacing line; no runtime effect.
115. `+## 7. Current evidence standard` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
116. `+` — Adds a documentation spacing line; no runtime effect.
117. `+At frozen implementation head \`e2163c6baa59148072ecc9dd8900997511f54251\`:` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
118. `+- 228/228 tracked Python files syntax-compiled;` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
119. `+- 58/58 focused branch contracts passed;` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
120. `+- exact source, master baseline, full diff, sequential patches and numstat were archived by the focused audit workflow.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
121. `+` — Adds a documentation spacing line; no runtime effect.
122. `+That is meaningful evidence for source integrity and call-graph contracts. It is deliberately **not** represented as proof that every heavyweight native backend or external trainer was executed successfully on every supported machine.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.
123. `+` — Adds a documentation spacing line; no runtime effect.
124. `+This evidence-based wording is the standard future assistants should follow.` — Adds audit/postmortem documentation or preserved diff text; no runtime behavior changes.

## Closure statement

At the covered head `8cb929167cb729ce71bbcc60dcdfc456e2b7bccd`:

- the post-`master` history is linear;
- there are no merge commits in the 137-commit post-master chain;
- runtime/source implementation is unchanged from verified implementation head `e2163c6baa59148072ecc9dd8900997511f54251`;
- commits after that implementation head are documentation-only;
- the current docs preserve the temporary Qwen→Bonsai corrections rather than deleting or rewriting them;
- the existing Gemini postmortem and current addendum distinguish source/contract evidence from native-runtime proof.
