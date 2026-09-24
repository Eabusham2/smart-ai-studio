# Current-head sequential diff appendix — commits 121–134

**Repository:** `Eabusham2/smart-ai-studio`  
**Branch:** `fix/real-benchmarks-final-32k`  
**Baseline already documented:** commits 1–120 in the existing 2026-09-19 audit and four line-annotation parts.  
**Current audited head before this documentation commit:** `e2163c6baa59148072ecc9dd8900997511f54251`  
**Master baseline:** `06390e5357a07f80a8089ac28fc16c75461a48a6`

This appendix does not replace or rewrite the earlier audit. It extends it. Every added/removed source line in commits 121–134 is reproduced below from GitHub's canonical commit diff, in chronological order. This records both temporary changes and later corrections, so the history is auditable rather than silently rewritten.


## 121. `371881796cd9` — fix: bound MLX Phase 3B conversational training

**Disposition:** retained as history. Where a later commit changes the same policy, the later current-head state is authoritative; no commit was deleted or squashed.

```diff
@@ -99,21 +99,118 @@ def phase3_then_conversation_teach(self):
             backend.is_mlx_available = True
         backend.adapter_path = p4.RSI_ADAPTER_PATH
 
-        consolidator = AwakeOnlineConsolidator(
-            mlx_engine=backend,
-            memory_db=None,
-            max_context=8192,
-        )
         train_tokens = _training_tokens(self.engine.tokenizer)
         teach_started = time.perf_counter()
-        consolidator._run_shadow_consolidation(_chat_history())
+
+        if str(getattr(self.engine, "backend_key", "mlx") or "mlx").lower() == "mlx":
+            # Reuse the same bounded-memory LoRA gradient path that Phase 3 already
+            # uses successfully. Never fall back to full-model value_and_grad here:
+            # Qwen3.5 inference CustomKernel has no VJP and can explode unified RAM.
+            trainable = dict(
+                p4.mlx.utils.tree_flatten(
+                    self.engine.model.trainable_parameters()
+                )
+            )
+            before = {}
+            for key, value in trainable.items():
+                try:
+                    before[key] = p4.mx.copy(value)
+                except Exception:
+                    before[key] = value + p4.mx.zeros_like(value)
+            if before:
+                p4.mx.eval(*before.values())
+
+            opt = p4.optim.AdamW(learning_rate=1e-4)
+            total_targets = 0
+            prepared = []
+            for user, assistant, _ in CONVERSATION_TEACH_EXAMPLES:
+                prefix = (
+                    f"<|im_start|>user\n{user}<|im_end|>\n"
+                    f"<|im_start|>assistant\n"
+                )
+                text = prefix + assistant + "<|im_end|>"
+                ids = self.engine.tokenizer.encode(text)
+                prefix_ids = self.engine.tokenizer.encode(prefix)
+                if len(ids) <= 1:
+                    continue
+                completion_loss_start = max(0, len(prefix_ids) - 1)
+                selected_start = max(0, len(ids) - 16_384)
+                row_targets = max(
+                    0,
+                    min(len(ids), 16_384) - 1
+                    - max(0, completion_loss_start - selected_start),
+                )
+                if row_targets <= 0:
+                    continue
+                prepared.append((ids, completion_loss_start, row_targets))
+                total_targets += row_targets
+
+            if not prepared or total_targets <= 0:
+                raise RuntimeError("Phase 3B conversational teach found no trainable completion targets")
+
+            completed_targets = 0
+            for _step in range(3):
+                for item_index, (ids, completion_loss_start, row_targets) in enumerate(prepared, 1):
+                    with p4.METAL_STREAM_LOCK:
+                        _loss, grads, trained_targets = p4._phase3_bounded_gradients(
+                            self,
+                            ids,
+                            completion_loss_start,
+                            item_index,
+                            len(prepared),
+                            completed_targets,
+                            total_targets * 3,
+                            teach_started,
+                        )
+                        opt.update(self.engine.model, grads)
+                        p4.mx.eval(self.engine.model.parameters(), opt.state)
+                        completed_targets += int(trained_targets)
+                        _loss = None
+                        grads = None
+                        try:
+                            p4.mx.clear_cache()
+                        except Exception:
+                            pass
+
+            after = dict(
+                p4.mlx.utils.tree_flatten(
+                    self.engine.model.trainable_parameters()
+                )
+            )
+            delta_sq = p4.mx.array(0.0)
+            matched = 0
+            for key, old_value in before.items():
+                new_value = after.get(key)
+                if new_value is None:
+                    continue
+                diff = new_value - old_value
+                delta_sq = delta_sq + p4.mx.sum(
+                    diff.astype(p4.mx.float32) * diff.astype(p4.mx.float32)
+                )
+                matched += 1
+            if matched <= 0:
+                raise RuntimeError("Phase 3B could not match post-update LoRA trainables")
+            p4.mx.eval(delta_sq)
+            delta = float(p4.mx.sqrt(delta_sq).item())
+            persisted = bool(p4._save_rsi_adapter(self))
+        else:
+            # Preserve the existing non-MLX production path unchanged.
+            consolidator = AwakeOnlineConsolidator(
+                mlx_engine=backend,
+                memory_db=None,
+                max_context=8192,
+            )
+            consolidator._run_shadow_consolidation(_chat_history())
+            delta = float(consolidator.total_param_shift or 0.0)
+            persisted = bool(os.path.exists(p4.RSI_ADAPTER_PATH))
+            if consolidator.consolidation_count != 1:
+                raise RuntimeError("Phase 3B conversational teach produced no real parameter update")
+
         teach_seconds = max(0.001, time.perf_counter() - teach_started)
         teach_tps = train_tokens / teach_seconds
 
         p4._assert_same_model(self, model_identity, "after conversational teach")
-        delta = float(consolidator.total_param_shift or 0.0)
-        persisted = bool(os.path.exists(p4.RSI_ADAPTER_PATH))
-        if consolidator.consolidation_count != 1 or delta <= 0.0:
+        if delta <= 0.0:
             raise RuntimeError("Phase 3B conversational teach produced no real parameter update")
         if not persisted:
             raise RuntimeError("Phase 3B conversational teach did not persist the updated adapter")
```

## 122. `c9d79f265268` — Correct BitNet learning runtime documentation

**Disposition:** retained as history. Where a later commit changes the same policy, the later current-head state is authoritative; no commit was deleted or squashed.

```diff
@@ -1,9 +1,10 @@
-"""Real Microsoft bitnet.cpp inference backend.
+"""Real Microsoft bitnet.cpp inference and persistent-learning backend.
 
 This replaces the old synthetic BitNet placeholder in ProReasoningEngine routing.
-It talks to bitnet.cpp's llama-server and never fabricates output. Parameter training
-is intentionally fail-closed until bitnet.cpp exposes a compatible persistent adapter
-training path.
+It talks to bitnet.cpp's llama-server and never fabricates output. Learning is
+fail-closed unless metadata provides a compatible BF16 training lineage; when it
+does, the backend performs a real PEFT update, rebuilds I2_S deployment weights,
+hot-reloads them, and rolls back atomically on failure or cancellation.
 """
 from __future__ import annotations
 
```

## 123. `a8d6033e01c4` — Lock real benchmark and bounded Phase 3B invariants

**Disposition:** retained as history. Where a later commit changes the same policy, the later current-head state is authoritative; no commit was deleted or squashed.

```diff
@@ -179,3 +179,36 @@ def test_end_to_end_call_chain_is_wired_through_existing_components():
     assert "▶ RSI: RECURSIVE SELF-IMPROVEMENT ON PHASE-1 MISSES" in phase
     assert "▶ PHASE 3: LEARN + RSI PARAMETRIC CONSOLIDATION" in phase
     assert "Phase 4: Post-Consolidation" in phase
+
+def test_real_benchmark_repair_cannot_fall_back_to_legacy_synthetic_rows():
+    suite = _src("master_4000_eval_suite.py")
+    real = _src("eval/real_benchmark_runtime.py")
+    assert "runtime_module._repair_suite = repair_real_suite" in real
+    install = suite.index(
+        "real_benchmark_runtime.install(BenchmarkDatasetProvider, master_runtime, phase4_pro_rsi, Master4000EvaluationEngine)"
+    )
+    capture = suite.index("_real_repair_suite = master_runtime._repair_suite")
+    assert install < capture
+    assert 'if not item.get("real_source"):' in real
+    assert "refusing to fall back to synthetic data" in real
+
+
+def test_mlx_phase3b_conversation_teach_uses_bounded_completion_only_lora_path():
+    teach = _src("eval/conversation_teach_hardening.py")
+    mlx_branch = teach.index(
+        'if str(getattr(self.engine, "backend_key", "mlx") or "mlx").lower() == "mlx":',
+        teach.index("teach_started = time.perf_counter()"),
+    )
+    non_mlx = teach.index("else:", mlx_branch)
+    mlx_body = teach[mlx_branch:non_mlx]
+    assert "p4._phase3_bounded_gradients(" in mlx_body
+    assert "with p4.METAL_STREAM_LOCK:" in mlx_body
+    assert "p4.optim.AdamW(learning_rate=1e-4)" in mlx_body
+    assert "completion_loss_start" in mlx_body
+    assert "p4._save_rsi_adapter(self)" in mlx_body
+    assert "_run_shadow_consolidation(" not in mlx_body
+
+    non_mlx_body = teach[non_mlx:]
+    assert "AwakeOnlineConsolidator(" in non_mlx_body
+    assert "consolidator._run_shadow_consolidation(_chat_history())" in non_mlx_body
+
```

## 124. `15db4ed565b4` — Expand focused branch audit coverage

**Disposition:** retained as history. Where a later commit changes the same policy, the later current-head state is authoritative; no commit was deleted or squashed.

```diff
@@ -5,8 +5,15 @@ on:
     branches: [fix/real-benchmarks-final-32k]
     paths:
       - .github/workflows/feature-audit.yml
-      - tools/feature_audit.py
-      - tests/audit/**
+      - app_gui.py
+      - master_4000_eval_suite.py
+      - run_studio_complete.py
+      - core/**
+      - eval/**
+      - consolidation/**
+      - tests/unit/test_*contract.py
+      - requirements.txt
+      - pyproject.toml
   workflow_dispatch:
 
 permissions:
@@ -89,6 +96,11 @@ jobs:
             tests/unit/test_app_eval_integration_contract.py \
             tests/unit/test_diverged_branch_reconciliation_contract.py \
             tests/unit/test_full_branch_audit_contract.py \
+            tests/unit/test_custom_backend_metadata_contract.py \
+            tests/unit/test_prism_gguf_and_bitnet_contract.py \
+            tests/unit/test_non_mlx_learning_contract.py \
+            tests/unit/test_memory_limit_control_contract.py \
+            tests/unit/test_temperature_policy_contract.py \
             --junitxml="$AUDIT_OUTPUT/contracts.xml" 2>&1 | tee "$AUDIT_OUTPUT/contracts.log"
       - name: Upload audit evidence even on failure
         if: always()
```

## 125. `7c711c83f809` — Restore standalone 4K eval Qwen3.8 target

**Disposition:** retained as history. Where a later commit changes the same policy, the later current-head state is authoritative; no commit was deleted or squashed.

```diff
@@ -34,7 +34,7 @@ def compute_auto_kv_budget(total_ram_gb: float)->int:
 @dataclass
 class EngineSettings:
     total_ram_gb: float=field(default_factory=lambda: psutil.virtual_memory().total/(1024**3))
-    mlx_model_path: str="prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"
+    mlx_model_path: str="penkia/TernaryQuench-Qwen3.8-27B-MLX"
     max_kv_tokens: int=field(init=False)
     h2o_sink_tokens:int=4; h2o_heavy_tokens:int=64; h2o_max_budget:int=128
     lora_rank:int=4; lora_alpha:float=8.0; lora_chunk_size:int=6; total_layers:int=60
```

## 126. `261fe3c56757` — Align standalone eval report with Qwen3.8 target

**Disposition:** retained as history. Where a later commit changes the same policy, the later current-head state is authoritative; no commit was deleted or squashed.

```diff
@@ -501,7 +501,7 @@ def _generate_master_report(self, base_scores: Dict[str, float], post_scores: Di
             getattr(
                 self,
                 "_eval_target_model_label",
-                "Bonsai 2 27B Ternary Multimodal",
+                "penkia/TernaryQuench-Qwen3.8-27B-MLX",
             )
         )
         md.append(f"**Target Model:** `{target_label}`  ")
```

## 127. `931595f8c5ba` — Lock standalone eval Qwen target separately from app Eval

**Disposition:** retained as history. Where a later commit changes the same policy, the later current-head state is authoritative; no commit was deleted or squashed.

```diff
@@ -212,3 +212,15 @@ def test_mlx_phase3b_conversation_teach_uses_bounded_completion_only_lora_path()
     assert "AwakeOnlineConsolidator(" in non_mlx_body
     assert "consolidator._run_shadow_consolidation(_chat_history())" in non_mlx_body
 
+def test_standalone_eval_keeps_qwen38_while_app_eval_uses_selected_model():
+    runtime = _src("run_studio_complete.py")
+    report = _src("eval/_master_4000_base.py")
+    bridge = _src("eval/app_cross_platform_bridge.py")
+    qwen = "penkia/TernaryQuench-Qwen3.8-27B-MLX"
+    assert f'mlx_model_path: str="{qwen}"' in runtime
+    assert f'"{qwen}"' in report
+    # Desktop Eval remains model-aware and loads the app-selected model metadata.
+    assert 'info = dict(config.get("model_info") or {})' in bridge
+    assert "self.pro.load_model(" in bridge
+    assert "model_info=info" in bridge
+
```

## 128. `94e0f8c5ef40` — Restore standalone 4K eval to Bonsai 2

**Disposition:** retained as history. Where a later commit changes the same policy, the later current-head state is authoritative; no commit was deleted or squashed.

```diff
@@ -34,7 +34,7 @@ def compute_auto_kv_budget(total_ram_gb: float)->int:
 @dataclass
 class EngineSettings:
     total_ram_gb: float=field(default_factory=lambda: psutil.virtual_memory().total/(1024**3))
-    mlx_model_path: str="penkia/TernaryQuench-Qwen3.8-27B-MLX"
+    mlx_model_path: str="prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"
     max_kv_tokens: int=field(init=False)
     h2o_sink_tokens:int=4; h2o_heavy_tokens:int=64; h2o_max_budget:int=128
     lora_rank:int=4; lora_alpha:float=8.0; lora_chunk_size:int=6; total_layers:int=60
```

## 129. `e074242477a3` — Align standalone eval report with Bonsai 2

**Disposition:** retained as history. Where a later commit changes the same policy, the later current-head state is authoritative; no commit was deleted or squashed.

```diff
@@ -501,7 +501,7 @@ def _generate_master_report(self, base_scores: Dict[str, float], post_scores: Di
             getattr(
                 self,
                 "_eval_target_model_label",
-                "penkia/TernaryQuench-Qwen3.8-27B-MLX",
+                "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit",
             )
         )
         md.append(f"**Target Model:** `{target_label}`  ")
```

## 130. `6142402da1d3` — Lock standalone eval Bonsai 2 target

**Disposition:** retained as history. Where a later commit changes the same policy, the later current-head state is authoritative; no commit was deleted or squashed.

```diff
@@ -212,13 +212,13 @@ def test_mlx_phase3b_conversation_teach_uses_bounded_completion_only_lora_path()
     assert "AwakeOnlineConsolidator(" in non_mlx_body
     assert "consolidator._run_shadow_consolidation(_chat_history())" in non_mlx_body
 
-def test_standalone_eval_keeps_qwen38_while_app_eval_uses_selected_model():
+def test_standalone_eval_keeps_bonsai2_while_app_eval_uses_selected_model():
     runtime = _src("run_studio_complete.py")
     report = _src("eval/_master_4000_base.py")
     bridge = _src("eval/app_cross_platform_bridge.py")
-    qwen = "penkia/TernaryQuench-Qwen3.8-27B-MLX"
-    assert f'mlx_model_path: str="{qwen}"' in runtime
-    assert f'"{qwen}"' in report
+    bonsai = "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"
+    assert f'mlx_model_path: str="{bonsai}"' in runtime
+    assert f'"{bonsai}"' in report
     # Desktop Eval remains model-aware and loads the app-selected model metadata.
     assert 'info = dict(config.get("model_info") or {})' in bridge
     assert "self.pro.load_model(" in bridge
```

## 131. `f7e3f4ef472d` — Keep metadata contract independent of full runtime imports

**Disposition:** retained as history. Where a later commit changes the same policy, the later current-head state is authoritative; no commit was deleted or squashed.

```diff
@@ -1,10 +1,18 @@
 from pathlib import Path
-
-from core.model_policy import derive_runtime_metadata
+import importlib.util
 
 
 ROOT = Path(__file__).resolve().parents[2]
 
+_spec = importlib.util.spec_from_file_location(
+    "smartai_model_policy_contract",
+    ROOT / "core" / "model_policy.py",
+)
+_model_policy = importlib.util.module_from_spec(_spec)
+assert _spec is not None and _spec.loader is not None
+_spec.loader.exec_module(_model_policy)
+derive_runtime_metadata = _model_policy.derive_runtime_metadata
+
 
 def _src(path: str) -> str:
     return (ROOT / path).read_text(encoding="utf-8")
```

## 132. `72950772121d` — Load standalone Bonsai eval through Prism runtime

**Disposition:** retained as history. Where a later commit changes the same policy, the later current-head state is authoritative; no commit was deleted or squashed.

```diff
@@ -197,11 +197,38 @@ def cleanup_worktree(self,d): subprocess.run(['git','worktree','remove','--force
 
 class UnifiedMasterEngine:
     def __init__(self,settings:Optional[EngineSettings]=None):
-        self.settings=settings or EngineSettings(); self.kg=RelationalKnowledgeGraph(self.settings.db_path); self.sandbox=POSIXHardenedSandbox(self.settings.sandbox_timeout_seconds,self.settings.sandbox_max_memory_mb); self.mcp=FastMCPDispatcher(self.sandbox,self.kg); self.lif=NeuromorphicLIFController(); self.drafter=ASTPrefixTrieDrafter(); self.h2o=H2OKVCacheArena(self.settings.h2o_sink_tokens,self.settings.h2o_heavy_tokens,self.settings.h2o_max_budget); self.ogp_projector=GramSchmidtOGPProjector(self.settings.ogp_ortho_tolerance); self.mcts=SymbolicMCTSSearchEngine(self.sandbox); self.model=self.tokenizer=self.moe_manager=self.moe_router=self.grpo_trainer=self.ogp_daemon=None; self._initialize_runtime()
+        self.settings=settings or EngineSettings(); self.kg=RelationalKnowledgeGraph(self.settings.db_path); self.sandbox=POSIXHardenedSandbox(self.settings.sandbox_timeout_seconds,self.settings.sandbox_max_memory_mb); self.mcp=FastMCPDispatcher(self.sandbox,self.kg); self.lif=NeuromorphicLIFController(); self.drafter=ASTPrefixTrieDrafter(); self.h2o=H2OKVCacheArena(self.settings.h2o_sink_tokens,self.settings.h2o_heavy_tokens,self.settings.h2o_max_budget); self.ogp_projector=GramSchmidtOGPProjector(self.settings.ogp_ortho_tolerance); self.mcts=SymbolicMCTSSearchEngine(self.sandbox); self.model=self.tokenizer=self.moe_manager=self.moe_router=self.grpo_trainer=self.ogp_daemon=None; self._runtime_backend=None; self._initialize_runtime()
+    def _load_primary_model(self):
+        path=str(self.settings.mlx_model_path or "")
+        if "Ternary-Bonsai-2-27B-mlx-2bit" in path:
+            # Prism Bonsai-2 requires the runtime bundled in the HF pack. Plain
+            # mlx_lm.load() skips its Hadamard activation transform and can emit
+            # incorrect text without raising, so reuse the app's proven loader.
+            from huggingface_hub import snapshot_download
+            from core.mlx_engine import MLXReasoningBackend
+            source=path if os.path.exists(path) else snapshot_download(repo_id=path)
+            backend=MLXReasoningBackend(
+                model_path=source,
+                model_info={
+                    "name":"Bonsai 2 27B Ternary Multimodal",
+                    "repo_id":path,
+                    "runtime_family":"bonsai2_hadamard",
+                    "input_modalities":["text","image","video"],
+                },
+            )
+            if not backend.load_model():
+                raise RuntimeError("Bonsai-2 bundled MLX runtime failed to load")
+            model=backend.get_training_model()
+            tokenizer=backend.tokenizer
+            if model is None or tokenizer is None:
+                raise RuntimeError("Bonsai-2 bundled runtime exposed no trainable language model/tokenizer")
+            self._runtime_backend=backend
+            return model,tokenizer
+        return load(path)
     def _initialize_runtime(self):
         if not MLX_AVAILABLE:return
         try:
-            self.model,self.tokenizer=load(self.settings.mlx_model_path)
+            self.model,self.tokenizer=self._load_primary_model()
             self.moe_manager=MoEDualBufferManager(self.model,self.settings); self.moe_router=HierarchicalMoERouter(self.model); self.grpo_trainer=GRPOTrainingEngine(self.model,self.tokenizer,self.sandbox)
             if self.settings.enable_awake_ogp_daemon:
                 self.ogp_daemon=ProjectedSleepConsolidationDaemon(self.moe_manager,self.ogp_projector,self.kg,self.tokenizer,self.settings,METAL_STREAM_LOCK); self.ogp_daemon.start()
```

## 133. `031dc53829e1` — Align eval target contract with exact Bonsai repo

**Disposition:** retained as history. Where a later commit changes the same policy, the later current-head state is authoritative; no commit was deleted or squashed.

```diff
@@ -82,7 +82,7 @@ def test_bonsai2_is_the_default_eval_model_and_current_model_is_reported():
     report = _src("eval/_master_4000_base.py")
     assert 'mlx_model_path: str="prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"' in runtime
     assert "_eval_target_model_label" in report
-    assert "Bonsai 2 27B Ternary Multimodal" in report
+    assert "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit" in report
 
 
 def test_top_app_controls_explicitly_include_context_limit_and_memory_watcher():
```

## 134. `e2163c6baa59` — Fix BitNet runtime audit assertion

**Disposition:** retained as history. Where a later commit changes the same policy, the later current-head state is authoritative; no commit was deleted or squashed.

```diff
@@ -58,7 +58,8 @@ def test_bitnet_backend_is_real_and_training_rebuilds_deployment_weights():
     trainer = _src("core/bitnet_rebuild_trainer.py")
     assert '"https://github.com/microsoft/BitNet.git"' in src
     assert '"setup_env.py"' in src
-    assert '"llama-server"' in src
+    assert '"build/bin/llama-server"' in src
+    assert '"build/bin/Release/llama-server.exe"' in src
     assert "def training_ready(self) -> bool:" in src
     assert "BitNetRebuildTrainer" in src
     assert "def train_mini_batch(" in src
```

## Current interpretation after commit 134

- Bonsai 2 remains the requested main/default app model (`model_1`) and the requested standalone 4K-eval target.
- Commits 125–127 temporarily switched standalone eval back to Qwen3.8; commits 128–130 intentionally superseded that and restored Bonsai 2. Both histories are preserved.
- Commit 132 does not change model choice. It makes the existing Bonsai-2 choice use Prism's bundled Hadamard-aware loader before exposing the loaded language module to the unchanged eval/training pipeline.
- Commits 131, 133 and 134 only repair focused audit contracts; they do not alter application model choices or runtime behavior.
- The successful focused audit at commit 134 compiled 228/228 tracked Python files and passed 58/58 focused contracts.
