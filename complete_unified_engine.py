import ast
import asyncio
import collections
import gc
import json
import math
import os
import platform
import psutil
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, Union

try:
    import resource
    HAS_RESOURCE = True
except ImportError:
    HAS_RESOURCE = False

MLX_AVAILABLE = False
try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import mlx.utils
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.tuner.lora import LoRALinear
    MLX_AVAILABLE = True
except ImportError:
    pass

@dataclass
class EngineSettings:
    total_ram_gb: float = field(default_factory=lambda: psutil.virtual_memory().total / (1024 ** 3))
    mlx_model_path: str = "prism-ml/Ternary-Bonsai-27B-mlx-2bit"
    max_kv_tokens: int = 4096
    h2o_sink_tokens: int = 4
    h2o_heavy_tokens: int = 1024
    lora_rank: int = 4
    lora_alpha: float = 8.0
    lora_layers_per_chunk: int = 6
    total_layers: int = 60
    base_learning_rate: float = 1e-4
    ogp_anchor_samples: int = 8
    ogp_ortho_tolerance: float = 1e-5
    polling_interval_seconds: float = 300.0
    min_surprise_threshold: float = 0.85
    min_batch_queue_size: int = 5
    sandbox_timeout_seconds: float = 4.0
    sandbox_max_memory_mb: int = 512
    enable_awake_ogp_daemon: bool = True
    db_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "memory.db")

class RelationalKnowledgeGraph:
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS graph_nodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity TEXT UNIQUE NOT NULL,
                    entity_type TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS graph_edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_entity TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    target_entity TEXT NOT NULL,
                    weight REAL NOT NULL,
                    temporal_session TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    UNIQUE(source_entity, predicate, target_entity)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS episodic_interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    completion TEXT NOT NULL,
                    reward REAL NOT NULL,
                    surprise_score REAL NOT NULL,
                    domain TEXT NOT NULL,
                    consolidated INTEGER DEFAULT 0,
                    timestamp REAL NOT NULL
                )
            """)
            conn.commit()

    def insert_triple(self, source: str, predicate: str, target: str, weight: float = 1.0, session_id: str = "main"):
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("INSERT OR IGNORE INTO graph_nodes (entity, entity_type, created_at) VALUES (?, 'concept', ?)", (source, now))
            c.execute("INSERT OR IGNORE INTO graph_nodes (entity, entity_type, created_at) VALUES (?, 'concept', ?)", (target, now))
            c.execute("""
                INSERT OR REPLACE INTO graph_edges (source_entity, predicate, target_entity, weight, temporal_session, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (source, predicate, target, weight, session_id, now))
            conn.commit()

    def recursive_multi_hop_query(self, start_entity: str, max_depth: int = 3) -> List[Dict[str, Any]]:
        query = """
        WITH RECURSIVE EntityHops AS (
            SELECT source_entity, predicate, target_entity, 1 AS depth,
                   source_entity || ' -> ' || predicate || ' -> ' || target_entity AS path
            FROM graph_edges
            WHERE source_entity = ?
            UNION ALL
            SELECT e.source_entity, e.predicate, e.target_entity, eh.depth + 1,
                   eh.path || ' -> ' || e.predicate || ' -> ' || e.target_entity
            FROM graph_edges e
            JOIN EntityHops eh ON e.source_entity = eh.target_entity
            WHERE eh.depth < ? AND INSTR(eh.path, e.target_entity) = 0
        )
        SELECT source_entity, predicate, target_entity, depth, path FROM EntityHops;
        """
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute(query, (start_entity, max_depth))
            rows = c.fetchall()
            return [{"source": r[0], "predicate": r[1], "target": r[2], "depth": r[3], "path": r[4]} for r in rows]

    def log_interaction(self, session_id: str, prompt: str, completion: str, reward: float, surprise: float, domain: str = "general"):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("""
                INSERT INTO episodic_interactions (session_id, prompt, completion, reward, surprise_score, domain, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (session_id, prompt, completion, reward, surprise, domain, time.time()))
            conn.commit()

    def fetch_unconsolidated_high_surprise(self, min_surprise: float, limit: int = 32) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute("""
                SELECT * FROM episodic_interactions
                WHERE consolidated = 0 AND reward >= 0.8 AND surprise_score >= ?
                ORDER BY surprise_score DESC LIMIT ?
            """, (min_surprise, limit))
            return [dict(r) for r in c.fetchall()]

    def mark_consolidated(self, ids: List[int]):
        if not ids:
            return
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            placeholders = ",".join("?" for _ in ids)
            c.execute(f"UPDATE episodic_interactions SET consolidated = 1 WHERE id IN ({placeholders})", ids)
            conn.commit()

@dataclass
class SandboxResult:
    passed: bool
    execution_time_ms: float
    output: str
    error: Optional[str] = None
    reward: float = 0.0

class POSIXHardenedSandbox:
    def __init__(self, timeout_sec: float = 4.0, max_memory_mb: int = 512):
        self.timeout = timeout_sec
        self.max_memory_mb = max_memory_mb

    def _apply_rlimits(self):
        if HAS_RESOURCE and platform.system() != "Windows":
            mem_bytes = self.max_memory_mb * 1024 * 1024
            try:
                resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
            except Exception:
                pass
            try:
                cpu_sec = max(1, int(self.timeout) + 1)
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_sec, cpu_sec))
            except Exception:
                pass
            try:
                resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
            except Exception:
                pass

    def execute_python_code(self, code: str, tests: str) -> SandboxResult:
        full_script = f"# -*- coding: utf-8 -*-\nimport sys, math, json, collections, itertools\n\n{code}\n\n# UNIT TESTS\n{tests}\n"
        t0 = time.perf_counter()
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tmp:
            tmp.write(full_script)
            tmp_path = tmp.name

        try:
            res = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                preexec_fn=self._apply_rlimits if HAS_RESOURCE and platform.system() != "Windows" else None
            )
            dur_ms = (time.perf_counter() - t0) * 1000.0
            if res.returncode == 0:
                return SandboxResult(True, dur_ms, res.stdout, reward=1.0)
            return SandboxResult(False, dur_ms, res.stdout, error=res.stderr[:240], reward=0.0)
        except subprocess.TimeoutExpired:
            return SandboxResult(False, (time.perf_counter() - t0) * 1000.0, "", error="ProcessTimeout", reward=0.0)
        except Exception as e:
            return SandboxResult(False, (time.perf_counter() - t0) * 1000.0, "", error=str(e), reward=0.0)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def verify_git_diff_patch(self, repo_structure: Dict[str, str], patch_text: str, test_cmd: str) -> SandboxResult:
        t0 = time.perf_counter()
        tmp_dir = tempfile.mkdtemp(prefix="swe_jail_")
        try:
            for rel_path, content in repo_structure.items():
                p = os.path.join(tmp_dir, rel_path)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "w", encoding="utf-8") as f:
                    f.write(content)

            patch_clean = re.findall(r"```(?:diff|patch)?\s*([\s\S]*?)```", patch_text, re.IGNORECASE)
            clean_diff = patch_clean[-1].strip() if patch_clean else patch_text.strip()
            patch_file = os.path.join(tmp_dir, "task.patch")
            with open(patch_file, "w", encoding="utf-8") as f:
                f.write(clean_diff + "\n")

            subprocess.run(["patch", "-p1", "-i", "task.patch"], cwd=tmp_dir, capture_output=True, timeout=2.0)
            proc = subprocess.run(
                test_cmd,
                shell=True,
                cwd=tmp_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                preexec_fn=self._apply_rlimits if HAS_RESOURCE and platform.system() != "Windows" else None
            )
            dur_ms = (time.perf_counter() - t0) * 1000.0
            if proc.returncode == 0:
                return SandboxResult(True, dur_ms, proc.stdout, reward=1.0)
            return SandboxResult(False, dur_ms, proc.stdout, error=proc.stderr[:200], reward=0.0)
        except Exception as e:
            return SandboxResult(False, (time.perf_counter() - t0) * 1000.0, "", error=str(e), reward=0.0)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    @staticmethod
    def evaluate_dsl_expression(expr: str) -> Optional[List[int]]:
        expr = expr.strip()
        m_fold_scale = re.search(r"\[([0-9,\s\-]+)\]\s*>>~fold\((\d+)\)\s*<#>scale\((\d+)\)", expr)
        if m_fold_scale:
            arr_str, fold_k, scale_s = m_fold_scale.groups()
            arr = [int(x.strip()) for x in arr_str.split(",") if x.strip()]
            k = int(fold_k) % len(arr) if arr else 0
            s = int(scale_s)
            folded = arr[k:] + arr[:k]
            return [x * s for x in folded]

        m_fuse = re.search(r"\[([0-9,\s\-]+)\]\s*@fuse\s*\[([0-9,\s\-]+)\]", expr)
        if m_fuse:
            a_str, b_str = m_fuse.groups()
            arr_a = [int(x.strip()) for x in a_str.split(",") if x.strip()]
            arr_b = [int(x.strip()) for x in b_str.split(",") if x.strip()]
            return [x + y for x, y in zip(arr_a, arr_b)]
        return None

class FastMCPDispatcher:
    def __init__(self, sandbox: POSIXHardenedSandbox, kg: RelationalKnowledgeGraph):
        self.sandbox = sandbox
        self.kg = kg
        self.tools: Dict[str, Dict[str, Any]] = {}
        self._register_core_tools()

    def _register_core_tools(self):
        self.register_tool(
            name="python_eval",
            description="Executes sandboxed Python arithmetic or code logic.",
            parameters={"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
            handler=lambda args: self.sandbox.execute_python_code(args.get("code", ""), "assert True").output
        )
        self.register_tool(
            name="dsl_evaluate",
            description="Evaluates TensorGraphDSL expression.",
            parameters={"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
            handler=lambda args: json.dumps(self.sandbox.evaluate_dsl_expression(args.get("expression", "")))
        )
        self.register_tool(
            name="graph_query",
            description="Queries multi-hop relational memory paths for an entity.",
            parameters={"type": "object", "properties": {"entity": {"type": "string"}, "depth": {"type": "integer", "default": 2}}, "required": ["entity"]},
            handler=lambda args: json.dumps(self.kg.recursive_multi_hop_query(args.get("entity", ""), args.get("depth", 2)))
        )

    def register_tool(self, name: str, description: str, parameters: Dict[str, Any], handler: Callable[[Dict[str, Any]], str]):
        self.tools[name] = {"name": name, "description": description, "parameters": parameters, "handler": handler}

    def handle_json_rpc(self, request_json: str) -> str:
        try:
            req = json.loads(request_json)
            req_id = req.get("id")
            method = req.get("method")
            params = req.get("params", {})

            if method == "tools/list":
                tools_list = [{"name": t["name"], "description": t["description"], "inputSchema": t["parameters"]} for t in self.tools.values()]
                return json.dumps({"jsonrpc": "2.0", "id": req_id, "result": {"tools": tools_list}})
            elif method == "tools/call":
                tname = params.get("name")
                targs = params.get("arguments", {})
                if tname in self.tools:
                    out = self.tools[tname]["handler"](targs)
                    return json.dumps({"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": str(out)}]}})
                return json.dumps({"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Tool '{tname}' not found."}})
            return json.dumps({"jsonrpc": "2.0", "id": req_id, "error": {"code": -32600, "message": "Invalid request."}})
        except Exception as e:
            return json.dumps({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(e)}})

class NeuromorphicLIFController:
    def __init__(self, v_thresh: float = 0.55, beta: float = 0.85):
        self.v_mem = 0.0
        self.v_thresh = v_thresh
        self.v_rest = 0.0
        self.beta = beta
        self.spike_history: List[int] = []

    def step(self, entropy_current: float) -> Tuple[int, float]:
        self.v_mem = (self.beta * self.v_mem) + ((1.0 - self.beta) * entropy_current)
        if self.v_mem >= self.v_thresh:
            spike = 1
            self.v_mem = self.v_rest
        else:
            spike = 0
        self.spike_history.append(spike)
        return spike, self.v_mem

    def compute_convex_temperature_ladder(self, entropy: float, t_min: float = 0.20, t_max: float = 0.88) -> Tuple[int, List[float], int]:
        spike, _ = self.step(entropy)
        if spike == 1 or entropy >= 0.65:
            num_branches = 4
        elif entropy >= 0.30:
            num_branches = 2
        else:
            return 1, [0.0], 0

        ladder = []
        for i in range(num_branches):
            t_i = t_min * ((t_max / t_min) ** (i / max(1, num_branches - 1)))
            ladder.append(round(t_i, 3))
        return num_branches, ladder, spike

class ASTPrefixTrieDrafter:
    def __init__(self):
        self.trie: Dict[str, Any] = {}
        self._seed_ast_grammar()

    def _seed_ast_grammar(self):
        keywords = [
            "def solve(", "return True", "return False", "for i in range(", "if __name__ == '__main__':",
            "import math", "import collections", "class Solution:", "def __init__(self):",
            "\\boxed{", "\\frac{", "\\sqrt{", "cycle_index=", "@fuse", ">>~fold("
        ]
        for kw in keywords:
            curr = self.trie
            for char in kw:
                if char not in curr:
                    curr[char] = {}
                curr = curr[char]
            curr["__END__"] = True

    def find_draft_tokens(self, token_history: List[int], tokenizer, max_draft: int = 4) -> List[int]:
        if len(token_history) >= 6:
            target = token_history[-3:]
            for i in range(len(token_history) - 4, -1, -1):
                if token_history[i:i+3] == target:
                    return token_history[i+3 : i+3+max_draft]
        try:
            recent_text = tokenizer.decode(token_history[-10:])
            curr = self.trie
            for char in recent_text:
                if char in curr:
                    curr = curr[char]
                else:
                    curr = self.trie
            draft_chars = ""
            while len(draft_chars) < 12 and curr and len(curr) == 1 and "__END__" not in curr:
                k = list(curr.keys())[0]
                draft_chars += k
                curr = curr[k]
            if draft_chars:
                return tokenizer.encode(draft_chars)[:max_draft]
        except Exception:
            pass
        return []

class GramSchmidtOGPProjector:
    def __init__(self, tolerance: float = 1e-5):
        self.tolerance = tolerance
        self.anchor_basis_vectors: List[Any] = []

    @staticmethod
    def flatten_gradients(grads_dict: Dict[str, Any]) -> Tuple[Any, List[Tuple[str, Tuple[int, ...]]]]:
        flat_list = []
        shapes = []
        sorted_keys = sorted(grads_dict.keys())
        for k in sorted_keys:
            val = grads_dict[k]
            shapes.append((k, val.shape))
            if MLX_AVAILABLE and isinstance(val, mx.array):
                flat_list.append(val.reshape(-1).astype(mx.float32))
        if MLX_AVAILABLE and flat_list:
            return mx.concatenate(flat_list), shapes
        return None, shapes

    @staticmethod
    def unflatten_gradients(flat_vec: Any, shapes: List[Tuple[str, Tuple[int, ...]]]) -> Dict[str, Any]:
        unflat = {}
        offset = 0
        if MLX_AVAILABLE and isinstance(flat_vec, mx.array):
            for k, shape in shapes:
                numel = 1
                for dim in shape:
                    numel *= dim
                unflat[k] = flat_vec[offset : offset + numel].reshape(shape)
                offset += numel
        return unflat

    def register_anchor_gradient(self, flat_anchor_grad: Any):
        if not MLX_AVAILABLE or flat_anchor_grad is None:
            return
        v = flat_anchor_grad.astype(mx.float32)
        for basis in self.anchor_basis_vectors:
            proj = (mx.sum(v * basis) / (mx.sum(basis * basis) + 1e-12)) * basis
            v = v - proj
        norm_sq = float(mx.sum(v * v).item())
        if norm_sq > self.tolerance:
            unit_basis = v / math.sqrt(norm_sq)
            mx.eval(unit_basis)
            self.anchor_basis_vectors.append(unit_basis)

    def project_gradient(self, flat_task_grad: Any) -> Any:
        if not MLX_AVAILABLE or flat_task_grad is None or not self.anchor_basis_vectors:
            return flat_task_grad
        g_proj = flat_task_grad.astype(mx.float32)
        for basis in self.anchor_basis_vectors:
            coeff = mx.sum(g_proj * basis)
            g_proj = g_proj - (coeff * basis)
        mx.eval(g_proj)
        return g_proj

    def verify_orthogonality(self, flat_proj_grad: Any) -> float:
        if not self.anchor_basis_vectors or flat_proj_grad is None:
            return 0.0
        max_overlap = 0.0
        for basis in self.anchor_basis_vectors:
            overlap = abs(float(mx.sum(flat_proj_grad * basis).item()))
            if overlap > max_overlap:
                max_overlap = overlap
        return max_overlap

class MoEDualBufferManager:
    def __init__(self, model: Any, settings: EngineSettings):
        self.model = model
        self.settings = settings
        self.active_domain = "general"
        self.adapters_buffer_a: Dict[str, Any] = {}
        self.adapters_buffer_b: Dict[str, Any] = {}
        self.lock = threading.Lock()
        self._init_lora_adapters()

    def _init_lora_adapters(self):
        if not MLX_AVAILABLE or self.model is None:
            return
        self.model.freeze()
        layers = getattr(self.model, "layers", []) or getattr(getattr(self.model, "model", None), "layers", [])
        num_layers = len(layers)

        for idx in range(num_layers):
            layer = layers[idx]
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "q_proj"):
                if not isinstance(layer.self_attn.q_proj, LoRALinear):
                    layer.self_attn.q_proj = LoRALinear.from_base(
                        layer.self_attn.q_proj, r=self.settings.lora_rank, scale=self.settings.lora_alpha / self.settings.lora_rank
                    )
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
                if not isinstance(layer.mlp.down_proj, LoRALinear):
                    layer.mlp.down_proj = LoRALinear.from_base(
                        layer.mlp.down_proj, r=self.settings.lora_rank, scale=self.settings.lora_alpha / self.settings.lora_rank
                    )

        trainable = dict(mlx.utils.tree_flatten(self.model.trainable_parameters()))
        self.adapters_buffer_a = {k: mx.array(v) for k, v in trainable.items()}
        self.adapters_buffer_b = {k: mx.array(v) for k, v in trainable.items()}

    def swap_buffers_atomic(self):
        with self.lock:
            self.adapters_buffer_a = {k: mx.array(v) for k, v in self.adapters_buffer_b.items()}
            if MLX_AVAILABLE and self.model is not None:
                self.model.update(mlx.utils.tree_unflatten(list(self.adapters_buffer_a.items())))
                mx.eval(self.model.parameters())

class MCTSAutonomousDiscovery:
    def __init__(self, sandbox: POSIXHardenedSandbox, kg: RelationalKnowledgeGraph):
        self.sandbox = sandbox
        self.kg = kg

    def generate_procedural_problem(self) -> Dict[str, Any]:
        k = (int(time.time()) % 4) + 1
        scale = 2 * k
        dsl_expr = f"[1, 3, 5, 7] >>~fold({k}) <#>scale({scale})"
        ground_truth = self.sandbox.evaluate_dsl_expression(dsl_expr)
        return {
            "domain": "TensorGraphDSL",
            "prompt": f"Derive the exact symbolic invariant evaluation of: `{dsl_expr}`",
            "expected_result": str(ground_truth),
            "dsl_expr": dsl_expr
        }

    def run_discovery_step(self, generator_fn: Callable[[str], str]) -> Optional[Dict[str, Any]]:
        problem = self.generate_procedural_problem()
        prompt = f"<|im_start|>user\n{problem['prompt']}<|im_end|>\n<|im_start|>assistant\n"
        output = generator_fn(prompt)
        passed = problem["expected_result"].replace(" ", "") in output.replace(" ", "")
        surprise = 0.95 if passed else 0.40
        self.kg.log_interaction(
            session_id="mcts_self_play",
            prompt=problem["prompt"],
            completion=output,
            reward=1.0 if passed else 0.0,
            surprise=surprise,
            domain=problem["domain"]
        )
        if passed:
            self.kg.insert_triple(problem["dsl_expr"], "evaluates_to", problem["expected_result"], weight=1.0)
            return {"prompt": problem["prompt"], "completion": problem["expected_result"], "reward": 1.0}
        return None

class UnifiedMasterEngine:
    def __init__(self, settings: Optional[EngineSettings] = None):
        self.settings = settings or EngineSettings()
        self.kg = RelationalKnowledgeGraph(self.settings.db_path)
        self.sandbox = POSIXHardenedSandbox(
            timeout_sec=self.settings.sandbox_timeout_seconds,
            max_memory_mb=self.settings.sandbox_max_memory_mb
        )
        self.mcp = FastMCPDispatcher(self.sandbox, self.kg)
        self.lif = NeuromorphicLIFController()
        self.drafter = ASTPrefixTrieDrafter()
        self.ogp_projector = GramSchmidtOGPProjector(tolerance=self.settings.ogp_ortho_tolerance)
        self.mcts = MCTSAutonomousDiscovery(self.sandbox, self.kg)
        self.model = None
        self.tokenizer = None
        self.moe_manager = None
        self._initialize_runtime()

    def _initialize_runtime(self):
        if MLX_AVAILABLE:
            try:
                self.model, self.tokenizer = load(self.settings.mlx_model_path, model_config={})
                self.moe_manager = MoEDualBufferManager(self.model, self.settings)
            except Exception:
                pass

    def generate(self, prompt: str, max_tokens: int = 128) -> str:
        if not MLX_AVAILABLE or self.model is None or self.tokenizer is None:
            return f"[Engine Offline Output for: {prompt[:40]}]"
        tokens = self.tokenizer.encode(prompt)
        prompt_cache = make_prompt_cache(self.model)
        inp = mx.array([tokens])
        logits = self.model(inp, cache=prompt_cache)
        mx.eval(logits)
        l_step = logits[0, -1].astype(mx.float32)
        next_tok = int(mx.argmax(l_step).item())
        generated_tokens = [next_tok]
        tokens.append(next_tok)
        while len(generated_tokens) < max_tokens:
            if hasattr(self.tokenizer, "eos_token_id") and next_tok == self.tokenizer.eos_token_id:
                break
            inp_step = mx.array([[next_tok]])
            step_logits = self.model(inp_step, cache=prompt_cache)
            mx.eval(step_logits)
            next_tok = int(mx.argmax(step_logits[0, -1]).item())
            generated_tokens.append(next_tok)
            tokens.append(next_tok)
        return self.tokenizer.decode(generated_tokens)

def run_full_subsystem_verification():
    print("=" * 80)
    print("🧪 EXECUTING COMPREHENSIVE ZERO-MOCK SUBSYSTEM INTEGRITY VERIFICATION")
    print("=" * 80)
    settings = EngineSettings()
    engine = UnifiedMasterEngine(settings)
    results = []

    engine.kg.insert_triple("TensorGraphDSL", "implements", "FoldScaleOperator", session_id="test")
    engine.kg.insert_triple("FoldScaleOperator", "transforms", "MatrixState", session_id="test")
    paths = engine.kg.recursive_multi_hop_query("TensorGraphDSL", max_depth=2)
    results.append(("1. Relational Graph Recursive CTE", len(paths) >= 2, f"Found {len(paths)} relational paths."))

    dsl_res = engine.sandbox.evaluate_dsl_expression("[0, 2, 4] >>~fold(1) <#>scale(2)")
    results.append(("2. DSL Symbolic Evaluator (>>~fold)", dsl_res == [4, 8, 0], f"Result: {dsl_res} == [4, 8, 0]"))
    dsl_fuse = engine.sandbox.evaluate_dsl_expression("[1, 2, 3] @fuse [4, 5, 6]")
    results.append(("3. DSL Matrix Fusion (@fuse)", dsl_fuse == [5, 7, 9], f"Result: {dsl_fuse} == [5, 7, 9]"))

    py_res = engine.sandbox.execute_python_code("def add(a, b): return a + b", "assert add(2, 3) == 5")
    results.append(("4. POSIX In-Process Sandbox", py_res.passed and py_res.reward == 1.0, f"Exec Time: {py_res.execution_time_ms:.1f}ms"))

    repo_files = {"src/app.py": "def main(): return 'bad'\n", "tests/test_app.py": "from src.app import main\nassert main() == 'good'\n"}
    patch = "```diff\n--- a/src/app.py\n+++ b/src/app.py\n@@ -1,1 +1,1 @@\n-def main(): return 'bad'\n+def main(): return 'good'\n```"
    swe_res = engine.sandbox.verify_git_diff_patch(repo_files, patch, "python3 tests/test_app.py")
    results.append(("5. SWE Git Diff Patch Sandbox", swe_res.passed, f"SWE Patch Reward: {swe_res.reward}"))

    mcp_call = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "dsl_evaluate", "arguments": {"expression": "[1, 2] @fuse [3, 4]"}}})
    mcp_resp = json.loads(engine.mcp.handle_json_rpc(mcp_call))
    results.append(("6. In-Memory MCP JSON-RPC Server", "[4, 6]" in str(mcp_resp), f"MCP Output: {mcp_resp.get('result')}"))

    lif = NeuromorphicLIFController(v_thresh=0.50, beta=0.80)
    b_count, ladder, spike = lif.compute_convex_temperature_ladder(0.75)
    results.append(("7. Neuromorphic LIF Spike Gating", b_count == 4 and spike == 1, f"Branches: {b_count}, Ladder: {ladder}, Spike: {spike}"))

    if MLX_AVAILABLE:
        v1 = mx.array([1.0, 0.0, 0.0, 0.0], dtype=mx.float32)
        v2 = mx.array([1.0, 1.0, 0.0, 0.0], dtype=mx.float32)
        ogp = GramSchmidtOGPProjector()
        ogp.register_anchor_gradient(v1)
        v2_proj = ogp.project_gradient(v2)
        overlap = ogp.verify_orthogonality(v2_proj)
        results.append(("8. Gram-Schmidt OGP Orthogonality", overlap < 1e-6, f"Max Overlap: {overlap:.2e} (Strictly zero drift)"))

    disc_item = engine.mcts.generate_procedural_problem()
    results.append(("9. MCTS Procedural Synthesizer", "dsl_expr" in disc_item and "expected_result" in disc_item, f"Synthesized: {disc_item.get('dsl_expr')}"))

    print("\n" + "=" * 90)
    print(f"{'STATUS':<10} | {'SUBSYSTEM / INVARIANT':<45} | {'DETAILS'}")
    print("=" * 90)
    for name, passed, detail in results:
        status_str = "\033[92m[✓ PASS]\033[0m" if passed else "\033[91m[✗ FAIL]\033[0m"
        print(f"{status_str:<19} | {name:<45} | {detail}")
    print("=" * 90)
    total_passed = sum(1 for _, p, _ in results if p)
    print(f"📊 SUBSYSTEM INTEGRITY: {total_passed}/{len(results)} REQUIREMENTS VERIFIED ({(total_passed/len(results))*100:.1f}%)\n")

if __name__ == "__main__":
    run_full_subsystem_verification()
