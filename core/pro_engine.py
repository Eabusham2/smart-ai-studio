"""Rich cross-platform Pro engine with recovered LIF/MoE/OGP architecture layered on top."""
from __future__ import annotations
import threading
from core._pro_engine_base import *
from core import _pro_engine_base as _base
from core.lif_gating import LIFNeuronState, convex_temperature_ladder
from core.drafter import GrammarGuidedASTTrieDrafter
from core.round_robin_lora import RoundRobinLoRAController
from consolidation.dual_buffer import DualAdapterBufferManager
from consolidation.moe_dual_buffer import HierarchicalMoELoRAManager
from consolidation.projected_daemon import GramSchmidtOGPProjector, ProjectedSleepConsolidationDaemon
from memory.db import EpisodicMemoryDB
from memory.knowledge_graph import RelationalKnowledgeGraph


def _continuous_ladder(n, t_min=.20, t_max=.88, gamma=1.35):
    return convex_temperature_ladder(n, t_min, t_max)

_base.get_ladder_temperatures = _continuous_ladder
get_ladder_temperatures = _continuous_ladder


class _LIFRouterAdapter:
    def __init__(self, base_router, lif):
        self.base_router = base_router
        self.lif = lif

    def __getattr__(self, name):
        return getattr(self.base_router, name)

    def route(self, entropy, has_test_cases=False):
        x = max(0.0, min(1.0, float(entropy)))
        lif_tier, _, _ = self.lif.determine_branch_budget(x)
        if lif_tier <= 1:
            branches = int(getattr(self.base_router, "instant_branches", 1))
        elif lif_tier == 2:
            branches = int(getattr(self.base_router, "pro_branches_mid", 8))
        else:
            branches = int(getattr(self.base_router, "pro_branches_high", 16))
        if has_test_cases:
            branches = max(branches, int(getattr(self.base_router, "pro_branches_high", 16)))
        if branches <= 1:
            return "Instant (N=1)", 1
        return ("Pro-RLVR" if has_test_cases else "Pro-Search") + f" (N={branches})", branches


class _MemoryReplayBridge:
    def __init__(self, db):
        self.db = db

    def fetch_unconsolidated_high_surprise(self, min_surprise, limit=32):
        rows = self.db.fetch_surprise_replay_data(limit=limit, unconsolidated_only=True)
        return [r for r in rows if float(r.get("surprise_score", 0)) >= float(min_surprise)]

    def mark_consolidated(self, ids):
        self.db.mark_consolidated([int(x) for x in ids if isinstance(x, (int, float))])


class ProReasoningEngine(_base.ProReasoningEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lif_state = LIFNeuronState(v_thresh=.55, beta=.85)
        if getattr(self.settings, "enable_neuromorphic_lif", True):
            self.router = _LIFRouterAdapter(self.router, self.lif_state)
        self.grammar_drafter = GrammarGuidedASTTrieDrafter(n_gram=3, max_draft=3)
        try:
            self.speculative_engine.pld_drafter = self.grammar_drafter
        except Exception:
            pass
        self.knowledge_graph = RelationalKnowledgeGraph(self.settings.database_path)
        self.round_robin_lora = None
        self.dual_adapter_buffer = None
        self.moe_lora_manager = None
        self.ogp_daemon = None
        self._continual_lock = threading.RLock()

    def _continual_stream_lock(self):
        backend = getattr(self, "mlx_backend", None)
        return getattr(backend, "stream_lock", self._continual_lock)

    def _start_ogp_daemon(self):
        if self.ogp_daemon is not None and self.ogp_daemon.is_alive():
            return self.ogp_daemon
        if not self.moe_lora_manager or not self.round_robin_lora:
            return None
        backend = getattr(self, "mlx_backend", None)
        if backend is None or getattr(backend, "tokenizer", None) is None:
            return None
        replay = _MemoryReplayBridge(EpisodicMemoryDB(self.settings.database_path))
        projector = GramSchmidtOGPProjector(float(getattr(self.settings, "ogp_ortho_tolerance", 1e-5)))
        anchor_texts=[
            "Write a correct Python function that returns x + 1.",
            "What is 17 + 25? Answer 42.",
            "State that water freezes at 0 degrees Celsius at standard pressure.",
            "If all mammals are warm-blooded and whales are mammals, whales are warm-blooded.",
            "Return valid JSON with keys name and arguments.",
            "Summarize the previous user request without changing its meaning.",
            "Compute [1, 3, 5] rotated left by one: [3, 5, 1].",
            "A secure code sandbox must reject a failing assertion.",
        ]
        self.ogp_daemon = ProjectedSleepConsolidationDaemon(
            self.moe_lora_manager,
            projector,
            replay,
            backend.tokenizer,
            self.settings,
            stream_lock=self._continual_stream_lock(),
            round_robin_controller=self.round_robin_lora,
            dual_buffer=self.dual_adapter_buffer,
            anchor_texts=anchor_texts,
        )
        self.ogp_daemon.initialize_anchor_basis()
        self.ogp_daemon.start()
        return self.ogp_daemon

    def set_awake_ogp_enabled(self, enabled: bool) -> bool:
        enabled = bool(enabled)
        try:
            self.settings.enable_awake_ogp_daemon = enabled
        except Exception:
            pass
        if not enabled:
            if self.ogp_daemon is not None:
                try:
                    self.ogp_daemon.stop()
                except Exception:
                    pass
                self.ogp_daemon = None
            return False
        return self._start_ogp_daemon() is not None

    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        if (
            result.get("status") == "loaded"
            and result.get("backend") == "mlx"
            and self.mlx_backend
            and self.mlx_backend.model is not None
        ):
            try:
                self.grammar_drafter.bind_tokenizer(self.mlx_backend.tokenizer)
                self.speculative_engine.pld_drafter = self.grammar_drafter
                self.speculative_engine.target_model = self.mlx_backend.model
                self.speculative_engine.tokenizer = self.mlx_backend.tokenizer

                rr = RoundRobinLoRAController(
                    self.mlx_backend.model,
                    rank=int(getattr(self.settings, "lora_rank", 4)),
                    alpha=float(getattr(self.settings, "lora_alpha", 8.0)),
                    chunk_size=int(getattr(self.settings, "lora_chunk_size", 6)),
                )
                rr.attach_all_layers()
                self.round_robin_lora = rr

                if rr.targets:
                    lock = self._continual_stream_lock()
                    self.dual_adapter_buffer = DualAdapterBufferManager(self.mlx_backend.model, rr, lock)
                    self.moe_lora_manager = HierarchicalMoELoRAManager(
                        self.mlx_backend.model,
                        rr,
                        getattr(self.settings, "moe_adapter_bank_dir", "adapter_banks"),
                    )
                    if getattr(self.settings, "enable_awake_ogp_daemon", True):
                        self._start_ogp_daemon()
            except Exception as exc:
                result["continual_learning_warning"] = f"{type(exc).__name__}: {exc}"
        return result

    def unload_model(self):
        if self.ogp_daemon is not None:
            try:
                self.ogp_daemon.stop()
            except Exception:
                pass
            self.ogp_daemon = None
        self.round_robin_lora = None
        self.dual_adapter_buffer = None
        self.moe_lora_manager = None
        return super().unload_model()

    def _route_domain(self, prompt: str):
        if self.moe_lora_manager is None:
            return None
        try:
            return self.moe_lora_manager.route_and_activate(prompt or "")
        except Exception:
            return None

    def solve(self, *args, **kwargs):
        prompt = kwargs.get("prompt") or (args[0] if args else "")
        lock = self._continual_stream_lock()
        with lock:
            domain = self._route_domain(prompt)
            response, meta = super().solve(*args, **kwargs)
        backend = self.mlx_backend if self.mlx_backend and getattr(self.mlx_backend, "is_mlx_available", False) else None
        meta["tok_speed"] = float(getattr(backend, "last_tok_per_sec", 0.0) or 0.0)
        meta["lif_spikes"] = self.lif_state.spike_count
        meta["lif_membrane"] = self.lif_state.v_mem
        meta["moe_expert"] = domain or getattr(self.moe_lora_manager, "active_domain", None)
        meta["ogp"] = self.ogp_daemon.status() if self.ogp_daemon is not None else {"running": False}
        return response, meta

    def stream_solve(self, *args, **kwargs):
        prompt = kwargs.get("prompt") or (args[0] if args else "")
        lock = self._continual_stream_lock()
        with lock:
            self._route_domain(prompt)
            try:
                entropy = float(self.calculate_token_entropy(prompt))
            except Exception:
                entropy = 0.0
        threshold = float(getattr(self.settings, "entropy_low_threshold", .25))
        if self.is_model_loaded and entropy >= threshold:
            call_kwargs = dict(kwargs)
            call_kwargs["prompt"] = prompt
            response, meta = self.solve(**call_kwargs)
            self.last_result_metadata = meta
            if response:
                yield response
            return

        generated=[]; t0=time.perf_counter()
        with lock:
            for chunk in super().stream_solve(*args, **kwargs):
                generated.append(chunk); yield chunk
        elapsed=max(.001,time.perf_counter()-t0)
        backend=self.mlx_backend if self.mlx_backend and getattr(self.mlx_backend,"is_mlx_available",False) else None
        measured=float(getattr(backend,"last_tok_per_sec",0.0) or 0.0)
        self.last_result_metadata={
            "mode":"Instant Stream (N=1)","entropy":entropy,"branch_count":1,
            "tok_speed":measured,"execution_time_ms":elapsed*1000.0,
            "lif_spikes":self.lif_state.spike_count,"lif_membrane":self.lif_state.v_mem,
            "moe_expert":getattr(self.moe_lora_manager,"active_domain",None),
            "ogp":self.ogp_daemon.status() if self.ogp_daemon is not None else {"running":False},
        }
