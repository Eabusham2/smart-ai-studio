"""Benchmark compatibility exports merged with the recovered architecture."""
from __future__ import annotations
from eval._run_studio_compat import *
from memory.knowledge_graph import RelationalKnowledgeGraph as _Graph
from core.swe_sandbox import HardenedSWESandbox as _Sandbox
from core.mcp_dispatcher import InProcessMCPDispatcher as _MCP
from core.lif_gating import LIFNeuronState as _LIF, convex_temperature_ladder
from core.drafter import GrammarGuidedASTTrieDrafter as _Drafter
from core.h2o_cache import H2OAttentionSinkPolicy as _H2O
from core.mcts_discovery import ProceduralMCTSDiscovery as _MCTS
from core.grpo_engine import GroupRelativePolicyOptimizer
from core.round_robin_lora import RoundRobinLoRAController
from consolidation.dual_buffer import DualAdapterBufferManager
from consolidation.moe_dual_buffer import HierarchicalMoELoRAManager

RelationalKnowledgeGraph = _Graph
POSIXHardenedSandbox = _Sandbox
FastMCPDispatcher = _MCP
ASTPrefixTrieDrafter = _Drafter

class NeuromorphicLIFController(_LIF):
    def compute_convex_temperature_ladder(self, entropy, t_min=.20, t_max=.88):
        n, ladder, spike = self.determine_branch_budget(entropy, t_min, t_max)
        return n, ladder, spike

class H2OKVCacheArena(_H2O):
    def __init__(self, sink_size=4, heavy_size=1024, max_budget=2048):
        super().__init__(sink_tokens=sink_size, heavy_tokens=heavy_size, recent_tokens=32, max_budget=max_budget)

class SymbolicMCTSSearchEngine(_MCTS):
    pass

def _merged_dialogue_ingest(self):
    rows = [
        ("ASUS ROG GT-BE19000", "runs_service", "AdGuard Home DNS", "network_session"),
        ("AdGuard Home DNS", "hosted_in", "Portainer Docker AI Board", "network_session"),
        ("BD PROCHOT Sensor", "state_decision", "Disabled via ThrottleStop", "hardware_session"),
        ("ROG Z790 Motherboard", "paired_with", "Thermal Grizzly Contact Frame", "hardware_session"),
        ("Ternary-Bonsai-27B", "quantization_format", "1.58-bit ternary MLX", "ml_architecture"),
        ("Omni-agi Engine", "combines", "Spiking Neural Networks & Liquid Networks", "ml_architecture"),
        ("3.6TB BitLocker Partition", "recovered_via", "DiskGenius Sector Editing & repair-bde", "recovery_session"),
        ("banana-mcp", "deployed_on", "Vercel Serverless Handler", "mcp_session"),
        ("banana-mcp", "integrates_with", "Zapier Claude Google Gemini MCP", "mcp_session"),
        ("MLX Metal", "uses", "Apple unified memory", "ml_architecture"),
        ("TensorGraphDSL", "supports", "fold scale fuse", "eval"),
    ]
    for s,p,t,session in rows:
        self.kg.insert_triple(s,p,t,1.0,session_id=session)
        try:
            self.kg.log_interaction(session_id=session,prompt=f"What is the stored developer fact about {s}?",completion=f"{s} {p.replace('_', ' ')} {t}",reward=1.0,surprise=0.95,domain="lore")
        except Exception:pass
    return len(rows)
DialogueTimelineGraphIngester.ingest_developer_sessions = _merged_dialogue_ingest

_original_engine_init = UnifiedMasterEngine.__init__

def _merged_engine_init(self, settings=None):
    _original_engine_init(self, settings)
    self.kg = RelationalKnowledgeGraph(self.settings.db_path)
    self.sandbox = POSIXHardenedSandbox(self.settings.sandbox_timeout_seconds, self.settings.sandbox_max_memory_mb)
    self.mcp = FastMCPDispatcher(self.sandbox, self.kg)
    self.lif = NeuromorphicLIFController(v_thresh=.55, beta=.85)
    self.drafter = ASTPrefixTrieDrafter(n_gram=3, max_draft=3, tokenizer=self.tokenizer)
    self.h2o = H2OKVCacheArena(sink_size=getattr(self.settings,"h2o_sink_tokens",4),heavy_size=max(1024,getattr(self.settings,"h2o_heavy_tokens",1024)),max_budget=max(2048,getattr(self.settings,"h2o_max_budget",2048)))
    self.mcts = SymbolicMCTSSearchEngine(self.sandbox, self.kg, simulations=32)
    self.grpo_optimizer = GroupRelativePolicyOptimizer(group_size=4, clip_eps=.2)
    self.round_robin_lora = None;self.dual_adapter_buffer = None;self.moe_lora_manager = None
    if MLX_AVAILABLE and self.model is not None:
        try:
            rr=RoundRobinLoRAController(self.model,rank=4,alpha=8.0,chunk_size=6);rr.attach_all_layers();self.round_robin_lora=rr
            if rr.targets:
                self.dual_adapter_buffer=DualAdapterBufferManager(self.model,rr,METAL_STREAM_LOCK)
                self.moe_lora_manager=HierarchicalMoELoRAManager(self.model,rr,bank_dir="adapter_banks")
        except Exception:
            self.round_robin_lora=None;self.dual_adapter_buffer=None;self.moe_lora_manager=None
UnifiedMasterEngine.__init__ = _merged_engine_init
