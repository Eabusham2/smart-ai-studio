"""Canonical benchmark compatibility exports.

The implementation is kept under eval/ so the desktop app does not depend on the
legacy monolith. Unique LoRA and dialogue-ingest behavior from the old runtime is
merged here.
"""
from eval._run_studio_compat import *


def _merged_init_lora(self):
    if not MLX_AVAILABLE or self.model is None:
        return
    try:
        self.model.freeze()
        layers = getattr(self.model, "layers", []) or getattr(getattr(self.model, "model", None), "layers", [])
        for layer in layers:
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "q_proj"):
                if not isinstance(layer.self_attn.q_proj, LoRALinear):
                    layer.self_attn.q_proj = LoRALinear.from_base(
                        layer.self_attn.q_proj,
                        r=self.settings.lora_rank,
                        scale=self.settings.lora_alpha / max(1, self.settings.lora_rank),
                    )
                if hasattr(layer.self_attn.q_proj, "unfreeze"):
                    layer.self_attn.q_proj.unfreeze()
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
                if not isinstance(layer.mlp.down_proj, LoRALinear):
                    layer.mlp.down_proj = LoRALinear.from_base(
                        layer.mlp.down_proj,
                        r=self.settings.lora_rank,
                        scale=self.settings.lora_alpha / max(1, self.settings.lora_rank),
                    )
                if hasattr(layer.mlp.down_proj, "unfreeze"):
                    layer.mlp.down_proj.unfreeze()
        self.adapters_buffer_a = {
            k: mx.array(v) for k, v in dict(mlx.utils.tree_flatten(self.model.trainable_parameters())).items()
        }
        self.adapters_buffer_b = {k: mx.array(v) for k, v in self.adapters_buffer_a.items()}
    except Exception as exc:
        self.adapters_buffer_a = {}
        self.adapters_buffer_b = {}
        self.init_error = f"{type(exc).__name__}: {exc}"


MoEDualBufferManager._init = _merged_init_lora


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
    for src, pred, target, session in rows:
        self.kg.insert_triple(src, pred, target, weight=1.0, session_id=session)
    return len(rows)


DialogueTimelineGraphIngester.ingest_developer_sessions = _merged_dialogue_ingest


def _active_mlx_gb():
    try:
        if hasattr(mx, "get_active_memory"):
            return float(mx.get_active_memory()) / (1024 ** 3)
        if hasattr(mx, "metal") and hasattr(mx.metal, "get_active_memory"):
            return float(mx.metal.get_active_memory()) / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def _strict_initialize_runtime(self):
    """Load the real model first and never erase it because an optional subsystem failed."""
    if not MLX_AVAILABLE:
        raise RuntimeError("MLX/MLX-LM is unavailable; refusing to run the real benchmark offline.")

    # Match the known-working baseline: normal model load, no generation-cache options
    # incorrectly passed through model_config.
    try:
        self.model, self.tokenizer = load(self.settings.mlx_model_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load real model {self.settings.mlx_model_path}: {exc!r}") from exc

    if self.model is None or self.tokenizer is None:
        raise RuntimeError(f"Model loader returned no model/tokenizer for {self.settings.mlx_model_path}")

    # MLX is lazy. Force all weights to materialize before Phase 1 so model loading cannot
    # masquerade as a frozen first benchmark item.
    try:
        mx.eval(self.model.parameters())
    except Exception as exc:
        raise RuntimeError(f"27B model loaded but weight materialization failed: {exc!r}") from exc

    rss_gb = psutil.Process().memory_info().rss / (1024 ** 3)
    active_gb = _active_mlx_gb()
    print(
        f"[✓] REAL MODEL READY: {self.settings.mlx_model_path} | RSS {rss_gb:.2f} GB | MLX active {active_gb:.2f} GB",
        flush=True,
    )

    # Initialize later-stage pieces separately. A failure can never null out the already-loaded model.
    try:
        self.moe_manager = MoEDualBufferManager(self.model, self.settings)
    except Exception as exc:
        self.moe_manager = None
        raise RuntimeError(f"Model is loaded, but MoE/LoRA stage initialization failed: {exc!r}") from exc

    try:
        self.moe_router = HierarchicalMoERouter(self.model)
    except Exception as exc:
        self.moe_router = None
        raise RuntimeError(f"Model is loaded, but MoE router initialization failed: {exc!r}") from exc

    try:
        self.grpo_trainer = GRPOTrainingEngine(self.model, self.tokenizer, self.sandbox)
    except Exception as exc:
        self.grpo_trainer = None
        raise RuntimeError(f"Model is loaded, but GRPO stage initialization failed: {exc!r}") from exc

    if self.settings.enable_awake_ogp_daemon:
        self.ogp_daemon = ProjectedSleepConsolidationDaemon(
            self.moe_manager,
            self.ogp_projector,
            self.kg,
            self.tokenizer,
            self.settings,
            METAL_STREAM_LOCK,
        )
        self.ogp_daemon.start()


UnifiedMasterEngine._initialize_runtime = _strict_initialize_runtime
