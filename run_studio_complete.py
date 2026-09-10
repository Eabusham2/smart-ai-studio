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
        self.adapters_buffer_a = {k: mx.array(v) for k, v in dict(mlx.utils.tree_flatten(self.model.trainable_parameters())).items()}
        self.adapters_buffer_b = {k: mx.array(v) for k, v in self.adapters_buffer_a.items()}
    except Exception:
        self.adapters_buffer_a = {}
        self.adapters_buffer_b = {}


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
