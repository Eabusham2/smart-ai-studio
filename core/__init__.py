from core.entropy_router import EntropyRouter
from core.verifier import GroundTruthVerifier, VerificationResult
from core.pro_engine import ProReasoningEngine
from core.mlx_engine import MLXReasoningBackend
from core.awake_auto_hook import install_awake_auto_learning
from core.pro_runtime_hardening import install_pro_runtime_hardening
from core.mlx_adapter_persistence import install_mlx_adapter_persistence
from core.mlx_runtime_lock import install_mlx_runtime_lock
from core.speculative_engine import SpeculativeEngine, PromptLookupDrafter, LookaheadJacobiDrafter
from core.platform import PlatformRouter, get_platform_router, detect_hardware

install_mlx_adapter_persistence(MLXReasoningBackend)
install_mlx_runtime_lock(MLXReasoningBackend)
install_awake_auto_learning(ProReasoningEngine)
install_pro_runtime_hardening(ProReasoningEngine)

__all__ = [
    "EntropyRouter",
    "GroundTruthVerifier",
    "VerificationResult",
    "ProReasoningEngine",
    "SpeculativeEngine",
    "PromptLookupDrafter",
    "LookaheadJacobiDrafter",
    "PlatformRouter",
    "get_platform_router",
    "detect_hardware",
]
