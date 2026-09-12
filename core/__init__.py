from core.entropy_router import EntropyRouter
from core.verifier import GroundTruthVerifier, VerificationResult
from core.pro_engine import ProReasoningEngine
from core.awake_auto_hook import install_awake_auto_learning
from core.speculative_engine import SpeculativeEngine, PromptLookupDrafter, LookaheadJacobiDrafter
from core.platform import PlatformRouter, get_platform_router, detect_hardware

install_awake_auto_learning(ProReasoningEngine)

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
