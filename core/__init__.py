import core.pro_engine as pro_engine_module
from core.entropy_router import EntropyRouter
from core.verifier import GroundTruthVerifier, VerificationResult
from core.pro_engine import ProReasoningEngine
from core.mlx_engine import MLXReasoningBackend
from core.awake_auto_hook import install_awake_auto_learning
from core.full_context_hardening import install as install_full_context_hardening
from core.pro_runtime_hardening import install_pro_runtime_hardening
from core.mlx_adapter_persistence import install_mlx_adapter_persistence
from core.mlx_runtime_lock import install_mlx_runtime_lock
from core.unified_context_budget import install as install_unified_context_budget
from core.temperature_policy import install_chat as install_chat_temperature_policy
from core.speculative_engine import SpeculativeEngine, PromptLookupDrafter, LookaheadJacobiDrafter
from core.platform import PlatformRouter, get_platform_router, detect_hardware

install_mlx_adapter_persistence(MLXReasoningBackend)
install_mlx_runtime_lock(MLXReasoningBackend)
install_unified_context_budget(MLXReasoningBackend)
install_chat_temperature_policy(pro_engine_module)
install_awake_auto_learning(ProReasoningEngine)
install_full_context_hardening(ProReasoningEngine)
install_pro_runtime_hardening(ProReasoningEngine)

# Native speed measurements share the same exclusion as real inference/learning.
from core.native_speed import install_native_runtime_lock
from core.speed_control import install_speed_control
from core.engines.gguf_engine import GGUFReasoningBackend
from core.engines.prism_gguf_engine import PrismGGUFReasoningBackend
from core.engines.bitnet_cpp_engine import BitNetCppReasoningBackend
# Strata serializes HTTP requests itself. Its unload must close the active
# transport before waiting, so a native inference lock would block cancellation.
for _backend in (MLXReasoningBackend, GGUFReasoningBackend, PrismGGUFReasoningBackend, BitNetCppReasoningBackend):
    install_native_runtime_lock(_backend)
install_speed_control(ProReasoningEngine)

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
