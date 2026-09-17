"""Pier custom agent used only to verify an already-selected DeepSWE patch.

No model call or branch selection happens here. The selected patch is applied to a
fresh official DeepSWE task environment and committed so the benchmark's untouched
collect hook + separate verifier can grade it once.
"""
from __future__ import annotations

import base64
from pathlib import Path

try:
    from pier.agents.base import BaseAgent
    from pier.environments.base import BaseEnvironment
    from pier.models.agent.context import AgentContext
except Exception:  # Imported only by Pier on the opt-in DeepSWE path.
    BaseAgent = object  # type: ignore
    BaseEnvironment = object  # type: ignore
    AgentContext = object  # type: ignore


class PatchReplayAgent(BaseAgent):
    SUPPORTS_WINDOWS = False

    def __init__(self, logs_dir: Path, patch_file: str, **kwargs):
        super().__init__(logs_dir=logs_dir, **kwargs)
        self.patch_file = Path(patch_file)

    @staticmethod
    def name() -> str:
        return "smartai-deepswe-patch-replay"

    def version(self) -> str:
        return "1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        return None

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        patch = self.patch_file.read_bytes()
        encoded = base64.b64encode(patch).decode("ascii")
        command = (
            "git config user.name 'Smart AI Studio' && "
            "git config user.email 'smartai@local.invalid' && "
            f"printf '%s' '{encoded}' | base64 -d > /tmp/smartai-selected.patch && "
            "git apply --whitespace=nowarn /tmp/smartai-selected.patch && "
            "git add -A && "
            "git diff --cached --quiet || git commit -m 'Smart AI Studio selected DeepSWE patch'"
        )
        result = await environment.exec(command)
        if result.return_code != 0:
            raise RuntimeError(
                f"Selected DeepSWE patch replay failed ({result.return_code}): "
                f"{result.stderr or result.stdout}"
            )
