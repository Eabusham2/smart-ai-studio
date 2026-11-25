"""Keep legacy benchmark sandbox classes on the mandatory container boundary."""
from __future__ import annotations

from core.container_sandbox import DEFAULT_SANDBOX_IMAGE
from core.swe_sandbox import HardenedSWESandbox


def _install_one(cls, result_cls, *, use_git_apply: bool = False) -> None:
    """Adapt legacy result types without restoring host test execution."""
    if getattr(cls, "_smartai_patch_timeout_hardened", False):
        return

    def verify_git_diff_patch(self, repo_structure, patch_text, test_cmd):
        sandbox = HardenedSWESandbox(
            timeout_sec=float(getattr(self, "timeout", 4.0) or 4.0),
            max_memory_mb=int(getattr(self, "max_memory_mb", 512) or 512),
            max_files=int(getattr(self, "max_files", 64) or 64),
            docker_image=str(
                getattr(self, "docker_image", DEFAULT_SANDBOX_IMAGE)
                or DEFAULT_SANDBOX_IMAGE
            ),
        )
        result = sandbox.verify_git_diff_patch(repo_structure, patch_text, test_cmd)
        return result_cls(
            result.passed,
            result.execution_time_ms,
            result.output,
            result.error,
            result.reward,
        )

    cls.verify_git_diff_patch = verify_git_diff_patch
    cls._smartai_patch_timeout_hardened = True


def install(compat_module) -> None:
    """Patch only legacy classes that do not already inherit the secure runner."""
    _install_one(
        compat_module.POSIXHardenedSandbox,
        compat_module.SandboxResult,
        use_git_apply=False,
    )

    try:
        from core.swe_sandbox import SandboxResult
    except Exception:
        return

    _install_one(HardenedSWESandbox, SandboxResult, use_git_apply=True)
