"""Optional, publisher-supplied Bonsai 2.5 MTP for the existing MLX-VLM path."""
from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path, PurePosixPath
import sys

from core.speculative_runtime import normalize_speculative_mode, sha256_file


class MLXMTP:
    """Own only the verified draft head; the active model verifies its proposals.

    All methods which touch MLX objects run on the backend's existing Metal owner
    thread. No older DFlash, LoRA adapter, or similarly named head is substituted.
    """

    def __init__(self):
        self.head = None
        self.requested_mode = "auto"
        self.max_draft_tokens = 1
        self.reason = "MTP is not configured."

    def configure(self, model, source, info, *, mode="auto", max_draft_tokens=None):
        requested = normalize_speculative_mode(mode)
        if max_draft_tokens is not None and (
                type(max_draft_tokens) is not int or not 1 <= max_draft_tokens <= 64):
            raise ValueError("MTP draft tokens must be an integer from 1 through 64")
        if requested == "none":
            self.head = None
            self.requested_mode = requested
            self.reason = "MTP is off."
            return self.telemetry()
        filename, expected_hash = info.get("mtp_file"), info.get("mtp_sha256")
        if not filename or not expected_hash:
            self.head = None
            self.requested_mode = requested
            self.reason = "This selected MLX model has no verified MTP sidecar."
            if requested == "mtp":
                raise ValueError(self.reason)
            return self.telemetry()
        if requested in {"lookup", "dflash2"}:
            raise ValueError("This MLX pack supports Auto/MTP/Off; it has no compatible lookup or DFlash2 drafter")

        if self.head is None:
            from mlx_vlm.generate import generate_step
            parameters = inspect.signature(generate_step).parameters
            if not {"draft_model", "draft_kind", "draft_block_size"} <= set(parameters):
                raise RuntimeError("Bonsai 2.5 MTP requires the bundled mlx-vlm 0.6.3 runtime; update dependencies")
            root = Path(source).resolve()
            relative = PurePosixPath(filename)
            if relative.is_absolute() or ".." in relative.parts or "\\" in filename or ":" in filename:
                raise ValueError("The MTP sidecar must be inside the selected model snapshot")
            # Hugging Face snapshots contain symlinks into their shared blob
            # store. The pinned checksum verifies those bytes too.
            head_path = root.joinpath(*relative.parts)
            if sha256_file(str(head_path)) != expected_hash:
                raise ValueError("Bonsai 2.5 MTP head checksum mismatch; re-download the selected model")
            runtime_dir = root / "runtime"
            module_file = runtime_dir / "mtp_artifact.py"
            spec = importlib.util.spec_from_file_location("_smartai_bonsai25_mtp", module_file)
            if spec is None or spec.loader is None:
                raise RuntimeError("The selected model is missing its publisher MTP loader")
            module = importlib.util.module_from_spec(spec)
            from core.repository_runtime_imports import repository_runtime_imports
            old_path = list(sys.path)
            try:
                sys.path.insert(0, str(runtime_dir))
                with repository_runtime_imports(runtime_dir):
                    spec.loader.exec_module(module)
                    head = module.load_mtp_head(model, head_path)
            finally:
                sys.path[:] = old_path
            if head is None:
                raise RuntimeError("The publisher MTP loader returned no draft head")
            self.head = head
        self.requested_mode = requested
        # Prism's verified example uses block size 2: one proposed token plus
        # the target bonus token. Larger depths are explicitly selectable.
        self.max_draft_tokens = max_draft_tokens or 1
        self.reason = "Publisher MTP head loaded; target verification is active. Speed and hardware parity are unmeasured."
        return self.telemetry()

    def generation_kwargs(self):
        if self.head is None:
            return {}
        return {"draft_model": self.head, "draft_kind": "mtp",
                "draft_block_size": self.max_draft_tokens + 1}

    def telemetry(self):
        return {"mode": "mtp" if self.head is not None else "none",
                "requested_mode": self.requested_mode,
                "runtime_contract": "mlx-vlm-publisher-target-verification",
                "status": "active" if self.head is not None else "disabled",
                "speedup": None, "output_equivalence_measured": False,
                "message": self.reason}

    def clear(self):
        self.head = None
        self.reason = "MTP head unloaded."
