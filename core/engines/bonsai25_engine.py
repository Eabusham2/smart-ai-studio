"""Bonsai 2.5 publisher runtime with embedded MTP, separate from the strict verifier.

This uses the publisher's target-verified speculative decoding. It deliberately
does not advertise Studio's older one-row exactness contract or a measured
speedup. The original strict Bonsai 2 / DFlash backend remains unchanged.
"""
from __future__ import annotations

import subprocess

from config.bonsai_models import GGUF_MTP_HASHES, MMPROJ_SHA256
from core.bonsai_native_runtime import ensure_bonsai_server
from core.engines.gguf_engine import GGUFReasoningBackend
from core.engines.prism_speculative_engine import PrismSpeculativeReasoningBackend, _NativeModelProxy
from core.speculative_runtime import SpeculativePlan, sha256_file

PUBLISHER_CONTRACT = "prism-bonsai25-target-verification"


class _BonsaiModelProxy(_NativeModelProxy):
    def create_chat_completion(self, **payload):
        payload.setdefault("stream", False)
        return self.backend._request("/v1/chat/completions", payload, timeout=1800.0)


class Bonsai25ReasoningBackend(PrismSpeculativeReasoningBackend):
    # Preserve existing native ownership, cancellation and usage integration.
    runtime = "prism_speculative"

    def __init__(self, model_path, server_path=None, *, model_sha256=None, **kwargs):
        self._managed_server = not server_path
        self._selected_model_sha256 = model_sha256
        defaults = {"top_k": 20, "min_p": 0.0, "repeat_penalty": 1.0}
        defaults.update(kwargs.pop("sampling_defaults", {}) or {})
        super().__init__(model_path, server_path or "unprepared-bonsai-server",
                         sampling_defaults=defaults, **kwargs)

    def preflight(self, cancelled=None):
        if self.mode not in {"auto", "mtp", "none"}:
            raise ValueError("Bonsai 2.5 uses its embedded MTP head; older DFlash2 and lookup presets are not compatible")
        if self.draft_model_path:
            raise ValueError("Bonsai 2.5 MTP is embedded in the selected GGUF; remove the external draft model")
        if self.max_draft_tokens is not None and (
                type(self.max_draft_tokens) is not int or not 1 <= self.max_draft_tokens <= 64):
            raise ValueError("MTP draft tokens must be an integer from 1 through 64")
        target_hash = sha256_file(self.model_path, cancelled)
        if target_hash not in GGUF_MTP_HASHES.values() or (
                self._selected_model_sha256 and target_hash != self._selected_model_sha256):
            raise ValueError("Selected Bonsai 2.5 GGUF checksum does not match the pinned MTP model")
        if self.mmproj_path and sha256_file(self.mmproj_path, cancelled) != MMPROJ_SHA256:
            raise ValueError("The selected projector does not match Bonsai 2.5")
        if cancelled and cancelled():
            raise RuntimeError("Bonsai model loading cancelled")
        # No weights are loaded by --help. An explicitly selected binary is
        # accepted only if it exposes the embedded-MTP and template interfaces.
        result = subprocess.run([self.server_path, "--help"], capture_output=True,
                                text=True, timeout=30, env=self._runtime_env())
        help_text = (result.stdout or "") + (result.stderr or "")
        required = {"--spec-type", "draft-mtp", "--spec-draft-n-max", "--jinja"}
        if result.returncode or not all(flag in help_text for flag in required):
            raise RuntimeError("This llama-server does not expose the required Bonsai 2.5 MTP flags")
        mode = "none" if self.mode == "none" else "mtp"
        self._plan = SpeculativePlan(self.mode, mode, target_hash, None,
                                     (self.max_draft_tokens or 2) if mode == "mtp" else 0,
                                     runtime_contract=PUBLISHER_CONTRACT)
        return {"status": "ready", "plan": self._plan.as_dict(), "model_loaded": False,
                "message": "Pinned Bonsai 2.5 weights and MTP-capable runtime verified; hardware output and speed are unmeasured."}

    def _load_selected_model(self, cancelled):
        if self._managed_server:
            try:
                self.server_path = ensure_bonsai_server(
                    lambda: self._cancelled.is_set() or self._external_cancelled(cancelled))
            except Exception as exc:
                self.last_error = str(exc)
                return False
        loaded = super()._load_selected_model(cancelled)
        if loaded:
            self.model = self.tokenizer = _BonsaiModelProxy(self)
            self.chat_handler = bool(self.mmproj_path)
        return loaded

    def _build_command(self, port):
        command = super()._build_command(port)
        command += ["--jinja", "--no-webui"]
        if self._plan and self._plan.active_mode == "mtp":
            command += ["--spec-draft-n-min", "0"]
        if self.mmproj_path:
            command += ["--mmproj", self.mmproj_path]
        return command

    def _completion_payload(self, prompt, max_tokens, temperature, top_p, **overrides):
        # Use the model's actual EOS markers; ordinary source lines beginning
        # with 'user' or 'Human:' must not prematurely stop an external harness.
        overrides.setdefault("stop", ["<|im_end|>", "<|endoftext|>"])
        return super()._completion_payload(prompt, max_tokens, temperature, top_p, **overrides)

    def prepare_request(self):
        """Reset a completed cancellation before template/tokenizer API calls."""
        with self._io_lock:
            if not self.is_loaded or self._process is None or self._process.poll() is not None:
                raise RuntimeError("The selected Bonsai server is not running")
            self._cancelled.clear()

    def get_speculative_telemetry(self):
        result = super().get_speculative_telemetry()
        result["runtime_contract"] = PUBLISHER_CONTRACT
        result["message"] = (
            "Publisher MTP verifies proposals with the selected target. "
            "No claim of bit-identical outputs or measured speedup is made."
        )
        return result

    def configure_auto_speed(self, enabled=True, **kwargs):
        del kwargs
        return {"status": "configured" if enabled else "disabled", "settings": {},
                "message": "Bonsai 2.5 uses the publisher's native ternary kernels and configured MTP; speed is not yet measured."}

    # Preserve image and sampled-video review using the same loaded projector
    # and existing review parser, with the native chat endpoint as its transport.
    supports_media_input = GGUFReasoningBackend.supports_media_input
    _image_data_url_from_bytes = staticmethod(GGUFReasoningBackend._image_data_url_from_bytes)
    _video_frame_data_urls = GGUFReasoningBackend._video_frame_data_urls

    def review_media_input(self, path, kind, prompt=""):
        with self._io_lock:
            self.prepare_request()
            return GGUFReasoningBackend.review_media_input(self, path, kind, prompt)
