"""Verify a local HF training parent against the GGUF actually serving the user."""
from __future__ import annotations

import json
import os
import signal
from pathlib import Path
import subprocess

from core.torch_context_learning import check_cancelled


def local_training_parent(identifier):
    from core.controller_runtime import resolve_local_snapshot
    source = Path(resolve_local_snapshot(str(identifier)))
    if (not (source / "config.json").is_file()
            or not (any(source.glob("*.safetensors")) or any(source.glob("pytorch_model*.bin")))):
        raise RuntimeError("Download the declared Hugging Face training parent before enabling learning: " + str(identifier))
    return str(source.absolute())


def standalone_training_parent(identifier):
    """An explicit standalone train call can acquire its configured Hugging Face parent."""
    try:
        return local_training_parent(identifier)
    except RuntimeError:
        path = Path(identifier).expanduser()
        offline = any(os.getenv(name, "").upper() in {"1", "ON", "TRUE", "YES"}
                      for name in ("OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"))
        if offline or path.exists() or path.is_absolute() or str(identifier).startswith((".", "~")):
            raise
    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().model_info(identifier)
    files = {entry.rfilename for entry in info.siblings}
    # Match Transformers' preference for safetensors. Some repos also ship the
    # same model as PyTorch bins and multiple quantized deployments.
    if "model.safetensors" in files or "model.safetensors.index.json" in files:
        weights = ["model.safetensors", "model-*.safetensors"]
    elif "pytorch_model.bin" in files or "pytorch_model.bin.index.json" in files:
        weights = ["pytorch_model.bin", "pytorch_model-*.bin"]
    else:
        raise RuntimeError("The selected Hugging Face parent has no Transformers model weights")
    snapshot = snapshot_download(repo_id=identifier, revision=info.sha, allow_patterns=[
        *weights, "*.json", "*.jsonl", "*.py", "*.model", "*.txt", "*.tiktoken", "*.jinja", "*.vocab", "*.bpe",
    ])
    return local_training_parent(snapshot)


def verify_parent(backend, trainer, path, native_tokens, state_root):
    from transformers import AutoTokenizer
    from core.gguf_metadata import inspect_gguf
    from core.learning_state import identity_digest, model_identity

    source = local_training_parent(trainer.base_model_id)
    metadata = inspect_gguf(path)
    config = json.loads((Path(source) / "config.json").read_text(encoding="utf-8"))
    language = config.get("text_config") or config
    for key, fields in (("layer_count", ("num_hidden_layers", "n_layer")),
                        ("embedding_length", ("hidden_size", "n_embd")),
                        ("vocab_size", ("vocab_size",))):
        value = next((language.get(field) for field in fields if language.get(field)), None)
        if not metadata[key] or value != metadata[key]:
            raise RuntimeError("HF training parent does not match selected GGUF " + key)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=True)
    for text in ("A number: 42.\n", "prefix λ東京🧠", "def run(x):\n    return x + 1"):
        expected = list(tokenizer.encode(text, add_special_tokens=False))
        if not expected or list(native_tokens(text)) != expected:
            raise RuntimeError("HF training parent token IDs differ from the deployed model")

    fingerprint = identity_digest(model_identity(source, adapter_format="training-parent", state_root=state_root))
    manifest = backend.learning_state.current_manifest(verify=False) or {}
    previous = manifest.get("metadata", {}).get("metrics", {}).get("training_parent_sha256")
    if manifest and previous != fingerprint:
        raise RuntimeError("The saved PEFT generation belongs to a different training parent")
    trainer.training_source = source
    return fingerprint


def run_conversion(command, *, cwd=None, cancel_event=None):
    """Interrupt and reap a converter before deleting its staging directory."""
    check_cancelled(cancel_event)
    # Conversion helpers may fan out over shards. Keep descendants inside
    # the conversion's own process group so none outlive a rejected stage.
    options = {"start_new_session": True} if os.name != "nt" else {
        "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP
    }
    with subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **options) as proc:
        try:
            while True:
                try:
                    stdout, stderr = proc.communicate(timeout=0.25)
                    return subprocess.CompletedProcess(command, proc.returncode, stdout, stderr)
                except subprocess.TimeoutExpired:
                    check_cancelled(cancel_event)
        except BaseException:
            if os.name == "nt":
                import psutil
                try:
                    children = psutil.Process(proc.pid).children(recursive=True)
                except psutil.NoSuchProcess:
                    children = []
                proc.kill()
                for child in children:
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
            else:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            proc.communicate()
            raise
