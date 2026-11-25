"""Regression contract for model repositories as data rather than executable code."""
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
TRANSFORMERS_LOADERS = (
    "consolidation/daemon.py",
    "core/bitnet_rebuild_trainer.py",
    "core/controller_runtime.py",
    "core/gguf_lora_trainer.py",
    "core/platform.py",
    "core/pro_engine.py",
    "core/training_lineage.py",
)


@pytest.mark.parametrize("relative_path", TRANSFORMERS_LOADERS)
def test_transformers_loaders_never_enable_repository_code(relative_path):
    source = (ROOT / relative_path).read_text(encoding="utf-8")
    assert "trust_remote_code=True" not in source
    assert "trust_remote_code=False" in source


def test_repository_module_execution_sink_is_removed():
    controller = (ROOT / "core/controller_runtime.py").read_text(encoding="utf-8")
    assert "exec_module(" not in controller
    assert not (ROOT / "core/repository_runtime_imports.py").exists()
