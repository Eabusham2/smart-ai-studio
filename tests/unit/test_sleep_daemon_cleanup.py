"""Real tiny PyTorch standalone Sleep work releases its owned second model."""
import gc
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest

from config.settings import get_settings
from consolidation.daemon import SleepConsolidationDaemon


@pytest.mark.parametrize("fail_save", [False, True])
def test_repeated_standalone_sleep_close_releases_tensor_owners(tmp_path, monkeypatch, fail_save):
    torch = pytest.importorskip("torch")
    import consolidation.daemon as module
    monkeypatch.setattr(module, "get_anchor_texts", lambda: ["one anchor", "second anchor"])

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(16, 4)
            self.head = torch.nn.Linear(4, 16)

        def forward(self, input_ids, labels=None):
            logits = self.head(self.embedding(input_ids))
            loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 16),
                                                       labels[:, 1:].reshape(-1), ignore_index=-100)
            return SimpleNamespace(loss=loss)

        def save_pretrained(self, folder):
            if fail_save:
                raise OSError("injected checkpoint failure")
            torch.save(self.state_dict(), Path(folder) / "fixture.pt")

    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": torch.tensor([[ord(c) % 16 for c in text]])}

    def initialize(self):
        self.model = self.base_model = TinyModel()
        self.tokenizer = Tokenizer()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=.001)

    monkeypatch.setattr(SleepConsolidationDaemon, "_init_model_and_lora", initialize)
    # Retain the daemon objects, as a GUI callback/exception record could do.
    # close() must free their large members even before those owners disappear.
    daemons, refs = [], []
    for index in range(5):
        settings = get_settings(use_mock=False, device="cpu", database_path=str(tmp_path / f"{index}.db"),
                                lora_adapter_path=str(tmp_path / f"adapter-{index}"))
        daemon = SleepConsolidationDaemon(settings=settings)
        daemons.append(daemon)
        daemon.db.log_interaction("fixture prompt", "fixture answer", verified_reward=1)
        try:
            if fail_save:
                with pytest.raises(OSError, match="checkpoint failure"):
                    daemon.run_consolidation_cycle(anchor_dataset=[{"prompt": "anchor", "completion": "answer"}])
            else:
                result = daemon.run_consolidation_cycle(anchor_dataset=[{"prompt": "anchor", "completion": "answer"}])
                assert result["status"] == "success" and result["memories_consolidated"] == 1
            refs.extend(weakref.ref(value) for value in (
                daemon.model, daemon.optimizer, *daemon.model.parameters(),
                *daemon.fisher_estimator.fisher_matrix.values(), *daemon.fisher_estimator.anchor_weights.values()))
        finally:
            daemon.close()
        gc.collect()
        assert all(ref() is None for ref in refs)
        assert daemon.model is daemon.base_model is daemon.optimizer is daemon.tokenizer is None
        assert not daemon.fisher_estimator.is_computed()
    daemons[-1].close()  # Idempotent cleanup leaves the saved checkpoint intact.
    if not fail_save:
        assert (tmp_path / "adapter-4" / "fixture.pt").is_file()
