"""Transactional scalar backend for testing orchestration and crash ordering only."""
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

from core.learning_state import LearningStateStore


class TransactionalBackend:
    def __init__(self, state_root, *, identity="chat-test"):
        self.learning_state = LearningStateStore({"source": identity}, state_root=state_root)
        self.model = object()
        self.tokenizer = SimpleNamespace(encode=lambda text: list(text.split()))
        self.train_calls = 0
        self.adapter_path = self.learning_state.current_checkpoint()
        self.weight = float(Path(self.adapter_path).read_text()) if self.adapter_path else 0.05
        self.adapters = {"layer_0_lora": self.weight}
        self.received_batches = []
        self.before_validation = None
        self.regress = False
        self.training_metrics = {}

    def score_learning_probe(self, prompt, completion):
        weight = -self.weight if self.regress and "quadratic formula" in prompt else self.weight
        return math.log1p(math.exp(-weight))

    def stream_generate_tokens(self, prompt, **kwargs):
        """Scripted exact recall for the synthetic cedar-label orchestration cases."""
        match = re.search(r"sample (\d+)", prompt)
        if not match:
            yield "unknown"
            return
        idx = match[1]
        answer = f"cedar{idx}"
        question = f"Which archive label belongs to sample {idx}?"
        if "Form one short recall question" in prompt:
            yield json.dumps({"question": question, "answer": answer, "evidence": answer})
        elif "Quote the entire original user" in prompt:
            yield f"The archive label for sample {idx} is cedar{idx}."
        elif "Quote the entire original assistant" in prompt:
            yield f"I recorded sample {idx} as cedar{idx}."
        else:
            yield answer

    def count_tokens(self, messages):
        return sum(len(self.tokenizer.encode(m["content"])) for m in messages)

    def train_mini_batch(self, *, data, learning_metadata, validation_callback, **kwargs):
        self.train_calls += 1
        self.received_batches.append(data)
        before = self.weight
        with self.learning_state.begin(metadata=learning_metadata) as txn:
            try:
                self.weight += 1
                if self.before_validation:
                    self.before_validation()
                evidence = validation_callback(self)
                assert evidence["accepted"]
                (txn.directory / "scalar.json").write_text(json.dumps(self.weight))
                txn.commit("scalar.json", metadata={"validation": evidence, "metrics": self.training_metrics})
            except BaseException:
                self.weight = before
                raise
        self.adapter_path = self.learning_state.current_checkpoint()
        self.adapters = {"layer_0_lora": self.weight}
        return {"trainable_parameters_touched": 1}, abs(self.weight - before)
