"""Complete, bounded causal training examples shared by inference-time learners.

Windows overlap by exactly one predecessor token. No old conversation is silently
truncated to fit an optimizer step: every eligible next-token target is visited.
The module deliberately imports no model framework.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Mapping, Sequence


IGNORE_INDEX = -100


@dataclass(frozen=True)
class TrainingWindow:
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    row_index: int
    start: int
    stop: int

    @property
    def supervised_tokens(self) -> int:
        return sum(value != IGNORE_INDEX for value in self.labels[1:])

    @property
    def loss_start(self) -> int:
        return next((i for i, value in enumerate(self.labels) if i and value != IGNORE_INDEX), len(self.labels))


def encode_text(tokenizer, text: str) -> list[int]:
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        try:
            result = encode(text, add_special_tokens=False)
        except TypeError:
            result = encode(text)
    else:
        tokenize = getattr(tokenizer, "tokenize", None)
        if not callable(tokenize):
            raise RuntimeError("Learning requires the selected model's tokenizer")
        try:
            result = tokenize(text.encode("utf-8"), add_bos=False)
        except TypeError:
            result = tokenize(text.encode("utf-8"))
    if hasattr(result, "tolist"):
        result = result.tolist()
    if result and isinstance(result[0], (list, tuple)):
        if len(result) != 1:
            raise ValueError("Learning tokenizer returned a batch instead of one sequence")
        result = result[0]
    return [int(value) for value in result]


def format_training_row(tokenizer, item: Mapping) -> tuple[str, str]:
    """Return native-template full text and its assistant-generation prefix."""
    prompt = str(item.get("prompt") or "")
    completion = str(item.get("completion") or "")
    if not prompt.strip() or not completion.strip():
        raise ValueError("Learning rows require nonempty prompt and completion")
    messages = item.get("messages")
    if messages is not None:
        messages = [dict(turn) for turn in messages]
        if not messages or messages[-1].get("role") != "assistant":
            raise ValueError("Training messages must end with the supervised assistant response")
        prefix_messages = messages[:-1]
    else:
        prefix_messages = [{"role": "user", "content": prompt}]
        messages = prefix_messages + [{"role": "assistant", "content": completion}]
    template = getattr(tokenizer, "apply_chat_template", None)
    if callable(template):
        try:
            full = template(messages, tokenize=False, add_generation_prompt=False)
            prefix = template(prefix_messages, tokenize=False, add_generation_prompt=True)
            if isinstance(full, str) and isinstance(prefix, str) and full and prefix:
                return full, prefix
        except (ValueError, TypeError, AttributeError, NotImplementedError):
            pass
    # This fallback is for tokenizers without a chat template, not a claim that
    # every imported model uses ChatML. Plain causal continuation stays neutral.
    prefix = prompt + "\n\n"
    eos = getattr(tokenizer, "eos_token", None)
    return prefix + completion + (eos if isinstance(eos, str) else ""), prefix


def tokenize_training_row(tokenizer, item: Mapping, *, supervise_prompt: bool = True) -> tuple[list[int], list[int]]:
    full, prefix = format_training_row(tokenizer, item)
    tokens = encode_text(tokenizer, full)
    labels = list(tokens)
    if labels:
        labels[0] = IGNORE_INDEX
    if not supervise_prompt:
        prefix_ids = encode_text(tokenizer, prefix)
        # BPE can merge at the assistant boundary. The longest common token
        # prefix masks only tokens that are demonstrably part of the prompt.
        boundary = 0
        for left, right in zip(tokens, prefix_ids):
            if left != right:
                break
            boundary += 1
        if boundary == 0 and prefix_ids:
            raise RuntimeError("Cannot prove the completion boundary for this tokenizer")
        for index in range(boundary):
            labels[index] = IGNORE_INDEX
    return tokens, labels


def windows_from_tokens(tokens: Sequence[int], labels: Sequence[int] | None = None, *, max_tokens: int = 256, row_index: int = 0) -> Iterator[TrainingWindow]:
    if isinstance(max_tokens, bool) or int(max_tokens) < 2:
        raise ValueError("A causal training window requires at least two tokens")
    size = int(max_tokens)
    ids = tuple(int(value) for value in tokens)
    targets = tuple(int(value) for value in (labels if labels is not None else ids))
    if len(ids) != len(targets):
        raise ValueError("Training token and label lengths differ")
    for start in range(0, max(0, len(ids) - 1), size - 1):
        stop = min(len(ids), start + size)
        local_labels = (IGNORE_INDEX,) + targets[start + 1:stop]
        window = TrainingWindow(ids[start:stop], local_labels, row_index, start, stop)
        if window.supervised_tokens:
            yield window


def iter_training_windows(tokenizer, data: Iterable[Mapping], *, max_tokens: int = 256, supervise_prompt: bool = True) -> Iterator[TrainingWindow]:
    for index, item in enumerate(data):
        tokens, labels = tokenize_training_row(tokenizer, item, supervise_prompt=supervise_prompt)
        yield from windows_from_tokens(tokens, labels, max_tokens=max_tokens, row_index=index)


def iter_probe_windows(tokenizer, prompt: str, completion: str, *, max_tokens: int = 256) -> Iterator[TrainingWindow]:
    """Teacher-force only an explicit answer, without an EOS or role-label target."""
    _, prefix = format_training_row(tokenizer, {"prompt": prompt, "completion": completion})
    prefix_ids = encode_text(tokenizer, prefix)
    tokens = encode_text(tokenizer, prefix + str(completion))
    boundary = 0
    for left, right in zip(tokens, prefix_ids):
        if left != right:
            break
        boundary += 1
    if not boundary or boundary >= len(tokens):
        raise ValueError("Probe has no independently tokenized answer targets")
    labels = [IGNORE_INDEX] * boundary + tokens[boundary:]
    yield from windows_from_tokens(tokens, labels, max_tokens=max_tokens)


def training_coverage(tokenizer, data: Iterable[Mapping], *, max_tokens: int = 256, supervise_prompt: bool = True) -> dict:
    rows = list(data)
    windows = list(iter_training_windows(tokenizer, rows, max_tokens=max_tokens, supervise_prompt=supervise_prompt))
    visited = {window.row_index for window in windows}
    return {
        "rows": len(rows), "trained_rows": len(visited), "windows": len(windows),
        "supervised_tokens": sum(window.supervised_tokens for window in windows),
        "max_window_tokens": max((len(window.input_ids) for window in windows), default=0),
        "complete": len(visited) == len(rows) and bool(rows),
        "supervise_prompt": bool(supervise_prompt),
    }
