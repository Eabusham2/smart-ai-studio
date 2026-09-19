from typing import List

class PromptLookupDrafter:
    """
    Speculative N-Gram Prompt Drafter:
    Matches repetitive n-grams in context to emit candidate tokens in a single forward step.
    """
    def __init__(self, n_gram: int = 3, max_draft: int = 4):
        self.n_gram = n_gram
        self.max_draft = max_draft

    def find_draft_tokens(self, token_history: List[int]) -> List[int]:
        if len(token_history) < self.n_gram * 2:
            return []
        target_ngram = token_history[-self.n_gram:]
        for i in range(len(token_history) - self.n_gram - 1, -1, -1):
            if token_history[i:i + self.n_gram] == target_ngram:
                draft_start = i + self.n_gram
                return token_history[draft_start : draft_start + self.max_draft]
        return []

# Compatibility/diagnostic superset recovered from the older architecture staging
# branch. Production speculative decoding remains disabled in core.speculative_engine;
# these helpers only produce proposals for diagnostics or future equivalence testing.
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

DEFAULT_SCAFFOLDS = (
    "def ", "\n    return ", "\n    if ", "\n    for ", "\n    while ",
    "\n    else:", "\n    elif ", "\n    try:", "\n    except ", "assert ",
    "```python\n", "\\boxed{", " = ", " == ", " in ",
)


@dataclass
class DraftTelemetry:
    attempts: int = 0
    hits: int = 0
    proposed_tokens: int = 0
    accepted_tokens: int = 0

    @property
    def hit_rate(self) -> float:
        return 100.0 * self.hits / max(1, self.attempts)

    @property
    def acceptance_rate(self) -> float:
        return 100.0 * self.accepted_tokens / max(1, self.proposed_tokens)


class GrammarGuidedASTTrieDrafter:
    """N-gram proposals plus tokenizer-bound grammar scaffolds; proposal-only."""

    def __init__(
        self,
        n_gram: int = 3,
        max_draft: int = 3,
        scaffolds: Optional[Iterable[str]] = None,
        tokenizer=None,
    ):
        self.n_gram = max(1, int(n_gram))
        self.max_draft = max(1, int(max_draft))
        self.scaffolds = tuple(scaffolds or DEFAULT_SCAFFOLDS)
        self.tokenizer = None
        self._sequences: List[List[int]] = []
        self.telemetry = DraftTelemetry()
        if tokenizer is not None:
            self.bind_tokenizer(tokenizer)

    def bind_tokenizer(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        sequences: List[List[int]] = []
        for text in self.scaffolds:
            try:
                try:
                    ids = tokenizer.encode(text, add_special_tokens=False)
                except TypeError:
                    ids = tokenizer.encode(text)
                ids = [int(x) for x in ids]
                if ids:
                    sequences.append(ids)
            except Exception:
                continue
        sequences.sort(key=len, reverse=True)
        self._sequences = sequences

    def _ngram_draft(self, history: Sequence[int]) -> List[int]:
        n = self.n_gram
        if len(history) < n * 2:
            return []
        target = list(history[-n:])
        current_start = len(history) - n
        for idx in range(current_start - 1, -1, -1):
            if list(history[idx:idx + n]) == target:
                start = idx + n
                end = min(start + self.max_draft, current_start)
                draft = list(history[start:end])
                if draft:
                    return [int(x) for x in draft]
        return []

    def _grammar_draft(self, history: Sequence[int]) -> List[int]:
        hist = list(history)
        for seq in self._sequences:
            for prefix_len in range(min(len(seq) - 1, len(hist)), 0, -1):
                if hist[-prefix_len:] == seq[:prefix_len]:
                    rest = seq[prefix_len:prefix_len + self.max_draft]
                    if rest:
                        return [int(x) for x in rest]
        return []

    def find_draft_tokens(self, token_history, tokenizer=None, max_draft=None) -> List[int]:
        if tokenizer is not None and tokenizer is not self.tokenizer:
            self.bind_tokenizer(tokenizer)
        self.telemetry.attempts += 1
        previous = self.max_draft
        if max_draft is not None:
            self.max_draft = max(1, int(max_draft))
        try:
            draft = self._ngram_draft(token_history) or self._grammar_draft(token_history)
        finally:
            self.max_draft = previous
        if draft:
            self.telemetry.hits += 1
            self.telemetry.proposed_tokens += len(draft)
        return draft

    def note_acceptance(self, accepted: int, proposed: Optional[int] = None) -> None:
        self.telemetry.accepted_tokens += max(0, int(accepted))
        if proposed is not None and proposed > 0 and self.telemetry.proposed_tokens < proposed:
            self.telemetry.proposed_tokens += int(proposed)

    def get_telemetry(self):
        return {
            "attempts": self.telemetry.attempts,
            "hits": self.telemetry.hits,
            "hit_rate": self.telemetry.hit_rate,
            "proposed_tokens": self.telemetry.proposed_tokens,
            "accepted_tokens": self.telemetry.accepted_tokens,
            "acceptance_rate": self.telemetry.acceptance_rate,
        }


ASTPrefixTrieDrafter = GrammarGuidedASTTrieDrafter

