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
