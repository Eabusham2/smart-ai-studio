"""Bounded, ranked PUCT archive for online discovery on one verified task."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class DiscoveryState:
    id: int
    parent: int | None
    text: str | None
    reward: float
    seed: bool = False
    visits: int = 0
    best_child_reward: float | None = None

    @property
    def q(self):
        return self.reward if self.best_child_reward is None else self.best_child_reward

    def report(self):
        return {"id": self.id, "parent": self.parent, "reward": self.reward, "seed": self.seed,
                "visits": self.visits, "best_child_reward": self.best_child_reward, "q": self.q}


class DiscoveryArchive:
    """Keep candidate text only for selected archive members, retaining small ancestry records."""

    def __init__(self, seeds, capacity, exploration, children_per_expansion=2):
        if not seeds or capacity <= len(seeds):
            raise ValueError("archive_size must leave room for a child in addition to the seeds")
        if type(children_per_expansion) is not int or children_per_expansion < 1:
            raise ValueError("children_per_expansion must be a positive integer")
        self.children_per_expansion = children_per_expansion
        self.capacity = capacity
        self.exploration = exploration
        self.expansions = 0
        self.states = {}
        self.active = set()
        for text, reward in seeds:
            self._add(text, reward, parent=None)

    def _add(self, text, reward, parent):
        state_id = len(self.states)
        self.states[state_id] = DiscoveryState(state_id, parent, text, reward, seed=parent is None)
        self.active.add(state_id)
        return state_id

    def ranked(self):
        return sorted(self.active, key=lambda i: (-self.states[i].reward, i))

    def select(self):
        ranked = self.ranked()
        n = len(ranked)
        scale = self.states[ranked[0]].reward - self.states[ranked[-1]].reward
        total = n * (n + 1) / 2
        scores = []
        for rank, i in enumerate(ranked):
            state = self.states[i]
            prior = (n - rank) / total
            bonus = self.exploration * scale * prior * math.sqrt(1 + self.expansions) / (1 + state.visits)
            scores.append({"id": i, "q": state.q, "prior": prior, "bonus": bonus, "score": state.q + bonus})
        pick = max(scores, key=lambda row: (row["score"], -self.states[row["id"]].visits, -row["id"]))
        return self.states[pick["id"]], pick

    def expand(self, parent, children):
        if not children or parent not in self.active:
            raise ValueError("Expand an active state with newly scored candidates")
        best = max(r for _, r in children)
        node = self.states[parent]
        node.best_child_reward = best if node.best_child_reward is None else max(node.best_child_reward, best)
        while True:
            node.visits += 1
            if node.parent is None:
                break
            node = self.states[node.parent]
        self.expansions += 1
        seen = {self.states[i].text: i for i in self.active}
        inserted = []
        for text, reward in sorted(children, key=lambda item: -item[1]):
            if text in seen:
                previous = self.states[seen[text]]
                previous.reward = max(previous.reward, reward)
            elif len(inserted) < self.children_per_expansion:
                state_id = self._add(text, reward, parent)
                seen[text] = state_id
                inserted.append(state_id)
        ranked = self.ranked()
        kept = {i for i in ranked if self.states[i].seed}
        for i in ranked:
            if len(kept) >= self.capacity:
                break
            kept.add(i)
        for i in self.active - kept:
            self.states[i].text = None
        self.active = kept
        return inserted
