"""Prompt-aware Pro branch synthesis using the already-existing Pro machinery.

This module does not replace entropy routing, branch generation, temperature
laddering, model backends, or runtime hardening. It only changes how N>1 branches
collapse: the same loaded model hierarchically synthesizes candidate responses using
the real user prompt/history. No hidden answer, reward, verifier result, or tests are
available during synthesis. If explicit tests were supplied to solve(), the existing
solver may verify only the single finished synthesis afterward.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def install_pro_model_synthesis(cls) -> None:
    if getattr(cls, "_pro_model_synthesis_installed", False):
        return

    original_generate = cls.generate_parallel_branches
    original_stream = cls.stream_solve
    original_solve = cls.solve

    def _merge_pair(self, prompt: str, history, left: str, right: str) -> str:
        merge_prompt = (
            "Use the original user request and the two independent candidate solutions below as parallel "
            "reasoning attempts. Deliberate over them, reconcile disagreements, preserve the strongest "
            "correct reasoning from either one, fix weaknesses you can identify from the request itself, "
            "and synthesize one best final response. You have NO hidden answer, reward, verifier result, "
            "or test outcome. Do not mention the candidates or this merge process. Return only the final "
            "response that should be sent to the user, preserving any format the user requested.\n\n"
            f"ORIGINAL USER REQUEST:\n{prompt}\n\n"
            f"CANDIDATE A:\n{left}\n\n"
            f"CANDIDATE B:\n{right}"
        )
        merged = original_generate(
            self,
            merge_prompt,
            branch_count=1,
            history=history,
            temperatures=[0.20],
        )
        if not merged:
            raise RuntimeError("Pro synthesis produced no response")
        return str(merged[0])

    def _synthesize(self, prompt: str, history, candidates: List[str]) -> tuple[str, int]:
        if not candidates:
            raise RuntimeError("Pro branch generation returned no candidates")
        if len(candidates) == 1:
            return str(candidates[0]), 0

        contenders = [str(x) for x in candidates]
        merge_calls = 0
        while len(contenders) > 1:
            next_round: List[str] = []
            for idx in range(0, len(contenders), 2):
                if idx + 1 >= len(contenders):
                    next_round.append(contenders[idx])
                    continue
                next_round.append(
                    _merge_pair(self, prompt, history, contenders[idx], contenders[idx + 1])
                )
                merge_calls += 1
            contenders = next_round
        return contenders[0], merge_calls

    def generate_parallel_branches_with_synthesis(
        self,
        prompt: str,
        branch_count: int = 16,
        history: Optional[List[Dict[str, str]]] = None,
        temperatures: Optional[List[float]] = None,
    ) -> List[str]:
        raw = original_generate(
            self,
            prompt,
            branch_count=branch_count,
            history=history,
            temperatures=temperatures,
        )
        self._last_pro_raw_branches = list(raw or [])
        self._last_pro_requested_branch_count = int(branch_count)
        self._last_pro_synthesis_merge_calls = 0

        if branch_count <= 1 or len(raw or []) <= 1:
            return list(raw or [])

        final, merge_calls = _synthesize(self, prompt, history, list(raw))
        self._last_pro_synthesis_merge_calls = int(merge_calls)
        self._last_pro_synthesized_response = final

        # Return only the completed synthesis to the existing solve() selection layer.
        # Therefore any explicit verifier/test supplied by a caller can run only once,
        # after all branch generation and synthesis are finished.
        return [final]

    def solve_with_synthesis_metadata(self, *args, **kwargs):
        self._last_pro_raw_branches = []
        self._last_pro_requested_branch_count = 0
        self._last_pro_synthesis_merge_calls = 0
        self._last_pro_synthesized_response = None

        response, metadata = original_solve(self, *args, **kwargs)
        raw = list(getattr(self, "_last_pro_raw_branches", []) or [])
        requested = int(getattr(self, "_last_pro_requested_branch_count", 0) or 0)
        merges = int(getattr(self, "_last_pro_synthesis_merge_calls", 0) or 0)

        if requested > 1 and raw:
            metadata = dict(metadata or {})
            metadata["branch_count"] = requested
            metadata["raw_branches"] = raw
            metadata["synthesis_merge_calls"] = merges
            metadata["selection"] = "prompt-aware hierarchical model synthesis"
            metadata["synthesized"] = True
            metadata["synthesis_generation_units"] = requested + merges
            # Keep legacy numeric winner fields compatible with existing UI/DB readers;
            # the selection field above is authoritative: output is a new synthesis,
            # not one of the original branch indices.
            metadata["winning_branch_is_synthesis"] = True
        return response, metadata

    def stream_solve_with_pro_routing(
        self,
        prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.75,
        top_p: float = 0.92,
        cancel_event: Optional[Any] = None,
    ):
        # Restore the intended Gemini-era contract: every chat prompt reaches the
        # entropy router. Easy prompts remain the existing low-latency stream; only
        # prompts routed to N>1 pay the Pro search/synthesis cost.
        entropy = float(self.calculate_token_entropy(prompt))
        mode, branch_count = self.router.route(entropy, has_test_cases=False)
        if int(branch_count) <= 1:
            yield from original_stream(
                self,
                prompt,
                history=history,
                temperature=temperature,
                top_p=top_p,
                cancel_event=cancel_event,
            )
            return

        response, metadata = self.solve(
            prompt=prompt,
            history=history,
            cancel_event=cancel_event,
            force_branch_count=int(branch_count),
            # N>1 intentionally uses the existing convex temperature ladder rather
            # than forcing every branch to the stream's single temperature.
            temperature=None,
        )
        metadata = dict(metadata or {})
        metadata["entropy"] = entropy
        metadata["mode"] = mode
        self._last_stream_pro_meta = metadata

        # Pro must finish deliberating before there is a single answer to stream.
        # Emit the final synthesis without fabricating an intermediate answer.
        text = str(response or "")
        step = 64
        for idx in range(0, len(text), step):
            if cancel_event and cancel_event.is_set():
                break
            yield text[idx : idx + step]

    cls.generate_parallel_branches = generate_parallel_branches_with_synthesis
    cls.solve = solve_with_synthesis_metadata
    cls.stream_solve = stream_solve_with_pro_routing
    cls._pro_model_synthesis_installed = True
