"""Prompt-aware answer-hidden Pro synthesis for RSI and Phase-4 evaluation.

The existing entropy router, temperature ladder, branch generator, RSI rounds, and
post-selection scorer remain unchanged. This module changes only branch collapse:
the same loaded model sees the public task plus candidate responses and synthesizes
a stronger final response. Expected answers, hidden tests, verifier output, rewards,
and pass/fail signals are never included. Only the synthesized final response is
scored afterward by the existing pipeline.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple


def install(phase4_module) -> None:
    if getattr(phase4_module, "_model_pro_synthesis_installed", False):
        return

    def _merge_two(
        self,
        split: str,
        item: Dict[str, Any],
        left: str,
        right: str,
        round_idx: int,
    ) -> str:
        original_task = phase4_module._task_user_prompt(split, item)
        merge_user = (
            "Solve the ORIGINAL TASK using both independently generated candidate reasoning paths as working material. "
            "Synthesize one stronger final response: preserve useful/correct insights, repair contradictions or weak steps, "
            "and reason through the task yourself. Do not vote by wording and do not merely choose A or B. "
            "You have NO expected answer, hidden tests, verifier result, reward, pass/fail signal, or reference solution. "
            "Do not mention the candidates in the final response. Obey the ORIGINAL TASK output format exactly.\n\n"
            f"ORIGINAL TASK:\n{original_task}\n\n"
            f"CANDIDATE A:\n{left}\n\n"
            f"CANDIDATE B:\n{right}\n\n"
            f"SYNTHESIS ROUND: {round_idx}"
        )
        formatted = phase4_module._chat(
            self.engine.tokenizer,
            merge_user,
            system=phase4_module.SYSTEM_PROMPT,
        )
        ceiling = int(phase4_module._benchmark_ceiling(self))
        try:
            prompt_tokens = len(self.engine.tokenizer.encode(formatted))
        except Exception:
            prompt_tokens = 0
        if prompt_tokens and prompt_tokens >= ceiling:
            raise RuntimeError(
                f"Pro synthesis prompt requires {prompt_tokens} tokens but benchmark ceiling is {ceiling}; "
                "refusing to truncate, summarize, or fabricate candidate reasoning"
            )
        max_out = min(16384, max(1, ceiling - prompt_tokens)) if prompt_tokens else min(16384, ceiling)
        merged = phase4_module._generate_branches_same_model(
            self,
            formatted,
            [0.20],
            max_tokens=max_out,
            top_p=0.92,
        )
        if not merged:
            raise RuntimeError("Pro synthesis generated no merged response")
        return str(merged[0])

    def synthesize_without_ground_truth(
        self,
        split: str,
        item: Dict[str, Any],
        branches: List[str],
    ) -> Tuple[str, int, bool, str]:
        if not branches:
            return "", 0, False, "no branches"
        if len(branches) == 1:
            return branches[0], 0, False, "single Pro branch"

        contenders = [str(x) for x in branches]
        merge_round = 1
        while len(contenders) > 1:
            next_round: List[str] = []
            for pos in range(0, len(contenders), 2):
                if pos + 1 >= len(contenders):
                    next_round.append(contenders[pos])
                    continue
                next_round.append(
                    _merge_two(
                        self,
                        split,
                        item,
                        contenders[pos],
                        contenders[pos + 1],
                        merge_round,
                    )
                )
            contenders = next_round
            merge_round += 1

        return (
            contenders[0],
            0,
            False,
            f"prompt-aware model synthesis across {len(branches)} Pro branches; no hidden scoring used",
        )

    phase4_module._choose_without_ground_truth = synthesize_without_ground_truth
    phase4_module._model_pro_synthesis_installed = True
