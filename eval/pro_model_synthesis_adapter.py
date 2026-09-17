"""Prompt-aware model synthesis for RSI and Phase-4 Pro branch collapse.

Branch generation, entropy routing, temperature ladders, RSI rounds, training, and
scoring stay unchanged. This adapter replaces only branch collapse: the already-
loaded model hierarchically synthesizes the candidate responses while seeing the
real benchmark task prompt and existing system/task instructions. Hidden expected
answers, verifier results, rewards, and test outcomes are not available during any
synthesis call. The existing hidden scorer/verifier runs only after the completed
synthesis has been selected.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple


def install(phase4_module) -> None:
    if getattr(phase4_module, "_pro_model_synthesis_adapter_installed", False):
        return

    def synthesize_pair(self, split: str, item: Dict[str, Any], left: str, right: str) -> str:
        original_user = phase4_module._task_user_prompt(split, item)
        user = (
            original_user
            + "\n\nPro synthesis step: below are two independent candidate responses generated for "
              "this exact task. Deliberate over both, reconcile disagreements, preserve the strongest "
              "reasoning from either one, and correct weaknesses you can identify from the task itself. "
              "You have NO hidden answer, reward, verifier result, pass/fail signal, or test outcome. "
              "Synthesize one best final response in the task's originally requested output format. "
              "Do not mention the candidates or the merge process.\n\n"
            + "CANDIDATE A:\n"
            + str(left)
            + "\n\nCANDIDATE B:\n"
            + str(right)
        )
        formatted = phase4_module._chat(
            self.engine.tokenizer,
            user,
            system=phase4_module.SYSTEM_PROMPT,
        )
        outputs = phase4_module._generate_branches_same_model(
            self,
            formatted,
            [0.20],
            max_tokens=min(phase4_module._benchmark_ceiling(self), 16384),
            top_p=0.92,
        )
        if not outputs:
            raise RuntimeError("Pro synthesis generated no response")
        return str(outputs[0])

    def choose_by_model_synthesis(
        self,
        split: str,
        item: Dict[str, Any],
        branches: List[str],
    ) -> Tuple[str, int, bool, str]:
        if not branches:
            return "", 0, False, "no branches"
        if len(branches) == 1:
            return str(branches[0]), 0, False, "single branch; no synthesis needed"

        contenders = [str(x) for x in branches]
        merge_calls = 0
        while len(contenders) > 1:
            next_round: List[str] = []
            for idx in range(0, len(contenders), 2):
                if idx + 1 >= len(contenders):
                    next_round.append(contenders[idx])
                    continue
                next_round.append(
                    synthesize_pair(self, split, item, contenders[idx], contenders[idx + 1])
                )
                merge_calls += 1
            contenders = next_round

        self._last_pro_synthesis_merge_calls = int(merge_calls)
        self._last_pro_synthesis_source_branches = len(branches)
        return (
            contenders[0],
            -1,
            False,
            f"prompt-aware hierarchical model synthesis ({len(branches)} branches + {merge_calls} merge calls)",
        )

    phase4_module._choose_without_ground_truth = choose_by_model_synthesis
    phase4_module._pro_model_synthesis_adapter_installed = True
