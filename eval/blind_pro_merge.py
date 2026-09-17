"""Answer-blind model synthesis for RSI and Phase-4 Pro branch merging.

The branch generator/router remain untouched. This replaces only branch selection:
independent candidates are recursively merged by the already-loaded model using the
public task prompt and candidate text. No benchmark expected answer, verifier result,
reward, hidden test, or pass/fail signal is available during generation or merging.
The single merged output is verified/scored only afterward by the existing pipeline.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple


def _token_count(tokenizer, text: str) -> int:
    try:
        return len(tokenizer.encode(str(text)))
    except Exception:
        return max(1, len(str(text)) // 4)


def merge_candidates(
    owner,
    task_prompt: str,
    candidates: List[str],
    *,
    phase4_module,
    runtime_module,
    context_limit: int,
    max_output_tokens: int,
    label: str = "Pro",
) -> str:
    """Hierarchically synthesize candidates without any correctness oracle."""
    if not candidates:
        return ""
    if len(candidates) == 1:
        return str(candidates[0])

    tokenizer = owner.engine.tokenizer
    contenders = [str(x) for x in candidates]
    round_idx = 0

    while len(contenders) > 1:
        round_idx += 1
        merged: List[str] = []
        for pos in range(0, len(contenders), 2):
            if pos + 1 >= len(contenders):
                merged.append(contenders[pos])
                continue

            a = contenders[pos]
            b = contenders[pos + 1]
            user = (
                f"{label} answer-blind branch merge, round {round_idx}.\n"
                "Solve the PUBLIC TASK by critically comparing the two independent candidate attempts below. "
                "Synthesize a single stronger final answer using the best reasoning, corrections, and useful parts "
                "from either candidate. You have NO expected answer, verifier result, reward, hidden tests, reference "
                "solution, or pass/fail feedback. Do not guess what a grader wants and do not mention the branch "
                "comparison in the final response. Preserve the output format requested by the public task.\n\n"
                f"PUBLIC TASK:\n{task_prompt}\n\n"
                f"CANDIDATE A:\n{a}\n\n"
                f"CANDIDATE B:\n{b}\n"
            )
            formatted = runtime_module._chat(
                tokenizer,
                user,
                system=phase4_module.SYSTEM_PROMPT,
            )
            prompt_tokens = _token_count(tokenizer, formatted)
            available = int(context_limit) - prompt_tokens
            if available <= 0:
                raise RuntimeError(
                    f"{label} blind merge pair exceeds the {context_limit:,}-token context; "
                    "refusing to truncate, compact, use a verifier, or fabricate a winner"
                )
            out_tokens = max(1, min(int(max_output_tokens), available))
            result = phase4_module._generate_branches_same_model(
                owner,
                formatted,
                [0.20],
                max_tokens=out_tokens,
                top_p=0.92,
            )
            if not result:
                raise RuntimeError(f"{label} blind merge produced no synthesis")
            merged.append(str(result[0]))
        contenders = merged

    return contenders[0]


def install(phase4_module, runtime_module) -> None:
    """Replace only _choose_without_ground_truth; scoring/reward stays downstream."""
    original = phase4_module._choose_without_ground_truth

    def choose(self, split: str, item: Dict[str, Any], branches: List[str]) -> Tuple[str, int, bool, str]:
        if not branches:
            return "", 0, False, "no branches"
        if len(branches) == 1:
            return branches[0], 0, False, "single branch"

        # Build the exact public task instruction already used by RSI/Phase 4.
        task_prompt = phase4_module._task_user_prompt(split, item)
        try:
            ceiling = int(phase4_module._benchmark_ceiling(self))
        except Exception:
            ceiling = int(runtime_module._benchmark_ceiling(self))
        ceiling = max(1024, ceiling)

        merged = merge_candidates(
            self,
            task_prompt,
            branches,
            phase4_module=phase4_module,
            runtime_module=runtime_module,
            context_limit=ceiling,
            max_output_tokens=min(16384, ceiling),
            label=f"{split} Pro",
        )

        # The synthesis is an additional virtual branch after the N sampled branches.
        # verified=False is intentional: verification/scoring happens only downstream.
        return (
            merged,
            len(branches),
            False,
            f"answer-blind model synthesis across {len(branches)} branches; no verifier/reward used",
        )

    choose.__name__ = getattr(original, "__name__", "_choose_without_ground_truth")
    phase4_module._choose_without_ground_truth = choose
