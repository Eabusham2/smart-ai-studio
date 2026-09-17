"""Answer-blind DeepSWE branch selection without touching the shared Pro engine.

Exact duplicate patches still use consensus. When candidate patches are all unique,
the already-loaded model runs a pairwise tournament using only the public DeepSWE
instruction and the two candidate diffs. No verifier, test result, reward, reference
solution, or hidden answer is available until after one winner has been selected.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter


def install(deep_module, phase4_module, runtime_module) -> None:
    original_generate = deep_module._generate_patch
    current_task = {"dir": None}

    def generate(pier, task_dir, base_url, universal_rule, stage, temperature, prior_patch=""):
        current_task["dir"] = task_dir
        return original_generate(
            pier, task_dir, base_url, universal_rule, stage, temperature, prior_patch
        )

    def choose(phase4, owner, patches):
        if not patches:
            return "", 0, "no branches"

        hashes = [hashlib.sha256(p.encode("utf-8", errors="replace")).hexdigest() for p in patches]
        counts = Counter(hashes)
        winner_hash, votes = counts.most_common(1)[0]
        if votes > 1:
            idx = hashes.index(winner_hash)
            return patches[idx], idx, f"answer-blind exact-patch consensus {votes}/{len(patches)}"

        task_dir = current_task.get("dir")
        if task_dir is None:
            return patches[0], 0, "answer-blind deterministic fallback (task unavailable)"
        instruction = deep_module._task_instruction(task_dir)
        contenders = list(enumerate(patches))
        tokenizer = owner.engine.tokenizer
        detected = runtime_module._model_context_limit(owner.engine)
        ceiling = min(
            int(getattr(deep_module, "DEEPSWE_CONTEXT_TOKENS", 226000)),
            int(detected) if detected else int(getattr(deep_module, "DEEPSWE_CONTEXT_TOKENS", 226000)),
        )

        while len(contenders) > 1:
            next_round = []
            for pos in range(0, len(contenders), 2):
                if pos + 1 >= len(contenders):
                    next_round.append(contenders[pos])
                    continue
                a_idx, a_patch = contenders[pos]
                b_idx, b_patch = contenders[pos + 1]
                user = (
                    "You are selecting between two independently generated candidate patches for the same "
                    "software-engineering task. You have NO verifier result, hidden tests, reference patch, "
                    "reward, or correct answer. Judge only from the public task and the diffs. Pick the patch "
                    "more likely to correctly solve the issue without regressions. Output ONLY A or B.\n\n"
                    f"PUBLIC TASK:\n{instruction}\n\n"
                    f"CANDIDATE A:\n```diff\n{a_patch}\n```\n\n"
                    f"CANDIDATE B:\n```diff\n{b_patch}\n```"
                )
                formatted = runtime_module._chat(
                    tokenizer,
                    user,
                    system=phase4_module.SYSTEM_PROMPT,
                )
                try:
                    prompt_tokens = len(tokenizer.encode(formatted))
                except Exception:
                    prompt_tokens = 0
                if prompt_tokens and prompt_tokens + 32 > ceiling:
                    # Never truncate/compact/fabricate candidate patches. If the honest
                    # pair does not fit, preserve deterministic answer-blind behavior.
                    next_round.append((a_idx, a_patch))
                    continue
                output = phase4_module._generate_branches_same_model(
                    owner,
                    formatted,
                    [0.20],
                    max_tokens=32,
                    top_p=0.90,
                )[0]
                cleaned = str(output).strip()
                if "</think>" in cleaned:
                    cleaned = cleaned.split("</think>", 1)[1].strip()
                letters = re.findall(r"\b([AB])\b", cleaned.upper())
                pick_b = bool(letters and letters[-1] == "B")
                next_round.append((b_idx, b_patch) if pick_b else (a_idx, a_patch))
            contenders = next_round

        idx, patch = contenders[0]
        return patch, idx, f"answer-blind pairwise model tournament across {len(patches)} branches"

    deep_module._generate_patch = generate
    deep_module._blind_select = choose
