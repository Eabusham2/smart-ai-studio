"""Prompt-aware DeepSWE branch synthesis without touching shared Pro code.

The official public DeepSWE task instruction and candidate patches are visible to
the already-loaded model. Hidden tests, verifier output, rewards, reference patches,
and pass/fail signals are never provided during synthesis. Candidate patches are
hierarchically merged into a NEW final patch; only that completed patch is verified
by the existing DeepSWE runner afterward.
"""
from __future__ import annotations


def install(deep_module, phase4_module, runtime_module) -> None:
    original_generate = deep_module._generate_patch
    current_task = {"dir": None}

    def generate(pier, task_dir, base_url, universal_rule, stage, temperature, prior_patch=""):
        current_task["dir"] = task_dir
        return original_generate(
            pier, task_dir, base_url, universal_rule, stage, temperature, prior_patch
        )

    def merge_pair(owner, instruction: str, left: str, right: str) -> str:
        user = (
            "Solve the public software-engineering task below by synthesizing the strongest parts of two "
            "independently generated candidate patches. Reconcile disagreements, preserve correct changes, "
            "remove regressions or unnecessary edits you can identify from the public task/code diff alone, "
            "and produce one best final patch. You have NO hidden tests, verifier result, reward, reference "
            "patch, correct answer, or pass/fail signal. Do not mention the candidates or merge process. "
            "Output ONLY the final unified diff patch.\n\n"
            f"PUBLIC TASK:\n{instruction}\n\n"
            f"CANDIDATE A:\n```diff\n{left}\n```\n\n"
            f"CANDIDATE B:\n```diff\n{right}\n```"
        )
        formatted = runtime_module._chat(
            owner.engine.tokenizer,
            user,
            system=phase4_module.SYSTEM_PROMPT,
        )
        ceiling = min(
            int(getattr(deep_module, "DEEPSWE_CONTEXT_TOKENS", 226000)),
            int(runtime_module._model_context_limit(owner.engine) or getattr(deep_module, "DEEPSWE_CONTEXT_TOKENS", 226000)),
        )
        try:
            prompt_tokens = len(owner.engine.tokenizer.encode(formatted))
        except Exception:
            prompt_tokens = 0
        if prompt_tokens and prompt_tokens >= ceiling:
            raise RuntimeError(
                f"DeepSWE Pro synthesis prompt requires {prompt_tokens} tokens but model context is {ceiling}; "
                "refusing truncation/compaction."
            )
        max_output = min(
            int(getattr(deep_module, "DEEPSWE_MAX_OUTPUT_TOKENS", 8192)),
            max(1, ceiling - prompt_tokens),
        )
        merged = phase4_module._generate_branches_same_model(
            owner,
            formatted,
            [0.20],
            max_tokens=max_output,
            top_p=0.92,
        )
        if not merged:
            raise RuntimeError("DeepSWE Pro synthesis produced no patch")
        return str(merged[0])

    def choose(phase4, owner, patches):
        if not patches:
            return "", -1, "no branches"
        if len(patches) == 1:
            return str(patches[0]), -1, "single branch; no synthesis needed"

        task_dir = current_task.get("dir")
        if task_dir is None:
            raise RuntimeError("DeepSWE synthesis has no current public task context")
        instruction = deep_module._task_instruction(task_dir)
        contenders = [str(p) for p in patches]
        merge_calls = 0
        while len(contenders) > 1:
            next_round = []
            for pos in range(0, len(contenders), 2):
                if pos + 1 >= len(contenders):
                    next_round.append(contenders[pos])
                    continue
                next_round.append(
                    merge_pair(owner, instruction, contenders[pos], contenders[pos + 1])
                )
                merge_calls += 1
            contenders = next_round

        owner._last_deepswe_synthesis_merge_calls = merge_calls
        owner._last_deepswe_synthesis_source_branches = len(patches)
        return (
            contenders[0],
            -1,
            f"prompt-aware hierarchical DeepSWE model synthesis ({len(patches)} branches + {merge_calls} merge calls)",
        )

    deep_module._generate_patch = generate
    deep_module._blind_select = choose
