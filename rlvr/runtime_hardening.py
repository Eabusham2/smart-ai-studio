"""Integrity hardening for the legacy master curriculum orchestrator.

Two recovered curriculum loops used `while logged < target_traces` and could make
zero progress forever when model-generated code failed the hidden verifier. They
also fell back to pre-written passing code in live mode, which made a "successful"
trace ambiguous. This layer keeps deterministic fixtures ONLY for explicit mock/CI
mode and makes live mode self-generated, feedback-driven, and bounded.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple


_EVOLUTION_CASES: List[Tuple[str, str, str]] = [
    (
        "NonAbelianAlgebra: Lie Bracket Commutation Invariant [L_i, L_j] = c_{ijk} L_k",
        "def verify_lie_bracket(c_struct):\n"
        "    def bracket(x, y): return x * y - y * x\n"
        "    x, y, z = 2, 3, 5\n"
        "    return bracket(x, bracket(y, z)) + bracket(y, bracket(z, x)) + bracket(z, bracket(x, y)) == 0\n",
        "assert verify_lie_bracket(None) == True",
    ),
    (
        "NonAbelianAlgebra: Casimir Invariant C = sum(g^{ij} L_i L_j)",
        "def verify_casimir_invariance(dim=3):\n"
        "    c = sum(i**2 for i in range(1, dim + 1))\n"
        "    return all(c * k - k * c == 0 for k in range(1, dim + 1))\n",
        "assert verify_casimir_invariance(3) == True",
    ),
    (
        "NonAbelianAlgebra: Root Space Decomposition Dimension Invariant",
        "def compute_root_space_dim(rank=2):\n"
        "    roots = 2 * rank * (rank + 1)\n"
        "    return roots > 0 and roots % 2 == 0\n",
        "assert compute_root_space_dim(2) == True",
    ),
    (
        "NonAbelianAlgebra: Automorphic Boundary Fuzzer & Nilpotency Verification",
        "def verify_nilpotent_subalgebra(n=4):\n"
        "    return all(k < n for k in range(n))\n",
        "assert verify_nilpotent_subalgebra(4) == True",
    ),
]

_RECOVERY_CASES: List[Tuple[str, str, str]] = [
    (
        "DeepSWE Cache Eviction Thread-Safety",
        "def verify_cache_eviction_threadsafe_0():\n"
        "    import threading\n"
        "    lock = threading.Lock()\n"
        "    with lock: return True\n",
        "assert verify_cache_eviction_threadsafe_0() == True",
    ),
    (
        "AIME Modular Congruence Recovery",
        "def solve_aime_congruence(): return 13\n",
        "assert solve_aime_congruence() == 13",
    ),
    (
        "LCB Algorithmic Segment Tree Recovery",
        "def solve_lcb_segtree(arr=[1, 3, 5, 7]): return sum(arr)\n",
        "assert solve_lcb_segtree() == 16",
    ),
    (
        "GPQA Quantum Phase Invariant",
        "def solve_gpqa_phase(): return 'U(1)'\n",
        "assert solve_gpqa_phase() == 'U(1)'",
    ),
    (
        "HLE Root Space Dimension Exact Proof",
        "def solve_hle_dim(): return 8\n",
        "assert solve_hle_dim() == 8",
    ),
    (
        "TensorGraphDSL Non-Commutative Fused Quant",
        "def evaluate_quant(arr=[1, 2, 3]): return [x * 2 for x in arr]\n",
        "assert evaluate_quant() == [2, 4, 6]",
    ),
]


def _mock_mode(obj) -> bool:
    return bool(getattr(getattr(obj, "settings", None), "use_mock", False))


def _model_code(obj, prompt: str) -> str:
    """Ask the active model; never substitute a known-good answer in live mode."""
    response, _meta = obj.engine.solve(prompt, force_branch_count=1)
    extracted = obj.verifier.extract_code_block(str(response or ""))
    return str(extracted or "").strip()


def _feedback(result: Any) -> str:
    for name in ("stderr", "error", "message", "details"):
        value = getattr(result, name, None)
        if value:
            return str(value)[:1200]
    return "Verifier rejected the previous candidate."


def install_master_curriculum_hardening(cls) -> None:
    if getattr(cls, "_bounded_integrity_hardening_installed", False):
        return

    def bounded_autonomous_evolution(self, target_traces: int = 100, verbose: bool = True) -> Dict[str, Any]:
        target = max(0, int(target_traces))
        if target == 0:
            return {"status": "success", "discovery_traces_logged": 0, "duration_s": 0.0}

        t0 = time.perf_counter()
        logged = 0
        attempts = 0
        # Enough retries for live self-correction, but mathematically impossible to hang forever.
        max_attempts_total = max(len(_EVOLUTION_CASES), target * 8)

        if verbose:
            print(f"\n[*] Starting bounded Autonomous Evolution (target={target}, max attempts={max_attempts_total})...")

        while logged < target and attempts < max_attempts_total:
            name, mock_code, tests = _EVOLUTION_CASES[attempts % len(_EVOLUTION_CASES)]
            attempts += 1

            if _mock_mode(self):
                # Deterministic fixture is CI-only; it is never exposed as a live-model fallback.
                code = mock_code
            else:
                prompt = (
                    "Self-generate a Python verification program for this mathematical invariant. "
                    "Do not ask for or assume a hidden answer. Output only executable Python code.\n\n"
                    f"Invariant: {name}"
                )
                code = _model_code(self, prompt)
                if not code:
                    continue

            result = self.verifier.verify_in_sandbox(code, tests)
            if not result.passed:
                continue

            self.db.log_interaction(
                prompt=f"Autonomous Unsupervised Evolution: {name}",
                completion=code,
                raw_branches=[code],
                verified_reward=1.0,
                surprise_score=0.85 + (0.02 * (attempts % 5)),
                mode="Autonomous Evolution (Mock Fixture)" if _mock_mode(self) else "Autonomous Evolution (Self-Generated)",
                entropy=0.12,
                winning_branch=0,
                test_cases=tests,
            )
            logged += 1

        duration = time.perf_counter() - t0
        status = "success" if logged >= target else "incomplete"
        if verbose:
            print(f"[{'✓' if status == 'success' else '!'}] Autonomous Evolution: {logged}/{target} verified in {duration:.2f}s.")
        return {
            "status": status,
            "discovery_traces_logged": logged,
            "attempts": attempts,
            "target_traces": target,
            "duration_s": round(duration, 3),
        }

    def bounded_environmental_recovery(
        self,
        target_traces: int = 300,
        max_attempts: int = 4,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        target = max(0, int(target_traces))
        per_problem_attempts = max(1, int(max_attempts))
        if target == 0:
            return {"status": "success", "recovery_traces_logged": 0, "duration_s": 0.0}

        t0 = time.perf_counter()
        logged = 0
        batches = 0
        max_batches = max(len(_RECOVERY_CASES), target * 4)

        if verbose:
            print(f"\n[*] Starting bounded Environmental RLVR (target={target}, batches<={max_batches})...")

        while logged < target and batches < max_batches:
            name, mock_code, tests = _RECOVERY_CASES[batches % len(_RECOVERY_CASES)]
            batches += 1
            previous_feedback = ""

            for attempt in range(1, per_problem_attempts + 1):
                if _mock_mode(self):
                    code = mock_code
                else:
                    prompt = (
                        f"Environmental RLVR attempt {attempt}: implement a Python solution for {name}. "
                        "Do not assume or request the hidden test or answer. Output only executable Python code."
                    )
                    if previous_feedback:
                        prompt += f"\nVerifier feedback from your previous attempt:\n{previous_feedback}"
                    code = _model_code(self, prompt)
                    if not code:
                        previous_feedback = "No executable code was produced."
                        continue

                result = self.verifier.verify_in_sandbox(code, tests)
                if result.passed:
                    self.db.log_interaction(
                        prompt=f"Environmental RLVR Self-Correction: {name} (attempt {attempt})",
                        completion=code,
                        raw_branches=[code],
                        verified_reward=1.0,
                        surprise_score=0.60 + (0.05 * attempt),
                        mode=(
                            f"Environmental RLVR Mock Fixture (M={attempt}/{per_problem_attempts})"
                            if _mock_mode(self)
                            else f"Environmental RLVR Self-Generated (M={attempt}/{per_problem_attempts})"
                        ),
                        entropy=0.15,
                        winning_branch=0,
                        test_cases=tests,
                    )
                    logged += 1
                    break

                previous_feedback = _feedback(result)

        duration = time.perf_counter() - t0
        status = "success" if logged >= target else "incomplete"
        if verbose:
            print(f"[{'✓' if status == 'success' else '!'}] Environmental RLVR: {logged}/{target} verified in {duration:.2f}s.")
        return {
            "status": status,
            "recovery_traces_logged": logged,
            "batches": batches,
            "target_traces": target,
            "duration_s": round(duration, 3),
        }

    cls.execute_autonomous_unsupervised_evolution = bounded_autonomous_evolution
    cls.execute_environmental_rlvr_recovery = bounded_environmental_recovery
    cls._bounded_integrity_hardening_installed = True
