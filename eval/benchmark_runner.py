"""
Benchmark Evaluation Runner for Coding (HumanEval) and Mathematical Reasoning (GSM8K/MATH).
Computes pass@1 accuracy, entropy distribution, throughput latency, and memory telemetry.
"""

import json
import math
import os
import platform
import resource
import time
from typing import Any, Dict, List, Optional, Tuple

from config.settings import Settings, get_settings
from core.pro_engine import ProReasoningEngine
from core.verifier import GroundTruthVerifier
from eval.benchmark_data import HUMANEVAL_50_SUBSET, MATH_50_SUBSET


def get_current_rss_mb() -> float:
    """Returns current process Resident Set Size in Megabytes."""
    try:
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() == "Darwin":
            return ru / (1024 * 1024)
        else:
            return ru / 1024
    except Exception:
        return 0.0


class BenchmarkRunner:
    def __init__(self, engine: Optional[ProReasoningEngine] = None, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.engine = engine or ProReasoningEngine(settings=self.settings)
        self.verifier = GroundTruthVerifier(sandbox_timeout=self.settings.sandbox_timeout_seconds)

    def evaluate_subset(
        self,
        dataset_name: str,
        problem_subset: List[Dict[str, Any]],
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Evaluates model over a problem dataset and computes pass@1, entropy, tok/s, and memory."""
        total_problems = len(problem_subset)
        passed_count = 0
        total_tokens = 0
        total_time_s = 0.0
        entropy_values = []
        branch_counts = []
        detailed_results = []

        start_mem_mb = get_current_rss_mb()

        for idx, item in enumerate(problem_subset, 1):
            pid = item.get("id", f"item_{idx}")
            prompt = item["prompt"]
            test_cases = item.get("tests", "")

            t0 = time.perf_counter()
            response, meta = self.engine.solve(prompt, test_cases=test_cases)
            duration = max(0.001, time.perf_counter() - t0)

            # Token count approximation
            toks = len(response.split()) * 2
            total_tokens += toks
            total_time_s += duration

            is_verified = meta.get("verified", False)
            if not is_verified and test_cases:
                # Direct verification run if not run in engine
                res = self.verifier.verify_in_sandbox(response, test_cases)
                is_verified = res.passed

            if is_verified:
                passed_count += 1

            ent = meta.get("entropy", 0.35)
            entropy_values.append(ent)
            branch_counts.append(meta.get("branch_count", 1))

            detailed_results.append({
                "id": pid,
                "passed": is_verified,
                "duration_s": round(duration, 3),
                "tokens": toks,
                "entropy": round(ent, 3),
                "branch_count": meta.get("branch_count", 1)
            })

            if verbose and idx % 10 == 0:
                print(f"  [{dataset_name}] Processed {idx}/{total_problems} | Current Accuracy: {passed_count}/{idx} ({(passed_count/idx)*100:.1f}%)")

        peak_mem_mb = max(start_mem_mb, get_current_rss_mb())
        pass_at_1 = (passed_count / max(1, total_problems)) * 100.0
        tok_per_sec = total_tokens / max(0.001, total_time_s)
        mean_entropy = sum(entropy_values) / max(1, len(entropy_values))
        mean_branches = sum(branch_counts) / max(1, len(branch_counts))

        return {
            "dataset": dataset_name,
            "total_samples": total_problems,
            "passed_samples": passed_count,
            "pass_at_1_accuracy": round(pass_at_1, 2),
            "mean_entropy": round(mean_entropy, 3),
            "mean_branch_count": round(mean_branches, 1),
            "throughput_tok_per_sec": round(tok_per_sec, 1),
            "total_tokens": total_tokens,
            "total_duration_s": round(total_time_s, 2),
            "peak_rss_mb": round(peak_mem_mb, 1),
            "details": detailed_results
        }

    def run_full_suite(self, verbose: bool = True) -> Dict[str, Any]:
        """Runs the complete coding and math benchmark evaluation."""
        if verbose:
            print("================================================================")
            print("  STARTING ZERO-SHOT BENCHMARK EVALUATION (HumanEval + Math)")
            print("================================================================")

        t_start = time.perf_counter()

        humaneval_res = self.evaluate_subset("HumanEval-50", HUMANEVAL_50_SUBSET, verbose=verbose)
        math_res = self.evaluate_subset("GSM8K/MATH-50", MATH_50_SUBSET, verbose=verbose)

        total_samples = humaneval_res["total_samples"] + math_res["total_samples"]
        total_passed = humaneval_res["passed_samples"] + math_res["passed_samples"]
        overall_accuracy = (total_passed / max(1, total_samples)) * 100.0

        overall_metrics = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "overall_accuracy_percent": round(overall_accuracy, 2),
            "overall_passed_samples": total_passed,
            "overall_total_samples": total_samples,
            "total_evaluation_time_s": round(time.perf_counter() - t_start, 2),
            "humaneval": humaneval_res,
            "math": math_res
        }

        if verbose:
            print("\n================================================================")
            print("  BENCHMARK EVALUATION COMPLETED")
            print(f"  ► Coding (HumanEval-50): {humaneval_res['pass_at_1_accuracy']}% ({humaneval_res['passed_samples']}/{humaneval_res['total_samples']})")
            print(f"  ► Math (GSM8K/MATH-50):   {math_res['pass_at_1_accuracy']}% ({math_res['passed_samples']}/{math_res['total_samples']})")
            print(f"  ► Overall pass@1:         {overall_metrics['overall_accuracy_percent']}% ({total_passed}/{total_samples})")
            print(f"  ► Throughput:             {humaneval_res['throughput_tok_per_sec']} tok/s")
            print(f"  ► Peak Memory RSS:        {humaneval_res['peak_rss_mb']} MB")
            print("================================================================\n")

        return overall_metrics
