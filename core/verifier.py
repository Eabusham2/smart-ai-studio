import ast
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

@dataclass
class VerificationResult:
    passed: bool
    execution_time_ms: float
    output: str
    error: Optional[str] = None
    details: str = ""
    reward: float = 0.0

class MultiFileGitSWESandbox:
    def __init__(self, timeout: float = 4.0, max_memory_mb: int = 512):
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb

    def verify_patch(self, repo_structure: Dict[str, str], patch_text: str, test_cmd: str) -> VerificationResult:
        t0 = time.perf_counter()
        tmp_dir = tempfile.mkdtemp(prefix="swe_sandbox_")
        try:
            for rel_path, file_content in repo_structure.items():
                full_p = os.path.join(tmp_dir, rel_path)
                os.makedirs(os.path.dirname(full_p), exist_ok=True)
                with open(full_p, "w", encoding="utf-8") as f:
                    f.write(file_content)

            patch_clean = re.findall(r"```(?:diff|patch)?\s*([\s\S]*?)```", patch_text, re.IGNORECASE)
            actual_patch = patch_clean[-1].strip() if patch_clean else patch_text.strip()
            
            patch_file = os.path.join(tmp_dir, "change.patch")
            with open(patch_file, "w", encoding="utf-8") as f:
                f.write(actual_patch + "\n")

            subprocess.run(["patch", "-p1", "-i", "change.patch"], cwd=tmp_dir, capture_output=True, text=True, timeout=2.0)
            proc = subprocess.run(test_cmd, shell=True, cwd=tmp_dir, capture_output=True, text=True, timeout=self.timeout)
            exec_time_ms = (time.perf_counter() - t0) * 1000.0

            if proc.returncode == 0:
                return VerificationResult(True, exec_time_ms, proc.stdout, details="SWE repository assertions passed.", reward=1.0)
            return VerificationResult(False, exec_time_ms, proc.stdout, error=proc.stderr, details=f"SWE Test failed (rc={proc.returncode}): {proc.stderr[:180]}", reward=0.0)
        except Exception as e:
            exec_time_ms = (time.perf_counter() - t0) * 1000.0
            return VerificationResult(False, exec_time_ms, "", error=str(e), details=f"SWE Error: {str(e)}", reward=0.0)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

class TensorGraphDSLEvaluator:
    @staticmethod
    def evaluate_expression(expr: str) -> Optional[List[int]]:
        expr = expr.strip()
        m_fold_scale = re.search(r"\[([0-9,\s\-]+)\]\s*>>~fold\((\d+)\)\s*<#>scale\((\d+)\)", expr)
        if m_fold_scale:
            arr = [int(x.strip()) for x in m_fold_scale.group(1).split(",") if x.strip()]
            k = int(m_fold_scale.group(2)) % len(arr) if arr else 0
            s = int(m_fold_scale.group(3))
            return [x * s for x in (arr[k:] + arr[:k])]
        
        m_fuse = re.search(r"\[([0-9,\s\-]+)\]\s*@fuse\s*\[([0-9,\s\-]+)\]", expr)
        if m_fuse:
            a = [int(x.strip()) for x in m_fuse.group(1).split(",") if x.strip()]
            b = [int(x.strip()) for x in m_fuse.group(2).split(",") if x.strip()]
            return [x + y for x, y in zip(a, b)]
        return None

class MathAnswerNormalizer:
    @staticmethod
    def extract_boxed(text: str) -> Optional[str]:
        idx = text.rfind("\\boxed{")
        if idx == -1:
            return None
        depth = 0
        start = idx + len("\\boxed{")
        for i in range(start - 1, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i].strip()
        return None

class GroundTruthVerifier:
    def __init__(self, sandbox_timeout: float = 4.0, max_memory_mb: int = 512):
        self.sandbox_timeout = sandbox_timeout
        self.max_memory_mb = max_memory_mb
        self.swe_sandbox = MultiFileGitSWESandbox(timeout=sandbox_timeout, max_memory_mb=max_memory_mb)
        self.dsl_evaluator = TensorGraphDSLEvaluator()
        self.math_normalizer = MathAnswerNormalizer()

    def extract_code(self, text: str, language: str = "python") -> str:
        matches = re.findall(rf"```{language}\s*([\s\S]*?)```", text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()
        generic = re.findall(r"```\s*([\s\S]*?)```", text)
        if generic:
            return generic[-1].strip()
        lines = []
        capture = False
        for line in text.splitlines():
            if line.startswith(("def ", "class ", "import ", "from ")):
                capture = True
            if capture:
                lines.append(line)
        return "\n".join(lines).strip() if lines else text.strip()

    def verify_in_sandbox(self, code: str, test_harness: str) -> VerificationResult:
        script = f"# -*- coding: utf-8 -*-\nimport sys, math, json, collections, itertools\n\n{code}\n\n# TESTS\n{test_harness}\n"
        t0 = time.perf_counter()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tmp:
            tmp.write(script)
            tmp_p = tmp.name
        try:
            res = subprocess.run([sys.executable, tmp_p], capture_output=True, text=True, timeout=self.sandbox_timeout)
            duration_ms = (time.perf_counter() - t0) * 1000.0
            if res.returncode == 0:
                return VerificationResult(True, duration_ms, res.stdout, details="Unit assertions passed.", reward=1.0)
            return VerificationResult(False, duration_ms, res.stdout, error=res.stderr, details=f"Failed (rc={res.returncode}): {res.stderr[:180]}", reward=0.0)
        except subprocess.TimeoutExpired:
            return VerificationResult(False, (time.perf_counter() - t0) * 1000.0, "", error="TimeoutExpired", details="Sandbox timed out.", reward=0.0)
        except Exception as e:
            return VerificationResult(False, (time.perf_counter() - t0) * 1000.0, "", error=str(e), details=str(e), reward=0.0)
        finally:
            if os.path.exists(tmp_p):
                os.remove(tmp_p)

    def verify_tool_schema(self, output: str, expected_tool: str) -> VerificationResult:
        passed = expected_tool.strip().lower() in output.lower()
        return VerificationResult(passed, 0.1, output, details=f"Tool '{expected_tool}' match: {passed}", reward=1.0 if passed else 0.0)

    def verify_math_olympiad(self, output: str, expected_answer: Union[str, int]) -> VerificationResult:
        exp_str = str(expected_answer).strip().lower()
        boxed = self.math_normalizer.extract_boxed(output)
        if boxed and boxed.strip().lower() == exp_str:
            return VerificationResult(True, 0.1, output, details=f"Boxed answer \\boxed{{{boxed}}} verified.", reward=1.0)
        passed = exp_str in output.lower()
        return VerificationResult(passed, 0.1, output, details=f"Target '{exp_str}' in output: {passed}", reward=1.0 if passed else 0.0)

    def verify_item(self, item: Dict[str, Any], output: str) -> VerificationResult:
        if "repo_files" in item and "test_patch" in item:
            return self.swe_sandbox.verify_patch(item["repo_files"], output, item.get("test_command", "python3 -m unittest"))
            
        if "test_cases" in item or "test_patch" in item:
            code = self.extract_code(output, "python")
            return self.verify_in_sandbox(code, item.get("test_cases") or item.get("test_patch", ""))

        if "correct_letter" in item:
            let = str(item["correct_letter"]).strip().upper()
            patterns = [rf"\*\*({let})\*\*", rf"\(({let})\)", rf"\bOption\s*({let})\b", rf"\bChoice\s*({let})\b", rf":\s*({let})\b"]
            passed = any(re.search(p, output, re.IGNORECASE) for p in patterns) or output.strip().startswith(let)
            return VerificationResult(passed, 0.1, output, details=f"Option [{let}] matched: {passed}", reward=1.0 if passed else 0.0)

        if "expected_integer" in item:
            exp_int = f"{int(item['expected_integer']):03d}"
            nums = re.findall(r"\b\d{1,3}\b", output)
            passed = any(f"{int(n):03d}" == exp_int for n in nums)
            return VerificationResult(passed, 0.1, output, details=f"AIME Target: {exp_int} in output -> {passed}", reward=1.0 if passed else 0.0)

        if "expected_answer" in item:
            return self.verify_math_olympiad(output, item["expected_answer"])

        if "expected_tool" in item:
            return self.verify_tool_schema(output, str(item["expected_tool"]))

        if "expected_result" in item:
            exp_res = str(item["expected_result"]).strip().replace(" ", "")
            passed = exp_res in output.replace(" ", "")
            return VerificationResult(passed, 0.1, output, details=f"DSL Invariant: {exp_res} -> {passed}", reward=1.0 if passed else 0.0)

        if "expected_fact" in item:
            exp_fact = str(item["expected_fact"]).lower()
            kws = [w for w in exp_fact.split() if len(w) > 3]
            match_count = sum(1 for kw in kws if kw in output.lower())
            passed = match_count >= max(1, len(kws) // 2)
            return VerificationResult(passed, 0.1, output, details=f"Memory keywords: {match_count}/{len(kws)}", reward=1.0 if passed else 0.0)

        if "discovery_target" in item or "expected_target" in item:
            tgt = str(item.get("discovery_target") or item.get("expected_target")).lower().strip()
            passed = tgt in output.lower()
            return VerificationResult(passed, 0.1, output, details=f"Discovery target [{tgt}] -> {passed}", reward=1.0 if passed else 0.0)

        return VerificationResult(False, 0.1, output, details="No verifier matched.", reward=0.0)
