"""
Deterministic Ground-Truth Verifier.
Provides objective RLVR validation to prevent neural verifier hacking via:
1. Subprocess/Docker Python sandbox execution against deterministic test assertions.
2. SymPy symbolic mathematics equivalence verification (with multi-point fallback).
3. Code AST validation.
4. Majority consensus voting fallback.
"""

import ast
import collections
import math
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class VerificationResult:
    passed: bool
    verifier_type: str
    details: str
    execution_time_ms: float = 0.0
    stdout: Optional[str] = None
    stderr: Optional[str] = None


def _limit_sandbox_resources(max_memory_mb: int = 512):
    """Sets OS-level memory and execution resource bounds for subprocess sandboxing."""
    try:
        import resource
        max_bytes = max_memory_mb * 1024 * 1024
        # Virtual memory limit (RLIMIT_AS) — skip on macOS where it crashes dyld
        import platform as _plat
        if hasattr(resource, "RLIMIT_AS") and _plat.system() != "Darwin":
            resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))
        # Data segment limit (RLIMIT_DATA)
        if hasattr(resource, "RLIMIT_DATA"):
            resource.setrlimit(resource.RLIMIT_DATA, (max_bytes, max_bytes))
        # Core dump limit = 0
        if hasattr(resource, "RLIMIT_CORE"):
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except Exception:
        pass


class GroundTruthVerifier:
    def __init__(
        self,
        sandbox_timeout: float = 4.0,
        max_memory_mb: int = 512,
        use_docker: bool = False,
        docker_image: str = "python:3.10-slim"
    ):
        self.sandbox_timeout = sandbox_timeout
        self.max_memory_mb = max_memory_mb
        self.use_docker = use_docker
        self.docker_image = docker_image

    @staticmethod
    def extract_code_block(text: str, language: Optional[str] = None) -> Optional[str]:
        """Extracts code block from markdown formatting."""
        if language and language.lower() not in ("any", "all"):
            pattern = rf"```{language}\s*\n(.*?)\n```"
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # Fallback to any language code block (```python, ```rust, ```ts, ``` etc.)
        generic_pattern = r"```(?:[a-zA-Z0-9_-]+)?\s*\n(.*?)\n```"
        match = re.search(generic_pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

        return None

    def validate_syntax_ast(self, code: str) -> Tuple[bool, Optional[str]]:
        """Verifies code parses into a valid Python AST without syntax errors."""
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"SyntaxError at line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, str(e)

    def verify_in_sandbox(
        self,
        code_block: str,
        test_assertions: str
    ) -> VerificationResult:
        """
        Executes code + assertions inside an isolated Python subprocess or Docker container.
        Enforces strict timeouts (default 4.0s) and memory caps (default 512MB).
        """
        start_time = time.perf_counter()

        # Step 1: AST validation
        valid_ast, ast_err = self.validate_syntax_ast(code_block)
        if not valid_ast:
            return VerificationResult(
                passed=False,
                verifier_type="sandbox_ast",
                details=f"AST Syntax Validation Failed: {ast_err}",
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
                stderr=ast_err
            )

        test_harness = f"""# Auto-generated RLVR Test Harness
{code_block}

# --- Deterministic Ground-Truth Assertions ---
{test_assertions}
"""

        if self.use_docker:
            return self._verify_in_docker(test_harness, start_time)
        else:
            return self._verify_in_subprocess(test_harness, start_time)

    def _verify_in_subprocess(self, test_harness: str, start_time: float) -> VerificationResult:
        """Executes test harness in local isolated subprocess with memory and time caps."""
        try:
            # Set memory limits on POSIX platforms where preexec_fn is supported
            preexec = None
            if sys.platform != "win32":
                preexec = lambda: _limit_sandbox_resources(self.max_memory_mb)

            res = subprocess.run(
                [sys.executable, "-c", test_harness],
                capture_output=True,
                text=True,
                timeout=self.sandbox_timeout,
                preexec_fn=preexec
            )
            exec_time = (time.perf_counter() - start_time) * 1000
            passed = (res.returncode == 0)

            details = "All assertions passed successfully." if passed else f"Process exited with code {res.returncode}"
            return VerificationResult(
                passed=passed,
                verifier_type="subprocess_sandbox",
                details=details,
                execution_time_ms=exec_time,
                stdout=res.stdout,
                stderr=res.stderr
            )
        except subprocess.TimeoutExpired:
            exec_time = (time.perf_counter() - start_time) * 1000
            return VerificationResult(
                passed=False,
                verifier_type="subprocess_sandbox",
                details=f"Execution timed out after {self.sandbox_timeout}s (max resource budget exceeded)",
                execution_time_ms=exec_time,
                stderr="TimeoutExpired"
            )
        except Exception as e:
            exec_time = (time.perf_counter() - start_time) * 1000
            return VerificationResult(
                passed=False,
                verifier_type="subprocess_sandbox",
                details=f"Sandbox execution error: {str(e)}",
                execution_time_ms=exec_time,
                stderr=str(e)
            )

    def _verify_in_docker(self, test_harness: str, start_time: float) -> VerificationResult:
        """Executes test harness in isolated Docker container with strict memory bounds."""
        try:
            res = subprocess.run(
                [
                    "docker", "run", "--rm",
                    "--network", "none",
                    "--memory", f"{self.max_memory_mb}m",
                    "--cpus", "1.0",
                    self.docker_image,
                    "python", "-c", test_harness
                ],
                capture_output=True,
                text=True,
                timeout=self.sandbox_timeout
            )
            exec_time = (time.perf_counter() - start_time) * 1000
            passed = (res.returncode == 0)
            return VerificationResult(
                passed=passed,
                verifier_type="docker_sandbox",
                details="Docker sandbox assertions passed" if passed else f"Docker exited with {res.returncode}",
                execution_time_ms=exec_time,
                stdout=res.stdout,
                stderr=res.stderr
            )
        except Exception as e:
            return self._verify_in_subprocess(test_harness, start_time)

    def verify_sympy_equivalence(
        self,
        candidate_expr: str,
        ground_truth_expr: str
    ) -> VerificationResult:
        """
        Verifies mathematical equality using SymPy symbolic simplification:
        Checks if simplify(candidate - ground_truth) == 0.
        Falls back to multi-point randomized evaluation if SymPy is not present.
        """
        start_time = time.perf_counter()
        try:
            import sympy
            from sympy.parsing.sympy_parser import parse_expr

            expr_a = parse_expr(candidate_expr.strip())
            expr_b = parse_expr(ground_truth_expr.strip())

            diff = sympy.simplify(expr_a - expr_b)
            passed = bool(diff == 0)

            return VerificationResult(
                passed=passed,
                verifier_type="sympy_symbolic",
                details=f"Symbolic diff: {diff}",
                execution_time_ms=(time.perf_counter() - start_time) * 1000
            )
        except ImportError:
            return self._fallback_multi_point_math_eval(candidate_expr, ground_truth_expr, start_time)
        except Exception as e:
            return VerificationResult(
                passed=False,
                verifier_type="sympy_symbolic",
                details=f"SymPy evaluation error: {str(e)}",
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
                stderr=str(e)
            )

    def _fallback_multi_point_math_eval(self, a_str: str, b_str: str, start_time: float) -> VerificationResult:
        """
        Multi-point numeric evaluation fallback when SymPy is not installed.
        Evaluates expressions at multiple test points for detected variables.
        """
        # Find variable names (identifiers that are not Python builtins)
        identifiers = set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", a_str + " " + b_str))
        allowed_math = {"sin", "cos", "tan", "exp", "log", "sqrt", "abs", "pi", "e"}
        variables = list(identifiers - allowed_math)

        # Test points
        test_points = [
            {v: val for v, val in zip(variables, sample)}
            for sample in [
                [1.5 + i * 0.7 for i in range(len(variables))],
                [2.3 - i * 0.4 for i in range(len(variables))],
                [-1.8 + i * 1.1 for i in range(len(variables))],
                [3.14 + i * 0.5 for i in range(len(variables))],
            ]
        ] or [{}]

        safe_env = {
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "exp": math.exp, "log": math.log, "sqrt": math.sqrt,
            "abs": abs, "pi": math.pi, "e": math.e
        }

        all_passed = True
        for point in test_points:
            env = {**safe_env, **point}
            try:
                val_a = eval(a_str, {"__builtins__": {}}, env)
                val_b = eval(b_str, {"__builtins__": {}}, env)
                if abs(val_a - val_b) > 1e-5:
                    all_passed = False
                    break
            except Exception:
                all_passed = (a_str.strip() == b_str.strip())
                break

        return VerificationResult(
            passed=all_passed,
            verifier_type="numeric_point_eval_fallback",
            details="Verified across numeric test points" if all_passed else "Point evaluation mismatch",
            execution_time_ms=(time.perf_counter() - start_time) * 1000
        )

    def consensus_voting(self, candidates: List[str]) -> Tuple[str, int, float]:
        """
        Computes majority consensus among candidate branches.
        Returns: (winning_candidate, count, consensus_ratio)
        """
        if not candidates:
            return "", 0, 0.0

        normalized = [re.sub(r"\s+", " ", c).strip() for c in candidates]
        counts = collections.Counter(normalized)
        most_common_norm, count = counts.most_common(1)[0]

        for orig, norm in zip(candidates, normalized):
            if norm == most_common_norm:
                ratio = count / len(candidates)
                return orig, count, ratio

        return candidates[0], 1, 1.0 / len(candidates)
