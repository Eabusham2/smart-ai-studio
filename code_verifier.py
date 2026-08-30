import os
import re
import sys
import time
import tempfile
import subprocess
from typing import Optional, Tuple

class CodeVerifier:
    def __init__(self, timeout: float = 4.0, max_mem_mb: int = 512):
        self.timeout = timeout
        self.max_mem_mb = max_mem_mb

    def extract_code(self, raw_text: str) -> str:
        match = re.findall(r"```(?:python)?\s*([\s\S]*?)```", raw_text, re.IGNORECASE)
        if match:
            return match[-1].strip()
        lines = []
        capture = False
        for line in raw_text.splitlines():
            if line.startswith(("def ", "import ", "from ", "class ")):
                capture = True
            if capture:
                lines.append(line)
        return "\n".join(lines).strip() if lines else raw_text.strip()

    def run_sandbox(self, code: str, tests: str) -> Tuple[bool, float, str, Optional[str]]:
        script = f"# -*- coding: utf-8 -*-\nimport sys, math, json, collections\n\n{code}\n\n# TESTS\n{tests}\n"
        t0 = time.perf_counter()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tmp:
            tmp.write(script)
            tmp_path = tmp.name

        try:
            res = subprocess.run([sys.executable, tmp_path], capture_output=True, text=True, timeout=self.timeout)
            duration_ms = (time.perf_counter() - t0) * 1000.0
            if res.returncode == 0:
                return True, duration_ms, res.stdout, None
            return False, duration_ms, res.stdout, res.stderr.strip()
        except subprocess.TimeoutExpired:
            return False, (time.perf_counter() - t0) * 1000.0, "", f"Timeout after {self.timeout}s"
        except Exception as e:
            return False, (time.perf_counter() - t0) * 1000.0, "", str(e)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

if __name__ == "__main__":
    verifier = CodeVerifier()
    print("=" * 70)
    print("🔍 STANDALONE DETERMINISTIC CODE VERIFIER")
    print("=" * 70)
    
    # Self-test 1: Valid assertions
    code_ok = "def add(a, b):\n    return a + b"
    test_ok = "assert add(2, 3) == 5\nassert add(-1, 1) == 0"
    passed, ms, out, err = verifier.run_sandbox(code_ok, test_ok)
    print(f"Test 1 (Valid Assertion)  : Passed={passed} ({ms:.2f}ms)")
    
    # Self-test 2: Failed assertions caught
    code_fail = "def add(a, b):\n    return a - b"
    test_fail = "assert add(2, 3) == 5"
    passed, ms, out, err = verifier.run_sandbox(code_fail, test_fail)
    print(f"Test 2 (Failed Assertion) : Passed={passed} ({ms:.2f}ms) | Caught: {err.splitlines()[-1] if err else 'None'}")
    print("=" * 70)
