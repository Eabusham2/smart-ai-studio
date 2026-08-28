"""
Unit tests for Deterministic Ground-Truth Verifier.
"""

import unittest
from core.verifier import GroundTruthVerifier, VerificationResult


class TestGroundTruthVerifier(unittest.TestCase):
    def setUp(self):
        self.verifier = GroundTruthVerifier(sandbox_timeout=3.0)

    def test_extract_code_block(self):
        markdown_text = """Here is the solution:
```python
def add(a, b):
    return a + b
```
Hope this helps!"""
        code = self.verifier.extract_code_block(markdown_text)
        self.assertIsNotNone(code)
        self.assertIn("def add(a, b):", code)

    def test_ast_validation_success(self):
        code = "def is_even(n):\n    return n % 2 == 0"
        valid, err = self.verifier.validate_syntax_ast(code)
        self.assertTrue(valid)
        self.assertIsNone(err)

    def test_ast_validation_syntax_error(self):
        bad_code = "def is_even(n)\n    return n % 2 == 0"
        valid, err = self.verifier.validate_syntax_ast(bad_code)
        self.assertFalse(valid)
        self.assertIn("SyntaxError", err)

    def test_sandbox_execution_success(self):
        code = "def multiply(x, y):\n    return x * y"
        assertions = "assert multiply(3, 4) == 12\nassert multiply(-2, 5) == -10"
        res = self.verifier.verify_in_sandbox(code, assertions)
        self.assertTrue(res.passed)
        self.assertEqual(res.verifier_type, "subprocess_sandbox")

    def test_sandbox_execution_assertion_failure(self):
        code = "def multiply(x, y):\n    return x + y"  # bug intentionally
        assertions = "assert multiply(3, 4) == 12"
        res = self.verifier.verify_in_sandbox(code, assertions)
        self.assertFalse(res.passed)

    def test_sandbox_timeout(self):
        infinite_loop_code = "import time\nwhile True:\n    time.sleep(0.1)"
        assertions = "assert True"
        short_verifier = GroundTruthVerifier(sandbox_timeout=0.5)
        res = short_verifier.verify_in_sandbox(infinite_loop_code, assertions)
        self.assertFalse(res.passed)
        self.assertIn("timed out", res.details.lower())

    def test_sympy_equivalence(self):
        # 2*x + 3*x == 5*x
        res = self.verifier.verify_sympy_equivalence("2*x + 3*x", "5*x")
        self.assertTrue(res.passed)

        # (x + 1)**2 == x**2 + 2*x + 1
        res2 = self.verifier.verify_sympy_equivalence("(x + 1)**2", "x**2 + 2*x + 1")
        self.assertTrue(res2.passed)

        # Inequality test
        res3 = self.verifier.verify_sympy_equivalence("x + 2", "x + 3")
        self.assertFalse(res3.passed)

    def test_consensus_voting(self):
        candidates = [
            "The answer is 42.",
            "The answer is 42.",
            "The answer is 42. ",
            "The answer is 100.",
        ]
        winner, count, ratio = self.verifier.consensus_voting(candidates)
        self.assertEqual(count, 3)
        self.assertAlmostEqual(ratio, 0.75)
        self.assertIn("42", winner)


if __name__ == "__main__":
    unittest.main()
