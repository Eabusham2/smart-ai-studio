"""
Interactive Domain-Specific Language (DSL) & Sandbox Execution Engine.
Supports:
1. TensorGraphDSL: Non-commutative array and tensor operators (>>~fold, <#>scale, @fuse_quant, ^mask_add).
2. GlyphScript: Symbolic logic, invariant checking, and structural graph transformations.
3. Interactive RLVR Sandbox Runner: Isolated execution with strict memory and timeout bounds.
"""

import ast
import json
import math
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from core.verifier import GroundTruthVerifier, VerificationResult, get_sandbox_preexec


def evaluate_tensorgraph_dsl(expr: str) -> List[Union[int, float]]:
    """
    Evaluates a TensorGraphDSL expression sequentially across non-commutative operators:
    - Initial array: [x1, x2, ...]
    - >>~fold(k): Left-to-right cyclic rotation by k elements (e.g. [2,4,6] >>~fold(1) -> [4,6,2])
    - <#>scale(s): Scalar multiplication of array elements
    - @fuse_quant(th): Clamps elements to ternary {-1, 0, +1} using absolute threshold th
    - ^mask_add(mask, val): Adds scalar val where mask element is 1
    """
    clean = expr.strip()
    if clean.startswith("`") and clean.endswith("`"):
        clean = clean[1:-1].strip()

    # Extract initial array
    arr_match = re.match(r"^\s*(\[[^\]]+\])", clean)
    if not arr_match:
        raise ValueError(f"Invalid TensorGraphDSL input array: {clean}")

    current_arr = ast.literal_eval(arr_match.group(1))
    current_arr = [float(x) if isinstance(x, (int, float)) else x for x in current_arr]
    remaining = clean[arr_match.end():].strip()

    # Sequential operator pipeline
    op_pattern = re.compile(
        r"(>>~fold\(\s*(\d+)\s*\)|<#>scale\(\s*([\d\.-]+)\s*\)|@fuse_quant\(\s*([\d\.-]+)\s*\)|\^mask_add\(\s*(\[[^\]]+\])\s*,\s*([\d\.-]+)\s*\))"
    )

    for match in op_pattern.finditer(remaining):
        op_str = match.group(1)
        if ">>~fold" in op_str:
            k = int(match.group(2))
            if current_arr:
                shift = k % len(current_arr)
                current_arr = current_arr[shift:] + current_arr[:shift]
        elif "<#>scale" in op_str:
            s = float(match.group(3))
            current_arr = [x * s for x in current_arr]
        elif "@fuse_quant" in op_str:
            th = float(match.group(4))
            quantized = []
            for x in current_arr:
                if x > th:
                    quantized.append(1)
                elif x < -th:
                    quantized.append(-1)
                else:
                    quantized.append(0)
            current_arr = quantized
        elif "^mask_add" in op_str:
            mask = ast.literal_eval(match.group(5))
            val = float(match.group(6))
            new_arr = []
            for idx, x in enumerate(current_arr):
                m_val = mask[idx] if idx < len(mask) else 0
                new_arr.append(x + (val if m_val else 0.0))
            current_arr = new_arr

    # Normalize integer floats to ints
    res = [int(x) if isinstance(x, (int, float)) and int(x) == x else round(x, 4) for x in current_arr]
    return res


def evaluate_glyph_script(script: str) -> Dict[str, Any]:
    """
    Evaluates GlyphScript symbolic invariant rules and graph AST transformations.
    Example:
      RULE: DAG_MONOTONIC_FLOW
      GRAPH: A -> B (5), B -> C (3)
      INVARIANT: ALL(weight > 0)
    """
    lines = [l.strip() for l in script.strip().split("\n") if l.strip() and not l.startswith("#")]
    nodes = set()
    edges = []
    invariants_passed = True
    invariants_checked = 0
    rule_name = "GLYPH_INVARIANT_RULE"

    for line in lines:
        if line.upper().startswith("RULE:"):
            rule_name = line.split(":", 1)[1].strip()
        elif "->" in line:
            parts = line.split("->")
            src = parts[0].strip()
            dest_weight = parts[1].strip()
            w_match = re.search(r"\(([-\d\.]+)\)", dest_weight)
            weight = float(w_match.group(1)) if w_match else 1.0
            if weight.is_integer():
                weight = int(weight)
            dest = re.sub(r"\([-\d\.]+\)", "", dest_weight).strip()
            nodes.add(src)
            nodes.add(dest)
            edges.append({"src": src, "dest": dest, "weight": weight})
        elif line.upper().startswith("INVARIANT:"):
            inv_text = line.split(":", 1)[1].strip()
            invariants_checked += 1
            if "weight > 0" in inv_text:
                if any(e["weight"] <= 0 for e in edges):
                    invariants_passed = False

    return {
        "rule": rule_name,
        "nodes_count": len(nodes),
        "edges_count": len(edges),
        "edges": edges,
        "invariants_checked": invariants_checked,
        "invariants_passed": invariants_passed,
        "status": "VALID_GLYPH_GRAPH" if invariants_passed else "INVARIANT_VIOLATION"
    }


class InteractiveDSLPlayground:
    """Provides isolated testing and execution for DSLs and RLVR assertions."""

    def __init__(self, sandbox_timeout: float = 4.0, max_memory_mb: int = 512):
        self.verifier = GroundTruthVerifier(sandbox_timeout=sandbox_timeout, max_memory_mb=max_memory_mb)

    def execute_dsl(self, dsl_type: str, code: str, assertions: Optional[str] = None) -> Dict[str, Any]:
        """Executes DSL snippet or runs Python RLVR verification."""
        start_time = time.perf_counter()
        clean_code = code.strip()

        if dsl_type.lower() == "tensorgraph":
            try:
                res = evaluate_tensorgraph_dsl(clean_code)
                elapsed = (time.perf_counter() - start_time) * 1000
                return {
                    "passed": True,
                    "result": res,
                    "stdout": f"✦ TensorGraphDSL Output: {res}\n",
                    "stderr": "",
                    "execution_time_ms": elapsed,
                    "exit_code": 0
                }
            except Exception as e:
                elapsed = (time.perf_counter() - start_time) * 1000
                return {
                    "passed": False,
                    "result": None,
                    "stdout": "",
                    "stderr": f"TensorGraphDSL Parsing Error: {e}",
                    "execution_time_ms": elapsed,
                    "exit_code": 1
                }
        elif dsl_type.lower() == "glyphscript":
            try:
                res = evaluate_glyph_script(clean_code)
                elapsed = (time.perf_counter() - start_time) * 1000
                return {
                    "passed": res["invariants_passed"],
                    "result": res,
                    "stdout": json.dumps(res, indent=2),
                    "stderr": "" if res["invariants_passed"] else "Invariant rule violation detected",
                    "execution_time_ms": elapsed,
                    "exit_code": 0 if res["invariants_passed"] else 2
                }
            except Exception as e:
                elapsed = (time.perf_counter() - start_time) * 1000
                return {
                    "passed": False,
                    "result": None,
                    "stdout": "",
                    "stderr": f"GlyphScript Error: {e}",
                    "execution_time_ms": elapsed,
                    "exit_code": 1
                }
        else:
            # Python RLVR Sandbox execution
            test_cases = assertions or "assert True"
            v_res = self.verifier.verify_in_sandbox(clean_code, test_cases)
            return {
                "passed": v_res.passed,
                "result": v_res.details,
                "stdout": v_res.stdout or ("All assertions passed successfully.\n" if v_res.passed else ""),
                "stderr": v_res.stderr or ("" if v_res.passed else "Assertions failed"),
                "execution_time_ms": v_res.execution_time_ms,
                "exit_code": 0 if v_res.passed else 1
            }
