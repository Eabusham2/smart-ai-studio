"""Checked training-task rewards for small pure Python functions."""
from __future__ import annotations

import ast
import json
import keyword
import math
import sys

from core.verifier import GroundTruthVerifier


_BUILTINS = set("abs all any bool dict divmod enumerate float int len list max min pow range reversed "
                "round set sorted str sum tuple zip".split())
_METHODS = set("append count extend get index items join keys lower pop reverse sort split strip upper values".split())
_NODES = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign, ast.AugAssign,
          ast.If, ast.IfExp, ast.For, ast.While, ast.Break, ast.Continue, ast.Pass, ast.Expr, ast.Constant,
          ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp, ast.List, ast.Set, ast.Dict, ast.Tuple,
          ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.comprehension, ast.Name,
          ast.Load, ast.Store, ast.Call, ast.keyword, ast.Attribute, ast.Subscript, ast.Slice,
          ast.operator, ast.unaryop, ast.cmpop, ast.boolop)


def pure_function(code, name):
    """Accept one function over ordinary values, with a closed set of call targets."""
    if len(code) > 12000:
        return False
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return False
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return False
    fn = tree.body[0]
    if (fn.name != name or fn.decorator_list or fn.args.defaults or any(fn.args.kw_defaults)
            or fn.args.vararg or fn.args.kwarg or getattr(fn, "type_params", [])):
        return False
    nodes = list(ast.walk(tree))
    local = {n.id for n in nodes if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    local.update(n.arg for n in nodes if isinstance(n, ast.arg))
    if len(nodes) > 512 or any(v.startswith("_") for v in local) or local & (_BUILTINS | {name}):
        return False
    names = local | _BUILTINS | {name}
    for node in nodes:
        if not isinstance(node, _NODES) or (isinstance(node, ast.FunctionDef) and node is not fn):
            return False
        if isinstance(node, ast.Name) and node.id not in names:
            return False
        if isinstance(node, ast.Attribute) and (node.attr not in _METHODS or not isinstance(node.ctx, ast.Load)):
            return False
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in (_BUILTINS | {name})
                    or isinstance(node.func, ast.Attribute) and node.func.attr in _METHODS):
                return False
    return True


class PythonTaskVerifier:
    def __init__(self, spec):
        self.function = spec.get("function")
        if (not isinstance(self.function, str) or not self.function.isidentifier()
                or self.function.startswith("_") or keyword.iskeyword(self.function)
                or self.function in _BUILTINS):
            raise ValueError("Provide a Python function name for the training task")
        cases = spec.get("cases")
        if not isinstance(cases, list) or not 1 <= len(cases) <= 32:
            raise ValueError("Provide 1–32 training-task input/output cases")
        if any(not isinstance(c, dict) or set(c) != {"args", "expected"}
               or not isinstance(c["args"], list) for c in cases):
            raise ValueError("Each case requires positional args and one expected JSON value")
        payload = json.dumps(cases, allow_nan=False)
        if len(payload) > 8192:
            raise ValueError("Python training cases exceed the 8192-character budget")
        timeout = spec.get("timeout_seconds", 2)
        memory = spec.get("memory_mb", 128)
        if type(timeout) not in (int, float) or not 0.1 <= timeout <= 5:
            raise ValueError("timeout_seconds must be 0.1–5")
        if type(memory) is not int or not 48 <= memory <= 512:
            raise ValueError("memory_mb must be 48–512")
        self.baseline = spec.get("baseline_ms")
        if self.baseline is not None and (type(self.baseline) not in (int, float)
                                          or not 0 < self.baseline <= 60000):
            raise ValueError("baseline_ms must be a positive measured training-task baseline")
        try:
            import resource
            limit_kind = "RLIMIT_DATA" if sys.platform == "darwin" else "RLIMIT_AS"
            getattr(resource, limit_kind)
        except (ImportError, AttributeError):
            raise RuntimeError("Python test rewards require OS-enforced per-process memory limits") from None
        self.verifier = GroundTruthVerifier(sandbox_timeout=timeout, max_memory_mb=memory)
        self.results = []
        # Set/verify a hard limit inside the child even if the portable
        # verifier's best-effort preexec limit was unavailable on this host.
        self.setup = (f"import resource as _resource\n"
                      f"_resource.setrlimit(_resource.{limit_kind}, ({memory * 1048576},) * 2)\n")
        passes = 3 if self.baseline is not None else 1
        self.harness = f"""import json as _json, time as _time
_times = []
for _trial in range({passes}):
    _cases = _json.loads({payload!r})
    _started = _time.perf_counter_ns()
    for _case in _cases:
        if {self.function}(*_case["args"]) != _case["expected"]:
            raise ValueError("incorrect output")
    _times.append((_time.perf_counter_ns() - _started) / 1000000)
print(sorted(_times)[len(_times) // 2])
"""

    def __call__(self, answer):
        code = self.verifier.extract_code_block(answer, language="python") or answer.strip()
        if not pure_function(code, self.function):
            self.results.append({"passed": False, "reason": "unsupported function or syntax"})
            return 0.0
        result = self.verifier.verify_in_sandbox(self.setup + code, self.harness)
        if not result.passed:
            self.results.append({"passed": False, "reason": result.details})
            return 0.0
        try:
            elapsed = float(result.stdout.strip())
        except (AttributeError, TypeError, ValueError):
            elapsed = math.nan
        if not math.isfinite(elapsed) or elapsed < 0:
            raise RuntimeError("The Python verifier produced no usable timing measurement")
        self.results.append({"passed": True, "median_cases_ms": elapsed})
        return self.baseline / (self.baseline + elapsed) if self.baseline is not None else 1.0
