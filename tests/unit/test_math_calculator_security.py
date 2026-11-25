import pytest

from core.tools import AgentToolRegistry


@pytest.fixture
def calculator(tmp_path):
    return AgentToolRegistry(workspace_dir=str(tmp_path))


@pytest.mark.parametrize(
    "expression",
    [
        "().__class__.__mro__[1].__subclasses__()",
        "(1, 2)[0]",
        "[value for value in (1, 2)]",
        "(lambda: 1)()",
        "__import__('os')",
        "open('secret.txt')",
        "subprocess.Popen('id')",
    ],
)
def test_calculator_rejects_executable_python(calculator, expression):
    ok, result = calculator.execute_tool("math_calculate", {"expression": expression})

    assert not ok
    assert "Failed to parse mathematical expression" in result


@pytest.mark.parametrize("tool_name", ["math_calculate", "calculate", "math", "calculator"])
def test_calculator_aliases_share_restricted_parser(calculator, tool_name):
    ok, result = calculator.execute_tool(
        tool_name, {"expression": "__import__('os').getenv('HOME')"}
    )

    assert not ok
    assert "Failed to parse mathematical expression" in result


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2**10 + 24", "1048"),
        ("x^2 + 1", "x**2 + 1"),
        ("diff(x**3 + 2*x, x)", "3*x**2 + 2"),
        ("integrate(3*x**2, x)", "x**3"),
        ("sin(pi / 2) + sqrt(9)", "4"),
        ("derivative of t**2", "2*t"),
    ],
)
def test_calculator_preserves_supported_math(calculator, expression, expected):
    ok, result = calculator.execute_tool("math_calculate", {"expression": expression})

    assert ok
    assert expected in result
