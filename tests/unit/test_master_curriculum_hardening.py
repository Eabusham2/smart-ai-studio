from types import SimpleNamespace

from rlvr.master_curriculum import MasterCurriculumOrchestrator


class FakeEngine:
    def __init__(self):
        self.calls = 0

    def solve(self, prompt, **kwargs):
        self.calls += 1
        return "```python\ndef wrong_name():\n    return False\n```", {}


class AlwaysFailVerifier:
    @staticmethod
    def extract_code_block(text):
        return "def wrong_name():\n    return False"

    @staticmethod
    def verify_in_sandbox(code, tests):
        return SimpleNamespace(passed=False, stderr="hidden verifier rejected candidate")


class FakeDB:
    def log_interaction(self, **kwargs):
        raise AssertionError("a failed live candidate must never be logged as verified")


def test_live_autonomous_evolution_is_bounded_and_never_uses_mock_answer_fallback():
    obj = object.__new__(MasterCurriculumOrchestrator)
    obj.settings = SimpleNamespace(use_mock=False)
    obj.engine = FakeEngine()
    obj.verifier = AlwaysFailVerifier()
    obj.db = FakeDB()

    result = obj.execute_autonomous_unsupervised_evolution(target_traces=1, verbose=False)
    assert result["status"] == "incomplete"
    assert result["discovery_traces_logged"] == 0
    assert result["attempts"] == 8
    assert obj.engine.calls == 8


def test_live_environmental_rlvr_is_bounded_and_feedback_driven_without_fixture_fallback():
    obj = object.__new__(MasterCurriculumOrchestrator)
    obj.settings = SimpleNamespace(use_mock=False)
    obj.engine = FakeEngine()
    obj.verifier = AlwaysFailVerifier()
    obj.db = FakeDB()

    result = obj.execute_environmental_rlvr_recovery(
        target_traces=1,
        max_attempts=2,
        verbose=False,
    )
    assert result["status"] == "incomplete"
    assert result["recovery_traces_logged"] == 0
    assert result["batches"] == 6
    assert obj.engine.calls == 12
