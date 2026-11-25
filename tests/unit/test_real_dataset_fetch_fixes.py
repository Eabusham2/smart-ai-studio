from __future__ import annotations

import base64
import json
import pickle
import zlib
from pathlib import Path
from types import SimpleNamespace

from eval import real_dataset_fetch_fixes as fixes


_REDUCER_CALLS = []


def _record_reducer_call():
    _REDUCER_CALLS.append(True)
    return "[]"


class _ReducerPayload:
    def __reduce__(self):
        return (_record_reducer_call, ())


def _normalizer():
    real_module = SimpleNamespace()
    fixes.install(real_module)
    return real_module._normalize_io_cases


def _compressed_json(value) -> str:
    payload = json.dumps(value).encode("utf-8")
    return base64.b64encode(zlib.compress(payload)).decode("ascii")


def test_runtime_decoder_has_no_pickle_sink():
    source = Path(fixes.__file__).read_text(encoding="utf-8")
    assert "import pickle" not in source
    assert "pickle.loads" not in source


def test_documented_json_case_formats_remain_supported():
    normalize = _normalizer()
    cases = [{"input": "2 3\n", "output": "5\n", "testtype": "stdin"}]
    expected = [("2 3\n", "5\n")]

    assert normalize(cases) == expected
    assert normalize(cases[0]) == expected
    assert normalize([{"stdin": "2 3\n", "stdout": "5\n"}]) == expected
    assert normalize(json.dumps(cases)) == expected
    assert normalize(_compressed_json(cases)) == expected
    assert normalize([
        {"input": [2, 3], "output": 5, "testtype": "functional"},
        *cases,
    ]) == expected


def test_compressed_pickle_reducer_is_rejected_without_execution():
    normalize = _normalizer()
    _REDUCER_CALLS.clear()
    payload = base64.b64encode(zlib.compress(pickle.dumps(_ReducerPayload()))).decode("ascii")

    assert normalize(payload) == []
    assert _REDUCER_CALLS == []


def test_compressed_json_limit_is_fail_closed(monkeypatch):
    normalize = _normalizer()
    monkeypatch.setattr(fixes, "_MAX_IO_JSON_BYTES", 64)
    cases = [{"input": "x" * 128, "output": "ok"}]

    assert normalize(_compressed_json(cases)) == []


def test_invalid_schema_does_not_preserve_partial_cases():
    normalize = _normalizer()
    mixed = [
        {"input": "valid", "output": "valid"},
        {"input": ["not", "stdin", "text"], "output": "invalid"},
    ]

    assert normalize(mixed) == []
