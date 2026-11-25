"""Source-level regression coverage for bounded web tool resource use."""

from email.message import Message

from core import tools as tools_module
from core.tools import AgentToolRegistry


class FakeResponse:
    def __init__(self, body: bytes, headers=None):
        self.body = body
        self.offset = 0
        self.read_sizes = []
        self.headers = Message()
        for name, value in (headers or {}).items():
            self.headers[name] = str(value)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, size):
        self.read_sizes.append(size)
        chunk = self.body[self.offset:self.offset + size]
        self.offset += len(chunk)
        return chunk


def _registry():
    return AgentToolRegistry.__new__(AgentToolRegistry)


def test_web_fetch_bounds_body_without_content_length(monkeypatch):
    response = FakeResponse(b"x" * (tools_module._WEB_RESPONSE_MAX_BYTES + 32), {
        "Content-Type": "text/html",
    })
    monkeypatch.setattr(tools_module.urllib.request, "urlopen", lambda *_args, **_kwargs: response)

    ok, result = _registry().execute_tool("web_fetch", {"url": "https://example.test/large"})

    assert not ok
    assert "exceeds the web content byte limit" in result
    assert response.offset == tools_module._WEB_RESPONSE_MAX_BYTES + 1
    assert max(response.read_sizes) <= tools_module._WEB_READ_CHUNK_BYTES


def test_web_fetch_preserves_bounded_html(monkeypatch):
    body = b"<html><body>Hello <strong>world</strong></body></html>"
    response = FakeResponse(body, {
        "Content-Type": "text/html; charset=utf-8",
        "Content-Length": len(body),
    })
    requests = []

    def open_response(request, **_kwargs):
        requests.append(request)
        return response

    monkeypatch.setattr(tools_module.urllib.request, "urlopen", open_response)

    ok, result = _registry().execute_tool("web_fetch", {"url": "https://example.test/page"})

    assert ok
    assert "Hello world" in result
    assert requests[0].headers["Accept-encoding"] == "identity"


def test_web_fetch_rejects_compressed_content_before_read(monkeypatch):
    response = FakeResponse(b"compressed", {
        "Content-Type": "text/html",
        "Content-Encoding": "gzip",
    })
    monkeypatch.setattr(tools_module.urllib.request, "urlopen", lambda *_args, **_kwargs: response)

    ok, result = _registry().execute_tool("web_fetch", {"url": "https://example.test/compressed"})

    assert not ok
    assert "Compressed responses are not accepted" in result
    assert response.read_sizes == []


def test_crawler_rejects_excessive_work_limits(monkeypatch):
    def unexpected_request(*_args, **_kwargs):
        raise AssertionError("crawler performed a request for rejected limits")

    monkeypatch.setattr(tools_module.urllib.request, "urlopen", unexpected_request)
    registry = _registry()

    pages_ok, pages_result = registry.execute_tool("web_crawler", {
        "query_or_url": "https://example.test/",
        "max_pages": tools_module._WEB_CRAWL_MAX_PAGES + 1,
        "max_depth": 1,
    })
    depth_ok, depth_result = registry.execute_tool("web_crawler", {
        "query_or_url": "https://example.test/",
        "max_pages": 1,
        "max_depth": tools_module._WEB_CRAWL_MAX_DEPTH + 1,
    })

    assert not pages_ok and "max_pages must be between" in pages_result
    assert not depth_ok and "max_depth must be between" in depth_result


def test_crawler_enforces_aggregate_byte_budget(monkeypatch):
    first_body = b'<a href="/next">next</a>'
    second_body = b"second page"
    responses = [
        FakeResponse(
            first_body,
            {"Content-Type": "text/html", "Content-Length": len(first_body)},
        ),
        FakeResponse(
            second_body,
            {"Content-Type": "text/html", "Content-Length": len(second_body)},
        ),
    ]
    monkeypatch.setattr(
        tools_module,
        "_WEB_CRAWL_MAX_TOTAL_BYTES",
        len(first_body) + len(second_body) - 1,
    )
    monkeypatch.setattr(
        tools_module.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: responses.pop(0),
    )

    ok, result = _registry().execute_tool("web_crawler", {
        "query_or_url": "https://example.test/",
        "max_pages": 2,
        "max_depth": 1,
    })

    assert not ok
    assert "exceeds the web content byte limit" in result
