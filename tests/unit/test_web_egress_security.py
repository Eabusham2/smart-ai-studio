import ipaddress
import socket

import pytest

from core import safe_http
from core import tools as tools_module
from core.best_work import _SIDE_EFFECT_TOOLS, _network_url_is_authorized


def _resolver(addresses_by_host):
    def resolve(host, port, family, socktype, protocol):
        assert family == socket.AF_UNSPEC
        return [
            (socket.AF_INET6 if ":" in address else socket.AF_INET,
             socket.SOCK_STREAM, socket.IPPROTO_TCP, "",
             (address, port, 0, 0) if ":" in address else (address, port))
            for address in addresses_by_host[host]
        ]
    return resolve


@pytest.mark.parametrize("address", [
    "127.0.0.1",
    "10.1.2.3",
    "169.254.169.254",
    "192.168.1.2",
    "::1",
    "::ffff:7f00:1",
    "fc00::1",
    "fe80::1",
])
def test_private_destinations_are_rejected(monkeypatch, address):
    monkeypatch.setattr(
        safe_http.socket, "getaddrinfo", _resolver({address: [address]}),
    )

    with pytest.raises(safe_http.UnsafeURLError, match="non-public"):
        safe_http._resolve_destination(f"http://[{address}]/" if ":" in address
                                       else f"http://{address}/")


def test_any_private_dns_answer_rejects_host(monkeypatch):
    monkeypatch.setattr(
        safe_http.socket,
        "getaddrinfo",
        _resolver({"mixed.example": ["93.184.216.34", "10.0.0.8"]}),
    )

    with pytest.raises(safe_http.UnsafeURLError, match="10.0.0.8"):
        safe_http._resolve_destination("https://mixed.example/docs")


@pytest.mark.parametrize("url", [
    "http://user@example.com/",
    "http://user:password@example.com/",
    "http://example.com\\@127.0.0.1/",
    "http://example.com/%0aHost:127.0.0.1\n",
])
def test_ambiguous_urls_are_rejected(monkeypatch, url):
    monkeypatch.setattr(
        safe_http.socket, "getaddrinfo",
        _resolver({"example.com": ["93.184.216.34"]}),
    )

    with pytest.raises(safe_http.UnsafeURLError):
        safe_http._resolve_destination(url)


def test_zero_port_is_not_reinterpreted(monkeypatch):
    monkeypatch.setattr(
        safe_http.socket, "getaddrinfo",
        _resolver({"example.com": ["93.184.216.34"]}),
    )

    with pytest.raises(safe_http.UnsafeURLError, match="port"):
        safe_http._resolve_destination("https://example.com:0/docs")


def test_public_resolution_retains_all_addresses(monkeypatch):
    monkeypatch.setattr(
        safe_http.socket,
        "getaddrinfo",
        _resolver({"example.com": ["93.184.216.34", "2606:2800:220:1:248:1893:25c8:1946"]}),
    )

    destination = safe_http._resolve_destination("https://example.com/docs?q=1")

    assert [str(address.ip) for address in destination.addresses] == [
        "93.184.216.34", "2606:2800:220:1:248:1893:25c8:1946",
    ]
    assert destination.authority == "example.com"
    assert destination.request_target == "/docs?q=1"


def test_socket_connects_to_validated_address(monkeypatch):
    class FakeSocket:
        def __init__(self):
            self.connected_to = None
            self.closed = False

        def settimeout(self, timeout):
            assert timeout == 4.0

        def connect(self, sockaddr):
            self.connected_to = sockaddr

        def getpeername(self):
            return self.connected_to

        def close(self):
            self.closed = True

    fake_socket = FakeSocket()
    monkeypatch.setattr(safe_http.socket, "socket", lambda *args: fake_socket)
    address = safe_http._ResolvedAddress(
        socket.AF_INET, ("93.184.216.34", 443),
        ipaddress.ip_address("93.184.216.34"),
    )

    connected = safe_http._connect_pinned_socket(address, 4.0)

    assert connected is fake_socket
    assert fake_socket.connected_to == ("93.184.216.34", 443)


def test_connected_peer_must_match_pin(monkeypatch):
    class MismatchedSocket:
        closed = False

        def settimeout(self, timeout):
            pass

        def connect(self, sockaddr):
            pass

        def getpeername(self):
            return ("10.0.0.8", 443)

        def close(self):
            self.closed = True

    fake_socket = MismatchedSocket()
    monkeypatch.setattr(safe_http.socket, "socket", lambda *args: fake_socket)
    address = safe_http._ResolvedAddress(
        socket.AF_INET, ("93.184.216.34", 443),
        ipaddress.ip_address("93.184.216.34"),
    )

    with pytest.raises(safe_http.UnsafeURLError, match="peer"):
        safe_http._connect_pinned_socket(address, 4.0)

    assert fake_socket.closed


def test_redirect_to_metadata_is_revalidated(monkeypatch):
    class RedirectResponse:
        status = 302

        def __init__(self):
            self.closed = False

        def getheader(self, name):
            assert name == "Location"
            return "http://169.254.169.254/latest/meta-data/"

        def close(self):
            self.closed = True

    redirect = RedirectResponse()
    opened = []
    monkeypatch.setattr(
        safe_http.socket,
        "getaddrinfo",
        _resolver({
            "public.example": ["93.184.216.34"],
            "169.254.169.254": ["169.254.169.254"],
        }),
    )

    def request_once(destination, headers, timeout):
        opened.append(destination.url)
        return redirect

    monkeypatch.setattr(safe_http, "_request_once", request_once)

    with pytest.raises(safe_http.UnsafeURLError, match="non-public"):
        safe_http.safe_urlopen("https://public.example/start")

    assert opened == ["https://public.example/start"]
    assert redirect.closed


@pytest.mark.parametrize(("name", "args"), [
    ("web_fetch", {"url": "https://example.com/docs"}),
    ("web_crawler", {"query_or_url": "https://example.com/docs"}),
])
def test_exact_requested_url_is_authorized(name, args):
    assert name in _SIDE_EFFECT_TOOLS
    assert _network_url_is_authorized(
        name, args, "Please fetch and inspect https://example.com/docs",
    )


@pytest.mark.parametrize("prompt", [
    "Explain networking without opening a website.",
    "Fetch https://example.com/docs",
])
def test_unapproved_model_url_is_rejected(prompt):
    assert not _network_url_is_authorized(
        "web_fetch", {"url": "https://example.com/docs?context=secret"}, prompt,
    )


def test_explicit_network_denial_is_respected():
    assert not _network_url_is_authorized(
        "web_fetch", {"url": "https://example.com/docs"},
        "Do not visit https://example.com/docs",
    )


def test_public_fetch_preserves_page_extraction(monkeypatch):
    class PublicResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return b"<html><title>Example</title><p>Public documentation</p></html>"

    calls = []

    def open_public(url, **options):
        calls.append((url, options))
        return PublicResponse()

    monkeypatch.setattr(tools_module, "safe_urlopen", open_public)
    registry = object.__new__(tools_module.AgentToolRegistry)

    ok, result = registry._tool_web_fetch("https://example.com/docs")

    assert ok and "Public documentation" in result
    assert calls[0][0] == "https://example.com/docs"
    assert calls[0][1]["headers"]["User-Agent"] == "SmartAI/2.0"


def test_public_search_preserves_structured_results(monkeypatch):
    class SearchResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return b'{"AbstractText":"Public result","Heading":"Example","RelatedTopics":[]}'

    calls = []

    def open_public(url, **options):
        calls.append((url, options))
        return SearchResponse()

    monkeypatch.setattr(tools_module, "safe_urlopen", open_public)
    registry = object.__new__(tools_module.AgentToolRegistry)

    ok, result = registry._tool_web_search("public topic")

    assert ok and "Public result" in result
    assert calls[0][0].startswith("https://api.duckduckgo.com/?q=public%20topic")


def test_public_crawl_preserves_page_extraction(monkeypatch):
    class PublicResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def geturl(self):
            return "https://example.com/docs"

        def read(self):
            return b"<html><title>Example Docs</title><p>Crawler content</p></html>"

    monkeypatch.setattr(
        tools_module, "safe_urlopen", lambda url, **options: PublicResponse(),
    )
    registry = object.__new__(tools_module.AgentToolRegistry)

    ok, result = registry._tool_web_crawler(
        "https://example.com/docs", max_pages=1, max_depth=0,
    )

    assert ok and "Example Docs" in result and "Crawler content" in result
