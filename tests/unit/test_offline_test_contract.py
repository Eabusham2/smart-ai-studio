"""The model-free suite permits local fixtures and blocks external resolution."""
import os
import socket
import http.client
import urllib.request

import pytest


@pytest.mark.skipif(os.environ.get("SMARTAI_TEST_OFFLINE", "1") == "0", reason="Explicit external integration run")
def test_external_resolution_is_refused_before_network_io():
    with pytest.raises(OSError, match="External network is disabled"):
        socket.getaddrinfo("example.invalid", 443)


def test_loopback_resolution_remains_available_for_native_fixtures():
    assert socket.getaddrinfo("127.0.0.1", 80)


@pytest.mark.skipif(os.environ.get("SMARTAI_TEST_OFFLINE", "1") == "0", reason="Explicit external integration run")
def test_external_proxy_destinations_are_refused_before_request_bytes(monkeypatch):
    def forbidden_io(*args, **kwargs):
        pytest.fail("An offline external request reached the network transport")

    monkeypatch.setattr(socket.socket, "connect", forbidden_io)
    connection = http.client.HTTPConnection("127.0.0.1", 9)
    with pytest.raises(OSError, match="External network is disabled"):
        connection.set_tunnel("example.invalid", 443)
    with pytest.raises(OSError, match="External network is disabled"):
        connection.putrequest("GET", "http://example.invalid/never-send")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({"https": "http://127.0.0.1:9"}))
    with pytest.raises(OSError, match="External network is disabled"):
        opener.open("https://example.invalid/never-send")


@pytest.mark.skipif(os.environ.get("SMARTAI_TEST_OFFLINE", "1") == "0", reason="Explicit external integration run")
@pytest.mark.parametrize("client_name", ["requests", "httpx"])
def test_optional_http_clients_refuse_external_urls_without_transport(monkeypatch, client_name):
    module = pytest.importorskip(client_name)

    def forbidden_io(*args, **kwargs):
        pytest.fail("An offline external request reached the network transport")

    monkeypatch.setattr(socket.socket, "connect", forbidden_io)
    with pytest.raises(OSError, match="External network is disabled"):
        module.get("https://example.invalid/never-send")
