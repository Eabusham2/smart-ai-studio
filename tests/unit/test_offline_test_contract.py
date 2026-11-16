"""The model-free suite permits local fixtures and blocks external resolution."""
import os
import socket

import pytest


@pytest.mark.skipif(os.environ.get("SMARTAI_TEST_OFFLINE", "1") == "0", reason="Explicit external integration run")
def test_external_resolution_is_refused_before_network_io():
    with pytest.raises(OSError, match="External network is disabled"):
        socket.getaddrinfo("example.invalid", 443)


def test_loopback_resolution_remains_available_for_native_fixtures():
    assert socket.getaddrinfo("127.0.0.1", 80)
