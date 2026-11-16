"""Keep normal CI/tests offline while allowing synthetic localhost services.

Some legacy tool tests exercise the search-service-unavailable fallback. They
must not send their prompts to a public service. Set SMARTAI_TEST_OFFLINE=0 only
for an explicitly requested external-service integration run.
"""
from __future__ import annotations

import ipaddress
import os
import socket


def _local_host(host):
    if host is None:
        return True
    if isinstance(host, bytes):
        host = host.decode("ascii", errors="strict")
    if str(host).lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(str(host).split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def pytest_sessionstart(session):
    if os.environ.get("SMARTAI_TEST_OFFLINE", "1") == "0":
        return
    original_resolve = socket.getaddrinfo
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_sendto = socket.socket.sendto

    def refuse_external(host):
        if not _local_host(host):
            raise OSError("External network is disabled for model-free tests; mock the service")

    def resolve(host, *args, **kwargs):
        refuse_external(host)
        return original_resolve(host, *args, **kwargs)

    def connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            refuse_external(address[0])
        return original_connect(sock, address)

    def connect_ex(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            refuse_external(address[0])
        return original_connect_ex(sock, address)

    def sendto(sock, data, *args):
        if sock.family in (socket.AF_INET, socket.AF_INET6) and args:
            refuse_external(args[-1][0])
        return original_sendto(sock, data, *args)

    # Deliberately stays installed through process exit, including teardown of
    # background GUI workers. No DNS query or payload leaves this test process.
    socket.getaddrinfo = resolve
    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.socket.sendto = sendto
