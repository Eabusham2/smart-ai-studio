"""Keep normal CI/tests offline while allowing synthetic localhost services.

Some legacy tool tests exercise the search-service-unavailable fallback. They
must not send their prompts to a public service. Set SMARTAI_TEST_OFFLINE=0 only
for an explicitly requested external-service integration run.
"""
from __future__ import annotations

import ipaddress
import http.client
import gc
import os
import socket
import urllib.parse
import urllib.request

import pytest


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    """Collect destroyed Tk object cycles on their owner between GUI tests."""
    try:
        import tkinter as tk
    except ImportError:
        yield
        return
    destroyed_widget = False
    original_destroy = tk.Misc.destroy

    def destroy(widget, *args, **kwargs):
        nonlocal destroyed_widget
        destroyed_widget = True
        return original_destroy(widget, *args, **kwargs)

    # Observe actual widget teardown, including roots created inside test bodies
    # and unittest class fixtures. Importing tkinter alone must not trigger GC.
    tk.Misc.destroy = destroy
    try:
        yield
    finally:
        tk.Misc.destroy = original_destroy
        if destroyed_widget:
            # Run after all fixture references are released. Otherwise a later
            # model worker's gc.collect() can finalize these dead interpreters'
            # Variables off the Tk thread and delay an unrelated model test.
            gc.collect()


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
    requested_mlx = os.environ.get("MLX_TEST_DEVICE")
    if requested_mlx:
        import mlx.core as mx
        if requested_mlx not in {"cpu", "metal"}:
            raise ValueError("MLX_TEST_DEVICE must be cpu or metal")
        if requested_mlx == "metal" and not mx.metal.is_available():
            raise RuntimeError("The native MLX Metal test lane cannot fall back to CPU")
        mx.set_default_device(mx.gpu if requested_mlx == "metal" else mx.cpu)
    if os.environ.get("SMARTAI_TEST_OFFLINE", "1") == "0":
        return
    original_resolve = socket.getaddrinfo
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_sendto = socket.socket.sendto

    def refuse_external(host):
        if not _local_host(host):
            raise OSError("External network is disabled for model-free tests; mock the service")

    def refuse_external_url(url):
        host = urllib.parse.urlsplit(str(url)).hostname
        if host is not None:
            refuse_external(host)

    # A loopback HTTP proxy can otherwise forward an external URL without the
    # socket layer ever seeing its remote hostname. Reject the destination
    # before creating a tunnel or writing a request, including redirects.
    original_tunnel = http.client.HTTPConnection.set_tunnel
    original_request = http.client.HTTPConnection.putrequest
    original_open = urllib.request.OpenerDirector.open

    def set_tunnel(connection, host, *args, **kwargs):
        refuse_external(host)
        return original_tunnel(connection, host, *args, **kwargs)

    def putrequest(connection, method, url, *args, **kwargs):
        refuse_external_url(url)
        return original_request(connection, method, url, *args, **kwargs)

    def open_url(opener, fullurl, *args, **kwargs):
        refuse_external_url(getattr(fullurl, "full_url", fullurl))
        return original_open(opener, fullurl, *args, **kwargs)

    http.client.HTTPConnection.set_tunnel = set_tunnel
    http.client.HTTPConnection.putrequest = putrequest
    urllib.request.OpenerDirector.open = open_url

    try:
        import requests.adapters
    except ImportError:
        pass
    else:
        original_send = requests.adapters.HTTPAdapter.send

        def send_request(adapter, request, *args, **kwargs):
            refuse_external_url(request.url)
            return original_send(adapter, request, *args, **kwargs)

        requests.adapters.HTTPAdapter.send = send_request

    try:
        import httpx
    except ImportError:
        pass
    else:
        original_httpx_send = httpx.Client.send
        original_httpx_async_send = httpx.AsyncClient.send
        original_httpx_transport = httpx.HTTPTransport.handle_request
        original_httpx_async_transport = httpx.AsyncHTTPTransport.handle_async_request

        def send_httpx(client, request, *args, **kwargs):
            refuse_external_url(request.url)
            return original_httpx_send(client, request, *args, **kwargs)

        async def send_httpx_async(client, request, *args, **kwargs):
            refuse_external_url(request.url)
            return await original_httpx_async_send(client, request, *args, **kwargs)

        def handle_httpx(transport, request):
            refuse_external_url(request.url)
            return original_httpx_transport(transport, request)

        async def handle_httpx_async(transport, request):
            refuse_external_url(request.url)
            return await original_httpx_async_transport(transport, request)

        httpx.Client.send = send_httpx
        httpx.AsyncClient.send = send_httpx_async
        httpx.HTTPTransport.handle_request = handle_httpx
        httpx.AsyncHTTPTransport.handle_async_request = handle_httpx_async

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
