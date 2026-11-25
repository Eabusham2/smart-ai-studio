"""Direct HTTP client with SSRF-resistant destination validation.

The model-callable web tools use this module instead of the process-wide
``urllib`` opener.  Every request and redirect is resolved before connecting,
all resolved addresses must be globally routable, and the selected address is
pinned through the socket connection and checked against the connected peer.
"""

from __future__ import annotations

from dataclasses import dataclass
import http.client
import ipaddress
import socket
import ssl
from typing import Any, Mapping, Optional, Tuple
import urllib.parse


class UnsafeURLError(ValueError):
    """Raised when a URL cannot safely cross the outbound network boundary."""


class SafeHTTPError(OSError):
    """Raised when a validated HTTP request does not produce a usable response."""


@dataclass(frozen=True)
class _ResolvedAddress:
    family: int
    sockaddr: Tuple[Any, ...]
    ip: ipaddress._BaseAddress


@dataclass(frozen=True)
class _Destination:
    url: str
    scheme: str
    hostname: str
    port: int
    authority: str
    request_target: str
    addresses: Tuple[_ResolvedAddress, ...]


class SafeHTTPResponse:
    """Small response adapter preserving the interface used by existing tools."""

    def __init__(self, connection: http.client.HTTPConnection,
                 response: http.client.HTTPResponse, url: str):
        self._connection = connection
        self._response = response
        self._url = url

    @property
    def headers(self):
        return self._response.headers

    @property
    def status(self) -> int:
        return self._response.status

    def getheader(self, name: str, default: Optional[str] = None):
        return self._response.getheader(name, default)

    def geturl(self) -> str:
        return self._url

    def read(self, amount: Optional[int] = None) -> bytes:
        if amount is None:
            return self._response.read()
        return self._response.read(amount)

    def close(self) -> None:
        try:
            self._response.close()
        finally:
            self._connection.close()

    def __enter__(self) -> "SafeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


# Explicitly cover ranges whose ``is_global`` classification has changed
# between supported Python releases or can tunnel an embedded private address.
_SENSITIVE_NETWORKS = tuple(ipaddress.ip_network(value) for value in (
    "0.0.0.0/8",
    "10.0.0.0/8",
    "100.64.0.0/10",
    "127.0.0.0/8",
    "169.254.0.0/16",
    "172.16.0.0/12",
    "192.0.0.0/24",
    "192.0.2.0/24",
    "192.168.0.0/16",
    "198.18.0.0/15",
    "198.51.100.0/24",
    "203.0.113.0/24",
    "224.0.0.0/4",
    "240.0.0.0/4",
    "::/128",
    "::1/128",
    "64:ff9b::/96",
    "64:ff9b:1::/48",
    "100::/64",
    "::/96",
    "::ffff:0:0/96",
    "2001::/32",
    "2001:2::/48",
    "2001:db8::/32",
    "2001:10::/28",
    "2001:20::/28",
    "2002::/16",
    "fc00::/7",
    "fe80::/10",
    "ff00::/8",
))
_REDIRECT_STATUSES = frozenset((301, 302, 303, 307, 308))
_RESERVED_HEADERS = frozenset((
    "authorization", "connection", "cookie", "host", "proxy-authorization",
    "proxy-connection", "transfer-encoding",
))


def _is_allowed_address(address: ipaddress._BaseAddress) -> bool:
    if not address.is_global:
        return False
    return not any(address in network for network in _SENSITIVE_NETWORKS
                   if address.version == network.version)


def _normalize_hostname(hostname: str) -> str:
    if not hostname or "%" in hostname:
        raise UnsafeURLError("URL host is missing or ambiguous")
    hostname = hostname.rstrip(".")
    if not hostname:
        raise UnsafeURLError("URL host is missing")
    try:
        return str(ipaddress.ip_address(hostname))
    except ValueError:
        try:
            ascii_hostname = hostname.encode("idna").decode("ascii").lower()
        except UnicodeError as exc:
            raise UnsafeURLError("URL host is not valid IDNA") from exc
        labels = ascii_hostname.split(".")
        if (len(ascii_hostname) > 253 or any(
                not label or len(label) > 63 or label.startswith("-")
                or label.endswith("-")
                or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-"
                       for char in label)
                for label in labels)):
            raise UnsafeURLError("URL host is malformed")
        return ascii_hostname


def _resolve_destination(url: str) -> _Destination:
    if not isinstance(url, str) or not url or url != url.strip():
        raise UnsafeURLError("URL must be a non-empty string without surrounding whitespace")
    if "\\" in url or any(ord(char) <= 32 or ord(char) == 127 for char in url):
        raise UnsafeURLError("URL contains ambiguous or control characters")
    try:
        parsed = urllib.parse.urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise UnsafeURLError("URL authority is malformed") from exc
    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https") or not parsed.netloc:
        raise UnsafeURLError("Only absolute HTTP(S) URLs are allowed")
    if parsed.username is not None or parsed.password is not None:
        raise UnsafeURLError("URL user information is not allowed")
    hostname = _normalize_hostname(parsed.hostname or "")
    if port is None:
        port = 443 if scheme == "https" else 80
    elif port == 0:
        raise UnsafeURLError("URL port must be between 1 and 65535")

    try:
        records = socket.getaddrinfo(
            hostname, port, socket.AF_UNSPEC, socket.SOCK_STREAM, socket.IPPROTO_TCP,
        )
    except socket.gaierror as exc:
        raise SafeHTTPError("URL host could not be resolved") from exc
    addresses = []
    seen = set()
    for family, socktype, protocol, _canonical, sockaddr in records:
        if family not in (socket.AF_INET, socket.AF_INET6):
            continue
        try:
            address = ipaddress.ip_address(sockaddr[0])
        except ValueError as exc:
            raise UnsafeURLError("Resolver returned a malformed address") from exc
        if not _is_allowed_address(address):
            raise UnsafeURLError(f"URL resolves to a non-public address: {address}")
        key = (family, address, port)
        if key not in seen:
            seen.add(key)
            if family == socket.AF_INET6:
                flowinfo = sockaddr[2] if len(sockaddr) > 2 else 0
                scope_id = sockaddr[3] if len(sockaddr) > 3 else 0
                pinned_sockaddr = (str(address), port, flowinfo, scope_id)
            else:
                pinned_sockaddr = (str(address), port)
            addresses.append(_ResolvedAddress(family, pinned_sockaddr, address))
    if not addresses:
        raise SafeHTTPError("URL host has no usable A or AAAA address")

    is_ipv6 = ":" in hostname
    authority = f"[{hostname}]" if is_ipv6 else hostname
    default_port = 443 if scheme == "https" else 80
    if port != default_port:
        authority += f":{port}"
    path = parsed.path or "/"
    request_target = urllib.parse.urlunsplit(("", "", path, parsed.query, ""))
    return _Destination(
        url=url,
        scheme=scheme,
        hostname=hostname,
        port=port,
        authority=authority,
        request_target=request_target,
        addresses=tuple(addresses),
    )


def _connect_pinned_socket(address: _ResolvedAddress, timeout: float) -> socket.socket:
    sock = socket.socket(address.family, socket.SOCK_STREAM, socket.IPPROTO_TCP)
    try:
        sock.settimeout(timeout)
        sock.connect(address.sockaddr)
        peer = ipaddress.ip_address(sock.getpeername()[0])
        if peer != address.ip:
            raise UnsafeURLError("Connected peer does not match the validated address")
        return sock
    except Exception:
        sock.close()
        raise


class _PinnedHTTPConnection(http.client.HTTPConnection):
    def __init__(self, destination: _Destination, address: _ResolvedAddress,
                 timeout: float):
        super().__init__(destination.hostname, destination.port, timeout=timeout)
        self._validated_address = address

    def connect(self) -> None:
        self.sock = _connect_pinned_socket(self._validated_address, self.timeout)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, destination: _Destination, address: _ResolvedAddress,
                 timeout: float):
        context = ssl.create_default_context()
        super().__init__(destination.hostname, destination.port, timeout=timeout,
                         context=context)
        self._validated_address = address

    def connect(self) -> None:
        raw_socket = _connect_pinned_socket(self._validated_address, self.timeout)
        try:
            self.sock = self._context.wrap_socket(
                raw_socket, server_hostname=self.host,
            )
            peer = ipaddress.ip_address(self.sock.getpeername()[0])
            if peer != self._validated_address.ip:
                raise UnsafeURLError("TLS peer does not match the validated address")
        except Exception:
            if self.sock is not None:
                self.sock.close()
            else:
                raw_socket.close()
            self.sock = None
            raise


def _validated_headers(headers: Optional[Mapping[str, str]]) -> Mapping[str, str]:
    result = {}
    for raw_name, raw_value in (headers or {}).items():
        name = str(raw_name)
        value = str(raw_value)
        if (not name or name.lower() in _RESERVED_HEADERS
                or "\r" in name or "\n" in name
                or "\r" in value or "\n" in value):
            raise UnsafeURLError("Request contains an unsafe header")
        result[name] = value
    return result


def _request_once(destination: _Destination, headers: Mapping[str, str],
                  timeout: float) -> SafeHTTPResponse:
    request_headers = dict(headers)
    request_headers.update({
        "Host": destination.authority,
        "Accept-Encoding": "identity",
        "Connection": "close",
    })
    last_error: Optional[BaseException] = None
    for address in destination.addresses:
        connection_class = (
            _PinnedHTTPSConnection if destination.scheme == "https"
            else _PinnedHTTPConnection
        )
        connection = connection_class(destination, address, timeout)
        try:
            connection.connect()
            connection.request("GET", destination.request_target,
                               headers=request_headers)
            response = connection.getresponse()
            return SafeHTTPResponse(connection, response, destination.url)
        except UnsafeURLError:
            connection.close()
            raise
        except (OSError, ssl.SSLError, http.client.HTTPException) as exc:
            last_error = exc
            connection.close()
    raise SafeHTTPError("Unable to connect to a validated destination") from last_error


def safe_urlopen(url: str, *, headers: Optional[Mapping[str, str]] = None,
                 timeout: float = 4.0, max_redirects: int = 5) -> SafeHTTPResponse:
    """Open a public HTTP(S) URL without proxies or automatic redirects."""
    if max_redirects < 0:
        raise ValueError("max_redirects must be nonnegative")
    request_headers = _validated_headers(headers)
    current_url = url
    for redirect_count in range(max_redirects + 1):
        destination = _resolve_destination(current_url)
        response = _request_once(destination, request_headers, timeout)
        if response.status not in _REDIRECT_STATUSES:
            if response.status >= 400:
                status = response.status
                response.close()
                raise SafeHTTPError(f"HTTP request failed with status {status}")
            return response
        location = response.getheader("Location")
        response.close()
        if not location:
            raise SafeHTTPError("Redirect response omitted its destination")
        if redirect_count >= max_redirects:
            raise SafeHTTPError("HTTP redirect limit exceeded")
        current_url = urllib.parse.urljoin(current_url, location)
    raise SafeHTTPError("HTTP redirect limit exceeded")
