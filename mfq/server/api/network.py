"""Network environment helpers shared by hub discovery and download jobs."""

from __future__ import annotations

import ipaddress
import os
import ssl
import urllib.request
from collections.abc import Mapping
from urllib.request import getproxies
from urllib.parse import urlparse

_LOCAL_BYPASS = ("127.0.0.1", "localhost", "::1")


def is_loopback_bind_host(value: str) -> bool:
    """Return whether a server bind target is explicitly loopback-only."""

    candidate = value.strip()
    if not candidate:
        return False
    if candidate.rstrip(".").casefold() == "localhost":
        return True
    try:
        address = ipaddress.ip_address(candidate)
    except ValueError:
        return False
    if address.is_loopback:
        return True
    return bool(
        isinstance(address, ipaddress.IPv6Address)
        and address.ipv4_mapped is not None
        and address.ipv4_mapped.is_loopback
    )


def network_auth_error(host: str, api_key: str | None) -> str | None:
    """Describe an unsafe unauthenticated network bind, if any."""

    if is_loopback_bind_host(host):
        return None
    if isinstance(api_key, str) and api_key.strip():
        return None
    return (
        f"an API key is required when binding MFQ Server to {host!r}; "
        "configure the selected API-key environment variable or bind to "
        "127.0.0.1"
    )


def system_proxy_environment(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return an environment that honors OS proxies without proxying loopback traffic."""

    environment = dict(os.environ if base is None else base)
    try:
        proxies = getproxies()
    except (OSError, ValueError, RuntimeError):
        proxies = {}
    # NO_PROXY alone makes urllib skip OS settings on macOS and Windows.
    # Merge native discovery without overwriting explicit environment proxies.
    for name in ("getproxies_macosx_sysconf", "getproxies_registry"):
        discover = getattr(urllib.request, name, None)
        if discover is not None:
            try:
                discovered = discover()
            except (OSError, ValueError, RuntimeError):
                continue
            for scheme, value in discovered.items():
                if scheme == "no":
                    proxies[scheme] = ",".join(filter(None, (proxies.get(scheme), value)))
                else:
                    proxies.setdefault(scheme, value)
    for scheme in ("http", "https"):
        lower = f"{scheme}_proxy"
        upper = lower.upper()
        configured = environment.get(lower) or environment.get(upper)
        discovered = proxies.get(scheme)
        value = configured or (
            discovered if not (environment.get("all_proxy") or environment.get("ALL_PROXY")) else None
        )
        if value:
            environment.setdefault(lower, value)
            environment.setdefault(upper, value)

    bypass: list[str] = []
    for name in ("NO_PROXY", "no_proxy"):
        bypass.extend(
            value.strip()
            for value in environment.get(name, "").split(",")
            if value.strip()
        )
    bypass.extend(
        value.strip()
        for value in str(proxies.get("no") or "").split(",")
        if value.strip()
    )
    for value in _LOCAL_BYPASS:
        if value not in bypass:
            bypass.append(value)
    serialized = ",".join(dict.fromkeys(bypass))
    environment["NO_PROXY"] = serialized
    environment["no_proxy"] = serialized
    return environment


def install_system_proxy_environment() -> None:
    """Install discovered proxy values once for in-process HTTP clients."""

    resolved = system_proxy_environment()
    for name in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY"):
        if value := resolved.get(name):
            os.environ.setdefault(name, value)
    os.environ["NO_PROXY"] = resolved["NO_PROXY"]
    os.environ["no_proxy"] = resolved["no_proxy"]


async def download_environment(endpoint: str, *, direct: bool = False) -> tuple[dict[str, str], bool]:
    import httpx

    environment = system_proxy_environment()
    direct_environment = dict(environment)
    for name in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
        direct_environment.pop(name, None)
    direct_environment["NO_PROXY"] = "*"
    direct_environment["no_proxy"] = "*"
    parsed = urlparse(endpoint)
    proxy = environment.get(f"{parsed.scheme}_proxy") or environment.get(f"{parsed.scheme.upper()}_PROXY")
    proxy = proxy or environment.get("all_proxy") or environment.get("ALL_PROXY")
    bypass = urllib.request.proxy_bypass_environment(parsed.hostname or "", {"no": environment["NO_PROXY"]})
    if direct or not proxy or bypass:
        return direct_environment, False
    verify: ssl.SSLContext | bool = True
    if environment.get("SSL_CERT_FILE") or environment.get("SSL_CERT_DIR"):
        verify = ssl.create_default_context(cafile=environment.get("SSL_CERT_FILE"), capath=environment.get("SSL_CERT_DIR"))
    try:
        async with httpx.AsyncClient(trust_env=False, verify=verify, timeout=4.0) as client:
            await client.head(endpoint)
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.RemoteProtocolError):
        return environment, True
    return direct_environment, False
