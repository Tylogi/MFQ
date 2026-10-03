from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from mfq.commands import serve
from mfq.server.api import network


@pytest.fixture(autouse=True)
def isolated_native_proxy_settings(monkeypatch):
    for name in ("getproxies_macosx_sysconf", "getproxies_registry"):
        monkeypatch.setattr(network.urllib.request, name, lambda: {}, raising=False)


@pytest.mark.parametrize(
    "host",
    (
        "127.0.0.1",
        "127.23.45.67",
        "::1",
        "::ffff:127.0.0.1",
        "localhost",
        "LOCALHOST.",
    ),
)
def test_loopback_bind_detection_accepts_only_local_targets(host: str) -> None:
    assert network.is_loopback_bind_host(host)
    assert network.network_auth_error(host, None) is None


@pytest.mark.parametrize(
    "host",
    ("0.0.0.0", "::", "192.168.1.20", "mfq.internal", ""),
)
def test_network_bind_requires_an_api_key(host: str) -> None:
    assert not network.is_loopback_bind_host(host)
    assert network.network_auth_error(host, None) is not None
    assert network.network_auth_error(host, "root-secret") is None


def test_serve_rejects_an_unauthenticated_network_bind_before_startup(
    monkeypatch,
) -> None:
    monkeypatch.delenv("MFQ_TEST_SERVER_KEY", raising=False)
    monkeypatch.setattr(
        network,
        "install_system_proxy_environment",
        lambda: pytest.fail("network startup ran before bind authentication validation"),
    )

    with pytest.raises(ValueError, match="API key is required"):
        serve._run(
            SimpleNamespace(
                host="0.0.0.0",
                api_key_env="MFQ_TEST_SERVER_KEY",
            )
        )


def test_system_proxy_environment_discovers_os_proxy_and_bypasses_loopback(monkeypatch) -> None:
    monkeypatch.setattr(
        network,
        "getproxies",
        lambda: {
            "http": "http://proxy.test:8080",
            "https": "http://proxy.test:8080",
            "no": "*.internal.test",
        },
    )

    environment = network.system_proxy_environment({})

    assert environment["HTTP_PROXY"] == "http://proxy.test:8080"
    assert environment["HTTPS_PROXY"] == "http://proxy.test:8080"
    assert environment["NO_PROXY"] == "*.internal.test,127.0.0.1,localhost,::1"
    assert environment["no_proxy"] == environment["NO_PROXY"]


def test_system_proxy_environment_preserves_explicit_proxy_and_bypass(monkeypatch) -> None:
    monkeypatch.setattr(
        network.urllib.request, "getproxies_macosx_sysconf", lambda: {}, raising=False
    )
    monkeypatch.setattr(
        network.urllib.request, "getproxies_registry", lambda: {}, raising=False
    )
    monkeypatch.setattr(
        network,
        "getproxies",
        lambda: {"https": "http://system-proxy.test:8080"},
    )

    environment = network.system_proxy_environment(
        {
            "HTTPS_PROXY": "http://explicit-proxy.test:9090",
            "NO_PROXY": "*.example.test,localhost",
        }
    )

    assert environment["HTTPS_PROXY"] == "http://explicit-proxy.test:9090"
    assert environment["https_proxy"] == "http://explicit-proxy.test:9090"
    assert environment["NO_PROXY"] == "*.example.test,localhost,127.0.0.1,::1"


@pytest.mark.parametrize("discovery", ("getproxies_macosx_sysconf", "getproxies_registry"))
def test_bypass_only_environment_does_not_hide_native_proxy(monkeypatch, discovery) -> None:
    for name in ("getproxies_macosx_sysconf", "getproxies_registry"):
        monkeypatch.setattr(network.urllib.request, name, lambda: {}, raising=False)
    monkeypatch.setattr(network, "getproxies", lambda: {"no": "localhost"})
    monkeypatch.setattr(
        network.urllib.request,
        discovery,
        lambda: {"https": "http://proxy.test:8080", "no": "*.internal.test"},
    )

    environment = network.system_proxy_environment({"NO_PROXY": "localhost"})

    assert environment["HTTPS_PROXY"] == "http://proxy.test:8080"
    assert environment["NO_PROXY"] == "localhost,*.internal.test,127.0.0.1,::1"


def test_explicit_environment_proxy_wins_over_native_settings(monkeypatch) -> None:
    monkeypatch.setattr(network, "getproxies", lambda: {"no": "localhost"})
    monkeypatch.setattr(
        network.urllib.request,
        "getproxies_macosx_sysconf",
        lambda: {"https": "http://system-proxy.test:8080"},
        raising=False,
    )
    monkeypatch.setattr(network.urllib.request, "getproxies_registry", lambda: {}, raising=False)

    environment = network.system_proxy_environment(
        {"https_proxy": "http://explicit-proxy.test:9090", "NO_PROXY": "localhost"}
    )

    assert environment["HTTPS_PROXY"] == "http://explicit-proxy.test:9090"
    assert environment["https_proxy"] == "http://explicit-proxy.test:9090"


@pytest.mark.parametrize("error", (OSError, ValueError, RuntimeError))
def test_proxy_discovery_failure_defaults_to_direct(monkeypatch, error) -> None:
    def unavailable():
        raise error("system settings unavailable")

    monkeypatch.setattr(network, "getproxies", unavailable)
    for name in ("getproxies_macosx_sysconf", "getproxies_registry"):
        monkeypatch.setattr(network.urllib.request, name, unavailable, raising=False)
    environment = network.system_proxy_environment({})
    assert "HTTP_PROXY" not in environment
    assert "HTTPS_PROXY" not in environment
    assert environment["NO_PROXY"] == "127.0.0.1,localhost,::1"
    explicit = network.system_proxy_environment({"HTTPS_PROXY": "http://proxy.test:8080"})
    assert explicit["https_proxy"] == "http://proxy.test:8080"


def test_explicit_all_proxy_takes_priority_over_system_proxy(monkeypatch) -> None:
    monkeypatch.setattr(network, "getproxies", lambda: {"https": "http://os.test:8080"})
    environment = network.system_proxy_environment({"ALL_PROXY": "http://explicit.test:8080"})
    assert environment["ALL_PROXY"] == "http://explicit.test:8080"
    assert "https_proxy" not in environment


@pytest.mark.parametrize("outcome", (200, 401, 404, 503, httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.RemoteProtocolError))
def test_download_uses_proxy_only_when_direct_connection_fails(monkeypatch, outcome) -> None:
    environment = {"HTTPS_PROXY": "http://proxy.test:8080", "NO_PROXY": "localhost", "HF_TOKEN": "test-token"}
    monkeypatch.setattr(network, "system_proxy_environment", lambda: dict(environment))
    original = httpx.AsyncClient
    calls = []

    def respond(request):
        calls.append(request)
        if isinstance(outcome, type):
            raise outcome("direct unavailable", request=request)
        return httpx.Response(outcome)

    def client(**kwargs):
        assert kwargs["trust_env"] is False and kwargs["timeout"] == 4.0
        return original(transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)
    selected, proxy = asyncio.run(network.download_environment("https://huggingface.co"))
    assert len(calls) == 1 and calls[0].method == "HEAD"
    assert selected["HF_TOKEN"] == "test-token"
    assert proxy is isinstance(outcome, type)
    if proxy:
        assert selected == environment
    else:
        assert "HTTPS_PROXY" not in selected and selected["NO_PROXY"] == "*"


@pytest.mark.parametrize("direct,environment,endpoint", (
    (True, {"HTTPS_PROXY": "http://proxy.test:8080", "NO_PROXY": ""}, "https://modelscope.cn"),
    (False, {"NO_PROXY": "localhost"}, "https://huggingface.co"),
    (False, {"HTTPS_PROXY": "http://proxy.test:8080", "NO_PROXY": "localhost"}, "https://localhost"),
))
def test_download_skips_probe_without_an_eligible_proxy(monkeypatch, direct, environment, endpoint) -> None:
    monkeypatch.setattr(network, "system_proxy_environment", lambda: dict(environment))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: pytest.fail("unexpected connectivity probe"))
    selected, proxy = asyncio.run(network.download_environment(endpoint, direct=direct))
    assert proxy is False and "HTTPS_PROXY" not in selected and selected["NO_PROXY"] == "*"
