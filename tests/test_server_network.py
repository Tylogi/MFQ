from __future__ import annotations

from types import SimpleNamespace

import pytest

from mfq.commands import serve
from mfq.server import network


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
