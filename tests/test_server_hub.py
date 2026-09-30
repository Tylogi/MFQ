from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from mfq.server.api import create_app
from mfq.server.protocol.models import (
    HubModelFile,
    HubModelInfo,
    HubModelSearchResult,
    HubModelSummary,
    HubModelVariant,
    ModelConfigurationStatus,
    OfficialModelInfo,
    OfficialModelList,
    OfficialModelSource,
)
from mfq.server.services.hub import (
    HubError,
    _model_variants,
    resolve_hub_reference,
    system_profile,
)
from mfq.server.services.service import ServerService
from mfq.server.state.storage import SessionStore
from tests.test_server_service import FakeBackend


class FakeHub:
    async def search(self, provider, query, *, limit):
        return HubModelSearchResult(
            data=[HubModelSummary(provider=provider, repo_id=f"owner/{query}")][:limit]
        )

    async def info(self, provider, repo_id, revision):
        return HubModelInfo(
            provider=provider,
            repo_id=repo_id,
            revision=revision or "main",
            total_bytes=42,
            files=[HubModelFile(name="weight.mfq", byte_size=42)],
        )

    async def official(self, *, profile, refresh):
        configuration = ModelConfigurationStatus(
            status="recommended",
            required_memory_bytes=42,
            available_memory_bytes=84,
            reasons=["fits"],
        )
        source = OfficialModelSource(
            provider="huggingface",
            repo_id="owner/model",
            revision="main",
            url="https://huggingface.co/owner/model",
        )
        return OfficialModelList(
            system=profile,
            data=[
                OfficialModelInfo(
                    id="model",
                    name="Model",
                    family="Test",
                    architecture="test",
                    description="Test model",
                    description_zh="测试模型",
                    sources=[source],
                    selected_source=source,
                    revision="main",
                    variants=[
                        HubModelVariant(
                            id="mfq:model",
                            label="model",
                            format="mfq",
                            files=["weight.mfq"],
                            byte_size=42,
                            configuration=configuration,
                        )
                    ],
                    configuration=configuration,
                )
            ],
        )


def test_hub_search_and_info_are_normalized(tmp_path: Path) -> None:
    async def run() -> None:
        service = ServerService(
            SessionStore(tmp_path / "mfq.server.sqlite3"),
            FakeBackend(),
            hub_catalog=FakeHub(),
        )
        transport = httpx.ASGITransport(app=create_app(service))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            search = await client.get(
                "/api/v1/hub/models",
                params={"provider": "modelscope", "query": "model"},
            )
            assert search.status_code == 200
            assert search.json()["data"][0]["repo_id"] == "owner/model"
            info = await client.get(
                "/api/v1/hub/models/huggingface/owner/model",
                params={"revision": "dev"},
            )
            assert info.status_code == 200
            assert info.json()["files"][0]["byte_size"] == 42
            assert info.json()["revision"] == "dev"
        await service.aclose()

    asyncio.run(run())


def test_hub_failures_are_retryable_gateway_errors(tmp_path: Path) -> None:
    class BrokenHub(FakeHub):
        async def search(self, provider, query, *, limit):
            raise HubError("offline")

        async def info(self, provider, repo_id, revision):
            raise HubError("offline")

    async def run() -> None:
        service = ServerService(
            SessionStore(tmp_path / "mfq.server.sqlite3"),
            FakeBackend(),
            hub_catalog=BrokenHub(),
        )
        transport = httpx.ASGITransport(app=create_app(service))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/api/v1/hub/models",
                params={"provider": "huggingface", "query": "model"},
            )
            assert response.status_code == 502
            assert response.json()["error"]["retryable"]
            resolved = await client.post(
                "/api/v1/hub/resolve",
                json={"reference": "owner/model"},
            )
            assert resolved.status_code == 502
            assert resolved.json()["error"]["retryable"]
        await service.aclose()

    asyncio.run(run())


def test_official_catalog_and_repository_link_resolution(tmp_path: Path) -> None:
    async def run() -> None:
        service = ServerService(
            SessionStore(tmp_path / "mfq.server.sqlite3"),
            FakeBackend(),
            hub_catalog=FakeHub(),
        )
        transport = httpx.ASGITransport(app=create_app(service))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            official = await client.get("/api/v1/hub/official")
            assert official.status_code == 200
            assert official.json()["data"][0]["variants"][0]["byte_size"] == 42
            resolved = await client.post(
                "/api/v1/hub/resolve",
                json={
                    "reference": "https://huggingface.co/owner/model/tree/dev",
                    "fallback_provider": "modelscope",
                },
            )
            assert resolved.status_code == 200
            assert resolved.json()["provider"] == "huggingface"
            assert resolved.json()["repo_id"] == "owner/model"
            assert resolved.json()["revision"] == "dev"
            rejected = await client.post(
                "/api/v1/hub/resolve",
                json={
                    "reference": "https://example.com/owner/model",
                    "fallback_provider": "huggingface",
                },
            )
            assert rejected.status_code == 422
            assert rejected.json()["error"]["code"] == "invalid_model_hub_reference"
        await service.aclose()

    asyncio.run(run())


def test_repository_link_resolution_rejects_untrusted_hosts_and_credentials() -> None:
    assert resolve_hub_reference("owner/model", "modelscope") == (
        "modelscope",
        "owner/model",
        None,
    )
    assert resolve_hub_reference(
        "https://modelscope.cn/models/owner/model/tree/v1", "huggingface"
    ) == ("modelscope", "owner/model", "v1")
    for reference in (
        "https://example.com/owner/model",
        "https://user:password@huggingface.co/owner/model",
    ):
        try:
            resolve_hub_reference(reference)
        except HubError:
            pass
        else:
            raise AssertionError(f"untrusted reference was accepted: {reference}")


def test_mfq_variants_keep_subdirectories_and_use_the_quantization_suffix() -> None:
    files = [
        HubModelFile(
            name=(
                "DeepSeek-V4-Flash-0731-EW-V2-S/"
                "DeepSeek-V4-Flash-0731-EW-V2-S-00001-of-00002.mfq"
            ),
            byte_size=10,
        ),
        HubModelFile(
            name=(
                "DeepSeek-V4-Flash-0731-EW-V2-S/"
                "DeepSeek-V4-Flash-0731-EW-V2-S-00002-of-00002.mfq"
            ),
            byte_size=12,
        ),
    ]
    variants = _model_variants(
        files,
        system_profile(backend="metal", runtime_memory_budget_bytes=64 << 30),
    )
    assert len(variants) == 1
    assert variants[0].label == "DeepSeek-V4-Flash-0731-EW-V2-S"
    assert variants[0].precision == "V2-S"
    assert variants[0].byte_size == 22


def test_unknown_repository_remains_downloadable_with_a_warning() -> None:
    variants = _model_variants(
        [HubModelFile(name="weights.custom", byte_size=42)],
        system_profile(backend="metal", runtime_memory_budget_bytes=64 << 30),
    )
    assert len(variants) == 1
    assert variants[0].format == "unknown"
    assert variants[0].files == []
    assert variants[0].configuration.status == "warning"


def test_third_party_variants_distinguish_unsupported_and_unverified_architectures() -> None:
    files = [HubModelFile(name="model.safetensors", byte_size=42)]
    profile = system_profile(
        backend="metal", runtime_memory_budget_bytes=64 << 30
    )
    unsupported = _model_variants(files, profile, runtime_compatible=False)
    unverified = _model_variants(files, profile, runtime_compatible=None)
    assert unsupported[0].configuration.status == "warning"
    assert unverified[0].configuration.status == "unknown"
