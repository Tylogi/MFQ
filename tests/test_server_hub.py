from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

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
    _OFFICIAL_MODELS,
    HubCatalog,
    HubError,
    _model_configuration_status,
    _model_variants,
    resolve_hub_reference,
    system_profile,
)
from mfq.server.services.service import ServerService
from mfq.server.state.storage import SessionStore
from tests.test_server_service import FakeBackend

_DISCOVER_OFFICIAL = HubCatalog._discover_official


def test_official_model_descriptions_use_exact_generations_and_model_capabilities():
    specs = {item.name: item for item in _OFFICIAL_MODELS}
    for version in ("3.6", "3.8"):
        spec = specs[f"Qwen{version}-27B"]
        assert spec.family == f"Qwen{version}"
        assert f"Qwen{version} 系列的 27B 稠密模型" in spec.description_zh
        assert "图像与视频理解" in spec.description_zh
        assert "27B 稠密 MFQ 版本" not in spec.description_zh
    catalog = HubCatalog()
    catalog._discovered_sources.add(("modelscope", "Tylogi/Qwen3.8-Flash-Next-EWQ-V1-MFQ"))
    flash = next(item for item in catalog._official_specs() if "Flash-Next" in item.name)
    assert "高性能、快速" in flash.description_zh
    assert "低精度" not in flash.description_zh
    assert "PLE" in flash.description_zh


@pytest.fixture(autouse=True)
def no_live_author_discovery(monkeypatch):
    async def discover(provider):
        return []
    monkeypatch.setattr(HubCatalog, '_discover_official', staticmethod(discover))


def test_new_author_repositories_appear_once_and_survive_offline_restart(tmp_path, monkeypatch):
    async def run():
        catalog = HubCatalog(cache_path=tmp_path / 'official.json')
        repo = 'Tylogi/Qwen3.8-Flash-Next-EWQ-V1-MFQ'
        async def discover(provider):
            return [repo, 'Other/Not-Official-MFQ', 'Tylogi/nonquantized']
        async def fetch(provider, repo_id, revision, profile):
            return HubModelInfo(provider=provider, repo_id=repo_id, revision='master',
                files=[HubModelFile(name='S4-L/model-00001-of-00002.mfq', byte_size=1024),
                       HubModelFile(name='S4-L/model-00002-of-00002.mfq', byte_size=1024)])
        monkeypatch.setattr(catalog, '_discover_official', discover)
        monkeypatch.setattr(catalog, '_fetch_info', fetch)
        assert len((await catalog.official()).data) == 4
        await catalog._refresh_task
        result = await catalog.official()
        assert len(result.data) == 5
        new = next(item for item in result.data if 'Flash-Next' in item.name)
        assert {source.provider for source in new.sources} == {'huggingface', 'modelscope'}
        assert len(new.variants) == 1 and new.variants[0].byte_size == 2048
        restored = HubCatalog(cache_path=tmp_path / 'official.json')
        assert len((await restored.official()).data) == 5
        async def offline(*args): raise HubError('offline')
        monkeypatch.setattr(restored, '_discover_official', offline)
        monkeypatch.setattr(restored, '_fetch_info', offline)
        await restored.official(refresh=True)
        await restored._refresh_task
        assert len((await restored.official()).data) == 5
        await catalog.aclose()
        await restored.aclose()
    asyncio.run(run())


def test_author_discovery_paginates_and_filters_private_and_other_owners(monkeypatch):
    import mfq.server.services.hub as hub
    requests = []
    def respond(request):
        requests.append(request)
        if request.url.path == '/openapi/v1/models':
            assert request.url.params['owner'] == 'Tylogi'
            page = int(request.url.params['page_number'])
            models = [{'id': f'Tylogi/model-{index}-MFQ'} for index in range(20)] if page == 1 else [
                {'id': 'Tylogi/new-MFQ'}, {'id': 'Other/model-MFQ'}, {'id': 'Tylogi/private-MFQ', 'private': True}, {'id': 'Tylogi/plain'}]
            return httpx.Response(200, json={'success': True, 'data': {'models': models, 'total_count': 24}})
        assert request.url.params['author'] == 'Tylogi'
        return httpx.Response(200, json=[{'id': 'Tylogi/hf-MFQ'}, {'id': 'Other/hf-MFQ'}])
    monkeypatch.setattr(hub, '_metadata_client', lambda *args, **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    async def run():
        assert len(await _DISCOVER_OFFICIAL('modelscope')) == 21
        assert await _DISCOVER_OFFICIAL('huggingface') == ['Tylogi/hf-MFQ']
    asyncio.run(run())
    assert len(requests) == 3


def test_official_refresh_is_incremental_coalesced_and_cancelable(monkeypatch) -> None:
    async def run() -> None:
        catalog = HubCatalog()
        started = asyncio.Event()
        blocked = asyncio.Event()
        calls = []
        canceled = []

        async def fetch(provider, repo_id, revision, profile):
            calls.append((provider, repo_id))
            if provider == "huggingface":
                started.set()
                try:
                    await blocked.wait()
                finally:
                    canceled.append(repo_id)
            return HubModelInfo(
                provider=provider, repo_id=repo_id, revision="master",
                files=[HubModelFile(name="S4.mfq", byte_size=42)],
            )

        monkeypatch.setattr(catalog, "_fetch_info", fetch)
        first = await asyncio.wait_for(catalog.official(), timeout=0.5)
        assert first.refreshing and len(first.data) == 4
        await started.wait()
        await asyncio.sleep(0)
        partial = await catalog.official(refresh=True)
        assert partial.refreshing and len(calls) == 5
        available = next(item for item in partial.data if item.id == "qwen3-8-27b")
        assert available.variants and available.selected_source.available
        cached = await catalog.info("modelscope", available.selected_source.repo_id, "master")
        assert cached.files[0].byte_size == 42 and len(calls) == 5
        await catalog.aclose()
        assert len(canceled) == 2 and catalog._refresh_task.done()

    asyncio.run(run())


def test_official_cache_survives_restart_and_failed_refresh(tmp_path, monkeypatch) -> None:
    async def run() -> None:
        path = tmp_path / "hub-official.json"
        catalog = HubCatalog(cache_path=path)

        async def fetch(provider, repo_id, revision, profile):
            return HubModelInfo(
                provider=provider, repo_id=repo_id, revision="master",
                files=[HubModelFile(name="S4.mfq", byte_size=42)],
            )

        monkeypatch.setattr(catalog, "_fetch_info", fetch)
        await catalog.official()
        await catalog._refresh_task
        assert path.exists()
        restored = HubCatalog(cache_path=path)
        assert not (await restored.official()).refreshing
        assert len(restored._official_cache) == 5
        original_times = restored._source_cache_times.copy()

        async def offline(*args):
            raise HubError("offline")

        monkeypatch.setattr(restored, "_fetch_info", offline)
        pending = await restored.official(refresh=True)
        assert pending.refreshing and all(item.variants for item in pending.data)
        await restored._refresh_task
        assert restored._source_cache_times == original_times
        assert all(item.variants for item in (await restored.official()).data)
        reloaded = HubCatalog(cache_path=path)
        assert all(item.variants for item in (await reloaded.official()).data)
        await catalog.aclose()
        await restored.aclose()
        await reloaded.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("content", ('{"version":99}', '{"version":1,"updated_at":NaN}', 'broken'))
def test_corrupt_official_cache_is_ignored(tmp_path, content) -> None:
    path = tmp_path / "hub-official.json"
    path.write_text(content)
    assert not HubCatalog(cache_path=path)._official_cache


def test_metadata_deadline_cancels_slow_source(monkeypatch) -> None:
    from mfq.server.services import hub

    async def run() -> None:
        canceled = asyncio.Event()

        async def slow(*args):
            try:
                await asyncio.Event().wait()
            finally:
                canceled.set()

        monkeypatch.setattr(hub, "_METADATA_DEADLINE", 0.01)
        monkeypatch.setattr(HubCatalog, "_info_huggingface", slow)
        with pytest.raises(HubError, match="timed out"):
            await HubCatalog().info("huggingface", "owner/model", None)
        assert canceled.is_set()

    asyncio.run(run())


def test_huggingface_metadata_uses_one_bounded_request(monkeypatch) -> None:
    from mfq.server.services import hub
    import huggingface_hub

    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={
            "id": "owner/model", "sha": "revision", "downloads": 3,
            "siblings": [{"rfilename": "S4.mfq", "size": 42,
                          "lfs": {"size": 42, "sha256": "a" * 64, "pointerSize": 1}}],
        })

    monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)
    monkeypatch.setattr(hub, "_metadata_client", lambda *args, **kwargs:
                        httpx.AsyncClient(transport=httpx.MockTransport(respond), **kwargs))
    info = asyncio.run(HubCatalog().info("huggingface", "owner/model", "feature/test"))
    assert len(requests) == 1
    assert requests[0].url.raw_path.startswith(b"/api/models/owner/model/revision/feature%2Ftest?")
    assert requests[0].url.params["blobs"] == "true"
    assert info.files[0].sha256 == "a" * 64 and info.total_bytes == 42


def test_modelscope_metadata_and_listing_run_in_parallel(monkeypatch) -> None:
    from mfq.server.services import hub
    from modelscope_hub import config
    from types import SimpleNamespace

    async def run() -> None:
        requests = []
        both_started = asyncio.Event()

        async def respond(request):
            requests.append(request)
            if len(requests) == 2:
                both_started.set()
            await both_started.wait()
            if "repo/files" in request.url.path:
                return httpx.Response(200, json={"Data": {"Files": [
                    {"Path": "folder", "Type": "tree"},
                    {"Path": "S4.mfq", "Size": 42, "Sha256": "b" * 64},
                ]}})
            return httpx.Response(200, json={"success": True, "data": {
                "license": "apache-2.0", "downloads": 7,
            }})

        monkeypatch.setattr(config, "get_default_config", lambda:
                            SimpleNamespace(endpoint="https://modelscope.cn", token=None))
        monkeypatch.setattr(hub, "_metadata_client", lambda *args, **kwargs:
                            httpx.AsyncClient(transport=httpx.MockTransport(respond), **kwargs))
        info = await asyncio.wait_for(HubCatalog().info("modelscope", "owner/model", None), 0.5)
        assert len(requests) == 2 and info.license == "apache-2.0"
        assert info.downloads == 7 and info.total_bytes == 42 and len(info.files) == 1
        assert info.files[0].sha256 == "b" * 64

    asyncio.run(run())


@pytest.mark.parametrize("endpoint,explicit,expected", (
    ("https://huggingface.co", None, None),
    ("https://huggingface.co", "http://proxy.test:8080", "http://proxy.test:8080"),
    ("http://localhost", "http://proxy.test:8080", None),
))
def test_metadata_client_defaults_to_direct_when_proxy_discovery_fails(
    monkeypatch, endpoint, explicit, expected
) -> None:
    from mfq.server.api import network
    from mfq.server.services import hub

    def unavailable():
        raise OSError("settings unavailable")

    for name in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "localhost")
    if explicit:
        monkeypatch.setenv("HTTPS_PROXY", explicit)
        monkeypatch.setenv("HTTP_PROXY", explicit)
    monkeypatch.setattr(network, "getproxies", unavailable)
    for name in ("getproxies_macosx_sysconf", "getproxies_registry"):
        monkeypatch.setattr(network.urllib.request, name, unavailable, raising=False)
    monkeypatch.setattr(hub.httpx, "AsyncClient", lambda **kwargs: kwargs)
    options = hub._metadata_client(endpoint)
    transport = options["transport"]
    assert isinstance(transport, hub._DirectFirstTransport) if expected else transport is None
    assert options["trust_env"] is False
    assert options["timeout"].connect == 4 and options["timeout"].read == 8


@pytest.mark.parametrize("failure", (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.RemoteProtocolError))
def test_unreachable_direct_connection_falls_back_to_proxy_once(monkeypatch, failure) -> None:
    from mfq.server.services import hub

    async def run() -> None:
        calls = []

        def transport(*, proxy=None, **kwargs):
            def respond(request):
                calls.append(proxy)
                if not proxy:
                    raise failure("direct unavailable", request=request)
                return httpx.Response(200, json={"ok": True})
            return httpx.MockTransport(respond)

        monkeypatch.setattr(hub.httpx, "AsyncHTTPTransport", transport)
        async with httpx.AsyncClient(transport=hub._DirectFirstTransport("http://proxy.test:8080", True)) as client:
            assert (await client.get("https://huggingface.co/api/models/owner/model")).status_code == 200
            assert (await client.get("https://huggingface.co/api/models/owner/other")).status_code == 200
        assert calls == [None, "http://proxy.test:8080", "http://proxy.test:8080"]

    asyncio.run(run())


def test_failed_proxy_fallback_is_not_retried(monkeypatch) -> None:
    from mfq.server.services import hub

    async def run() -> None:
        calls = []

        def transport(*, proxy=None, **kwargs):
            def respond(request):
                calls.append(proxy)
                raise httpx.ConnectError("unavailable", request=request)
            return httpx.MockTransport(respond)

        monkeypatch.setattr(hub.httpx, "AsyncHTTPTransport", transport)
        async with httpx.AsyncClient(transport=hub._DirectFirstTransport("http://proxy.test:8080", True)) as client:
            with pytest.raises(httpx.ConnectError):
                await client.get("https://huggingface.co/api/models/owner/model")
        assert calls == [None, "http://proxy.test:8080"]

    asyncio.run(run())


def test_concurrent_direct_failures_share_one_proxy_pool(monkeypatch) -> None:
    from mfq.server.services import hub

    async def run() -> None:
        created = []
        proxy_calls = []
        both_started = asyncio.Event()

        def transport(*, proxy=None, **kwargs):
            created.append(proxy)

            async def respond(request):
                if not proxy:
                    proxy_calls.append(request)
                    if len(proxy_calls) == 2:
                        both_started.set()
                    await both_started.wait()
                    raise httpx.ConnectError("unavailable", request=request)
                return httpx.Response(200, json={"ok": True})
            return httpx.MockTransport(respond)

        monkeypatch.setattr(hub.httpx, "AsyncHTTPTransport", transport)
        async with httpx.AsyncClient(transport=hub._DirectFirstTransport("http://proxy.test:8080", True)) as client:
            responses = await asyncio.gather(*(client.get(f"https://modelscope.cn/{index}") for index in range(2)))
        assert all(response.status_code == 200 for response in responses)
        assert created == [None, "http://proxy.test:8080"]

    asyncio.run(run())


@pytest.mark.parametrize("status", (200, 401, 404, 503))
def test_direct_http_response_does_not_trigger_proxy_retry(monkeypatch, status) -> None:
    from mfq.server.services import hub

    async def run() -> None:
        created = []

        def transport(*, proxy=None, **kwargs):
            created.append(proxy)
            return httpx.MockTransport(lambda request: httpx.Response(status))

        monkeypatch.setattr(hub.httpx, "AsyncHTTPTransport", transport)
        async with httpx.AsyncClient(transport=hub._DirectFirstTransport("http://proxy.test:8080", True)) as client:
            assert (await client.get("https://huggingface.co/api/models/owner/model")).status_code == status
        assert created == [None]

    asyncio.run(run())


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
            recommendation="three_stars",
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


def test_streamable_official_model_uses_each_tiers_full_residency_size() -> None:
    gib = 1 << 30
    spec = next(item for item in _OFFICIAL_MODELS if item.id == "deepseek-v4-flash-0731")
    sizes = (78 * gib, 88 * gib, 98 * gib)
    files = []
    for precision, size in zip(("V2-S", "V2-M", "V2-L"), sizes, strict=True):
        stem = f"DeepSeek-V4-Flash-0731-EW-{precision}"
        files.append(
            HubModelFile(
                name=f"{stem}/{stem}-00001-of-00001.mfq",
                byte_size=size,
            )
        )
    source = spec.sources[0]
    catalog = HubCatalog()
    catalog._official_cache[(source.provider, source.repo_id)] = HubModelInfo(
        provider=source.provider,
        repo_id=source.repo_id,
        revision="main",
        total_bytes=sum(sizes),
        files=files,
    )

    model = catalog._official_model(
        spec,
        system_profile(backend="metal", runtime_memory_budget_bytes=96 * gib),
    )
    requirements = [
        variant.configuration.required_memory_bytes for variant in model.variants
    ]
    expected = list(sizes)
    assert requirements == expected
    assert len(set(requirements)) == 3
    assert 48 * gib not in requirements
    assert model.configuration.required_memory_bytes == min(expected)
    assert model.configuration.recommendation == "two_stars"


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
    assert unsupported[0].configuration.recommendation == "not_recommended"
    assert unverified[0].configuration.status == "unknown"
    assert unverified[0].configuration.recommendation == "unknown"


def test_model_recommendation_counts_fitting_precision_tiers() -> None:
    def variant(label: str, required: int) -> HubModelVariant:
        return HubModelVariant(
            id=f"mfq:{label}",
            label=label,
            format="mfq",
            files=[f"{label}.mfq"],
            byte_size=required,
            configuration=ModelConfigurationStatus(
                status="recommended",
                recommendation="three_stars",
                required_memory_bytes=required,
                available_memory_bytes=required,
            ),
        )

    variants = [
        variant("small", 100),
        variant("medium", 200),
        variant("large", 300),
        variant("largest", 400),
    ]
    cases = (
        (400, "three_stars"),
        (300, "two_stars"),
        (200, "one_star"),
        (100, "one_star"),
        (70, "caution"),
        (69, "not_recommended"),
    )
    for capacity, expected in cases:
        profile = system_profile(
            backend="metal", runtime_memory_budget_bytes=capacity
        )
        configuration = _model_configuration_status(
            variants,
            profile,
        )
        assert configuration.recommendation == expected
        assert configuration.available_memory_bytes == capacity
        assert configuration.required_memory_bytes == 100


def test_model_recommendation_ignores_non_loadable_repository_artifacts() -> None:
    profile = system_profile(backend="metal", runtime_memory_budget_bytes=100)
    loadable = HubModelVariant(
        id="mfq:model",
        label="model",
        format="mfq",
        files=["model.mfq"],
        byte_size=100,
        configuration=ModelConfigurationStatus(
            status="recommended",
            recommendation="three_stars",
            required_memory_bytes=100,
            available_memory_bytes=100,
        ),
    )
    gguf = HubModelVariant(
        id="model.gguf",
        label="model",
        format="gguf",
        files=["model.gguf"],
        byte_size=200,
        configuration=ModelConfigurationStatus(
            status="warning",
            recommendation="not_recommended",
            required_memory_bytes=200,
            available_memory_bytes=100,
        ),
    )
    configuration = _model_configuration_status(
        [loadable, gguf],
        profile,
    )
    assert configuration.recommendation == "three_stars"
