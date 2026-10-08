import asyncio
import hashlib
from dataclasses import replace

import httpx
import pytest

from mfq.server.api import create_app
from mfq.server.services.evaluation_datasets import OFFICIAL_DATASETS, download_official_dataset
from mfq.server.services.jobs import JobContext, JobExecutionError
from mfq.server.services.service import ServerService
from mfq.server.state.storage import SessionStore
from tests.test_server_evaluations import WorkspaceTools
from tests.test_server_service import FakeBackend


def official_fixture(monkeypatch, data=b"official complete file"):
    spec = replace(OFFICIAL_DATASETS["wt2-raw-test"], sha256=hashlib.sha256(data).hexdigest(), byte_size=len(data))
    monkeypatch.setitem(OFFICIAL_DATASETS, spec.id, spec)
    return spec


def test_registration_rejects_self_declared_hashes_and_overrides_client_identity(tmp_path, monkeypatch):
    spec = official_fixture(monkeypatch)
    async def run():
        (tmp_path / "unofficial.txt").write_bytes(b"a private test collection")
        (tmp_path / "official.parquet").write_bytes(b"official complete file")
        store = SessionStore(tmp_path / "db.sqlite3")
        service = ServerService(store, FakeBackend(), tool_handlers=WorkspaceTools(tmp_path))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(service)), base_url="http://test") as client:
            body = {"name": "Official WT2", "kind": "wikitext2", "artifact_uri": "workspace://unofficial.txt", "metadata": {"sha256": spec.sha256, "official_id": spec.id}}
            rejected = await client.post("/api/v1/datasets", json=body)
            assert rejected.status_code == 422 and rejected.json()["error"]["code"] == "unofficial_dataset"
            assert not store.list_datasets()
            body.update(artifact_uri="workspace://official.parquet", source_uri="https://wrong.invalid", revision="user-version")
            accepted = await client.post("/api/v1/datasets", json=body)
            assert accepted.status_code == 201
            assert accepted.json()["name"] == spec.name
            assert accepted.json()["revision"] == spec.revision and accepted.json()["source_uri"] == spec.url
            catalog = (await client.get("/api/v1/datasets/catalog")).json()["data"]
            assert catalog[0]["sha256"] == spec.sha256 and len(catalog) == 7
            (tmp_path / "official.parquet").write_bytes(b"unofficial altered file")
            assert not (await client.get("/api/v1/datasets")).json()["data"]
    asyncio.run(run())


@pytest.mark.parametrize("case", ["valid", "same_size_corruption", "truncated", "oversized", "cancelled"])
def test_downloads_verify_whole_files_atomically_and_clean_partial_files(tmp_path, monkeypatch, case):
    data = b"official complete file"
    spec = official_fixture(monkeypatch, data)
    body = data if case in {"valid", "cancelled"} else data[:-1] if case == "truncated" else data + b"extra" if case == "oversized" else b"x" * len(data)
    original_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original_client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=body)), **kwargs))
    monkeypatch.setattr("mfq.server.api.network.system_proxy_environment", lambda: {})
    async def run():
        store = SessionStore(tmp_path / "db.sqlite3")
        job = store.create_job("dataset.download", {})
        store.claim_job(job.id)
        event = asyncio.Event()
        if case == "cancelled":
            event.set()
        context = JobContext(store, job.id, event)
        if case == "valid":
            result = await download_official_dataset(context, {"dataset": spec.id}, tmp_path)
            assert result["verified"] and result["sha256"] == spec.sha256
            second = await download_official_dataset(context, {"dataset": spec.id}, tmp_path)
            assert second["dataset_id"] == result["dataset_id"] and len(store.list_datasets()) == 1
        else:
            with pytest.raises(asyncio.CancelledError if case == "cancelled" else JobExecutionError):
                await download_official_dataset(context, {"dataset": spec.id}, tmp_path)
            assert not store.list_datasets()
            assert not list((tmp_path / "datasets").rglob("test.parquet"))
        assert not list((tmp_path / "datasets").rglob(".download-*"))
    asyncio.run(run())


@pytest.mark.parametrize("case", ["direct_success", "direct_timeout", "partial_timeout", "proxy_timeout", "http_error", "corrupt", "cancelled", "bypass"])
def test_download_falls_back_only_on_direct_transport_failure(tmp_path, monkeypatch, case):
    data = b"official complete file"
    spec = official_fixture(monkeypatch, data)
    proxy = "http://proxy.invalid:8080"
    monkeypatch.setattr("mfq.server.api.network.system_proxy_environment", lambda: {
        "https_proxy": proxy, "NO_PROXY": "huggingface.co" if case == "bypass" else "localhost",
    })
    attempts = []
    event = asyncio.Event()
    original_client = httpx.AsyncClient

    class InterruptedBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield data[:4]
            raise httpx.ReadTimeout("partial download stalled")

    def client(**kwargs):
        selected = kwargs.pop("proxy")
        attempts.append(selected)
        def respond(request):
            if case == "cancelled":
                event.set()
                raise httpx.ReadTimeout("cancelled during direct connection", request=request)
            if case in {"direct_timeout", "proxy_timeout", "bypass"} and (selected is None or case == "proxy_timeout"):
                raise httpx.ReadTimeout("actual GET timed out", request=request)
            if case == "partial_timeout" and selected is None:
                return httpx.Response(200, stream=InterruptedBody())
            return httpx.Response(404 if case == "http_error" else 200, content=b"x" * len(data) if case == "corrupt" else data)
        return original_client(transport=httpx.MockTransport(respond), **kwargs)
    monkeypatch.setattr(httpx, "AsyncClient", client)

    async def run():
        store = SessionStore(tmp_path / "db.sqlite3")
        job = store.create_job("dataset.download", {})
        store.claim_job(job.id)
        context = JobContext(store, job.id, event)
        if case in {"direct_success", "direct_timeout", "partial_timeout"}:
            result = await download_official_dataset(context, {"dataset": spec.id}, tmp_path)
            assert result["verified"]
            assert (tmp_path / "datasets" / "official" / spec.id / spec.revision / "test.parquet").read_bytes() == data
        else:
            with pytest.raises(asyncio.CancelledError if case == "cancelled" else JobExecutionError):
                await download_official_dataset(context, {"dataset": spec.id}, tmp_path)
            assert not store.list_datasets()
            assert not list((tmp_path / "datasets").rglob("test.parquet"))
        assert attempts == ([None, proxy] if case in {"direct_timeout", "partial_timeout", "proxy_timeout"} else [None])
        assert not list((tmp_path / "datasets").rglob(".download-*"))
    asyncio.run(run())
