from __future__ import annotations

import asyncio
import hashlib
import os
import ssl
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse
from urllib.request import proxy_bypass_environment

import httpx
from pydantic import BaseModel, ConfigDict

from mfq.server.protocol.models import CreateDatasetRequest, DatasetResource, ErrorDetail
from mfq.server.services.jobs import JobContext, JobExecutionError


@dataclass(frozen=True)
class OfficialDataset:
    id: str
    name: str
    kind: str
    repository: str
    revision: str
    filename: str
    sha256: str
    byte_size: int
    rows: int
    license: str
    format: str = "parquet"
    split: str = "test"
    task: str | None = None
    origin: str = "huggingface"

    @property
    def url(self) -> str:
        if self.origin == "github":
            return f"https://raw.githubusercontent.com/{self.repository}/{self.revision}/{self.filename}"
        return f"https://huggingface.co/datasets/{self.repository}/resolve/{self.revision}/{self.filename}"

    def metadata(self) -> dict[str, Any]:
        return {"official_id": self.id, "revision": self.revision, "source_sha256": self.sha256,
            "format": self.format, "split": self.split, "rows": self.rows, "license": self.license,
            "task": self.task, "origin": self.origin}


OFFICIAL_DATASETS = {
    "wt2-raw-test": OfficialDataset("wt2-raw-test", "WikiText-2 raw · test", "wikitext2", "Salesforce/wikitext",
        "b08601e04326c79dfdd32d625aee71d232d685c3", "wikitext-2-raw-v1/test-00000-of-00001.parquet",
        "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91", 732610, 4358, "CC BY-SA 3.0 / GFDL"),
    "mmlu-pro-test": OfficialDataset("mmlu-pro-test", "MMLU-Pro · test", "custom", "TIGER-Lab/MMLU-Pro",
        "24ac2da5bb7c7b42ea1a984c6b535e35a73d30b3", "data/test-00000-of-00001.parquet",
        "0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8", 4144185, 12032, "MIT", task="MMLU-Pro"),
    "mmlu-test": OfficialDataset("mmlu-test", "MMLU · test", "custom", "cais/mmlu",
        "c30699e8356da336a370243923dbaf21066bb9fe", "all/test-00000-of-00001.parquet",
        "74a41822ce7d3def56e1682f958469c04642a5336a5ce912fa375fdb90fb25d7", 3504718, 14042, "MIT", task="MMLU"),
    "livecodebench-v6": OfficialDataset("livecodebench-v6", "LiveCodeBench · v6 (2025-01–04)", "custom", "livecodebench/code_generation_lite",
        "0fe84c3912ea0c4d4a78037083943e8f0c4dd505", "test6.jsonl",
        "bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5", 134303240, 175, "MIT (dataset script) / original problem licenses",
        format="jsonl", task="LiveCodeBench"),
    "aime-2025": OfficialDataset("aime-2025", "AIME 2025 · I + II", "custom", "MathArena/aime_2025",
        "c94da77eb22bbd6439e62a323bec18493a421302", "data/train-00000-of-00001.parquet",
        "9f9066ff48ad2e31f9bf1b1ac6d5e80693195f987985f2859f89dd25ffa51c2d", 14313, 30, "CC BY-NC-SA 4.0",
        split="train", task="AIME2025"),
    "gpqa-diamond": OfficialDataset("gpqa-diamond", "GPQA-Diamond · author archive", "custom", "idavidrein/gpqa",
        "56686c06f5e19865c153de0fdb11be3890014df7", "dataset.zip",
        "461ae7329f15a3e35f8184d2dac24b990f34fdf12f366ca4062d8e6638cd08dc", 2348038, 198, "MIT (author repository)",
        format="gpqa_zip", task="GPQA-Diamond", origin="github"),
    "truthfulqa-mc1": OfficialDataset("truthfulqa-mc1", "TruthfulQA · MC1 / MC2", "custom", "truthfulqa/truthful_qa",
        "741b8276f2d1982aa3d5b832d3ee81ed3b896490", "multiple_choice/validation-00000-of-00001.parquet",
        "23f08e230ca4ed66babf3a72419af7cbde1f3d734dd396ac4cf6d088bd162afd", 271033, 817, "Apache-2.0",
        split="validation", task="TruthfulQA"),
}
WT2_TEXT_SHA256 = "7952b062817620ac99b47296dc945942b55068d0e85baf1f95f300c7ad1480ef"


class OfficialDatasetDownloadPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: Literal["wt2-raw-test", "mmlu-pro-test", "mmlu-test", "livecodebench-v6", "aime-2025", "gpqa-diamond", "truthfulqa-mc1"]


def dataset_error(code: str, message: str) -> JobExecutionError:
    return JobExecutionError(ErrorDetail(code=code, message=message))


def official_for_digest(kind: str, sha256: str, byte_size: int) -> OfficialDataset:
    for spec in OFFICIAL_DATASETS.values():
        if (kind, sha256, byte_size) == (spec.kind, spec.sha256, spec.byte_size):
            return spec
    raise dataset_error("unofficial_dataset", "the complete file must match an official dataset's pinned SHA256 and size; custom datasets are not accepted")


def read_official_file(path: Path, dataset: DatasetResource) -> tuple[OfficialDataset, bytes]:
    spec = official_for_digest(dataset.kind, dataset.sha256, dataset.byte_size)
    with path.open("rb") as file:
        raw = file.read(spec.byte_size + 1)
    if len(raw) != spec.byte_size or hashlib.sha256(raw).hexdigest() != spec.sha256:
        raise dataset_error("dataset_changed", "the complete dataset file no longer matches the official SHA256; download it again")
    return spec, raw


def parquet_rows(raw: bytes) -> list[dict[str, Any]]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise dataset_error("dataset_reader_unavailable", "official dataset evaluation requires pyarrow") from error
    return pq.read_table(pa.BufferReader(raw)).to_pylist()


def materialize_wt2(path: Path, dataset: DatasetResource) -> Path:
    spec, raw = read_official_file(path, dataset)
    if spec.id != "wt2-raw-test":
        raise dataset_error("wrong_dataset_kind", "WT2 quality requires the official WikiText-2 raw test split")
    rows = parquet_rows(raw)
    if len(rows) != spec.rows:
        raise dataset_error("invalid_official_dataset", "official WT2 row count is inconsistent")
    text = "".join(row["text"] or "\n" for row in rows).encode("utf-8")
    if hashlib.sha256(text).hexdigest() != WT2_TEXT_SHA256:
        raise dataset_error("wt2_conversion_mismatch", "WT2 text conversion differs from the fixed evaluation protocol")
    target = path.with_suffix(".mfq-wt2-v1.txt")
    if not target.exists() or target.read_bytes() != text:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".wt2-", delete=False) as file:
            temporary = Path(file.name)
            file.write(text)
        try:
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    return target


async def download_official_dataset(context: JobContext, payload: dict[str, Any], root: Path) -> dict[str, Any]:
    from mfq.server.api.network import system_proxy_environment
    request = OfficialDatasetDownloadPayload.model_validate(payload)
    spec = OFFICIAL_DATASETS[request.dataset]
    root = root.resolve()
    filename = {"parquet": "test.parquet", "jsonl": "test.jsonl", "gpqa_zip": "test.zip"}[spec.format]
    target = (root / "datasets" / "official" / spec.id / spec.revision / filename).resolve()
    if not target.is_relative_to(root):
        raise dataset_error("path_outside_workspace", "official dataset destination is outside the workspace")
    target.parent.mkdir(parents=True, exist_ok=True)
    valid = target.is_file() and target.stat().st_size == spec.byte_size and await asyncio.to_thread(lambda: hashlib.sha256(target.read_bytes()).hexdigest() == spec.sha256)
    if not valid:
        environment = system_proxy_environment()
        proxy = environment.get("https_proxy") or environment.get("HTTPS_PROXY") or environment.get("all_proxy") or environment.get("ALL_PROXY")
        if proxy_bypass_environment(urlparse(spec.url).hostname or "", {"no": environment.get("NO_PROXY") or environment.get("no_proxy") or ""}):
            proxy = None
        verify = ssl.create_default_context(cafile=environment.get("SSL_CERT_FILE"), capath=environment.get("SSL_CERT_DIR"))
        for selected_proxy in ([None, proxy] if proxy else [None]):
            context.raise_if_cancelled()
            temporary = None
            try:
                async with httpx.AsyncClient(trust_env=False, proxy=selected_proxy, verify=verify, follow_redirects=True, timeout=httpx.Timeout(5 if selected_proxy is None else 30, connect=5)) as client, client.stream("GET", spec.url) as response:
                    response.raise_for_status()
                    digest, received = hashlib.sha256(), 0
                    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".download-", delete=False) as file:
                        temporary = Path(file.name)
                        async for chunk in response.aiter_bytes(65536):
                            context.raise_if_cancelled()
                            received += len(chunk)
                            if received > spec.byte_size:
                                raise dataset_error("official_hash_mismatch", "download exceeds the official file size")
                            digest.update(chunk)
                            file.write(chunk)
                            await context.progress(received / spec.byte_size * .95, message=f"{spec.name}: {received} / {spec.byte_size}")
                    if received != spec.byte_size or digest.hexdigest() != spec.sha256:
                        raise dataset_error("official_hash_mismatch", "download does not match the complete official file SHA256")
                context.raise_if_cancelled()
                os.replace(temporary, target)
                break
            except httpx.HTTPError as error:
                if selected_proxy is not None or not proxy or not isinstance(error, httpx.TransportError):
                    raise dataset_error("dataset_download_failed", f"official dataset download failed: {error}") from error
                context.raise_if_cancelled()
                await context.progress(0, message=f"{spec.name}: retrying with the configured proxy")
            finally:
                if temporary:
                    temporary.unlink(missing_ok=True)
    uri = f"workspace://{target.relative_to(root).as_posix()}"
    from mfq.server.services.official_benchmarks import install_official_scoring
    await install_official_scoring(context, root, spec.id)
    existing = await asyncio.to_thread(context.store.list_datasets)
    dataset = next((item for item in existing if item.artifact_uri == uri and item.sha256 == spec.sha256), None)
    if dataset is None:
        dataset = await asyncio.to_thread(context.store.create_dataset,
            CreateDatasetRequest(name=spec.name, kind=spec.kind, artifact_uri=uri, source_uri=spec.url,
                revision=spec.revision, metadata=spec.metadata()), sha256=spec.sha256, byte_size=spec.byte_size)
    return {"dataset_id": str(dataset.id), "official_id": spec.id, "sha256": spec.sha256, "byte_size": spec.byte_size, "verified": True}
