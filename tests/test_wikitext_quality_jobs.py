import asyncio
import hashlib
import json
import sys

import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from dataclasses import replace
from mfq.server.services import evaluation_datasets, tool_jobs

from mfq.server.protocol.models import CreateDatasetRequest
from mfq.server.services.jobs import JobContext, JobExecutionError
from mfq.server.services.tool_jobs import ToolJobHandlers, ToolJobPaths
from mfq.server.state.catalog import ModelCatalog
from mfq.server.state.storage import SessionStore
from tests.test_server_tool_jobs import _model, _executable


@pytest.mark.parametrize("case", ["ok", "dataset_changed", "reference_mismatch", "geometry_mismatch", "missing_metrics"])
def test_wt2_reference_contract_and_top1(tmp_path, case, monkeypatch):
    async def run():
        root = tmp_path / "models"
        root.mkdir(); _model(root / "tiny.mfq")
        corpus = tmp_path / "wt2.parquet"
        pq.write_table(pa.Table.from_pylist([{"text": "WT2 test corpus\n"}]), corpus)
        text_digest = hashlib.sha256(b"WT2 test corpus\n").hexdigest()
        monkeypatch.setattr(evaluation_datasets, "WT2_TEXT_SHA256", text_digest)
        monkeypatch.setattr(tool_jobs, "WT2_TEXT_SHA256", text_digest)
        (tmp_path / "ref.logits").write_bytes(b"reference")
        digest = hashlib.sha256(corpus.read_bytes()).hexdigest()
        monkeypatch.setitem(evaluation_datasets.OFFICIAL_DATASETS, "wt2-raw-test", replace(evaluation_datasets.OFFICIAL_DATASETS["wt2-raw-test"], sha256=digest, byte_size=corpus.stat().st_size, rows=1))
        (tmp_path / "ref.json").write_text(json.dumps({"dataset": {"sha256": "0" * 64 if case == "reference_mismatch" else text_digest},
            "evaluation": {"n_ctx": 64 if case == "geometry_mismatch" else 512, "n_seq": 1}}))
        executable = _executable(tmp_path / "quality", "#!/usr/bin/env python3\n")
        handlers = ToolJobHandlers(ModelCatalog([root]), ToolJobPaths(tmp_path, __import__('pathlib').Path(sys.executable), None, None, None, executable))
        store = SessionStore(tmp_path / "db.sqlite3")
        dataset = store.create_dataset(CreateDatasetRequest(name="WT2", kind="wikitext2", artifact_uri="workspace://wt2.parquet"), sha256=digest, byte_size=corpus.stat().st_size)
        if case == "dataset_changed": corpus.write_text("changed")
        job = store.create_job("evaluate.wikitext2", {})
        context = JobContext(store, job.id, asyncio.Event())
        async def invoke(context, argv, **kwargs):
            assert "--kl-base" in argv and "--file" not in argv
            return [] if case == "missing_metrics" else ["cpp_kl_result kld=0.01 same_top=0.98 scored_tokens=100"]
        handlers._run = invoke
        payload = {"model": "tiny", "dataset_id": str(dataset.id), "reference_logits": "ref.logits", "reference_manifest": "ref.json"}
        if case != "ok":
            with pytest.raises(JobExecutionError): await handlers.wikitext_quality(context, payload)
            assert not store.list_evaluations()
        else:
            result = await handlers.wikitext_quality(context, payload)
            assert result["top1_agreement"] == .98 and result["kld"] == .01
            assert store.list_evaluations()[0].dataset_manifest["sha256"] == text_digest
            assert store.list_evaluations()[0].dataset_manifest["source_sha256"] == digest
    asyncio.run(run())
