from __future__ import annotations

import asyncio
import hashlib
import stat
import textwrap
from pathlib import Path

import numpy as np

from mfq.formats.header import FileHeader
from mfq.formats.io import save
from mfq.quantize.imatrix import ImportanceEntry, save_importance_matrix
from mfq.server.protocol.models import CreateJobRequest, JobStatus
from mfq.server.services.jobs import JobManager
from mfq.server.services.tool_jobs import (
    ImatrixCalibrationPayload,
    QuantizePayload,
    ToolJobHandlers,
    ToolJobPaths,
)
from mfq.server.state.catalog import ModelCatalog
from mfq.server.state.storage import SessionStore


def test_imatrix_jobs_default_to_compact_activation_aware_objective() -> None:
    assert ImatrixCalibrationPayload(
        model="model", corpus="corpus", output="output.imatrix"
    ).objective == "naq"
    payload = QuantizePayload(input="model", output="output.mfq")
    assert payload.imatrix_objective == "naq"


def test_standalone_cli_jobs_reinvoke_the_unified_mfq_binary(tmp_path: Path) -> None:
    catalog = ModelCatalog([tmp_path])
    executable = tmp_path / "mfq"
    handlers = ToolJobHandlers(
        catalog,
        ToolJobPaths(
            tmp_path,
            executable,
            None,
            None,
            None,
            None,
            standalone_cli=True,
        ),
    )

    assert handlers._mfq_command("quantize", "input", "output") == [
        str(executable),
        "quantize",
        "input",
        "output",
    ]


def _executable(path: Path, source: str) -> Path:
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _model(path: Path) -> None:
    save(
        path,
        FileHeader(version=2, model_arch="test"),
        {"weight": np.ones((2, 2), dtype=np.float16)},
    )


def test_tool_process_retains_only_a_bounded_output_tail(
    tmp_path: Path, monkeypatch
) -> None:
    import mfq.server.services.tool_jobs as tool_jobs

    class Context:
        def __init__(self) -> None:
            self.cleanup = []
            self.logs: list[str] = []

        def add_cleanup(self, callback) -> None:
            self.cleanup.append(callback)

        def raise_if_cancelled(self) -> None:
            return None

        async def log(self, message: str) -> None:
            self.logs.append(message)

        async def progress(self, _value: float, *, message: str) -> None:
            return None

    async def run() -> None:
        executable = _executable(
            tmp_path / "verbose-tool",
            """\
            #!/usr/bin/env python3
            for index in range(6):
                print(f'line-{index}')
            """,
        )
        handlers = ToolJobHandlers(
            ModelCatalog([]),
            ToolJobPaths(tmp_path, executable, None, None, None, None),
        )
        context = Context()
        lines = await handlers._run(context, [str(executable)])
        assert lines == ["line-3", "line-4", "line-5"]
        assert context.logs == [f"line-{index}" for index in range(6)]
        for callback in context.cleanup:
            await callback()

    monkeypatch.setattr(tool_jobs, "_TOOL_OUTPUT_TAIL_LINES", 3)
    asyncio.run(run())


async def _wait(store: SessionStore, job_id) -> object:
    for _ in range(300):
        job = store.get_job(job_id)
        if job.status in {
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        }:
            return job
        await asyncio.sleep(0.01)
    return store.get_job(job_id)


def test_container_validation_and_workspace_path_boundary(tmp_path: Path) -> None:
    async def run() -> None:
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        _model(model_dir / "tiny.mfq")
        runtime = _executable(
            tmp_path / "runtime",
            """\
            #!/usr/bin/env python3
            import sys
            print('MFQ container OK version=2 architecture=test shards=1 records=1')
            """,
        )
        diagnostics = _executable(
            tmp_path / "mfq-diagnostics",
            """\
            #!/usr/bin/env python3
            import sys
            assert '--check-mfq-container' in sys.argv
            print('MFQ container OK via diagnostics')
            """,
        )
        python = _executable(
            tmp_path / "python",
            """\
            #!/usr/bin/env python3
            import pathlib, sys
            assert '--staged-blobs' in sys.argv
            output = pathlib.Path(sys.argv[5])
            output.write_bytes(b'MFQ1')
            print('100% done')
            """,
        )
        catalog = ModelCatalog([model_dir], cache_seconds=0)
        handlers = ToolJobHandlers(
            catalog,
            ToolJobPaths(
                work_root=tmp_path,
                python=python,
                modelscope=None,
                huggingface=None,
                runtime=runtime,
                perplexity=None,
                diagnostics=diagnostics,
            ),
        )
        store = SessionStore(tmp_path / "jobs.sqlite3")
        manager = JobManager(store, handlers.handlers())
        kinds = {item.kind: item for item in manager.kinds().data}
        assert kinds["model.quantize"].payload_schema["additionalProperties"] is False
        assert "input" in kinds["model.quantize"].payload_schema["properties"]
        assert "staged_blobs" in kinds["model.quantize"].payload_schema["properties"]
        artifact = (await catalog.list()).data[0]

        checked = await manager.submit(
            CreateJobRequest(kind="model.validate", payload={"model": artifact.name})
        )
        result = await _wait(store, checked.id)
        assert result.status == JobStatus.SUCCEEDED
        assert result.result["summary"] == "MFQ container OK via diagnostics"

        source = tmp_path / "source"
        source.mkdir()
        quantized = await manager.submit(
            CreateJobRequest(
                kind="model.quantize",
                payload={
                    "input": "source",
                    "output": "output/model.mfq",
                    "staged_blobs": True,
                },
            )
        )
        result = await _wait(store, quantized.id)
        assert result.status == JobStatus.SUCCEEDED
        assert result.result["artifact"] == "workspace://output/model.mfq"
        lineage = store.list_artifact_lineage(artifact_uri="workspace://output/model.mfq")[0]
        assert lineage.producer_kind == "model.quantize"
        assert lineage.parameters["source_format"] == "auto"
        assert lineage.parameters["staged_blobs"] is True
        assert lineage.source_uris == ["workspace://source"]

        rejected = await manager.submit(
            CreateJobRequest(
                kind="model.quantize",
                payload={"input": str(source), "output": "output/other.mfq"},
            )
        )
        result = await _wait(store, rejected.id)
        assert result.status == JobStatus.FAILED
        assert result.error.code == "absolute_path_not_allowed"
        await manager.close()

    asyncio.run(run())



def test_imatrix_calibration_job_publishes_native_artifact(tmp_path: Path) -> None:
    async def run() -> None:
        (tmp_path / "hf-model").mkdir()
        (tmp_path / "corpus").mkdir()
        python = _executable(
            tmp_path / "python",
            """\
            #!/usr/bin/env python3
            import json, pathlib, sys
            assert sys.argv[1:5] == ['-m', 'mfq.cli', 'calibrate', 'imatrix']
            assert sys.argv[sys.argv.index('--backend') + 1] == 'metal'
            assert sys.argv[sys.argv.index('--device') + 1] == 'mps'
            output = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])
            output.write_bytes(b'imatrix')
            print(json.dumps({'event': 'imatrix_layer', 'layer': 0, 'layers': 2}))
            print(json.dumps({'event': 'imatrix_saved', 'entries': 9, 'tokens': 128}))
            """,
        )
        handlers = ToolJobHandlers(
            ModelCatalog([]),
            ToolJobPaths(tmp_path, python, None, None, None, None),
        )
        store = SessionStore(tmp_path / "jobs.sqlite3")
        manager = JobManager(store, handlers.handlers())
        kinds = {item.kind for item in manager.kinds().data}
        assert {"calibrate.imatrix", "artifact.import", "model.quantize"} <= kinds
        submitted = await manager.submit(
            CreateJobRequest(
                kind="calibrate.imatrix",
                payload={
                    "model": "hf-model",
                    "corpus": "corpus",
                    "output": "artifacts/model.imatrix",
                    "backend": "metal",
                    "train_tokens": 128,
                },
            )
        )
        result = await _wait(store, submitted.id)
        assert result.status == JobStatus.SUCCEEDED
        assert result.result["artifact"] == "workspace://artifacts/model.imatrix"
        assert result.result["entries"] == 9
        lineage = store.list_artifact_lineage()[0]
        assert lineage.producer_kind == "calibrate.imatrix"
        assert lineage.source_uris == ["workspace://hf-model", "workspace://corpus"]
        await manager.close()

    asyncio.run(run())


def test_quantize_can_collect_and_consume_imatrix_in_one_job(tmp_path: Path) -> None:
    async def run() -> None:
        (tmp_path / "hf-model").mkdir()
        (tmp_path / "corpus").mkdir()
        python = _executable(
            tmp_path / "python",
            """\
            #!/usr/bin/env python3
            import json, pathlib, sys
            if sys.argv[1:5] == ['-m', 'mfq.cli', 'calibrate', 'imatrix']:
                output = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])
                output.write_bytes(b'imatrix')
                print(json.dumps({'event': 'imatrix_saved', 'entries': 5, 'tokens': 64}))
            else:
                assert sys.argv[1:4] == ['-m', 'mfq.cli', 'quantize']
                imatrix = pathlib.Path(sys.argv[sys.argv.index('--imatrix') + 1])
                assert imatrix.read_bytes() == b'imatrix'
                pathlib.Path(sys.argv[5]).write_bytes(b'MFQ1')
                print('100% done')
            """,
        )
        handlers = ToolJobHandlers(
            ModelCatalog([]),
            ToolJobPaths(tmp_path, python, None, None, None, None),
        )
        store = SessionStore(tmp_path / "jobs.sqlite3")
        manager = JobManager(store, handlers.handlers())
        submitted = await manager.submit(
            CreateJobRequest(
                kind="model.quantize",
                payload={
                    "input": "hf-model",
                    "output": "outputs/model.mfq",
                    "calibrate_imatrix": True,
                    "imatrix_corpus": "corpus",
                    "imatrix_output": "artifacts/model.imatrix",
                    "imatrix_backend": "metal",
                    "imatrix_train_tokens": 64,
                },
            )
        )
        result = await _wait(store, submitted.id)
        assert result.status == JobStatus.SUCCEEDED
        assert result.result["artifact"] == "workspace://outputs/model.mfq"
        assert result.result["imatrix"] == "workspace://artifacts/model.imatrix"
        assert result.result["imatrix_calibration"]["entries"] == 5
        lineage = store.list_artifact_lineage()
        assert {item.artifact_uri for item in lineage} == {
            "workspace://artifacts/model.imatrix",
            "workspace://outputs/model.mfq",
        }
        await manager.close()

    asyncio.run(run())


def test_uploaded_imatrix_is_validated_and_imported(tmp_path: Path) -> None:
    async def run() -> None:
        source = tmp_path / "source.imatrix"
        save_importance_matrix(
            source,
            {
                "blk.0.attn_q.weight": ImportanceEntry(
                    np.ones((1, 2), dtype=np.float32),
                    np.asarray([2], dtype=np.int64),
                )
            },
        )
        store = SessionStore(tmp_path / "jobs.sqlite3")
        data = source.read_bytes()
        media = store.put_media(
            data,
            "application/x-mfq-imatrix",
            hashlib.sha256(data).hexdigest(),
        )
        handlers = ToolJobHandlers(
            ModelCatalog([]),
            ToolJobPaths(tmp_path, Path(__import__("sys").executable), None, None, None, None),
        )
        manager = JobManager(store, handlers.handlers())
        submitted = await manager.submit(
            CreateJobRequest(
                kind="artifact.import",
                payload={
                    "media_id": str(media.media.id),
                    "destination": "artifacts/imported.imatrix",
                },
            )
        )
        result = await _wait(store, submitted.id)
        assert result.status == JobStatus.SUCCEEDED
        assert result.result["artifact"] == "workspace://artifacts/imported.imatrix"
        assert result.result["entries"] == 1
        assert (tmp_path / "artifacts" / "imported.imatrix").read_bytes() == data
        await manager.close()

    asyncio.run(run())


def test_download_argv_does_not_execute_shell_metacharacters(tmp_path: Path) -> None:
    async def run() -> None:
        downloader = _executable(
            tmp_path / "modelscope",
            """\
            #!/usr/bin/env python3
            import pathlib, sys
            root = pathlib.Path(sys.argv[sys.argv.index('--local-dir') + 1])
            root.mkdir(parents=True, exist_ok=True)
            (root / 'weight.mfq').write_bytes(b'MFQ1')
            print('downloaded')
            """,
        )
        handlers = ToolJobHandlers(
            ModelCatalog([]),
            ToolJobPaths(
                work_root=tmp_path,
                python=Path(__import__("sys").executable),
                modelscope=downloader,
                huggingface=None,
                runtime=None,
                perplexity=None,
            ),
        )
        store = SessionStore(tmp_path / "jobs.sqlite3")
        manager = JobManager(store, handlers.handlers())
        submitted = await manager.submit(
            CreateJobRequest(
                kind="download.modelscope",
                payload={
                    "repo_id": "owner/model",
                    "destination": "downloads/value;touch-owned",
                },
            )
        )
        result = await _wait(store, submitted.id)
        assert result.status == JobStatus.SUCCEEDED
        assert not (tmp_path / "owned").exists()
        assert result.result["files"] == 1
        lineage = store.list_artifact_lineage()[0]
        assert lineage.source_uris == ["modelscope://owner/model@master"]
        removed = handlers.remove_workspace_artifact(result.result["artifact"])
        assert removed == {"files": 1, "total_bytes": 4}
        assert not (tmp_path / "downloads" / "value;touch-owned").exists()
        await manager.close()

    asyncio.run(run())


def test_packaged_download_reinvokes_the_unified_cli(tmp_path: Path) -> None:
    async def run() -> None:
        cli = _executable(
            tmp_path / "mfq-cli",
            """\
            #!/usr/bin/env python3
            import pathlib, sys
            assert sys.argv[1:6] == [
                '_hub-download', '--provider', 'modelscope', '--repo-id', 'owner/model'
            ]
            root = pathlib.Path(sys.argv[sys.argv.index('--local-dir') + 1])
            root.mkdir(parents=True, exist_ok=True)
            (root / 'config.json').write_text('{}')
            """,
        )
        model_root = tmp_path / "managed" / "models"
        (tmp_path / "work").mkdir()
        handlers = ToolJobHandlers(
            ModelCatalog([model_root]),
            ToolJobPaths(
                work_root=tmp_path / "work",
                python=cli,
                modelscope=None,
                huggingface=None,
                runtime=None,
                perplexity=None,
                standalone_cli=True,
                internal_modelscope=True,
            ),
        )
        store = SessionStore(tmp_path / "jobs.sqlite3")
        manager = JobManager(store, handlers.handlers())
        assert "download.modelscope" in {item.kind for item in manager.kinds().data}
        submitted = await manager.submit(
            CreateJobRequest(
                kind="download.modelscope",
                payload={"repo_id": "owner/model", "destination": "models/owner/model"},
            )
        )
        result = await _wait(store, submitted.id)
        assert result.status == JobStatus.SUCCEEDED
        assert result.result["files"] == 1
        assert result.result["artifact"] == "workspace://models/owner/model"
        assert (model_root / "owner" / "model" / "config.json").is_file()
        await manager.close()

    asyncio.run(run())


def test_perplexity_job_parses_result_and_publishes_logits(tmp_path: Path) -> None:
    async def run() -> None:
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        _model(model_dir / "tiny.mfq")
        (tmp_path / "dataset.txt").write_text("evaluation text", encoding="utf-8")
        perplexity = _executable(
            tmp_path / "perplexity",
            """\
            #!/usr/bin/env python3
            import pathlib, sys
            if '--logits-file' in sys.argv:
                path = pathlib.Path(sys.argv[sys.argv.index('--logits-file') + 1])
                path.write_bytes(b'logits')
            print('[1]12.3000,')
            print('Final estimate: PPL = 12.3000 +/- 0.12000 (255 scored tokens)')
            """,
        )
        catalog = ModelCatalog([model_dir], cache_seconds=0)
        handlers = ToolJobHandlers(
            catalog,
            ToolJobPaths(
                work_root=tmp_path,
                python=Path(__import__("sys").executable),
                modelscope=None,
                huggingface=None,
                runtime=None,
                perplexity=perplexity,
            ),
        )
        store = SessionStore(tmp_path / "jobs.sqlite3")
        manager = JobManager(store, handlers.handlers())
        artifact = (await catalog.list()).data[0]
        submitted = await manager.submit(
            CreateJobRequest(
                kind="evaluate.perplexity",
                payload={
                    "model": artifact.name,
                    "dataset_file": "dataset.txt",
                    "dataset": "fixture",
                    "chunks": 1,
                    "logits_file": "results/reference.logits",
                },
            )
        )
        result = await _wait(store, submitted.id)
        assert result.status == JobStatus.SUCCEEDED
        assert result.result["perplexity"] == 12.3
        assert result.result["uncertainty"] == 0.12
        assert result.result["model_id"] == artifact.name
        assert result.result["artifact_id"] == artifact.id
        assert result.result["logits"] == "workspace://results/reference.logits"
        evaluation = store.list_evaluations()[0]
        assert str(evaluation.id) == result.result["evaluation_id"]
        assert evaluation.kind == "perplexity"
        assert evaluation.metrics["perplexity"] == 12.3
        assert evaluation.dataset_manifest["byte_size"] == 15
        assert evaluation.hardware_identity["machine"]
        assert evaluation.runtime_identity["executable"] == "perplexity"
        assert evaluation.model_id == artifact.name
        assert evaluation.runtime_identity["artifact_id"] == artifact.id
        events = store.list_job_events(submitted.id)
        assert any(event.type.value == "artifact" for event in events)
        await manager.close()

    asyncio.run(run())
