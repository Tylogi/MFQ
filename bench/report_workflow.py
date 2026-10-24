"""Archive a benchmark, validate its samples, and publish an offline report."""

import argparse
import contextlib
import hashlib
import html
import json
import os
import re
import shutil
import sys
import tempfile
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench import render_model_report as report

RUN_ID = re.compile(r"[0-9]{8}T[0-9]{6}Z-[a-f0-9]{12}")


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


def preflight(metadata, protocol):
    """Check report inputs before a caller spends time on GPU work."""
    import matplotlib  # noqa: F401

    report.public_metadata(metadata, dict(protocol))
    template = Path(report.__file__).with_name("templates") / "inference-report.html"
    report.require(template.is_file(), "Report template is missing")


def verify_report(folder):
    folder = Path(folder)
    hashes = report.read_json(folder / "checksums.json")
    required = {
        "bundle.json",
        "index.html",
        "charts.pdf",
        "samples.csv",
        "summary.csv",
        "coverage.csv",
        "theme.json",
        "report.md",
    }
    required |= {
        f"{name}.{extension}" for name, _, _ in report.CHARTS for extension in ("svg", "png")
    }
    report.require(set(hashes) == required, "Report file manifest is incomplete")
    report.require(
        {p.name for p in folder.iterdir()} == required | {"checksums.json"},
        "Unexpected report files",
    )
    for name, expected in hashes.items():
        report.require(
            hashlib.sha256((folder / name).read_bytes()).hexdigest() == expected,
            "Report checksum mismatch",
        )
    page = (folder / "index.html").read_text(encoding="utf-8")
    for target in re.findall(r'(?:href|src)="([^"]+)"', page):
        if target.startswith("#") or target == "../index.html":
            continue
        report.require(target in required, "Report contains an external or missing asset")
    pages = len(re.findall(rb"/Type /Page\b", (folder / "charts.pdf").read_bytes()))
    report.require(pages == len(report.CHARTS), "Unexpected chart PDF page count")


@contextlib.contextmanager
def publication_lock(root):
    lock = root / ".publish-lock"
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise ValueError(
            "Report archive is busy; retry after the other publisher finishes"
        ) from error
    try:
        yield
    finally:
        with contextlib.suppress(OSError):
            lock.rmdir()


def refresh_catalog(public, current, before_commit):
    entries = []
    for folder in sorted(public.iterdir(), reverse=True):
        if not folder.is_dir() or not RUN_ID.fullmatch(folder.name):
            continue
        bundle = report.read_json(folder / "bundle.json")
        report.require(bundle["run"]["id"] == folder.name, "Archive run identity mismatch")
        entries.append((folder.name, bundle))
    rows = []
    for run_id, bundle in entries:
        identity, protocol = bundle["identity"], bundle["protocol"]
        values = (
            bundle["run"]["created_at"],
            identity["model"],
            identity["hardware"],
            ", ".join(map(str, protocol["input_tokens"])),
            protocol["output_tokens"],
            bundle["validation"]["measured"],
        )
        cells = "".join(f"<td>{html.escape(str(value))}</td>" for value in values)
        rows.append(f'<tr>{cells}<td><a href="{run_id}/index.html">開啟報告</a></td></tr>')
    page = """<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MFQ benchmark reports</title><style>
body{max-width:1200px;margin:48px auto;padding:0 24px;color:#122544;background:white;font:16px/1.55 "Songti TC",Georgia,serif}
h1{font-size:36px;font-weight:500}a{color:#2563eb}p{max-width:720px}.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;min-width:760px;font:13px/1.5 system-ui}td,th{padding:14px 10px;text-align:left;border-bottom:1px solid #dce3ed}th{font-weight:500}
</style><h1>Benchmark 報告檔案庫</h1><p>每次成功量測都有獨立報告與資料包。不同模型、硬體或設定的分數需先核對條件，再作比較。</p>"""
    page += f'<p><a href="{current}/index.html">最新成功報告 ↗</a> · <a href="latest.json">最新報告索引 JSON</a></p>'
    page += '<div class="scroll"><table><thead><tr><th>報告時間（UTC）</th><th>模型</th><th>硬體</th><th>Input tokens</th><th>Output tokens</th><th>正式樣本</th><th>報告</th></tr></thead><tbody>'
    page += "".join(rows) + "</tbody></table></div></html>\n"
    latest = {
        "schema": "mfq.benchmark-latest.v1",
        "run_id": current,
        "report": f"{current}/index.html",
    }
    contents = {"index.html": page, "latest.json": json.dumps(latest, indent=2) + "\n"}
    with tempfile.TemporaryDirectory(
        prefix=".catalog-", dir=public, ignore_cleanup_errors=True
    ) as folder:
        staged = Path(folder)
        for name, text in contents.items():
            (staged / name).write_text(text, encoding="utf-8")
            if (public / name).exists():
                shutil.copyfile(public / name, staged / f"{name}.backup")
        replaced = []
        try:
            os.replace(staged / "index.html", public / "index.html")
            replaced.append("index.html")
            before_commit()
            os.replace(staged / "latest.json", public / "latest.json")
        except Exception:
            for name in reversed(replaced):
                backup = staged / f"{name}.backup"
                if backup.exists():
                    os.replace(backup, public / name)
                else:
                    (public / name).unlink(missing_ok=True)
            raise


def archive_report(
    raw,
    metadata,
    root,
    *,
    modes=("decode", "mtp"),
    model_index=0,
    gpu_samples=None,
    gpu_before=None,
    gpu_after=None,
):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    private_root = root / "private"
    private_root.mkdir(mode=0o700, exist_ok=True)
    private_root.chmod(0o700)
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:12]
    private = private_root / run_id
    private.mkdir(mode=0o700)
    status = {"run_id": run_id, "state": "saved", "created_at": now.isoformat(timespec="seconds")}
    try:
        write_json(private / "raw.json", raw)
        write_json(private / "metadata.json", metadata)
        for name, value in (
            ("gpu-samples", gpu_samples),
            ("gpu-before", gpu_before),
            ("gpu-after", gpu_after),
        ):
            if value is not None:
                write_json(private / f"{name}.json", value)
        write_json(private / "status.json", status)
        bundle = report.normalize(raw, metadata, modes, model_index)
        if gpu_samples is not None:
            if isinstance(gpu_before, list):
                gpu_before = gpu_before[0] if gpu_before else None
            bundle["gpu"] = report.gpu_telemetry(gpu_samples, gpu_before, gpu_after)
        for metric in ("GPU use (%)", "VRAM Total Used Memory (B)", "VRAM Total Memory (B)"):
            bundle["coverage"].append(
                {
                    "metric": metric,
                    "chart": "gpu",
                    "state": "recorded" if bundle["gpu"] else "not_measured",
                }
            )
        bundle["run"] = {"id": run_id, "created_at": status["created_at"]}
        theme = report.read_json(Path(report.__file__).with_name("reporting-theme.json"))
        staged = private / "report"
        report.build_report(bundle, staged, theme)
        verify_report(staged)
        public = root / "public"
        public.mkdir(exist_ok=True)
        with publication_lock(root):
            target = public / run_id
            report.require(not target.exists(), "Report run already exists")
            staged.rename(target)

            def complete():
                write_json(private / "status.json", {**status, "state": "complete"})

            refresh_catalog(public, run_id, complete)
        return target
    except Exception:
        status["state"] = "failed"
        write_json(private / "status.json", status)
        (private / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--archive-root", type=Path, default=Path(".mfq/benchmarks"))
    parser.add_argument("--modes", nargs="+", choices=("decode", "mtp"), default=["decode", "mtp"])
    parser.add_argument("--model-index", type=int, default=0)
    for key in ("gpu-samples", "gpu-before", "gpu-after"):
        parser.add_argument(f"--{key}", type=Path)
    args = parser.parse_args()
    try:
        gpu = {
            key: report.read_json(getattr(args, key)) if getattr(args, key) else None
            for key in ("gpu_samples", "gpu_before", "gpu_after")
        }
        output = archive_report(
            report.read_json(args.raw),
            report.read_json(args.metadata) if args.metadata else {},
            args.archive_root,
            modes=args.modes,
            model_index=args.model_index,
            **gpu,
        )
    except (ValueError, OSError, ImportError) as error:
        parser.exit(
            1,
            f"Report failed ({type(error).__name__}); diagnostic details remain in the private archive.\n",
        )
    print(f"Report ready: {output / 'index.html'}")


if __name__ == "__main__":
    main()
