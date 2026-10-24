import copy
import json
from pathlib import Path

import pytest

from bench import render_model_report as report


def fixture():
    runs = []
    for mode in ("decode", "mtp"):
        for index, prefill in enumerate((10000, 10, 20, 40)):
            decode = prefill * 6
            drafted = (0, 10, 20, 30)[index] if mode == "mtp" else 0
            accepted = (0, 1, 10, 27)[index] if mode == "mtp" else 0
            runs.append(
                dict(
                    prompt_tokens_requested=32,
                    prompt_tokens=32,
                    prefill_tokens=32,
                    completion_tokens=64,
                    mode=mode,
                    warmup=index == 0,
                    status="ok",
                    prefill_ms=prefill,
                    decode_ms=decode,
                    ttft_ms=prefill + 2,
                    prefill_tps=32000 / prefill,
                    decode_tps=63000 / decode,
                    generation_tps=64000 / (prefill + decode + 2),
                    mtp_used=mode == "mtp",
                    mtp_drafted_tokens=drafted,
                    mtp_accepted_tokens=accepted,
                )
            )
    return dict(
        prefill_lengths=[32],
        decode_tokens=64,
        repetitions=3,
        warmup_runs=1,
        results=[dict(model="private-marker", error=None, runs=runs)],
        env={"PRIVATE_TOKEN": "secret-not-for-report"},
        summary=[{"decode_tps": 999999}],
    )


def test_recompute_statistics_and_weighted_acceptance():
    bundle = report.normalize(fixture(), {})
    ordinary, mtp = bundle["summary"]
    assert ordinary["stats"]["prefill_ms"]["median"] == 20
    assert ordinary["stats"]["decode_tps"]["median"] == 525
    assert ordinary["pooled"]["decode_tps"] == 450
    assert ordinary["stats"]["prefill_ms"]["q25"] == 15
    assert ordinary["stats"]["prefill_ms"]["q75"] == 30
    assert bundle["samples"][1]["generation_ms_derived"] == pytest.approx(72)
    assert bundle["samples"][1]["mean_decode_interval_ms"] == pytest.approx(60 / 63)
    assert mtp["mtp_acceptance"] == pytest.approx(38 / 60)
    assert bundle["validation"]["measured"] == 6
    assert bundle["validation"]["warmups"] == 2
    assert "secret-not-for-report" not in json.dumps(bundle)
    assert "private-marker" not in json.dumps(bundle)


@pytest.mark.parametrize(
    "field,value",
    [
        ("prefill_ms", float("nan")),
        ("decode_ms", 0),
        ("generation_tps", float("inf")),
        ("generation_tps", 0),
        ("prompt_tokens", 31),
        ("prefill_tokens", 0),
        ("completion_tokens", 63),
        ("decode_tps", 999),
        ("mtp_used", True),
        ("mtp_drafted_tokens", 2),
        ("status", "timeout"),
        ("warmup", 1),
        ("prompt_tokens_requested", True),
    ],
)
def test_reject_invalid_run(field, value):
    raw = fixture()
    raw["results"][0]["runs"][1][field] = value
    with pytest.raises(ValueError):
        report.normalize(raw, {})


@pytest.mark.parametrize("change", ["missing", "duplicate", "missing_mode"])
def test_reject_incomplete_matrix(change):
    raw = fixture()
    runs = raw["results"][0]["runs"]
    if change == "missing":
        runs.pop()
    elif change == "duplicate":
        runs.append(copy.deepcopy(runs[-1]))
    else:
        raw["results"][0]["runs"] = runs[:4]
    with pytest.raises(ValueError, match="Incomplete"):
        report.normalize(raw, {})


def test_optional_generation_not_filled_from_decode():
    raw = fixture()
    for run in raw["results"][0]["runs"]:
        run.pop("generation_tps")
    bundle = report.normalize(raw, {})
    assert all(row["stats"]["generation_tps"] is None for row in bundle["summary"])
    assert all(row["pooled"]["generation_tps"] is None for row in bundle["summary"])
    assert (
        next(row for row in bundle["coverage"] if row["metric"] == "generation_tps")["state"]
        == "missing"
    )


def test_metadata_allowlist_and_conflict(tmp_path):
    with pytest.raises(ValueError, match="Public label"):
        report.normalize(fixture(), {"model": str(tmp_path / "model")})
    with pytest.raises(ValueError, match="conflicts"):
        report.normalize(fixture(), {"protocol": {"output_tokens": 128}})
    with pytest.raises(ValueError, match="protocol object"):
        report.normalize(fixture(), {"protocol": None})
    bundle = report.normalize(fixture(), {"model": "Test model", "host": "private-marker"})
    assert "host" not in bundle["identity"]


def test_gpu_telemetry_keeps_relative_time_and_allowlisted_fields():
    reading = {
        "GPU use (%)": "50",
        "VRAM Total Used Memory (B)": "200",
        "VRAM Total Memory (B)": "1000",
        "device_serial": "private-marker",
    }
    samples = [{"time": 100, "gpu": reading}, {"time": 102, "gpu": reading}]
    result = report.gpu_telemetry(samples, reading, reading)
    assert [row["seconds"] for row in result["samples"]] == [0, 2]
    assert "device_serial" not in json.dumps(result)
    samples[1]["time"] = 100
    with pytest.raises(ValueError, match="increase"):
        report.gpu_telemetry(samples)
    samples[1]["time"] = 102
    with pytest.raises(ValueError, match="capacity"):
        report.gpu_telemetry(samples, {**reading, "VRAM Total Memory (B)": "2000"})


def test_json_rejects_duplicates_and_nonfinite(tmp_path):
    path = tmp_path / "raw.json"
    for text in ('{"x":1,"x":2}', '{"x":NaN}'):
        path.write_text(text)
        with pytest.raises(ValueError):
            report.read_json(path)


@pytest.mark.parametrize("ordinary_only", [False, True])
def test_chart_bundle_links_and_privacy(tmp_path, ordinary_only):
    pytest.importorskip("matplotlib")
    raw = fixture()
    if ordinary_only:
        raw["results"][0]["runs"] = raw["results"][0]["runs"][:4]
        for run in raw["results"][0]["runs"]:
            run.pop("generation_tps")
    modes = ["decode"] if ordinary_only else ["decode", "mtp"]
    bundle = report.normalize(raw, {"model": "Test model", "hardware": "Test GPU"}, modes)
    theme = report.read_json(Path(report.__file__).with_name("reporting-theme.json"))
    output = tmp_path / "report"
    report.build_report(bundle, output, theme)
    assert (output / "charts.pdf").stat().st_size > 1000
    gallery = (output / "index.html").read_text()
    if ordinary_only:
        assert "MTP not measured" in (output / "mtp.svg").read_text()
        assert "Generation timing not recorded" in (output / "warmup.svg").read_text()
    for name, _, _ in report.CHARTS:
        assert (output / f"{name}.png").exists()
        assert (output / f"{name}.svg").exists()
        assert f'src="{name}.svg"' in gallery
    for path in output.iterdir():
        if path.suffix in (".json", ".html", ".md", ".csv", ".svg"):
            text = path.read_text()
            assert "secret-not-for-report" not in text
            assert "private-marker" not in text
    with pytest.raises(ValueError, match="already exists"):
        report.build_report(bundle, output, theme)
