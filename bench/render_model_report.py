"""Render a standalone model benchmark as an offline chart gallery."""

import argparse
import csv
import hashlib
import html
import json
import math
import re
import statistics
import tempfile
from pathlib import Path

METRICS = {
    "prefill_tps": ("Prefill throughput", "tok/s", "higher"),
    "decode_tps": ("Decode throughput", "tok/s", "higher"),
    "generation_tps": ("Whole-request throughput", "tok/s", "higher"),
    "ttft_ms": ("Time to first token", "ms", "lower"),
    "prefill_ms": ("Prefill duration", "ms", "lower"),
    "decode_ms": ("Decode duration", "ms", "lower"),
    "generation_ms_derived": ("Whole-request duration · derived", "ms", "lower"),
    "mean_decode_interval_ms": ("Mean decode interval · derived", "ms/token", "lower"),
}
CHARTS = [
    ("throughput", "吞吐量", "Prefill、decode 與包含整段請求耗時的 generation throughput。"),
    (
        "latency",
        "延遲與耗時",
        "首 token、prefill、decode、整段請求與平均 decode 間隔，分開標示量測與推導值。",
    ),
    ("mtp", "MTP 效率", "草稿接受率、接受／拒絕數、吞吐量倍率，以及首 token 延遲成本。"),
    ("stability", "逐輪穩定性", "每輪相對同條件中位數的偏差；暖機排除，所有正式樣本保留。"),
    (
        "warmup",
        "暖機與執行順序",
        "逐次請求的 TTFT、prefill 與 generation throughput，顯示暖機成本。",
    ),
    ("aggregation", "統計口徑", "逐輪中位數與總 token／總時間的差異，兩種值都保留。"),
    ("gpu", "GPU 與顯存", "整個測試時段的裝置使用率與顯存取樣，無法對齊到個別請求。"),
    ("validation", "負載與資料完整性", "請求與實際 token 數、正式／暖機樣本數，以及模式檢查。"),
]
SOURCE_FIELDS = {
    "prompt_tokens_requested": "validation",
    "prompt_tokens": "validation",
    "prefill_tokens": "validation",
    "completion_tokens": "validation",
    "mode": "all",
    "warmup": "warmup, validation",
    "status": "validation",
    "mtp_used": "validation",
    "mtp_drafted_tokens": "mtp",
    "mtp_accepted_tokens": "mtp",
    "prefill_tps": "throughput, stability, warmup, aggregation",
    "decode_tps": "throughput, stability, aggregation",
    "generation_tps": "throughput, stability, warmup, aggregation",
    "ttft_ms": "latency, stability, warmup",
    "prefill_ms": "latency, stability",
    "decode_ms": "latency, stability",
}
MISSING = {
    "KLD / Top-1 agreement / perplexity": "需要同一模型的品質評測與 reference／dataset 身分。",
    "accuracy / pass@1 / MC1–MC3 / category scores": "需要任務評測結果、題數與 scorer 設定。",
    "Per-token p95 / p99 latency": "沒有逐 token 時間戳；請求層級的樣本不能替代逐 token 尾延遲。",
    "Power / energy / temperature / clocks": "這次 GPU 遙測沒有收集這些欄位。",
    "Concurrent request throughput / kernel timing": "這是單請求模型測試，未量測並行服務或個別核心。",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, name, *, minimum=0, integer=False):
    require(
        type(value) in (int, float) and math.isfinite(value) and value >= minimum, f"Invalid {name}"
    )
    require(not integer or int(value) == value, f"Noninteger {name}")
    return value


def quantile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def stats(values):
    if not values or any(value is None for value in values):
        return None
    return {
        "n": len(values),
        "median": statistics.median(values),
        "q25": quantile(values, 0.25),
        "q75": quantile(values, 0.75),
        "min": min(values),
        "max": max(values),
    }


def safe_label(value):
    require(isinstance(value, str) and 0 < len(value) <= 160, "Invalid public label")
    require(
        not re.search(
            r"[/\\@<>\r\n]|\b\d{1,3}(?:\.\d{1,3}){3}\b|(?i:token=|password|secret|api.key|ghp_|github_pat_)",
            value,
        ),
        "Public label contains a path, address or unsafe text",
    )
    return value


def public_metadata(metadata, protocol):
    require(isinstance(metadata, dict), "Expected metadata object")
    result = {
        key: safe_label(metadata.get(key, default))
        for key, default in (
            ("model", "Model"),
            ("hardware", "Hardware not recorded"),
            ("backend", "Backend not recorded"),
        )
    }
    for key, length in (("source_revision", 40), ("runtime_sha256", 64)):
        value = metadata.get(key)
        require(
            value is None
            or (isinstance(value, str) and re.fullmatch(f"[a-f0-9]{{{length}}}", value)),
            f"Invalid {key}",
        )
        result[key] = value
    source = metadata.get("protocol", {})
    require(isinstance(source, dict), "Expected protocol object")
    for key in ("input_tokens", "output_tokens", "repetitions", "warmup_runs"):
        require(key not in source or source[key] == protocol[key], f"Metadata conflicts with {key}")
    for key in ("context_tokens", "temperature", "seed", "batch", "cooldown_seconds"):
        if key in source:
            protocol[key] = number(source[key], key)
    for key in ("prefix_cache", "mma_attention"):
        if key in source:
            require(type(source[key]) is bool, f"Invalid {key}")
            protocol[key] = source[key]
    return result


def normalize(raw, metadata, modes=("decode", "mtp"), model_index=0):
    require(
        isinstance(raw, dict) and isinstance(raw.get("results"), list),
        "Expected standalone benchmark JSON",
    )
    require(0 <= model_index < len(raw["results"]), "Invalid model index")
    modes = list(modes)
    require(
        modes and len(set(modes)) == len(modes) and set(modes) <= {"decode", "mtp"}, "Invalid modes"
    )
    lengths = raw.get("prefill_lengths")
    require(
        isinstance(lengths, list) and lengths and len(set(lengths)) == len(lengths),
        "Invalid input lengths",
    )
    for length in lengths:
        number(length, "input length", minimum=1, integer=True)
    repetitions = number(raw.get("repetitions"), "repetitions", minimum=1, integer=True)
    warmups = number(raw.get("warmup_runs"), "warmup runs", integer=True)
    output_tokens = number(raw.get("decode_tokens"), "output limit", minimum=2, integer=True)
    protocol = dict(
        input_tokens=lengths,
        output_tokens=output_tokens,
        repetitions=repetitions,
        warmup_runs=warmups,
        modes=modes,
    )
    identity = public_metadata(metadata, protocol)
    model = raw["results"][model_index]
    require(
        not model.get("error"),
        "Runner failed; resolve the failure before making a comparison report",
    )
    samples = []
    for index, run in enumerate(model.get("runs", []), 1):
        length, mode = run.get("prompt_tokens_requested"), run.get("mode")
        number(length, "requested input", minimum=1, integer=True)
        require(run.get("status") == "ok", "Failed run; diagnostic logs must remain private")
        require(
            length in lengths and mode in modes and type(run.get("warmup")) is bool,
            "Unexpected case",
        )
        for key in ("prompt_tokens", "prefill_tokens", "completion_tokens"):
            number(run.get(key), key, integer=True)
        require(
            run["prompt_tokens"] == run["prefill_tokens"] == length,
            "Input count mismatch or prefix reuse",
        )
        require(
            run["completion_tokens"] == output_tokens, "Short output is not a fixed-workload sample"
        )
        require(
            type(run.get("mtp_used")) in (bool, int) and run["mtp_used"] in (0, 1),
            "Invalid MTP flag",
        )
        require(bool(run["mtp_used"]) == (mode == "mtp"), "MTP execution mismatch")
        sample = {
            key: run[key]
            for key in (
                "prompt_tokens_requested",
                "prompt_tokens",
                "prefill_tokens",
                "completion_tokens",
                "mode",
                "warmup",
                "status",
                "mtp_used",
            )
        }
        sample["index"] = index
        for key in ("prefill_ms", "decode_ms", "ttft_ms"):
            sample[key] = number(run.get(key), key, minimum=1e-12)
        sample["prefill_tps"] = length * 1000 / sample["prefill_ms"]
        sample["decode_tps"] = (output_tokens - 1) * 1000 / sample["decode_ms"]
        for key in ("prefill_tps", "decode_tps"):
            reported = number(run.get(key), key, minimum=1e-12)
            require(math.isclose(reported, sample[key], rel_tol=1e-6), f"Inconsistent {key}")
        generation = run.get("generation_tps")
        sample["generation_tps"] = (
            None if generation is None else number(generation, "generation_tps", minimum=1e-12)
        )
        sample["generation_ms_derived"] = (
            1000 * output_tokens / generation if generation is not None else None
        )
        sample["mean_decode_interval_ms"] = sample["decode_ms"] / (output_tokens - 1)
        for key in ("mtp_drafted_tokens", "mtp_accepted_tokens"):
            sample[key] = number(run.get(key), key, integer=True)
        require(
            sample["mtp_accepted_tokens"] <= sample["mtp_drafted_tokens"],
            "Accepted drafts exceed drafted tokens",
        )
        require(
            mode == "mtp" or sample["mtp_drafted_tokens"] == 0,
            "Ordinary decode reported MTP tokens",
        )
        samples.append(sample)
    summary = []
    for length in lengths:
        for mode in modes:
            selected = [r for r in samples if r["prompt_tokens"] == length and r["mode"] == mode]
            measured = [r for r in selected if not r["warmup"]]
            require(
                len(measured) == repetitions and len(selected) - len(measured) == warmups,
                "Incomplete or duplicate case samples",
            )
            drafted = sum(r["mtp_drafted_tokens"] for r in measured)
            accepted = sum(r["mtp_accepted_tokens"] for r in measured)
            pooled = {}
            for key, duration, count in (
                ("prefill_tps", "prefill_ms", length),
                ("decode_tps", "decode_ms", output_tokens - 1),
                ("generation_tps", "generation_ms_derived", output_tokens),
            ):
                times = [r[duration] for r in measured]
                pooled[key] = (
                    count * len(times) * 1000 / sum(times)
                    if all(t is not None for t in times)
                    else None
                )
            summary.append(
                {
                    "prompt_tokens": length,
                    "mode": mode,
                    "n": len(measured),
                    "stats": {key: stats([r[key] for r in measured]) for key in METRICS},
                    "pooled": pooled,
                    "drafted_mean": drafted / len(measured),
                    "accepted_mean": accepted / len(measured),
                    "rejected_mean": (drafted - accepted) / len(measured),
                    "mtp_acceptance": accepted / drafted if drafted else None,
                }
            )
    coverage = [
        {
            "metric": key,
            "chart": chart,
            "state": "recorded" if all(r.get(key) is not None for r in samples) else "missing",
        }
        for key, chart in SOURCE_FIELDS.items()
    ]
    coverage += [
        {
            "metric": key,
            "chart": "latency",
            "state": "derived" if all(r[key] is not None for r in samples) else "missing",
        }
        for key in ("generation_ms_derived", "mean_decode_interval_ms")
    ]
    coverage += [
        {"metric": key, "chart": "—", "state": "not_measured", "reason": reason}
        for key, reason in MISSING.items()
    ]
    return {
        "schema": "mfq.inference-chart-report.v1",
        "identity": identity,
        "protocol": protocol,
        "samples": samples,
        "summary": summary,
        "coverage": coverage,
        "gpu": None,
        "validation": {
            "measured": len(summary) * repetitions,
            "warmups": len(summary) * warmups,
            "status": "complete",
            "native_finish_reason": "not retained by runner export",
        },
    }


def gpu_telemetry(samples, before=None, after=None):
    require(isinstance(samples, list) and samples, "No GPU telemetry samples")

    def reading(raw):
        def field(key):
            value = raw.get(key)
            require(isinstance(value, (str, int, float)), "Missing GPU counter")
            return number(float(value), key)

        used, total, use = (
            field("VRAM Total Used Memory (B)"),
            field("VRAM Total Memory (B)"),
            field("GPU use (%)"),
        )
        require(total > 0 and used <= total and use <= 100, "Invalid GPU counters")
        return {"used_bytes": used, "total_bytes": total, "use_percent": use}

    origin = number(samples[0].get("time"), "GPU timestamp")
    rows = []
    for source in samples:
        time = number(source.get("time"), "GPU timestamp")
        require(not rows or time - origin > rows[-1]["seconds"], "GPU timestamps must increase")
        rows.append({"seconds": time - origin, **reading(source.get("gpu", {}))})
    require(len({r["total_bytes"] for r in rows}) == 1, "GPU capacity changed")
    boundaries = {
        "before": reading(before) if before else None,
        "after": reading(after) if after else None,
    }
    require(
        all(r is None or r["total_bytes"] == rows[0]["total_bytes"] for r in boundaries.values()),
        "GPU capacity changed at run boundary",
    )
    return {"samples": rows, **boundaries}


def draw_charts(bundle, output, theme):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "text.color": theme["ink"],
            "axes.labelcolor": theme["muted"],
            "xtick.color": theme["muted"],
            "ytick.color": theme["muted"],
            "svg.hashsalt": "mfq-report-blue-rose-v1",
            "svg.fonttype": "path",
        }
    )
    rows, samples, protocol = bundle["summary"], bundle["samples"], bundle["protocol"]
    lengths, modes = protocol["input_tokens"], protocol["modes"]
    mode_name = {"decode": "MTP off", "mtp": "MTP on"}

    def color(mode):
        return theme[mode]

    legend = [
        Line2D(
            [],
            [],
            color=color(mode),
            marker="o" if mode == "decode" else "D",
            label=mode_name[mode],
        )
        for mode in modes
    ]
    revision = (bundle["identity"]["source_revision"] or "not recorded")[:12]
    identity = f"{bundle['identity']['model']}  ·  {bundle['identity']['hardware']}  ·  {bundle['identity']['backend']}  ·  {protocol['output_tokens']} output tokens"
    conditions = f"Revision {revision} · n={protocol['repetitions']} per cell · {protocol['warmup_runs']} warmup(s)/cell"
    if "mma_attention" in protocol:
        conditions += f" · MMA attention {'on' if protocol['mma_attention'] else 'off'}"

    def frame(title, nrows=1, ncols=3, height=5.8, telemetry=False):
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(15, height), squeeze=False, facecolor=theme["background"]
        )
        fig.subplots_adjust(
            left=0.065, right=0.975, bottom=0.16, top=0.75, wspace=0.32, hspace=0.52
        )
        fig.text(0.065, 0.94, title, fontsize=24, weight="bold")
        fig.text(0.065, 0.88, identity, fontsize=11, color=theme["muted"])
        handles = (
            [Line2D([], [], color=theme["telemetry"], marker="o", label="Device-wide samples")]
            if telemetry
            else legend
        )
        fig.legend(
            handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.865), ncol=2, frameon=False
        )
        for ax in axes.flat:
            ax.set_axisbelow(True)
            ax.grid(axis="y", color=theme["grid"], linewidth=0.8)
            ax.tick_params(length=0, pad=7)
            for spine in ax.spines.values():
                spine.set_visible(False)
        return fig, list(axes.flat)

    def missing(ax, message="Not recorded"):
        ax.set_axis_off()
        ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=theme["muted"],
        )

    def grouped(ax, metric):
        title, unit, direction = METRICS[metric]
        ax.set_title(title, loc="left", fontsize=12, weight="bold", pad=13)
        ax.set_ylabel(unit + (" ↑" if direction == "higher" else " ↓"))
        top = 0
        for j, mode in enumerate(modes):
            for i, length in enumerate(lengths):
                row = next(r for r in rows if r["prompt_tokens"] == length and r["mode"] == mode)
                stat = row["stats"][metric]
                x = i + (j - (len(modes) - 1) / 2) * 0.34
                if stat is None:
                    ax.text(x, 0, "N/A", ha="center", va="bottom", fontsize=8)
                    continue
                value = stat["median"]
                points = [
                    s[metric]
                    for s in samples
                    if not s["warmup"] and s["prompt_tokens"] == length and s["mode"] == mode
                ]
                top = max(top, *points)
                bar = ax.bar(x, value, width=0.27, color=color(mode), zorder=2)[0]
                bar.set_gid(f"{metric}-{length}-{mode}")
                ax.errorbar(
                    x,
                    value,
                    yerr=[[value - stat["q25"]], [stat["q75"] - value]],
                    fmt="none",
                    ecolor=theme["ink"],
                    capsize=3,
                    lw=1,
                )
                ax.scatter(
                    [x + (k - (len(points) - 1) / 2) * 0.018 for k in range(len(points))],
                    points,
                    s=15,
                    facecolor="white",
                    edgecolor=color(mode),
                    linewidth=0.7,
                    zorder=4,
                )
                label = (
                    f"{value:,.0f}"
                    if value >= 1000
                    else f"{value:.1f}" if value >= 10 else f"{value:.2f}"
                )
                ax.annotate(
                    label,
                    (x, max(points)),
                    xytext=(0, 7 + 13 * j),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                    weight="bold",
                    color=color(mode),
                )
        ax.set_xticks(range(len(lengths)), list(map(str, lengths)))
        ax.set_xlabel("Input tokens")
        ax.set_ylim(0, top * 1.34 if top else 1)

    with PdfPages(
        output / "charts.pdf",
        metadata={
            "Title": "Inference metrics",
            "Creator": "MFQ benchmark reporter",
            "CreationDate": None,
            "ModDate": None,
        },
    ) as pdf:

        def save(fig, name, note):
            fig.text(0.065, 0.065, note, fontsize=9, color=theme["muted"])
            fig.text(0.065, 0.025, conditions, fontsize=9, color=theme["muted"])
            fig.savefig(
                output / f"{name}.png", dpi=150, metadata={"Software": "MFQ benchmark reporter"}
            )
            fig.savefig(
                output / f"{name}.svg",
                metadata={"Date": None, "Creator": "MFQ benchmark reporter", "Title": name},
            )
            pdf.savefig(fig)
            plt.close(fig)

        fig, axes = frame("01  Throughput")
        for ax, metric in zip(axes, ("prefill_tps", "decode_tps", "generation_tps"), strict=True):
            grouped(ax, metric)
        save(
            fig,
            "throughput",
            "Median + 25–75% spread; dots show measured requests. Generation includes the full native request interval; decode does not.",
        )

        fig, axes = frame("02  Latency & duration", 2, 3, 9.4)
        for ax, metric in zip(axes, list(METRICS)[3:], strict=False):
            grouped(ax, metric)
        axes[-1].set_axis_off()
        axes[-1].text(0, 0.85, "Timing definitions", weight="bold", fontsize=14)
        axes[-1].text(
            0,
            0.68,
            "Generation duration = output tokens /\ngeneration tok/s × 1,000.\n\nMean decode interval = decode ms /\n(output tokens − 1).\n\nDerived values are not new timers.\nNo per-token p95/p99 is available.",
            va="top",
            linespacing=1.5,
        )
        save(
            fig,
            "latency",
            "Measured: TTFT, prefill and decode duration. Derived: whole-request duration and mean decode interval. Warmups excluded.",
        )

        fig, axes = frame("03  MTP efficiency", 2, 2, 9)
        mtp = [r for r in rows if r["mode"] == "mtp"]
        if not mtp:
            for ax in axes:
                missing(ax, "MTP not measured")
        else:
            x = list(range(len(mtp)))
            rates = [
                100 * r["mtp_acceptance"] if r["mtp_acceptance"] is not None else None for r in mtp
            ]
            for i, value in enumerate(rates):
                if value is not None:
                    axes[0].bar(i, value, color=color("mtp"), width=0.5)
                    axes[0].text(
                        i,
                        value + 3,
                        f"{value:.1f}%",
                        ha="center",
                        color=color("mtp"),
                        weight="bold",
                    )
            axes[0].set(title="Acceptance · accepted / drafted", ylabel="%", ylim=(0, 100))
            axes[1].bar(x, [r["accepted_mean"] for r in mtp], color=color("mtp"), label="Accepted")
            axes[1].bar(
                x,
                [r["rejected_mean"] for r in mtp],
                bottom=[r["accepted_mean"] for r in mtp],
                color=theme["rejected"],
                hatch="//",
                label="Rejected",
            )
            axes[1].set(
                title="Draft tokens per request · mean",
                ylabel="tokens",
                ylim=(0, max(1, *(r["drafted_mean"] for r in mtp)) * 1.35),
            )
            axes[1].legend(frameon=False, loc="upper right", ncol=2)
            for ax, metric, title in (
                (axes[2], "decode_tps", "Decode throughput · MTP on / off"),
                (axes[3], "ttft_ms", "TTFT cost · MTP on / off"),
            ):
                if "decode" not in modes:
                    missing(ax, "Ordinary decode baseline not measured")
                    continue
                ratios = [
                    r["stats"][metric]["median"]
                    / next(
                        b
                        for b in rows
                        if b["prompt_tokens"] == r["prompt_tokens"] and b["mode"] == "decode"
                    )["stats"][metric]["median"]
                    for r in mtp
                ]
                ax.axhline(1, color=theme["muted"], ls="--", lw=1)
                ax.bar(x, ratios, color=color("mtp"), width=0.5)
                for i, value in enumerate(ratios):
                    ax.text(
                        i,
                        value + 0.04,
                        f"{value:.3f}×",
                        ha="center",
                        color=color("mtp"),
                        weight="bold",
                    )
                ax.set(title=title, ylabel="ratio", ylim=(0, max(1, *ratios) * 1.25))
            for ax in axes:
                ax.set_xticks(x, [str(r["prompt_tokens"]) for r in mtp])
                ax.set_xlabel("Input tokens")
        save(
            fig,
            "mtp",
            "Acceptance uses pooled token counts. Throughput ratio >1 is faster; TTFT cost >1 is slower. Same measured workload, no cross-revision claim.",
        )

        fig, axes = frame("04  Repeat stability", 2, 3, 9.4)
        metrics = (
            "prefill_tps",
            "decode_tps",
            "generation_tps",
            "ttft_ms",
            "prefill_ms",
            "decode_ms",
        )
        for ax, metric in zip(axes, metrics, strict=True):
            ax.axhline(0, color=theme["muted"], lw=1)
            for i, row in enumerate(rows):
                stat = row["stats"][metric]
                if stat is None:
                    continue
                points = [
                    100 * (s[metric] / stat["median"] - 1)
                    for s in samples
                    if not s["warmup"]
                    and s["mode"] == row["mode"]
                    and s["prompt_tokens"] == row["prompt_tokens"]
                ]
                ax.scatter(
                    [i + (j - (len(points) - 1) / 2) * 0.045 for j in range(len(points))],
                    points,
                    color=color(row["mode"]),
                    marker="o" if row["mode"] == "decode" else "D",
                    s=22,
                )
            ax.set_title(METRICS[metric][0], loc="left", fontsize=12, weight="bold")
            ax.set_xticks(
                range(len(rows)),
                [f"{r['prompt_tokens']}\n{'off' if r['mode']=='decode' else 'on'}" for r in rows],
                fontsize=8,
            )
            ax.set_ylabel("Deviation from cell median (%)")
        save(
            fig,
            "stability",
            "Every measured request shown; each cell has its own 0% reference. Axes zoom into variation. Small-sample spread is not a confidence interval.",
        )

        fig, axes = frame("05  Warmup & execution order")
        for ax, metric in zip(axes, ("ttft_ms", "prefill_tps", "generation_tps"), strict=True):
            if not any(s[metric] is not None for s in samples):
                missing(ax, "Generation timing not recorded")
                continue
            for row in rows:
                group = [
                    s
                    for s in samples
                    if s["mode"] == row["mode"]
                    and s["prompt_tokens"] == row["prompt_tokens"]
                    and s[metric] is not None
                ]
                if not group:
                    continue
                ax.plot(
                    [s["index"] for s in group],
                    [s[metric] for s in group],
                    color=color(row["mode"]),
                    lw=1,
                    alpha=0.6,
                )
                for s in group:
                    ax.scatter(
                        s["index"],
                        s[metric],
                        marker="D" if s["warmup"] else "o",
                        s=55 if s["warmup"] else 20,
                        facecolor="white" if s["warmup"] else color(s["mode"]),
                        edgecolor=color(s["mode"]),
                        zorder=3,
                    )
            ax.set_title(METRICS[metric][0], loc="left", weight="bold", fontsize=12)
            ax.set_yscale("log")
            ax.set_ylabel(METRICS[metric][1] + " · log scale")
            ax.set_xlabel("Request order (warmups included)")
        save(
            fig,
            "warmup",
            "Hollow diamonds = warmups; filled dots = measured runs. Groups follow the saved execution order. Warmups never enter headline statistics.",
        )

        fig, axes = frame("06  Median vs pooled throughput")
        for ax, metric in zip(axes, ("prefill_tps", "decode_tps", "generation_tps"), strict=True):
            for i, row in enumerate(rows):
                stat = row["stats"][metric]
                if stat is None:
                    continue
                for offset, value, hatch in (
                    (-0.16, stat["median"], None),
                    (0.16, row["pooled"][metric], "///"),
                ):
                    ax.bar(
                        i + offset,
                        value,
                        width=0.28,
                        facecolor=color(row["mode"]) if not hatch else "white",
                        edgecolor=color(row["mode"]),
                        hatch=hatch,
                    )
            ax.set_title(METRICS[metric][0], loc="left", fontsize=12, weight="bold")
            ax.set_ylabel("tok/s")
            ax.set_xticks(
                range(len(rows)),
                [f"{r['prompt_tokens']}\n{'off' if r['mode']=='decode' else 'on'}" for r in rows],
                fontsize=8,
            )
            ax.legend(
                handles=[
                    Patch(facecolor=theme["ink"], label="Median"),
                    Patch(facecolor="white", edgecolor=theme["ink"], hatch="///", label="Pooled"),
                ],
                frameon=False,
                fontsize=8,
            )
        save(
            fig,
            "aggregation",
            "Solid = median of per-request throughput. Hatched = sum(tokens) / sum(time). They use the same warmup-free samples; neither is relabeled as the other.",
        )

        fig, axes = frame("07  GPU activity & memory", 1, 2, telemetry=True)
        if not bundle["gpu"]:
            for ax in axes:
                missing(ax, "GPU telemetry not recorded")
            note = "No device telemetry was supplied. No utilization, memory, power or energy value is inferred from model throughput."
        else:
            gpu = bundle["gpu"]
            points = gpu["samples"]
            times = [r["seconds"] for r in points]
            for ax, key, factor, title, unit in (
                (axes[0], "use_percent", 1, "Device utilization", "%"),
                (axes[1], "used_bytes", 2**30, "Device VRAM allocated", "GiB"),
            ):
                values = [r[key] / factor for r in points]
                ax.plot(times, values, "o-", color=theme["telemetry"], ms=4, lw=2)
                ax.fill_between(times, values, alpha=0.09, color=theme["telemetry"])
                ax.set(
                    title=title,
                    ylabel=unit,
                    xlabel="Seconds since first GPU sample",
                    ylim=(0, 105 if key == "use_percent" else max(values) * 1.2),
                )
            peak = max(r["used_bytes"] for r in points) / 2**30
            capacity = points[0]["total_bytes"] / 2**30
            before = f"{gpu['before']['used_bytes']/2**30:.3f}" if gpu["before"] else "N/A"
            after = f"{gpu['after']['used_bytes']/2**30:.3f}" if gpu["after"] else "N/A"
            note = f"{len(points)} device-wide samples; sampled peak {peak:.3f} / {capacity:.1f} GiB; before/after {before}/{after} GiB. Includes load and warmups; no per-request alignment."
        save(fig, "gpu", note)

        fig, axes = frame("08  Workload & sample validation")
        for i, row in enumerate(rows):
            mode, length = row["mode"], row["prompt_tokens"]
            group = [s for s in samples if s["mode"] == mode and s["prompt_tokens"] == length]
            axes[0].scatter(i, length, s=90, marker="s", facecolors="none", edgecolors=theme["ink"])
            axes[0].scatter(i, group[0]["prompt_tokens"], s=35, color=color(mode))
            axes[0].scatter(i, group[0]["prefill_tokens"], s=65, marker="x", color=color(mode))
            axes[1].bar(i, group[0]["completion_tokens"], width=0.55, color=color(mode))
            axes[2].bar(i, row["n"], color=color(mode), width=0.55)
            axes[2].bar(
                i,
                protocol["warmup_runs"],
                bottom=row["n"],
                facecolor="white",
                edgecolor=color(mode),
                hatch="///",
                width=0.55,
            )
        axes[0].set(title="Requested / input / prefill tokens", ylabel="tokens")
        axes[0].legend(
            handles=[
                Line2D(
                    [], [], color=theme["ink"], marker="s", mfc="none", ls="", label="Requested"
                ),
                Line2D([], [], color=theme["decode"], marker="o", ls="", label="Input"),
                Line2D([], [], color=theme["decode"], marker="x", ls="", label="Prefill"),
            ],
            frameon=False,
            fontsize=8,
        )
        axes[1].axhline(protocol["output_tokens"], ls="--", color=theme["ink"], lw=1)
        axes[1].set(
            title="Actual completion tokens",
            ylabel="tokens",
            ylim=(0, protocol["output_tokens"] * 1.25),
        )
        axes[2].set(
            title="Measured + warmup requests",
            ylabel="requests",
            ylim=(0, protocol["repetitions"] + protocol["warmup_runs"] + 1),
        )
        for ax in axes:
            ax.set_xticks(
                range(len(rows)),
                [f"{r['prompt_tokens']}\n{'off' if r['mode']=='decode' else 'on'}" for r in rows],
                fontsize=8,
            )
        save(
            fig,
            "validation",
            f"{len(samples)}/{len(samples)} saved runs passed count, finite-timing and MTP-mode checks. Dashed = output target. Hatched = warmup. Native finish reason was not retained.",
        )


def write_tables(bundle, output):
    flat = []
    for row in bundle["summary"]:
        item = {
            key: row[key]
            for key in (
                "prompt_tokens",
                "mode",
                "n",
                "drafted_mean",
                "accepted_mean",
                "rejected_mean",
                "mtp_acceptance",
            )
        }
        for metric, value in row["stats"].items():
            for key in ("median", "q25", "q75", "min", "max"):
                item[f"{metric}_{key}"] = value[key] if value else None
        item.update({f"{key}_pooled": value for key, value in row["pooled"].items()})
        flat.append(item)
    for name, rows in (
        ("summary", flat),
        ("samples", bundle["samples"]),
        ("coverage", bundle["coverage"]),
    ):
        keys = list(dict.fromkeys(key for row in rows for key in row))
        with (output / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)


def write_gallery(bundle, output, theme):
    from string import Template

    esc = html.escape
    identity, protocol = bundle["identity"], bundle["protocol"]
    nav = "".join(
        f'<a href="#{key}">{i:02} {title}</a>' for i, (key, title, _) in enumerate(CHARTS, 1)
    )
    cards = "".join(
        f'<section class="chart-section" id="{key}"><div class="chapter-num">CHART {i:02}</div>'
        f'<div class="figure-header"><div><h2>{title}</h2><p class="prose">{description}</p></div>'
        f'<div class="exports"><a href="{key}.svg">SVG ↗</a><a href="{key}.png">PNG ↗</a></div></div>'
        f'<figure><div class="chart-scroll" tabindex="0" role="region" aria-label="{title}圖表">'
        f'<a href="{key}.svg"><img loading="lazy" src="{key}.svg" alt="{title}：{description}"></a></div>'
        '<figcaption><span class="mobile-hint">圖表可橫向捲動。</span>開啟 SVG 可檢視完整尺寸。</figcaption></figure></section>'
        for i, (key, title, description) in enumerate(CHARTS, 1)
    )
    states = {
        "recorded": "已記錄",
        "derived": "由量測推導",
        "missing": "缺少欄位",
        "not_measured": "尚未量測",
    }
    coverage = "".join(
        f'<tr><td>{esc(r["metric"])}</td><td class="state {r["state"]}">{states[r["state"]]}</td><td>{esc(r.get("reason", r["chart"]))}</td></tr>'
        for r in bundle["coverage"]
    )
    methods = (
        "圖中柱高採逐輪中位數，細線為 25–75% 區間，小點保留所有正式樣本。"
        "Generation throughput 包含 native request 的整段耗時；decode 只使用首末 token 間的時間。"
        "整段耗時由 generation_tps 反推，並非新增 timer；平均 decode 間隔也不是逐 token 延遲分佈。"
        "GPU 圖是裝置層級取樣，包含載入及暖機，取樣峰值不等於真正的瞬時峰值。"
        "出圖只讀取已保存的量測資料，不會重新使用 GPU。"
    )
    rows = bundle["summary"]
    baseline = {r["prompt_tokens"]: r for r in rows if r["mode"] == "decode"}
    paired = [
        (r, baseline[r["prompt_tokens"]])
        for r in rows
        if r["mode"] == "mtp" and r["prompt_tokens"] in baseline
    ]
    findings = []
    if paired:
        for metric, label in (("decode_tps", "decode 吞吐量"), ("ttft_ms", "首 token 等待時間")):
            ratios = [
                r["stats"][metric]["median"] / b["stats"][metric]["median"] for r, b in paired
            ]
            findings.append(
                f"在本次負載下，MTP 的 {label}為普通模式的 {min(ratios):.3f}–{max(ratios):.3f} 倍。"
            )
    acceptance = [r["mtp_acceptance"] * 100 for r in rows if r["mtp_acceptance"] is not None]
    if acceptance:
        findings.append(
            f"各輸入長度的 MTP 草稿接受率為 {min(acceptance):.1f}–{max(acceptance):.1f}%，採總接受數除以總草稿數。"
        )
    if not findings:
        findings.append(
            "本次未提供可配對的普通模式與 MTP 結果；各圖呈現已量測模式，不推估模式間的差異。"
        )
    lengths = protocol["input_tokens"]
    input_range = str(min(lengths)) if len(lengths) == 1 else f"{min(lengths)}–{max(lengths)}"
    facts = [
        (bundle["validation"]["measured"], "正式樣本"),
        (len(CHARTS), "圖表主題"),
        (input_range, f"輸入長度（{len(lengths)} 組）"),
        (protocol["output_tokens"], "輸出 tokens／次"),
    ]
    modes = {"decode": "普通 decode / MTP off", "mtp": "MTP on"}
    legend = [(theme[mode], modes[mode]) for mode in protocol["modes"]]
    if bundle.get("gpu"):
        legend.append((theme["telemetry"], "GPU 裝置取樣"))
    settings = [
        ("Input tokens", ", ".join(map(str, lengths))),
        ("Output tokens", protocol["output_tokens"]),
        ("每格正式 / 暖機次數", f"{protocol['repetitions']} / {protocol['warmup_runs']}"),
    ]
    labels = {
        "context_tokens": "Context tokens",
        "temperature": "Temperature",
        "seed": "Seed",
        "batch": "Batch",
        "prefix_cache": "Prefix cache",
        "mma_attention": "MMA attention",
        "cooldown_seconds": "Cooldown seconds",
    }
    settings += [(label, protocol[key]) for key, label in labels.items() if key in protocol]
    if identity.get("runtime_sha256"):
        settings.append(("Runtime SHA-256", identity["runtime_sha256"]))
    if identity.get("source_revision"):
        settings.append(("Source revision", identity["source_revision"]))
    overview = (
        f"以 {', '.join(map(str, lengths))} 個 input tokens、每次 {protocol['output_tokens']} 個 output tokens 量測。"
        f"每個條件保留 {protocol['repetitions']} 次正式結果，共排除 {bundle['validation']['warmups']} 次暖機。"
        "以下比較只適用於本次模型、runtime 與設定。"
    )
    template = Template(
        Path(__file__)
        .with_name("templates")
        .joinpath("inference-report.html")
        .read_text(encoding="utf-8")
    )
    page = template.substitute(
        title=esc(f"推理基準測試 · {identity['model']}"),
        model=esc(identity["model"]),
        hardware=esc(identity["hardware"]),
        backend=esc(identity["backend"]),
        revision=esc((identity["source_revision"] or "未記錄")[:12]),
        ink=theme["ink"],
        muted=theme["muted"],
        brand=theme["decode"],
        facts="".join(
            f'<div><span class="metric-value">{value}</span><span class="metric-label">{label}</span></div>'
            for value, label in facts
        ),
        archive_link='<a href="../index.html">歷史報告 ↗</a>' if bundle.get("run") else "",
        nav=nav,
        overview=esc(overview),
        findings="".join(f"<li>{esc(text)}</li>" for text in findings),
        legend="".join(
            f'<span><i class="dot" style="background:{color}"></i>{label}</span>'
            for color, label in legend
        ),
        methods=methods,
        protocol="".join(
            f"<tr><th>{esc(label)}</th><td>{esc(str(value))}</td></tr>" for label, value in settings
        ),
        cards=cards,
        coverage=coverage,
        repetitions=protocol["repetitions"],
    )
    (output / "index.html").write_text(page, encoding="utf-8")
    markdown = [
        "# Inference chart report",
        "",
        f"{identity['model']} · {identity['hardware']}",
        "",
        "[Open chart gallery](index.html) · [All charts PDF](charts.pdf) · [Summary CSV](summary.csv) · [Samples CSV](samples.csv)",
        "",
        methods,
        "",
    ]
    for key, title, description in CHARTS:
        markdown += [f"## {title}", "", description, "", f"![{title}]({key}.png)", ""]
    (output / "report.md").write_text("\n".join(markdown), encoding="utf-8")


def build_report(bundle, output, theme):
    output = Path(output)
    require(not output.exists(), "Output directory already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".mfq-report-", dir=output.parent) as folder:
        staged = Path(folder)
        (staged / "bundle.json").write_text(
            json.dumps(bundle, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        write_tables(bundle, staged)
        draw_charts(bundle, staged, theme)
        write_gallery(bundle, staged, theme)
        (staged / "theme.json").write_text(json.dumps(theme, indent=2) + "\n", encoding="utf-8")
        hashes = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(staged.iterdir())
        }
        (staged / "checksums.json").write_text(
            json.dumps(hashes, indent=2) + "\n", encoding="utf-8"
        )
        staged.rename(output)


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Non-finite JSON number")),
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument(
        "--metadata", type=Path, help="Public model/hardware labels and run settings"
    )
    parser.add_argument("--modes", choices=("decode", "mtp"), nargs="+", default=["decode", "mtp"])
    parser.add_argument("--model-index", type=int, default=0)
    parser.add_argument("--gpu-samples", type=Path)
    parser.add_argument("--gpu-before", type=Path)
    parser.add_argument("--gpu-after", type=Path)
    parser.add_argument(
        "--theme", type=Path, default=Path(__file__).with_name("reporting-theme.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        bundle = normalize(
            read_json(args.raw),
            read_json(args.metadata) if args.metadata else {},
            args.modes,
            args.model_index,
        )
        if args.gpu_samples:
            before = read_json(args.gpu_before) if args.gpu_before else None
            if isinstance(before, list):
                before = before[0] if before else None
            after = read_json(args.gpu_after) if args.gpu_after else None
            bundle["gpu"] = gpu_telemetry(read_json(args.gpu_samples), before, after)
        for key in ("GPU use (%)", "VRAM Total Used Memory (B)", "VRAM Total Memory (B)"):
            bundle["coverage"].append(
                {
                    "metric": key,
                    "chart": "gpu",
                    "state": "recorded" if bundle["gpu"] else "not_measured",
                }
            )
        build_report(bundle, args.output, read_json(args.theme))
    except (ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(
        f"Rendered {len(CHARTS)} chart groups; {bundle['validation']['measured']} measured runs; {bundle['validation']['warmups']} warmups excluded."
    )


if __name__ == "__main__":
    main()
