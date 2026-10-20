"""Plot the revision-pinned MFQ, Strata, llama.cpp and oMLX Metal measurements.

uv run --no-project --no-config --with matplotlib python docs/figures/generate-metal-runtime-comparison.py
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Patch, PathPatch
from matplotlib.path import Path as PlotPath
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / "data/qwen38-flash-next-metal-runtime-comparison.json").read_text())
C = json.loads((ROOT / "studio-figure-theme.json").read_text())["colors"]
REPOSITORY_URL = "https://github.com/Tylogi/MFQ"
WIDTH, HEIGHT = 1600, 1224
COLORS = {
    "mfq": (C["accent"], C["accent_ink"]),
    "gsq-rco": ("#8b9fd0", "#5c74ad"),
    "unsloth-ud": ("#6da993", "#3f806c"),
    "jundot-oq": ("#a69bbf", "#76678f"),
}
FAMILIES = ["DejaVu Sans"]
for font_path in (
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
):
    if font_path.exists():
        font_manager.fontManager.addfont(font_path)
        family = font_manager.FontProperties(fname=font_path).get_name()
        if family not in FAMILIES:
            FAMILIES.append(family)

plt.rcParams.update({
    "font.family": FAMILIES,
    "font.size": 17,
    "text.color": C["ink"],
    "xtick.color": C["muted_light"],
    "axes.unicode_minus": False,
    "svg.fonttype": "path",
    "svg.hashsalt": "mfq-metal-runtime-comparison",
})
LANGUAGES = {
    "en": {
        "title": "Metal runtime comparison",
        "hardware": f"{DATA['model']} · M5 Max · 40-core GPU · 128 GB",
        "workload": "4,096 input · 128 output tokens · batch 1 · greedy",
        "configuration": "Runtime / quantization",
        "prefill": "Prompt processing¹",
        "generation_no_mtp": "Generation · MTP off",
        "generation_mtp": "Generation · MTP on",
        "prefill_note": "¹ MFQ: published prefill value; others: MTP off. oMLX includes first-token overhead.",
        "comparison_note": "Shaded rows: selected comparison tiers. Draft setups differ; oMLX uses adaptive MTP. See notes.",
    },
    "zh-CN": {
        "title": "Metal 运行时速度对比",
        "hardware": f"{DATA['model']} · M5 Max · 40 核 GPU · 128 GB",
        "workload": "输入 4,096 · 生成 128 tokens · batch 1 · 贪心采样",
        "configuration": "运行时／量化档位",
        "prefill": "提示词处理¹",
        "generation_no_mtp": "生成速度 · 关闭 MTP",
        "generation_mtp": "生成速度 · 开启 MTP",
        "prefill_note": "¹ MFQ 为已发布的提示词处理值；其余使用 MTP 关闭值。oMLX 计时包含首 token 开销。",
        "comparison_note": "浅色底纹标出重点对比档位；草稿配置不同，oMLX 使用自适应 MTP。详见基准说明。",
    },
}
GROUPS = (
    ("mfq", "MFQ", 268, 306),
    ("strata", "Strata · v0.1.44.1", 540, 578),
    ("llamacpp", "llama.cpp · f74e2fa6", 740, 778),
    ("omlx", "oMLX · 0.7.1.dev2", 940, 978),
)
ROWS = [
    (row, first_y + index * 36)
    for runtime, _, _, first_y in GROUPS
    for index, row in enumerate(r for r in DATA["measurements"] if r["runtime"] == runtime)
]
assert len(ROWS) == len(DATA["measurements"]) == 17


def bounds(x, y, width, height):
    return x / WIDTH, 1 - (y + height) / HEIGHT, width / WIDTH, height / HEIGHT


def label(fig, x, y, value, size=17, color=None, weight="normal", **kwargs):
    return fig.text(x / WIDTH, 1 - y / HEIGHT, value, fontsize=size,
                    color=color or C["ink"], weight=weight, **kwargs)


def surface(fig, x, y, width, height, fill, border=None, radius=12, zorder=-1):
    fig.add_artist(FancyBboxPatch(
        bounds(x, y, width, height)[:2], width / WIDTH, height / HEIGHT,
        boxstyle=f"round,pad=0,rounding_size={radius / WIDTH}",
        mutation_aspect=WIDTH / HEIGHT, transform=fig.transFigure,
        facecolor=fill, edgecolor=border or "none", linewidth=0.8, zorder=zorder,
    ))


def rule(fig, y):
    fig.add_artist(Line2D([76 / WIDTH, 1524 / WIDTH], [1 - y / HEIGHT] * 2,
                         transform=fig.transFigure, color=C["line_soft"], linewidth=0.8))


def brand(fig):
    ax = fig.add_axes(bounds(1434, 38, 32, 32), zorder=5)
    ax.set_xlim(0, 64)
    ax.set_ylim(64, 0)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((1, 1), 62, 62, boxstyle="round,pad=0,rounding_size=17",
                              facecolor="#0e1010", linewidth=0))
    ax.add_patch(PathPatch(PlotPath([(16, 44), (16, 20), (32, 37), (48, 20), (48, 44)]),
                           fill=False, edgecolor="#f1f2ef", linewidth=2.75,
                           capstyle="round", joinstyle="round"))
    ax.add_patch(Circle((48, 44), 3, facecolor="#d6aa73", linewidth=0))
    label(fig, 1480, 62, "MFQ", 21, C["strong"], "bold")


def selected_metric(row, metric):
    if metric == "prefill":
        return "prefill_reported" if row["runtime"] == "mfq" else "prefill_no_mtp"
    return metric


def comparable_size(row):
    return (row["method"], row["tier"]) in (
        ("mfq", "V3-XXS"), ("gsq-rco", "IQ3_S"),
        ("jundot-oq", "oQ3.5e"),
    )


def panel(fig, left, width, metric, text):
    limit = 3000 if metric == "prefill" else 140
    ticks = (0, 1000, 2000, 3000) if metric == "prefill" else (0, 40, 80, 120, 140)
    label(fig, left, 199, text[metric], 19, C["strong"], "bold")
    label(fig, left, 223, "tok/s", 14, C["muted_light"])
    ax = fig.add_axes(bounds(left, 284, width, 790), facecolor="none")
    ax.set_xlim(0, limit)
    ax.set_ylim(1074, 284)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.set_yticks([])
    ax.grid(axis="x", color=C["line_soft"], linewidth=0.65)
    ax.tick_params(axis="x", length=0, pad=10, labelsize=14)
    for spine in ax.spines.values():
        spine.set_visible(False)
    pixels_per_x = width / limit
    for row, y in ROWS:
        key = selected_metric(row, metric)
        value = row[key]
        assert value is not None and value > 0
        fill, ink = COLORS[row["method"]]
        bar = FancyBboxPatch((0, y - 9), value, 18,
                            boxstyle=f"round,pad=0,rounding_size={3 / pixels_per_x}",
                            mutation_aspect=pixels_per_x, facecolor=fill, linewidth=0, zorder=3)
        bar.set_gid(f'{row["runtime"]}-{row["method"]}-{row["tier"]}-{metric}')
        ax.add_patch(bar)
        ax.text(value + 6 / pixels_per_x, y, f"{value:,.1f}", fontsize=17, color=ink,
                weight="bold" if comparable_size(row) else "normal", va="center")
    return ax


def generate(locale, text):
    fig = plt.figure(figsize=(WIDTH / 72, HEIGHT / 72), facecolor=C["canvas"])
    label(fig, 48, 63, text["title"], 30, C["strong"], "bold")
    label(fig, 48, 96, text["hardware"], 17, C["muted"])
    label(fig, 48, 125, text["workload"], 16, C["muted"])
    brand(fig)
    fig.legend([Patch(color=COLORS[k][0]) for k in COLORS], ["MFQ", "GSQ-RCO", "Unsloth UD", "Jundot oQ"],
               loc="center right", bbox_to_anchor=(1564 / WIDTH, 1 - 123 / HEIGHT), ncol=4,
               frameon=False, fontsize=16, handlelength=1.2, columnspacing=1.7)
    surface(fig, 48, 156, 1504, 960, C["surface"], C["line"])
    label(fig, 76, 199, text["configuration"], 17, C["muted"])
    for y in (238, 514, 714, 914):
        rule(fig, y)
    for _, title, heading_y, _ in GROUPS:
        label(fig, 76, heading_y, title, 20, C["strong"], "bold")
    for row, y in ROWS:
        highlight = comparable_size(row)
        if highlight:
            tint = {"mfq": C["accent_soft"], "gsq-rco": "#eef1f8",
                    "jundot-oq": "#f1eef6"}[row["method"]]
            surface(fig, 64, y - 17, 1472, 34, tint, radius=6, zorder=0)
        method = {"mfq": "", "gsq-rco": "GSQ-RCO · ", "unsloth-ud": "Unsloth · ",
                  "jundot-oq": "Jundot · "}[row["method"]]
        label(fig, 76, y + 6, method + row["tier"], 17,
              COLORS[row["method"]][1] if highlight else C["soft"], "bold" if highlight else "normal")
    panel(fig, 360, 330, "prefill", text)
    panel(fig, 820, 300, "generation_no_mtp", text)
    panel(fig, 1210, 300, "generation_mtp", text)
    label(fig, 48, 1142, text["prefill_note"], 14, C["muted_light"])
    label(fig, 48, 1167, text["comparison_note"], 14, C["muted_light"])
    footer = label(fig, 1552, 1200, REPOSITORY_URL, 14, C["muted_light"], ha="right")
    footer.set_gid("repository-url")
    footer.set_url(REPOSITORY_URL)
    suffix = "" if locale == "en" else ".zh-CN"
    output = ROOT / f"qwen38-flash-next-metal-runtime-comparison{suffix}.svg"
    description = ". ".join((text["hardware"], text["workload"], text["prefill_note"],
                              text["comparison_note"], REPOSITORY_URL))
    fig.savefig(output, metadata={"Title": text["title"], "Description": description,
                                  "Creator": "MFQ", "Date": None})
    svg = output.read_text()
    svg = svg.replace("<svg ", f'<svg role="img" aria-labelledby="title desc" xml:lang="{locale}" ', 1)
    svg = re.sub(r'width="[^"]+" height="[^"]+"', f'width="{WIDTH}" height="{HEIGHT}"', svg, count=1)
    svg = svg.replace(" <title>", ' <title id="title">', 1)
    svg = svg.replace(" <metadata>", f' <desc id="desc">{escape(description)}</desc>\n <metadata>', 1)
    for row, _ in ROWS:
        for metric in ("prefill", "generation_no_mtp", "generation_mtp"):
            key = selected_metric(row, metric)
            group = f'<g id="{row["runtime"]}-{row["method"]}-{row["tier"]}-{metric}">'
            assert group in svg
            title = f'{row["runtime"]} / {row["method"]} {row["tier"]}: {key} = {row[key]:.1f} tok/s'
            svg = svg.replace(group, group[:-1] + f' data-metric="{key}" data-value="{row[key]}"><title>{escape(title)}</title>', 1)
    output.write_text("\n".join(line.rstrip() for line in svg.splitlines()) + "\n")
    png = ROOT / "qwen38-flash-next-metal-runtime-comparison.png" if locale == "en" else Path("/tmp/mfq-metal-runtime-comparison.zh-CN.png")
    fig.savefig(png, dpi=108, facecolor=C["canvas"], metadata={"Title": text["title"], "Description": description})
    plt.close(fig)
    print(output)
    print(png)


if __name__ == "__main__":
    for locale, text in LANGUAGES.items():
        generate(locale, text)
