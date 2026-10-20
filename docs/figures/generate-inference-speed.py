"""Regenerate the bilingual SVGs from the published MFQ benchmark snapshot.

uv run --no-project --no-config --with matplotlib python docs/figures/generate-inference-speed.py
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
DATA = json.loads((ROOT / "data/qwen38-flash-next-inference-speed.json").read_text())
THEME = json.loads((ROOT / "studio-figure-theme.json").read_text())
C = THEME["colors"]
REPOSITORY_URL = "https://github.com/Tylogi/MFQ"
WIDTH, HEIGHT = 1600, 1080
CARD_WIDTH = 740

# Register the local CJK fonts when available. SVG glyphs are saved as paths,
# so the exported files do not depend on the viewer having these fonts.
font_families = ["DejaVu Sans"]
for font_path in (
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
):
    if font_path.exists():
        font_manager.fontManager.addfont(font_path)
        family = font_manager.FontProperties(fname=font_path).get_name()
        if family not in font_families:
            font_families.append(family)

plt.rcParams.update(
    {
        "font.family": font_families,
        "font.size": 17,
        "svg.fonttype": "path",
        "svg.hashsalt": "mfq-flash-next-inference-speed",
        "axes.unicode_minus": False,
        "axes.edgecolor": C["line"],
        "text.color": C["ink"],
        "xtick.color": C["muted_light"],
        "ytick.color": C["soft"],
    }
)

COLORS = {"plain": C["neutral_bar"], "mtp": C["accent"], "prefill": C["muted"]}
LANGUAGES = {
    "en": {
        "title": "Inference speed",
        "subtitle": f"{DATA['model']} · {DATA['input_tokens']:,}-token input",
        "generation": "Generation (tok/s)",
        "prefill": "Prefill (tok/s)",
        "plain": "MTP off",
        "mtp": "MTP on",
        "metal": "M5 Max · Metal",
        "metal_details": "40-core GPU · 128 GB",
        "cuda": "RTX 3090 Ti · CUDA",
        "cuda_details": "RAM: 32 GiB · S3-L / V4-XS: 64 GiB",
        "footnote2": "Generation bars compare MTP settings. Hardware and RAM configurations are part of each result.",
    },
    "zh-CN": {
        "title": "推理速度",
        "subtitle": f"{DATA['model']} · 输入 {DATA['input_tokens']:,} tokens",
        "generation": "生成速度（tok/s）",
        "prefill": "提示词处理（tok/s）",
        "plain": "关闭 MTP",
        "mtp": "开启 MTP",
        "metal": "M5 Max · Metal",
        "metal_details": "40 核 GPU · 128 GB",
        "cuda": "RTX 3090 Ti · CUDA",
        "cuda_details": "内存：32 GiB · S3-L / V4-XS：64 GiB",
        "footnote2": "生成速度的两根柱分别对应 MTP 开关；硬件与内存配置是各项结果的一部分。",
    },
}


def bounds(x, y, width, height):
    """Use top-left canvas coordinates for both plots and UI elements."""
    return x / WIDTH, 1 - (y + height) / HEIGHT, width / WIDTH, height / HEIGHT


def label(fig, x, y, value, size=17, color=None, weight="normal", **kwargs):
    return fig.text(x / WIDTH, 1 - y / HEIGHT, value, fontsize=size,
                    color=color or C["ink"], weight=weight, **kwargs)


def surface(fig, x, y, width, height, fill, border=None, radius=12, zorder=-1):
    patch = FancyBboxPatch(
        bounds(x, y, width, height)[:2], width / WIDTH, height / HEIGHT,
        boxstyle=f"round,pad=0,rounding_size={radius / WIDTH}",
        mutation_aspect=WIDTH / HEIGHT, transform=fig.transFigure,
        facecolor=fill, edgecolor=border or "none", linewidth=0.8, zorder=zorder,
    )
    fig.add_artist(patch)


def rule(fig, x, y, width):
    fig.add_artist(Line2D(
        [x / WIDTH, (x + width) / WIDTH], [1 - y / HEIGHT] * 2,
        transform=fig.transFigure, color=C["line_soft"], linewidth=0.8,
    ))


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


def panel(fig, card_x, backend, metric):
    is_generation = metric == "generation"
    top, plot_height = (282, 324) if is_generation else (716, 264)
    plot_width = 608
    ax = fig.add_axes(bounds(card_x + 100, top, plot_width, plot_height), facecolor="none")
    rows = backend["measurements"]
    limit = 120 if is_generation else 3000
    ticks = range(0, 121, 20) if is_generation else (0, 1000, 2000, 3000)

    def bar(y, value, height, color, key, row):
        # Keep the data scale linear while giving bars the UI's subtle corner radius.
        pixels_per_x = plot_width / limit
        pixels_per_y = plot_height / len(rows)
        patch = FancyBboxPatch(
            (0, y - height / 2), value, height,
            boxstyle=f"round,pad=0,rounding_size={3 / pixels_per_x}",
            mutation_aspect=pixels_per_x / pixels_per_y,
            facecolor=color, linewidth=0, zorder=3,
        )
        patch.set_gid(f"{backend['id']}-{row['tier']}-{key}")
        ax.add_patch(patch)

    for i, row in enumerate(rows):
        if row["tier"] == "V3-XXS":
            row_height = plot_height / len(rows)
            center = top + (i + 0.55) * row_height
            surface(fig, card_x + 16, center - row_height * 0.46,
                    CARD_WIDTH - 32, row_height * 0.92, C["accent_soft"], radius=6, zorder=0)
        if is_generation:
            for offset, key, color in (
                (-0.17, "generation_no_mtp", COLORS["plain"]),
                (0.17, "generation_mtp", COLORS["mtp"]),
            ):
                value = row[key]
                bar(i + offset, value, 0.24, color, key, row)
                ink = C["muted"] if key == "generation_no_mtp" else C["accent_ink"]
                ax.text(value + 1.4, i + offset, f"{value:.1f}", va="center", fontsize=17, color=ink)
        else:
            value = row["prefill"]
            color = C["accent"] if row["tier"] == "V3-XXS" else COLORS["prefill"]
            bar(i, value, 0.32, color, "prefill", row)
            ax.text(value + 42, i, f"{value:,.1f}", va="center", fontsize=17,
                    color=C["accent_ink"] if row["tier"] == "V3-XXS" else C["muted"])

    labels = [row["tier"] for row in rows]
    ax.set_yticks(range(len(rows)), labels, fontsize=17)
    for tick, row in zip(ax.get_yticklabels(), rows):
        if row["tier"] == "V3-XXS":
            tick.set_fontweight("bold")
            tick.set_color(C["accent_ink"])
    ax.set_xlim(0, limit)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.set_ylim(len(rows) - 0.55, -0.55)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=C["line_soft"], linewidth=0.7)
    ax.tick_params(axis="y", length=0, pad=14)
    ax.tick_params(axis="x", length=0, pad=10, labelsize=14)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def generate(locale, text):
    fig = plt.figure(figsize=(WIDTH / 72, HEIGHT / 72), facecolor=C["canvas"])
    label(fig, 48, 63, text["title"], 30, C["strong"], "bold")
    label(fig, 48, 96, text["subtitle"], 17, C["muted"])
    brand(fig)
    fig.legend(
        [Patch(color=COLORS["plain"]), Patch(color=COLORS["mtp"])],
        [text["plain"], text["mtp"]],
        loc="center right", bbox_to_anchor=(1564 / WIDTH, 1 - 91 / HEIGHT), ncol=2,
        frameon=False, fontsize=16, handlelength=1.2, columnspacing=1.8,
    )

    for left, backend in zip((48, 812), DATA["backends"]):
        surface(fig, left, 128, CARD_WIDTH, 892, C["surface"], C["line"])
        label(fig, left + 28, 173, text[backend["id"]], 23, C["strong"], "bold")
        label(fig, left + 28, 202, text[f"{backend['id']}_details"], 15, C["muted"])
        rule(fig, left + 28, 224, CARD_WIDTH - 56)
        label(fig, left + 28, 261, text["generation"], 18, C["soft"])
        panel(fig, left, backend, "generation")
        rule(fig, left + 28, 655, CARD_WIDTH - 56)
        label(fig, left + 28, 693, text["prefill"], 18, C["soft"])
        panel(fig, left, backend, "prefill")

    repository_label = label(fig, 1552, 1056, REPOSITORY_URL, 14, C["muted_light"], ha="right")
    repository_label.set_gid("repository-url")
    repository_label.set_url(REPOSITORY_URL)

    suffix = "" if locale == "en" else ".zh-CN"
    target = ROOT / f"qwen38-flash-next-inference-speed{suffix}.svg"
    fig.savefig(target, metadata={"Title": text["title"], "Description": text["subtitle"], "Creator": "MFQ", "Date": None})
    svg = target.read_text()
    svg = svg.replace("<svg ", '<svg role="img" aria-labelledby="title desc" ', 1)
    svg = re.sub(r'width="[^"]+" height="[^"]+"', f'width="{WIDTH}" height="{HEIGHT}"', svg, count=1)
    svg = svg.replace(" <title>", ' <title id="title">', 1)
    description = ". ".join((text["subtitle"], text["metal"], text["metal_details"], text["cuda"], text["cuda_details"], text["footnote2"], REPOSITORY_URL))
    svg = svg.replace(" <metadata>", f' <desc id="desc">{escape(description)}</desc>\n <metadata>', 1)
    # Preserve exact values in the shareable artifact as well as its source snapshot.
    for backend in DATA["backends"]:
        for row in backend["measurements"]:
            for key in ("generation_no_mtp", "generation_mtp", "prefill"):
                group = f'<g id="{backend["id"]}-{row["tier"]}-{key}">'
                assert group in svg
                svg = svg.replace(group, group[:-1] + f' data-value="{row[key]}">', 1)
    target.write_text("\n".join(line.rstrip() for line in svg.splitlines()) + "\n")
    preview = Path(f"/tmp/mfq-inference-speed{suffix or '.en'}.png")
    fig.savefig(preview, dpi=72, facecolor=C["canvas"])
    plt.close(fig)
    print(target)
    print(preview)


if __name__ == "__main__":
    for locale, text in LANGUAGES.items():
        generate(locale, text)
