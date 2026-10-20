"""Redraw the published quality comparison with MFQ Studio's figure theme.

uv run --no-project --no-config --with matplotlib python docs/figures/generate-quantization-quality.py
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
from matplotlib.patches import Circle, FancyBboxPatch, PathPatch
from matplotlib.path import Path as PlotPath
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator


ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / "data/qwen38-flash-next-quality.json").read_text())
C = json.loads((ROOT / "studio-figure-theme.json").read_text())["colors"]
REPOSITORY_URL = "https://github.com/Tylogi/MFQ"
WIDTH, HEIGHT = 1600, 1020

families = ["DejaVu Sans"]
for font_path in (
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
):
    if font_path.exists():
        font_manager.fontManager.addfont(font_path)
        family = font_manager.FontProperties(fname=font_path).get_name()
        if family not in families:
            families.append(family)

plt.rcParams.update({
    "font.family": families,
    "font.size": 17,
    "svg.fonttype": "path",
    "svg.hashsalt": "mfq-flash-next-quality",
    "axes.unicode_minus": False,
    "text.color": C["ink"],
    "xtick.color": C["muted_light"],
    "ytick.color": C["muted_light"],
})

STYLES = {
    "mfq": {"color": C["accent"], "label_color": C["accent_ink"], "marker": "o", "linestyle": "-", "width": 5.2, "marker_size": 14, "label_size": 19},
    "unsloth-ud": {"color": "#6da993", "label_color": "#3f806c", "marker": "s", "linestyle": (0, (5, 3)), "width": 3, "marker_size": 10.5, "label_size": 17.5},
    "gsq-rco": {"color": "#8b9fd0", "label_color": "#5c74ad", "marker": "D", "linestyle": "-", "width": 3, "marker_size": 10.5, "label_size": 17.5},
}
LANGUAGES = {
    "en": {
        "title": "Quantization quality",
        "subtitle": f"{DATA['model']} · WikiText-2",
        "kld": "Mean KL divergence",
        "kld_hint": "Lower is better · log scale",
        "top1": "Top-1 agreement",
        "top1_hint": "Higher is better",
        "size": "Weight size (GB)",
        "exclusions": "Weight sizes exclude the vision encoder and MTP head. GB = 10⁹ bytes.",
        "protocol": "MFQ · EW / GSQ-RCO: WT2 · 580 × 512 · 147,900 positions.",
    },
    "zh-CN": {
        "title": "量化质量",
        "subtitle": f"{DATA['model']} · WikiText-2",
        "kld": "平均 KL 散度",
        "kld_hint": "越低越好 · 对数刻度",
        "top1": "Top-1 一致率",
        "top1_hint": "越高越好",
        "size": "权重大小（GB）",
        "exclusions": "权重大小不含视觉编码器与 MTP 层。GB = 10⁹ 字节。",
        "protocol": "MFQ · EW / GSQ-RCO：WT2 · 580 × 512 · 147,900 个位置。",
    },
}

# Every point shows its tier and metric in its series color. Offsets
# keep labels outside the neighboring curves without moving any data points.
OFFSETS = {
    "kld": {
        "mfq": {"V1-S": (-32, -20, "center", "top"), "V1-M": (0, -48, "center", "top"),
                "S2-L": (-32, -20, "center", "top"), "V3-XXS": (0, -32, "center", "top"),
                "S3-L": (-16, -20, "center", "top"), "V4-XS": (8, -12, "center", "top")},
        "unsloth-ud": {"IQ1_S": (0, 12, "center", "bottom"), "IQ1_M": (32, 20, "center", "bottom"),
                       "Q2_K_XL": (32, 0, "left", "center"), "IQ3_XXS": (8, 12, "left", "bottom"),
                       "Q3_K_XL": (32, 20, "center", "bottom"), "IQ4_XS": (8, 32, "center", "bottom")},
        "gsq-rco": {"Q2_0": (16, 12, "center", "bottom"), "IQ2_XS": (-8, -32, "center", "top"),
                    "IQ3_XXS": (32, 48, "left", "bottom"), "IQ3_S": (32, 12, "left", "bottom")},
    },
    "top1_agreement_percent": {
        "mfq": {"V1-S": (-32, 12, "center", "bottom"), "V1-M": (-8, 48, "center", "bottom"),
                "S2-L": (0, 32, "center", "bottom"), "V3-XXS": (32, 48, "center", "bottom"),
                "S3-L": (-8, 20, "center", "bottom"), "V4-XS": (8, 12, "center", "bottom")},
        "unsloth-ud": {"IQ1_S": (0, -12, "center", "top"), "IQ1_M": (32, -12, "center", "top"),
                       "Q2_K_XL": (32, 0, "left", "center"), "IQ3_XXS": (8, -12, "left", "top"),
                       "Q3_K_XL": (0, -32, "center", "top"), "IQ4_XS": (8, -20, "center", "top")},
        "gsq-rco": {"Q2_0": (16, -12, "center", "top"), "IQ2_XS": (32, -12, "center", "top"),
                    "IQ3_XXS": (32, -48, "left", "top"), "IQ3_S": (64, -48, "left", "top")},
    },
}


def bounds(x, y, width, height):
    return x / WIDTH, 1 - (y + height) / HEIGHT, width / WIDTH, height / HEIGHT


def label(fig, x, y, value, size=17, color=None, weight="normal", **kwargs):
    return fig.text(x / WIDTH, 1 - y / HEIGHT, value, fontsize=size,
                    color=color or C["ink"], weight=weight, **kwargs)


def card(fig, x):
    fig.add_artist(FancyBboxPatch(
        bounds(x, 128, 740, 772)[:2], 740 / WIDTH, 772 / HEIGHT,
        boxstyle=f"round,pad=0,rounding_size={12 / WIDTH}",
        mutation_aspect=WIDTH / HEIGHT, transform=fig.transFigure,
        facecolor=C["surface"], edgecolor=C["line"], linewidth=0.8, zorder=-1,
    ))
    fig.add_artist(Line2D(
        [(x + 28) / WIDTH, (x + 712) / WIDTH], [1 - 228 / HEIGHT] * 2,
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


def panel(fig, x, metric, text):
    card(fig, x)
    short = "kld" if metric == "kld" else "top1"
    label(fig, x + 28, 174, text[short], 25, C["strong"], "bold")
    label(fig, x + 28, 204, text[f"{short}_hint"], 17, C["muted"])
    ax = fig.add_axes(bounds(x + 92, 244, 616, 578), facecolor="none")
    ax.set_xlim(65, 97)
    ax.set_xticks((66, 70, 74, 78, 82, 86, 90, 94))
    if metric == "kld":
        ax.set_yscale("log")
        ax.set_ylim(0.06, 0.5)
        ax.yaxis.set_major_locator(FixedLocator((0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.yaxis.set_minor_locator(NullLocator())
    else:
        ax.set_ylim(75, 93)
        ax.set_yticks(range(76, 93, 2))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}%"))
    ax.set_axisbelow(True)
    ax.grid(color=C["line_soft"], linewidth=0.7)
    ax.tick_params(axis="both", length=0, pad=10, labelsize=17)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel(text["size"], fontsize=19, color=C["soft"], labelpad=18)

    for series in DATA["series"]:
        style = STYLES[series["id"]]
        rows = series["measurements"]
        line, = ax.plot([r["weight_size_gb"] for r in rows], [r[metric] for r in rows],
                        color=style["color"], linewidth=style["width"],
                        linestyle=style["linestyle"], solid_capstyle="round",
                        zorder=3 if series["id"] == "mfq" else 2)
        line.set_gid(f'quality-{short}-{series["id"]}-line')
        for row in rows:
            point, = ax.plot(row["weight_size_gb"], row[metric], linestyle="none",
                             marker=style["marker"], markersize=style["marker_size"],
                             color=style["color"], markeredgecolor=C["surface"],
                             markeredgewidth=1.6, zorder=5 if series["id"] == "mfq" else 4)
            point.set_gid(f'quality-{short}-{series["id"]}-{row["tier"]}')
            dx, dy, align, vertical = OFFSETS[metric][series["id"]][row["tier"]]
            value = f'{row[metric]:.5f}' if metric == "kld" else f'{row[metric]:.2f}%'
            caption = f'{row["tier"]}\n{value}'
            annotation = ax.annotate(caption, (row["weight_size_gb"], row[metric]),
                                    xytext=(dx, dy), textcoords="offset points", ha=align, va=vertical,
                                    color=style["label_color"], linespacing=1.15,
                                    weight="bold" if series["id"] == "mfq" else "normal", fontsize=style["label_size"],
                                    arrowprops={"arrowstyle": "-", "linewidth": 0.85,
                                                "color": style["color"], "shrinkA": 4, "shrinkB": 10})
            annotation.set_gid(f'quality-{short}-label-{series["id"]}-{row["tier"]}')
    return ax


def generate(locale, text):
    fig = plt.figure(figsize=(WIDTH / 72, HEIGHT / 72), facecolor=C["canvas"])
    label(fig, 48, 63, text["title"], 32, C["strong"], "bold")
    label(fig, 48, 96, text["subtitle"], 18, C["muted"])
    brand(fig)
    handles = [Line2D([], [], color=STYLES[s["id"]]["color"], marker=STYLES[s["id"]]["marker"],
                      linestyle=STYLES[s["id"]]["linestyle"], linewidth=STYLES[s["id"]]["width"],
                      markersize=STYLES[s["id"]]["marker_size"] * 0.8) for s in DATA["series"]]
    fig.legend(handles, [s["label"] for s in DATA["series"]], loc="center right",
               bbox_to_anchor=(1564 / WIDTH, 1 - 91 / HEIGHT), ncol=3,
               frameon=False, fontsize=18, handlelength=2, columnspacing=1.6)
    panel(fig, 48, "kld", text)
    panel(fig, 812, "top1_agreement_percent", text)
    label(fig, 48, 926, text["exclusions"], 14, C["muted_light"])
    label(fig, 48, 952, text["protocol"], 14, C["muted_light"])
    footer = label(fig, 1552, 996, REPOSITORY_URL, 14, C["muted_light"], ha="right")
    footer.set_gid("repository-url")
    footer.set_url(REPOSITORY_URL)

    suffix = "" if locale == "en" else ".zh-CN"
    target = ROOT / f"qwen38-flash-next-quality{suffix}.svg"
    description = ". ".join((text["subtitle"], text["kld_hint"], text["top1_hint"],
                              text["exclusions"], text["protocol"], REPOSITORY_URL))
    fig.savefig(target, metadata={"Title": text["title"], "Description": description,
                                  "Creator": "MFQ", "Date": None})
    svg = target.read_text()
    svg = svg.replace("<svg ", f'<svg role="img" aria-labelledby="title desc" xml:lang="{locale}" ', 1)
    svg = re.sub(r'width="[^"]+" height="[^"]+"', f'width="{WIDTH}" height="{HEIGHT}"', svg, count=1)
    svg = svg.replace(" <title>", ' <title id="title">', 1)
    svg = svg.replace(" <metadata>", f' <desc id="desc">{escape(description)}</desc>\n <metadata>', 1)
    for short in ("kld", "top1"):
        for series in DATA["series"]:
            for row in series["measurements"]:
                group = f'<g id="quality-{short}-{series["id"]}-{row["tier"]}">'
                assert group in svg
                title = (f'{series["label"]} {row["tier"]}: {row["weight_size_gb"]} GB, '
                         f'KLD {row["kld"]:.5f}, Top-1 {row["top1_agreement_percent"]:.2f}%')
                attributes = (f' data-size-gb="{row["weight_size_gb"]}" data-kld="{row["kld"]}"'
                              f' data-top1="{row["top1_agreement_percent"]}"')
                svg = svg.replace(group, group[:-1] + attributes + f'><title>{escape(title)}</title>', 1)
    target.write_text("\n".join(line.rstrip() for line in svg.splitlines()) + "\n")
    png = ROOT / "qwen38-flash-next-quality.png" if locale == "en" else Path("/tmp/mfq-quantization-quality.zh-CN.png")
    fig.savefig(png, dpi=108, facecolor=C["canvas"], metadata={"Title": text["title"], "Description": description})
    plt.close(fig)
    print(target)
    print(png)


if __name__ == "__main__":
    for locale, text in LANGUAGES.items():
        generate(locale, text)
