"""The showcase cover: one night, SpO2 over snore bars, 4:3, static PNG."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .analysis import minute_grid  # noqa: E402
from .night import Night  # noqa: E402
from .report import STAGE_Y  # noqa: E402

BG, INK, INK2, RULE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, VIOLET = "#2a78d6", "#eb6834", "#6250d6"


def render_cover(night: Night, out: Path) -> Path:
    tz = night.meta.get("timezone", "UTC")
    g = minute_grid(night)
    x = g.index.tz_convert(tz).tz_localize(None)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.edgecolor": RULE, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(12, 9), dpi=100, sharex=True,
                                        gridspec_kw={"height_ratios": [1, 2.2, 1.1], "hspace": 0.12}, facecolor=BG)
    for ax in (ax0, ax1, ax2):
        ax.set_facecolor(BG)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color=RULE, linewidth=1)

    xs, ys = [], []
    for s, e, st in night.stages.itertuples(index=False):
        xs += [s.tz_convert(tz).tz_localize(None), e.tz_convert(tz).tz_localize(None)]
        ys += [STAGE_Y.get(st, 1)] * 2
    ax0.plot(xs, ys, color=VIOLET, linewidth=2, drawstyle="steps-post")
    ax0.set_yticks([0, 1, 2, 3], ["Deep", "Light", "REM", "Awake"])
    ax0.set_ylim(-0.4, 3.4)

    ax1.plot(x, g["spo2"], color=BLUE, linewidth=2)
    ax1.set_ylabel("SpO2 %")
    ax1.set_ylim(min(89, np.nanmin(g["spo2"]) - 0.5), 99)

    ax2.bar(x, g["snore_s"], width=1 / 1440 * 0.9, color=ORANGE, linewidth=0)
    ax2.set_ylabel("snoring\ns / min")
    ax2.set_ylim(0, 60)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=None))

    fig.text(0.075, 0.955, "DO I SNORE?", fontsize=12, color=INK2, fontweight="bold", family="DejaVu Sans Mono")
    fig.text(0.075, 0.915, "Snore bouts (orange) and the SpO2 dips that follow them (blue)", fontsize=19, color=INK)
    fig.text(0.075, 0.025, "Sleep Lab · synthetic demo night · not a medical device", fontsize=11, color=INK2)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.88, bottom=0.08)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    return out
