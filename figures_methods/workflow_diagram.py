#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a conceptual workflow diagram of the decadal shoreline-projection
methodology (Our Changing Coasts), following Sections 1-5 of the methods doc."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "hatch.linewidth": 0.6,
})

# ---- style: grayscale, journal-figure conventions --------------------------
FILL = {
    "input": "white",
    "process": "0.92",
    "mc": "0.80",
    "output": "0.75",
}
HATCH = {
    "output": "......",
}
HATCH_COLOR = (0.35, 0.35, 0.35, 0.45)  # grey, semi-transparent so overlaid text stays legible
EDGE = "black"


def box(ax, xy, w, h, text, kind, fontsize=9, fontweight="normal"):
    x, y = xy
    rect = Rectangle((x - w / 2, y - h / 2), w, h,
                      facecolor=FILL[kind], edgecolor=EDGE, linewidth=0.9, zorder=2)
    ax.add_patch(rect)
    if kind in HATCH:
        overlay = Rectangle((x - w / 2, y - h / 2), w, h,
                             facecolor="none", edgecolor=HATCH_COLOR,
                             linewidth=0, hatch=HATCH[kind], zorder=2.5)
        ax.add_patch(overlay)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
             fontweight=fontweight, zorder=3, linespacing=1.4)
    return (x, y, w, h)


def arrow(ax, start, end, style="-|>", color="black"):
    a = FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=10,
        linewidth=0.9, color=color, shrinkA=1, shrinkB=1, zorder=1,
    )
    ax.add_patch(a)


def bottom(b):
    x, y, w, h = b
    return (x, y - h / 2)


def top(b):
    x, y, w, h = b
    return (x, y + h / 2)


def left(b):
    x, y, w, h = b
    return (x - w / 2, y)


def right(b):
    x, y, w, h = b
    return (x + w / 2, y)


def build_diagram():
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 22.4)
    ax.axis("off")

    ax.text(5, 21.9, "Decadal shoreline-projection workflow", ha="center",
             fontsize=12.5, fontweight="bold")
    ax.text(5, 21.4, "Our Changing Coasts \u2014 Aotearoa New Zealand", ha="center",
             fontsize=9.5, style="italic")

    # Row 1: raw inputs
    b_input = box(ax, (5, 20.2), 3.2, 0.7, "Input data", "input", fontweight="bold")

    b_sat = box(ax, (2.6, 18.7), 3.6, 1.15,
                "Satellite shorelines\n+ beach slopes\n(CoastSat, 2000\u20132025)", "input")
    b_slr = box(ax, (7.4, 18.7), 3.6, 1.15,
                "NZ SeaRise SLR projections\n(SSP1-1.9 \u2013 SSP5-8.5;\n$P_{17}$ / $P_{50}$ / $P_{83}$)", "input")

    arrow(ax, bottom(b_input), top(b_sat))
    arrow(ax, bottom(b_input), top(b_slr))

    # Row 2: preprocessing
    b_hist = box(ax, (2.6, 17.0), 3.6, 1.05,
                 "Historical shoreline trends\n(gap screening + 10 yr LOESS)", "process")
    b_ds = box(ax, (7.4, 17.0), 3.6, 1.05,
               "$\\Delta S$ scenarios\n(rebased to 2025)", "process")

    arrow(ax, bottom(b_sat), top(b_hist))
    arrow(ax, bottom(b_slr), top(b_ds))

    # Row 3: bootstrap (left branch only)
    b_boot = box(ax, (2.6, 15.5), 3.6, 0.75,
                 "Block-bootstrap regression\n($B$ replicates)", "process")
    arrow(ax, bottom(b_hist), top(b_boot))

    # Row 4: rate distributions
    b_dist = box(ax, (2.6, 13.75), 3.8, 1.35,
                 "$p_{hist}(r)=\\mathcal{U}[r_{min},r_{max}]$\n"
                 "$p_{recent}(r)$  (last $\\sim$10 yr)\n"
                 "$p_1 = w\\,p_{recent}+(1-w)\\,p_{hist}$", "process")
    arrow(ax, bottom(b_boot), top(b_dist))

    # Row 5: components
    b_intrinsic = box(ax, (2.6, 11.9), 3.8, 1.1,
                       "Intrinsic component\n$i_{lt,k}=r_k^{(m)}\\,\\Delta t_k$", "process")
    b_extrinsic = box(ax, (7.4, 12.75), 3.8, 2.35,
                       "Extrinsic SLR component\n"
                       "$e_{slr,k}=-\\dfrac{c}{\\tan\\beta^{(m)}}\\,\\Delta S_k^{(m)}$\n"
                       "$\\tan\\beta^{(m)}\\sim\\mathcal{U}[0.8,1.2]\\tan\\beta_0$\n"
                       "$\\Delta S_k^{(m)}\\sim\\mathcal{N}(\\mu_{S,k},\\sigma_{S,k})$", "process")

    arrow(ax, bottom(b_dist), top(b_intrinsic))
    arrow(ax, bottom(b_ds), top(b_extrinsic))

    # Cross-link: beach slope feeds the active-profile slope in the SLR component
    route_x = 9.7
    sx, sy = right(b_sat)
    ex, ey = right(b_extrinsic)
    ax.plot([sx, route_x], [sy, sy], linestyle="--", linewidth=0.8, color="0.35", zorder=1)
    ax.plot([route_x, route_x], [sy, ey], linestyle="--", linewidth=0.8, color="0.35", zorder=1)
    arrow(ax, (route_x, ey), (ex, ey), color="0.35")
    ax.text(route_x + 0.1, (sy + ey) / 2, "beach\nslope", ha="left", va="center",
            fontsize=7.5, color="0.25", style="italic")

    # Row 6: Monte Carlo
    b_mc = box(ax, (5, 9.5), 5.6, 0.95,
               "Monte Carlo propagation ($M=2{,}000$ realisations per interval)",
               "mc", fontweight="bold", fontsize=9.5)
    arrow(ax, bottom(b_intrinsic), (3.4, 9.975))
    arrow(ax, bottom(b_extrinsic), (6.6, 9.975))

    # Row 7: compound change
    b_compound = box(ax, (5, 7.85), 5.8, 1.05,
                      "Compound shoreline change\n"
                      "$\\Delta y_k^{(m)} = r_k^{(m)}\\Delta t_k - \\dfrac{c}{\\tan\\beta^{(m)}}\\Delta S_k^{(m)}$",
                      "mc")
    arrow(ax, bottom(b_mc), top(b_compound))

    # Row 8: horizons
    b_horizons = box(ax, (5, 6.3), 5.8, 0.8,
                      "Cumulative projection horizons: 2030 $\\rightarrow$ 2040 $\\rightarrow$ 2050",
                      "mc")
    arrow(ax, bottom(b_compound), top(b_horizons))

    # Row 9: output
    b_out = box(ax, (5, 4.6), 5.8, 1.3,
                "Output\nMedian projection + uncertainty bounds\n"
                "(534 sites, $\\sim$24,000 transects)", "output", fontweight="bold")
    arrow(ax, bottom(b_horizons), top(b_out))

    # Legend
    legend_items = [
        ("Input data", "input"),
        ("Individual components", "process"),
        ("Uncertainty propagation", "mc"),
        ("Final output", "output"),
    ]
    ly = 2.9
    lx0 = 1.0
    for i, (label, kind) in enumerate(legend_items):
        lx = lx0 + i * 2.2
        ax.add_patch(Rectangle((lx - 0.15, ly - 0.15), 0.3, 0.3,
                                facecolor=FILL[kind], edgecolor="black", linewidth=0.8))
        if kind in HATCH:
            ax.add_patch(Rectangle((lx - 0.15, ly - 0.15), 0.3, 0.3,
                                    facecolor="none", edgecolor=HATCH_COLOR,
                                    linewidth=0, hatch=HATCH[kind]))
        ax.text(lx + 0.25, ly, label, ha="left", va="center", fontsize=8)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    fig = build_diagram()
    fig.savefig("workflow_diagram.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig("workflow_diagram.pdf", bbox_inches="tight", facecolor="white")
    print("Saved workflow_diagram.png and workflow_diagram.pdf")
