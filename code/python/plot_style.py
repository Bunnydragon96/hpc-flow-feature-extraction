"""
Matplotlib style for report figures (copy next to make_plots.py and `import plot_style`).

Figures are saved as vector PDF at their FINAL printed size, so text in the
PDF is 7-9 pt exactly like the surrounding report:
    full text width (1 in margins, letter)  -> figsize width 6.5 in
    half width (two side by side)           -> figsize width 3.3 in
Include them with width=\\linewidth (or the minipage width): no rescaling,
so fonts never shrink.

Colours were checked with the dataviz skill's validator (light mode, white
surface). Use them by role:
    SEQ_COLOR  - the sequential / baseline series (categorical orange)
    RAMP5      - an ORDERED series such as p or n (single-hue blue ramp,
                 at most 5 steps; more than 5 fails the adjacent-step check,
                 so show 5 in the plot and the rest in a table)
    CAT4       - unordered categories, e.g. stacked time components (max 4)
    MUTED      - reference lines (ideal speedup, S = 1)
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#9a9993", "#e6e5e1"
SHADE = "#f3f2ef"                                                 # background band (e.g. "E-cores in use")
RAMP5 = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]  # light -> dark, ordinal
SEQ_COLOR = "#eb6834"
CAT4 = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]               # blue, orange, aqua, yellow
FULL_W, HALF_W = 6.5, 3.3

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 1.5, "lines.markersize": 5,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42,  # embed TrueType: text stays selectable/searchable in the PDF
})

# Keyword arguments for line plots: round markers with a thin white ring so
# overlapping points stay distinguishable.
LINE = dict(marker="o", markeredgecolor="white", markeredgewidth=0.8)


def log2_axis(ax, ticks):
    """Log-2 x axis with readable tick labels (512, 1K, 2K, ...)."""
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:,}" if t < 1000 else f"{t // 1024}K" for t in ticks])
    ax.minorticks_off()


def shared_legend_below(fig, ax, ncol=None, pad=0.035):
    """One legend for a multi-panel figure, centred under the panels."""
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=ncol or len(l), bbox_to_anchor=(0.5, -pad))
    fig.tight_layout(rect=(0, pad, 1, 1))


def subplots(nrows, ncols, **kw):
    """plt.subplots that always returns a 2-D axes array (no squeeze surprises
    when a configuration list has a single element)."""
    return plt.subplots(nrows, ncols, squeeze=False, **kw)
