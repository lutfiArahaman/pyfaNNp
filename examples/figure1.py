"""Generate Figure 1 of the paper.

    python examples/figure1.py [--simulations 1000] [--scale 0.10]

Produces ``figures/figure1.png`` (300 dpi) and ``figures/figure1.pdf``
(vector, for typesetting), plus ``figures/figure1_data.csv`` carrying every
plotted number so the figure has a table view rather than being readable by
colour alone.

Layout
------
Top row, spanning: PROMETHEE II net flows per alternative under the two
weight-derivation methods. This is the honest contrast. Comparing "baseline"
against "coupled pipeline" would show two near-identical sets of bars,
because under the surrogate coupling the network approximates the outranking
rather than changing it -- the pipeline's value there is cost on large
alternative sets, not a different answer. What *does* move the ranking is the
choice of weighting method, and showing that is the more useful panel.

Bottom row: rank distribution across Monte Carlo perturbations of the
criterion weights, one heatmap per method. This is the panel that carries the
argument. A manual workflow reports a single ordering; these show how much of
that ordering survives the imprecision already present in the expert
judgements.

Colour
------
The two series use the Okabe-Ito blue and orange, a published
colour-vision-deficiency-safe pair, rather than a palette chosen here by eye.
The heatmaps use a single-hue sequential ramp light-to-dark, since their
quantity is a magnitude with no meaningful midpoint.
"""

from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this runs in CI

import matplotlib.pyplot as plt
import numpy as np

from pyfap import FAHP, DecisionPipeline, Promethee
from pyfap.datasets import load_demo
from pyfap.preprocessing import minmax_normalize

# Okabe & Ito (2008), "Color Universal Design" -- safe under deuteranopia,
# protanopia and tritanopia.
BLUE = "#0072B2"
ORANGE = "#E69F00"

INK = "#222222"
MUTED = "#666666"
GRID = "#DDDDDD"

METHODS = [
    ("extent_analysis", "Extent analysis (Chang)", BLUE),
    ("geometric_mean", "Geometric mean (Buckley)", ORANGE),
]


def build(problem, method):
    """Run the pipeline for one weight-derivation method."""
    pipe = DecisionPipeline(
        weights=FAHP(method=method, consistency_check=True),
        ranker=Promethee(
            version="II",
            preference="v-shape",
            q=0.1,
            p=0.5,
            criteria_types=problem.criteria_types,
        ),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = pipe.fit_rank(
            judgments=problem.judgments,
            decision_matrix=minmax_normalize(problem.decision_matrix),
            alternatives=problem.alternatives,
            criteria=problem.criteria,
        )
    notes = [str(w.message) for w in caught]
    return result, notes


def style_axis(ax):
    """Recessive frame: the data should be the darkest thing on the panel."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9, length=3, width=0.8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(INK)


def panel_net_flows(ax, problem, results):
    """Grouped bars: net flow per alternative, one group per method."""
    labels = problem.alternatives
    x = np.arange(len(labels))
    width = 0.38

    for i, (key, name, colour) in enumerate(METHODS):
        offset = (i - 0.5) * width
        ax.bar(
            x + offset,
            results[key].net_flow,
            width * 0.94,  # 2px-equivalent gap between adjacent bars
            label=name,
            color=colour,
            edgecolor="white",
            linewidth=0.6,
        )

    ax.axhline(0.0, color=MUTED, linewidth=0.9, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("PROMETHEE II net flow  $\\phi$", fontsize=10, color=INK)
    ax.set_xlabel("Alternative", fontsize=10, color=INK)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    style_axis(ax)

    legend = ax.legend(
        frameon=False, fontsize=9, loc="upper right", ncol=2,
        handlelength=1.2, handleheight=1.0,
    )
    for text in legend.get_texts():
        text.set_color(INK)

    ax.set_title(
        "Net flows depend on how the criterion weights were derived",
        fontsize=11, color=INK, loc="left", pad=10,
    )


def panel_stability(ax, problem, report, name, colour_map, show_ylabel):
    """Heatmap: P(alternative takes rank r) across the simulations."""
    n = len(problem.alternatives)
    freq = report.rank_counts / report.ranks.shape[0]

    image = ax.imshow(
        freq.T,  # rows = rank, columns = alternative
        cmap=colour_map,
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
        origin="upper",
    )

    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(problem.alternatives)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels([str(r + 1) for r in range(n)])
    ax.set_xlabel("Alternative", fontsize=10, color=INK)
    if show_ylabel:
        ax.set_ylabel("Rank", fontsize=10, color=INK)

    # Selective labels only: annotating all 36 cells would be noise.
    for a in range(n):
        for r in range(n):
            value = freq[a, r]
            if value >= 0.15:
                ax.text(
                    a, r, f"{value:.0%}",
                    ha="center", va="center", fontsize=8,
                    color="white" if value > 0.55 else INK,
                )

    # Thin white separators so adjacent cells read as distinct marks.
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    style_axis(ax)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)

    ax.set_title(
        f"{name}\nfull ordering changes in "
        f"{report.rank_reversal_rate:.0%} of runs",
        fontsize=10, color=INK, loc="left", pad=8,
    )
    return image


def write_table(path, problem, results, reports):
    """The figure's table view -- every plotted number, as CSV."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["panel", "method", "alternative", "quantity", "value"])

        for key, name, _ in METHODS:
            result = results[key]
            for i, alt in enumerate(problem.alternatives):
                writer.writerow(
                    ["A", name, alt, "net_flow", f"{result.net_flow[i]:.6f}"]
                )
            for i, criterion in enumerate(problem.criteria):
                writer.writerow(
                    ["A", name, criterion, "weight", f"{result.weights[i]:.6f}"]
                )

        for key, name, _ in METHODS:
            report = reports[key]
            total = report.ranks.shape[0]
            for a, alt in enumerate(problem.alternatives):
                for r in range(len(problem.alternatives)):
                    writer.writerow(
                        [
                            "B", name, alt,
                            f"P(rank={r + 1})",
                            f"{report.rank_counts[a, r] / total:.6f}",
                        ]
                    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulations", type=int, default=1000)
    parser.add_argument("--scale", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    problem = load_demo()

    results, reports, notes = {}, {}, {}
    for key, name, _ in METHODS:
        result, warned = build(problem, key)
        results[key] = result
        notes[key] = warned
        reports[key] = result.stability(
            n=args.simulations, scale=args.scale, random_state=args.seed
        )

    fig = plt.figure(figsize=(7.5, 8.2))
    grid = fig.add_gridspec(
        2, 2, height_ratios=[1.0, 1.15], hspace=0.42, wspace=0.14,
        left=0.10, right=0.88, top=0.92, bottom=0.09,
    )

    panel_net_flows(fig.add_subplot(grid[0, :]), problem, results)

    image = None
    for i, (key, name, _) in enumerate(METHODS):
        ax = fig.add_subplot(grid[1, i])
        image = panel_stability(
            ax, problem, reports[key], name, "Blues", show_ylabel=(i == 0)
        )

    bar = fig.colorbar(image, ax=fig.axes[1:], fraction=0.030, pad=0.02)
    bar.set_label("Proportion of simulations", fontsize=9, color=INK)
    bar.ax.tick_params(colors=MUTED, labelsize=8, length=3)
    bar.outline.set_visible(False)

    for suffix in ("png", "pdf"):
        path = args.outdir / f"figure1.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"wrote {path}")

    table = args.outdir / "figure1_data.csv"
    write_table(table, problem, results, reports)
    print(f"wrote {table}")

    # Everything the caption needs to state, printed rather than inferred.
    print("\n--- values behind the figure ---")
    for key, name, _ in METHODS:
        result, report = results[key], reports[key]
        print(f"\n{name}")
        print("  weights:  " + "  ".join(
            f"{c}={w:.4f}" for c, w in zip(problem.criteria, result.weights)
        ))
        cr = result.consistency_ratio
        print(f"  consistency ratio: {cr:.4f}" if cr is not None else "  CR: n/a")
        print(f"  ranking:  {' > '.join(result.ranking)}")
        print(f"  rank reversal rate: {report.rank_reversal_rate:.1%} "
              f"of {args.simulations} runs at +/-{args.scale:.0%}")
        for note in notes[key]:
            print(f"  NOTE: {note}")


if __name__ == "__main__":
    main()
