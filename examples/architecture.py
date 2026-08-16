"""Generate the package architecture diagram.

    python examples/architecture.py

Produces ``figures/architecture.{png,pdf}``.

The diagram is drawn rather than hand-authored so it stays in the same
toolchain as the other figures, regenerates in CI, and cannot drift out of
step with the code silently. Box labels are the actual public names in the
package; if one is renamed, rename it here.

What the diagram has to carry: that the three stages share one data contract
and one object. The dashed enclosure is the argument -- the individual
methods are all published elsewhere, and what the package contributes is
their composition.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from _style import BLUE, GREEN, INK, MUTED, ORANGE, PURPLE

NEUTRAL = "#8A8A8A"

BOX_HALF_HEIGHT = 0.65


def box(ax, x, y, width, name, detail, colour, half_height=BOX_HALF_HEIGHT):
    """A rounded stage box: bold public name above, what it does below."""
    patch = FancyBboxPatch(
        (x - width / 2, y - half_height),
        width,
        half_height * 2,
        boxstyle="round,pad=0.02,rounding_size=0.14",
        facecolor=to_rgba(colour, 0.10),
        edgecolor=colour,
        linewidth=1.6,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        x, y + half_height * 0.34, name,
        ha="center", va="center", fontsize=10.5, color=INK,
        fontweight="bold", family="monospace", zorder=4,
    )
    ax.text(
        x, y - half_height * 0.28, detail,
        ha="center", va="center", fontsize=8.4, color=MUTED,
        linespacing=1.45, zorder=4,
    )
    return x - width / 2, x + width / 2


def arrow(ax, start, end, label=None, label_offset=(0.0, 0.18), ha="center"):
    ax.add_patch(
        FancyArrowPatch(
            start, end,
            arrowstyle="-|>", mutation_scale=13,
            color=MUTED, linewidth=1.4,
            shrinkA=2, shrinkB=2, zorder=2,
        )
    )
    if label:
        mid = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        ax.text(
            mid[0] + label_offset[0], mid[1] + label_offset[1], label,
            ha=ha, va="center", fontsize=8.2, color=INK, zorder=4,
        )


def output(ax, x, y, text):
    ax.text(
        x, y, text,
        ha="center", va="center", fontsize=8.4, color=INK, style="italic",
        zorder=4,
    )


def draw(path_stem):
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0.2, 7.0)
    ax.axis("off")

    # The composition, drawn first so every box sits on top of it.
    ax.add_patch(
        FancyBboxPatch(
            (3.9, 0.65), 8.6, 5.7,
            boxstyle="round,pad=0.02,rounding_size=0.2",
            facecolor="none", edgecolor=NEUTRAL,
            linewidth=1.3, linestyle=(0, (5, 4)), zorder=1,
        )
    )
    ax.text(
        3.9, 6.6, "DecisionPipeline  —  one object, one call",
        ha="left", va="center", fontsize=9.6, color=INK,
        fontweight="bold", zorder=4,
    )

    # Inputs -----------------------------------------------------------
    _, judgments_right = box(
        ax, 1.5, 5.4, 2.6,
        "judgments", "fuzzy pairwise matrix\n(m × m × 3)", NEUTRAL,
    )
    _, matrix_right = box(
        ax, 1.5, 2.9, 2.6,
        "decision_matrix", "N alternatives\n× m criteria", NEUTRAL,
    )

    # Stages ------------------------------------------------------------
    fahp_left, _fahp_right = box(
        ax, 5.9, 5.4, 3.0,
        "FAHP", "extent analysis or\ngeometric mean", BLUE,
    )
    promethee_left, promethee_right = box(
        ax, 5.9, 2.9, 3.0,
        "Promethee", "six preference functions\nthresholds q, p, s", ORANGE,
    )
    box(
        ax, 10.3, 4.9, 3.2,
        "ANNSurrogate", "fitted on (X, φ)", GREEN,
    )
    box(
        ax, 10.3, 1.9, 3.2,
        "rank_stability", "Monte Carlo on w", PURPLE,
    )

    # Flow ---------------------------------------------------------------
    arrow(ax, (judgments_right, 5.4), (fahp_left, 5.4))
    arrow(ax, (matrix_right, 2.9), (promethee_left, 2.9))
    arrow(
        ax, (5.9, 5.4 - BOX_HALF_HEIGHT), (5.9, 2.9 + BOX_HALF_HEIGHT),
        label="weights w,\nconsistency ratio",
        label_offset=(0.28, 0.0), ha="left",
    )
    arrow(
        ax, (promethee_right, 3.2), (10.3 - 1.6, 4.6),
        label="φ as target", label_offset=(-0.1, 0.42),
    )
    arrow(
        ax, (promethee_right, 2.6), (10.3 - 1.6, 2.1),
        label="w, X, ranker", label_offset=(0.0, -0.36),
    )

    # Outputs -------------------------------------------------------------
    output(ax, 5.9, 2.9 - BOX_HALF_HEIGHT - 0.34,
           "φ⁺, φ⁻, φ  ·  PROMETHEE I and II")
    output(ax, 10.3, 4.9 - BOX_HALF_HEIGHT - 0.34,
           "φ̂ for unseen alternatives")
    output(ax, 10.3, 1.9 - BOX_HALF_HEIGHT - 0.34,
           "rank distribution, reversal rate")

    fig.tight_layout()
    for suffix in ("png", "pdf"):
        out = f"{path_stem}.{suffix}"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"wrote {out}")


def main():
    parser = argparse.ArgumentParser(description="Architecture diagram")
    parser.add_argument("--outdir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    draw(str(args.outdir / "architecture"))


if __name__ == "__main__":
    main()
