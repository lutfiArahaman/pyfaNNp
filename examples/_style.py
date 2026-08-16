"""Shared plotting style for the example figures.

Colours are the Okabe-Ito "Color Universal Design" palette, which is
published as safe under deuteranopia, protanopia and tritanopia. They are
used in a fixed order rather than cycled, so a series keeps its colour
across figures.
"""

from __future__ import annotations

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"

SERIES = (BLUE, ORANGE, GREEN, VERMILLION, PURPLE)

INK = "#222222"
MUTED = "#666666"
GRID = "#DDDDDD"

SEQUENTIAL = "Blues"  # single hue, light to dark


def style_axis(ax, hide_spines=("top", "right")):
    """Recessive frame: the data should be the darkest thing on the panel."""
    for side in hide_spines:
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        if side in ax.spines:
            ax.spines[side].set_color(MUTED)
            ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9, length=3, width=0.8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(INK)


def legend(ax, **kwargs):
    """Frameless legend with text in ink, not series colour."""
    options = {"frameon": False, "fontsize": 9, "handlelength": 1.4}
    options.update(kwargs)
    result = ax.legend(**options)
    for text in result.get_texts():
        text.set_color(INK)
    return result


def title(ax, text, **kwargs):
    options = {"fontsize": 10, "color": INK, "loc": "left", "pad": 8}
    options.update(kwargs)
    ax.set_title(text, **options)
