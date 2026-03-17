from __future__ import annotations

import io

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import streamlit as st

from src.common.common import page_setup

matplotlib.rcParams["font.family"] = "DejaVu Sans"

page_setup()

# -----------------------------------
#   Helpers


def fig_to_st(fig, caption: str | None = None, dpi: int = 130) -> None:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    st.image(buf, caption=caption, use_container_width=True)
    plt.close(fig)


# -----------------------------------
#   Page content

st.title("Feature Scoring in DIA")
st.markdown(
    """
During targeted data extraction and peak picking, we identify candidate peak groups that may correspond to the peptides in our spectral library. During this process, we compute a variety of scores that quantify how well the observed data matches our expectations based on the library. These scores are crucial for distinguishing true peptide detections from false positives during downstream statistical validation.

This page walks through a variety of computed scores used by
[OpenSWATH](https://openswath.org) with interactive controls so you can
**see how each score responds** when chromatographic quality, library match, or
mass accuracy changes.
"""
)

st.info(
    "**How to use this page:** Each section has sliders that simulate different "
    "data quality scenarios. Adjust them and watch the scores update in real time."
)

# --------------------------------
# Scoring pipeline overview

st.markdown("---")
st.subheader("The OpenSWATH Scoring Pipeline")

st.markdown(
    """
`scorePeakgroups()` is the central driver that computes a few main categories of scores for each candidate peak group detected in the XICs:
"""
)

SCORE_CATS = [
    ("Chromatographic", "#4A90D9", "XIC shape & coelution\ncross-correlation"),
    ("Library", "#5BAD72", "Spectral angle, dot-product\nManhattan, RMSD, Pearson RT"),
    (
        "Single Spectrum\n(DIA)",
        "#E8A838",
        "Isotope correlation\nmass deviation, b/y ions",
    ),
    ("Ion Mobility", "#9B59B6", "Ion mobility drift time\ncorrelation and deviation"),
    ("UIS", "#D95A4A", "Unique ion signature\n(PTM identification)"),
    ("Intensity", "#16A085", "Peak group intensity\nvs total XIC"),
    ("Elution\nModel", "#7F8C8D", "EMG peak shape\nfit quality"),
]

fig_ov, ax_ov = plt.subplots(figsize=(14, 2.8))
ax_ov.set_xlim(0, 14)
ax_ov.set_ylim(0, 2.8)
ax_ov.axis("off")
box_w = 1.65
box_h = 1.35
xs_ov = np.linspace(0.9, 13.1, len(SCORE_CATS))

for i, ((label, color, sub), x) in enumerate(zip(SCORE_CATS, xs_ov)):
    ax_ov.add_patch(
        mpatches.FancyBboxPatch(
            (x - box_w / 2, 0.75),
            box_w,
            box_h,
            boxstyle="round,pad=0.10",
            facecolor=color,
            edgecolor="white",
            lw=1.5,
            alpha=0.88,
            zorder=3,
        )
    )
    ax_ov.text(
        x,
        1.44,
        label,
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color="white",
        zorder=4,
        linespacing=1.25,
    )
    ax_ov.text(
        x,
        0.50,
        sub,
        ha="center",
        va="top",
        fontsize=7,
        color="#444444",
        linespacing=1.25,
        zorder=4,
    )

ax_ov.text(
    7.0,
    2.65,
    "OpenSWATH scorePeakgroups()- scoring categories",
    ha="center",
    va="top",
    fontsize=10.5,
    fontweight="bold",
)
fig_ov.tight_layout()
fig_to_st(fig_ov)

st.markdown(
    """
All scores are fed into **PyProphet** (a semi-supervised machine learning
classifier) which learns a linear discriminant that maximally separates
target from decoy peak groups.  The resulting discriminant score is used for
**FDR control** via the target-decoy method.
"""
)
