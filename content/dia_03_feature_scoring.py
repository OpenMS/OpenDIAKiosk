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


def gauss(t, apex, sigma, amp=1.0):
    """Pure Gaussian chromatographic peak."""
    return amp * np.exp(-((t - apex) ** 2) / (2 * sigma**2))


def cross_correlation(x, y):
    """
    Normalised cross-correlation (delay=0 is central index).
    Returns (max_corr, optimal_delay_index).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sx = x.std()
    sy = y.std()
    if sx < 1e-12 or sy < 1e-12:
        return 0.0, 0
    xn = (x - x.mean()) / sx
    yn = (y - y.mean()) / sy
    cc = np.correlate(xn, yn, mode="full")
    cc /= len(x)
    best_idx = int(np.argmax(cc))
    delay = best_idx - (len(x) - 1)
    return float(cc[best_idx]), delay


def xcorr_shape_score(xics: list[np.ndarray]) -> float:
    """VAR_XCORR_SHAPE — mean of all pairwise max-cross-correlations."""
    n = len(xics)
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            v, _ = cross_correlation(xics[i], xics[j])
            vals.append(v)
    return float(np.mean(vals)) if vals else 0.0


def xcorr_coelution_score(xics: list[np.ndarray]) -> float:
    """VAR_XCORR_COELUTION — mean(delays) + std(delays)."""
    n = len(xics)
    delays = []
    for i in range(n):
        for j in range(i + 1, n):
            _, d = cross_correlation(xics[i], xics[j])
            delays.append(abs(d))
    if not delays:
        return 0.0
    return float(np.mean(delays) + np.std(delays))


def spectral_angle(lib: np.ndarray, obs: np.ndarray) -> float:
    """Normalised spectral angle ∈ [0,1].  1 = perfect match."""
    lib = np.asarray(lib, dtype=float)
    obs = np.asarray(obs, dtype=float)
    nl = np.linalg.norm(lib)
    no = np.linalg.norm(obs)
    if nl < 1e-12 or no < 1e-12:
        return 0.0
    cos_theta = np.dot(lib / nl, obs / no)
    cos_theta = float(np.clip(cos_theta, -1, 1))
    return 1.0 - (2.0 * np.arccos(cos_theta) / np.pi)


def dot_product(lib: np.ndarray, obs: np.ndarray) -> float:
    """Square-root + L2 normalised dot product."""
    lib_s = np.sqrt(np.maximum(lib, 0))
    obs_s = np.sqrt(np.maximum(obs, 0))
    nl = np.linalg.norm(lib_s)
    no = np.linalg.norm(obs_s)
    if nl < 1e-12 or no < 1e-12:
        return 0.0
    return float(np.dot(lib_s / nl, obs_s / no))


def manhattan_score(lib: np.ndarray, obs: np.ndarray) -> float:
    """Sqrt-normalised Manhattan distance (lower = better match)."""
    lib_s = np.sqrt(np.maximum(lib, 0))
    obs_s = np.sqrt(np.maximum(obs, 0))
    sl = lib_s.sum()
    so = obs_s.sum()
    if sl < 1e-12 or so < 1e-12:
        return 1.0
    return float(np.sum(np.abs(lib_s / sl - obs_s / so)))


def pearson_corr(lib: np.ndarray, obs: np.ndarray) -> float:
    """Pearson correlation between library and observed intensity vectors."""
    if np.std(lib) < 1e-12 or np.std(obs) < 1e-12:
        return 0.0
    return float(np.corrcoef(lib, obs)[0, 1])


def rmsd_score(lib: np.ndarray, obs: np.ndarray) -> float:
    """Normalised RMSD (lower = better)."""
    mu_l = lib.mean()
    mu_o = obs.mean()
    if mu_l < 1e-12 or mu_o < 1e-12:
        return 1.0
    return float(np.sqrt(np.mean(np.abs(obs / mu_o - lib / mu_l))))


def score_badge(value: float, good_high: bool = True) -> str:
    """Return a coloured Streamlit metric delta string."""
    return "↑" if good_high else "↓"


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

# --------------------------------
# 2. Chromatographic Scores

st.markdown("---")
st.subheader("Chromatographic Scores")

st.markdown(
    r"""
Chromatographic scores measure how well the transition XICs **co-elute** and
share the same **peak shape**.  They are based on pairwise normalised
cross-correlations between every pair of transitions.

For each pair $(x, y)$ the cross-correlation at delay $d$ is:

$$Z[d] = \sum_{i=0}^{N-1} x_i\, y_{i+d}$$

where $x$ and $y$ are **standardised** (zero-mean, unit-variance) intensity
vectors.  The **maximum** $Z[d]$ gives the shape score, and the **delay $d$**
at that maximum gives the coelution score.

| Score | Definition | Good detection |
|---|---|---|
| `VAR_XCORR_SHAPE` | Mean of all pairwise max cross-correlations | → 1 |
| `VAR_XCORR_SHAPE_WEIGHTED` | Library-intensity weighted mean | → 1 |
| `VAR_XCORR_COELUTION` | Mean(delays) + SD(delays) | → 0 |
| `VAR_XCORR_COELUTION_WEIGHTED` | Library-intensity weighted | → 0 |
"""
)

st.markdown("#### Chromatographic Quality Demo")

col_ctrl1, col_ctrl2 = st.columns(2)
with col_ctrl1:
    rt_offset_max = st.slider(
        "Max RT offset between transitions (scans)",
        min_value=0,
        max_value=20,
        value=0,
        step=1,
        help="Simulates how much transitions are staggered in RT. "
        "0 = perfect co-elution; higher = worse coelution score.",
    )
    noise_level = st.slider(
        "Noise level (% of peak height)",
        min_value=0,
        max_value=80,
        value=10,
        step=5,
        help="Gaussian noise added to each transition. "
        "High noise degrades both shape and coelution scores.",
    )
with col_ctrl2:
    n_transitions = st.slider(
        "Number of transitions",
        min_value=3,
        max_value=8,
        value=5,
        step=1,
    )
    apex_sigma = st.slider(
        "Peak width (sigma, scans)",
        min_value=2,
        max_value=15,
        value=5,
        step=1,
        help="Narrower peaks are harder to align → worse coelution.",
    )

# Simulate XICs
rng_xic = np.random.default_rng(42)
t = np.arange(0, 100)
APEX = 50.0
FRAG_COLORS = [
    "#CC1111",
    "#1199CC",
    "#22AA33",
    "#DD8800",
    "#8833CC",
    "#AA1177",
    "#117733",
    "#443399",
]
LIB_AMPS = np.array([1.00, 0.72, 0.55, 0.40, 0.28, 0.22, 0.18, 0.14])

xics_demo = []
offsets_used = []
for k in range(n_transitions):
    offset = (
        rng_xic.integers(-rt_offset_max, rt_offset_max + 1) if rt_offset_max > 0 else 0
    )
    offsets_used.append(offset)
    peak = gauss(t, APEX + offset, apex_sigma, amp=LIB_AMPS[k])
    noise = rng_xic.normal(0, noise_level / 100.0 * LIB_AMPS[k], len(t))
    xics_demo.append(np.maximum(peak + noise, 0))

# Compute scores
shape_score = xcorr_shape_score(xics_demo)
coelution_score = xcorr_coelution_score(xics_demo)

# Plot
fig_xic, axes_x = plt.subplots(
    2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]}
)

for k, (xic, col, off) in enumerate(
    zip(xics_demo, FRAG_COLORS[:n_transitions], offsets_used)
):
    axes_x[0].plot(
        t, xic, color=col, lw=1.8, alpha=0.85, label=f"y{k + 2} (offset={off:+d})"
    )
    axes_x[0].axvline(APEX + off, color=col, lw=0.9, linestyle=":", alpha=0.55)

axes_x[0].axvline(
    APEX, color="black", lw=2, linestyle="--", alpha=0.7, label="Expected apex"
)
axes_x[0].set_ylabel("Intensity", fontsize=10)
axes_x[0].set_title("Extracted Ion Chromatograms (XICs)", fontsize=11)
axes_x[0].legend(fontsize=8, ncol=4, loc="upper right")
axes_x[0].spines["top"].set_visible(False)
axes_x[0].spines["right"].set_visible(False)

# Summed XIC
summed = np.sum(xics_demo, axis=0)
axes_x[1].fill_between(t, summed, alpha=0.3, color="steelblue")
axes_x[1].plot(t, summed, color="navy", lw=1.8)
axes_x[1].axvline(APEX, color="black", lw=2, linestyle="--", alpha=0.7)
axes_x[1].set_ylabel("Summed", fontsize=10)
axes_x[1].set_xlabel("Retention Time (scans)", fontsize=10)
axes_x[1].spines["top"].set_visible(False)
axes_x[1].spines["right"].set_visible(False)

fig_xic.tight_layout()
fig_to_st(fig_xic)

# Score display
m1, m2 = st.columns(2)
m1.metric(
    "VAR_XCORR_SHAPE",
    f"{shape_score:.3f}",
    delta="↑ good (→ 1.0)" if shape_score > 0.7 else "↓ poor",
    delta_color="normal" if shape_score > 0.7 else "inverse",
    help="Mean of all pairwise max cross-correlations. Perfect co-elution = 1.",
)
m2.metric(
    "VAR_XCORR_COELUTION",
    f"{coelution_score:.2f}",
    delta="↑ poor (→ 0 is best)" if coelution_score > 1 else "↓ good",
    delta_color="inverse" if coelution_score > 1 else "normal",
    help="Mean(delays) + SD(delays). Perfect alignment = 0.",
)

with st.expander("How cross-correlation works — code"):
    st.code(
        '''def cross_correlation(x, y):
    """Normalised cross-correlation between two XIC arrays."""
    xn = (x - x.mean()) / x.std()   # standardise
    yn = (y - y.mean()) / y.std()
    cc = np.correlate(xn, yn, mode="full") / len(x)
    best_idx = np.argmax(cc)
    delay    = best_idx - (len(x) - 1)   # 0 = perfectly aligned
    return cc[best_idx], delay            # (shape_val, coelution_delay)

# VAR_XCORR_SHAPE    = mean of cc[best_idx] across all pairs
# VAR_XCORR_COELUTION = mean(|delays|) + std(|delays|) across all pairs''',
        language="python",
    )

# --------------------------------
#   Library Scores

st.markdown("---")
st.subheader("Library Scores")

st.markdown(
    r"""
Library scores compare the **observed peak-apex fragment intensities** against
the reference intensities stored in the spectral library.  A true detection
should produce intensities that closely mirror the library spectrum.

| Score | Formula (simplified) | Good detection |
|---|---|---|
| `VAR_LIBRARY_SANGLE` | $1 - \frac{2}{\pi}\arccos\!\left(\frac{b \cdot x}{\|b\|\|x\|}\right)$ | → 1 |
| `VAR_LIBRARY_DOTPROD` | $\frac{\sqrt{b}}{\|\sqrt{b}\|} \cdot \frac{\sqrt{x}}{\|\sqrt{x}\|}$ | → 1 |
| `VAR_LIBRARY_CORR` | Pearson $r(b, x)$ | → 1 |
| `VAR_LIBRARY_MANHATTAN` | $\sum_i \|\frac{\sqrt{b_i}}{\sum\sqrt{b}} - \frac{\sqrt{x_i}}{\sum\sqrt{x}}\|$ | → 0 |
| `VAR_LIBRARY_RMSD` | $\sqrt{\frac{1}{N}\sum_i\left(\frac{x_i}{\mu_x} - \frac{b_i}{\mu_b}\right)^2}$ | → 0 |

$b$ = library intensities, $x$ = observed intensities.
"""
)

st.markdown("#### Library Match Quality")

col_lib1, col_lib2 = st.columns(2)
with col_lib1:
    intensity_noise_pct = st.slider(
        "Intensity perturbation (% random deviation)",
        min_value=0,
        max_value=100,
        value=15,
        step=5,
        help="How much each observed fragment intensity deviates from the library. "
        "0% = perfect match; 100% = fully random.",
    )
with col_lib2:
    n_missing_frags = st.slider(
        "Missing/interfered transitions",
        min_value=0,
        max_value=5,
        value=0,
        step=1,
        help="Transitions with no signal (e.g. from co-eluting interference).",
    )

# Library intensities
N_FRAGS = 7
frag_labels = [f"y{i + 2}" for i in range(N_FRAGS)]
frag_mzs = [300 + i * 120 for i in range(N_FRAGS)]
lib_ints = np.array([1.00, 0.75, 0.88, 0.55, 0.42, 0.30, 0.20])

# Observed: perturb library + zero out missing
rng_lib = np.random.default_rng(17)
obs_ints = lib_ints.copy()
obs_ints += rng_lib.normal(0, intensity_noise_pct / 100.0, N_FRAGS)
obs_ints = np.maximum(obs_ints, 0.01)
for i in range(min(n_missing_frags, N_FRAGS)):
    obs_ints[-(i + 1)] = rng_lib.exponential(0.04)

obs_ints_norm = obs_ints / obs_ints.max()

# Scores
sa = spectral_angle(lib_ints, obs_ints_norm)
dp = dot_product(lib_ints, obs_ints_norm)
pr = pearson_corr(lib_ints, obs_ints_norm)
mh = manhattan_score(lib_ints, obs_ints_norm)
rmsd = rmsd_score(lib_ints, obs_ints_norm)

# Mirror plot
B_COLOR = "#2266CC"
Y_COLOR = "#CC2222"
fig_lib, ax_lib = plt.subplots(figsize=(11, 4.5))
x_pos = np.arange(N_FRAGS)
bar_w = 0.32

bars_lib = ax_lib.bar(
    x_pos - bar_w / 2,
    lib_ints,
    bar_w,
    color=B_COLOR,
    alpha=0.82,
    label="Library",
    edgecolor="white",
)
bars_obs = ax_lib.bar(
    x_pos + bar_w / 2,
    obs_ints_norm,
    bar_w,
    color=Y_COLOR,
    alpha=0.82,
    label="Observed (apex)",
    edgecolor="white",
)

for xi, (l, o) in zip(x_pos, zip(lib_ints, obs_ints_norm)):
    ax_lib.plot(
        [xi - bar_w / 2 + bar_w / 4, xi + bar_w / 2 + bar_w / 4],
        [l, o],
        color="#888888",
        lw=0.9,
        linestyle="--",
        alpha=0.6,
    )

ax_lib.set_xticks(x_pos)
ax_lib.set_xticklabels(frag_labels, fontsize=10)
ax_lib.set_ylabel("Relative Intensity", fontsize=10)
ax_lib.set_title("Library vs Observed Fragment Intensities", fontsize=11)
ax_lib.set_ylim(0, 1.25)
ax_lib.legend(fontsize=10)
ax_lib.spines["top"].set_visible(False)
ax_lib.spines["right"].set_visible(False)

# Spectral angle arc annotation
angle_deg = (1 - sa) * 90
ax_lib.text(
    N_FRAGS - 1 + 0.6,
    1.15,
    f"Spectral angle: {sa:.3f}\n(θ = {angle_deg:.1f}°)",
    ha="right",
    va="top",
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#EEF3FF", edgecolor=B_COLOR, lw=0.8),
)

fig_lib.tight_layout()
fig_to_st(
    fig_lib,
    caption=(
        "Blue = library reference intensities. Red = observed peak-apex intensities. "
        "Dashed grey lines connect matching transitions."
    ),
)

# Score metrics
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(
    "VAR_LIBRARY_SANGLE", f"{sa:.3f}", help="Spectral angle ∈ [0,1]. 1 = perfect."
)
c2.metric(
    "VAR_LIBRARY_DOTPROD", f"{dp:.3f}", help="Sqrt-normalised dot product. 1 = perfect."
)
c3.metric("VAR_LIBRARY_CORR", f"{pr:.3f}", help="Pearson correlation. 1 = perfect.")
c4.metric("VAR_LIBRARY_MANHATTAN", f"{mh:.3f}", help="Manhattan distance. 0 = perfect.")
c5.metric("VAR_LIBRARY_RMSD", f"{rmsd:.3f}", help="Normalised RMSD. 0 = perfect.")

with st.expander("Score equations & code"):
    st.code(
        """def spectral_angle(lib, obs):
    cos_theta = np.dot(lib/||lib||, obs/||obs||)
    return 1 - (2/π) * arccos(cos_theta)   # 1 = perfect

def dot_product(lib, obs):
    lib_s = sqrt(lib);  obs_s = sqrt(obs)
    return dot(lib_s/||lib_s||, obs_s/||obs_s||)

def pearson_corr(lib, obs):
    return np.corrcoef(lib, obs)[0, 1]

def manhattan_score(lib, obs):
    lib_s = sqrt(lib)/sum(sqrt(lib))
    obs_s = sqrt(obs)/sum(sqrt(obs))
    return sum(|lib_s - obs_s|)           # 0 = perfect

def rmsd_score(lib, obs):
    return sqrt(mean((obs/mean(obs) - lib/mean(lib))^2))  # 0 = perfect""",
        language="python",
    )
