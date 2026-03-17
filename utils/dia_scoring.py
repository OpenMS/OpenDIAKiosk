"""Chromatographic scoring utilities inspired by OpenSwath/MRMScoring.

This module provides helper functions to extract per-transition traces
for a peak boundary and compute simple approximations of the following
scores:

- Cross-correlation (shape) matrix and derived shape scores
- Coelution (lag) matrix and derived coelution scores
- Number of peaks (NR_PEAKS)
- Simple Signal-to-Noise estimator (log S/N)
- Mutual Information (MI) matrix and derived MI scores

These implementations are meant to be readable and practical for
interactive inspection in the Streamlit app. They are not byte-for-byte
reproductions of OpenMS internals but follow the same high-level
algorithms described in the OpenSwath documentation.

Usage example
-------------
from utils.dia_scoring import (
    extract_traces_in_peak,
    build_xcorr_matrices,
    calc_xcorr_shape_score,
    calc_nr_peaks,
)

rt, traces = extract_traces_in_peak(exp_df_targeted, peak_row)
corr_max, lag_at_max = build_xcorr_matrices(traces)
shape_mean = calc_xcorr_shape_score(corr_max)
nr = calc_nr_peaks(picked_peaks_df_for_group)
"""

from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
from scipy.stats import rankdata


def extract_traces_in_peak(
    exp_df: pd.DataFrame,
    peak_row: pd.Series,
    annotation_col: str = "annotation",
    rt_col: str = "rt",
    intensity_col: str = "intensity",
    annotations: List[str] = None,
    n_points: int = 101,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Extract per-transition traces inside a peak boundary and resample.

    Returns a common retention-time grid and a dict mapping `annotation` ->
    intensity vector sampled on that grid. If a transition has no points
    inside the window, a zero vector is returned.
    """
    left = (
        float(peak_row["leftWidth"])
        if "leftWidth" in peak_row
        else float(peak_row.get("left", 0))
    )
    right = (
        float(peak_row["rightWidth"])
        if "rightWidth" in peak_row
        else float(peak_row.get("right", 0))
    )
    if right <= left:
        # fallback: small window around apex if available
        apex = float(peak_row.get("apex", peak_row.get("rt", left)))
        left = apex - 5
        right = apex + 5

    if annotations is None:
        annotations = list(exp_df[annotation_col].unique())

    common_rt = np.linspace(left, right, n_points)
    traces = {}
    for ann in annotations:
        sub = exp_df[exp_df[annotation_col] == ann]
        sub = sub[(sub[rt_col] >= left) & (sub[rt_col] <= right)]
        if sub.shape[0] == 0:
            traces[ann] = np.zeros(n_points, dtype=float)
            continue
        # ensure RT ordering
        sub = sub.sort_values(rt_col)
        # interpolate missing values onto common_rt
        traces[ann] = np.interp(
            common_rt,
            sub[rt_col].to_numpy(dtype=float),
            sub[intensity_col].to_numpy(dtype=float),
            left=0.0,
            right=0.0,
        )

    return common_rt, traces


def standardize_traces_matrix(
    traces: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, List[str]]:
    """Convert dict of traces to 2D array and z-score each trace.

    Returns (matrix, annotations) where matrix shape is (n_traces, n_points).
    """
    annotations = list(traces.keys())
    arr = np.vstack([traces[a].astype(float) for a in annotations])
    means = arr.mean(axis=1, keepdims=True)
    stds = arr.std(axis=1, ddof=0, keepdims=True)
    stds[stds == 0] = 1.0
    arr_std = (arr - means) / stds
    return arr_std, annotations


def calculate_cross_correlation(
    x: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute sliding dot-product cross-correlation and corresponding lags.

    Uses numpy.correlate(mode='full'). Lags are in units of samples; if
    the two arrays are length N the lag vector spans (-(N-1), ..., N-1).
    """
    z = np.correlate(x, y, mode="full")
    n = x.shape[0]
    lags = np.arange(-n + 1, n)
    return z, lags


def build_xcorr_matrices(
    traces: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Compute pairwise maximum cross-correlation and lag-at-maximum.

    Returns (corr_max_matrix, lag_at_max_matrix, annotations)
    """
    arr_std, annotations = standardize_traces_matrix(traces)
    n = arr_std.shape[0]
    corr_max = np.zeros((n, n), dtype=float)
    lag_at_max = np.zeros((n, n), dtype=int)
    for i in range(n):
        xi = arr_std[i]
        for j in range(n):
            yj = arr_std[j]
            z, lags = calculate_cross_correlation(xi, yj)
            idx = int(np.argmax(z))
            corr_max[i, j] = z[idx]
            lag_at_max[i, j] = int(lags[idx])
    return corr_max, lag_at_max, annotations


def calc_xcorr_shape_score(corr_max: np.ndarray) -> float:
    """Mean of all entries in cross-correlation matrix (shape score)."""
    return float(np.mean(corr_max))


def calc_xcorr_shape_weighted(
    corr_max: np.ndarray, lib_intensities: np.ndarray
) -> float:
    """Weighted mean of cross-correlation matrix using library intensities L.

    Weighted = sum_{i,j} X_{i,j} * L_i * L_j  / sum_{i,j} L_i * L_j
    """
    L = np.asarray(lib_intensities, dtype=float)
    if L.ndim != 1 or L.size != corr_max.shape[0]:
        raise ValueError(
            "lib_intensities must be 1D array with same length as number of transitions"
        )
    w = np.outer(L, L)
    denom = np.sum(w)
    if denom == 0:
        return float(np.mean(corr_max))
    return float(np.sum(corr_max * w) / denom)


def calc_xcorr_coelution_score(lag_at_max: np.ndarray) -> float:
    """Estimate coelution score: mean(abs(lag)) + std(abs(lag))."""
    lags = np.abs(lag_at_max.astype(float))
    return float(np.mean(lags) + np.std(lags))


def calc_xcorr_coelution_weighted(
    lag_at_max: np.ndarray, lib_intensities: np.ndarray
) -> float:
    L = np.asarray(lib_intensities, dtype=float)
    if L.ndim != 1 or L.size != lag_at_max.shape[0]:
        raise ValueError(
            "lib_intensities must be 1D array with same length as number of transitions"
        )
    w = np.outer(L, L)
    lags = np.abs(lag_at_max.astype(float))
    denom = np.sum(w)
    if denom == 0:
        return calc_xcorr_coelution_score(lag_at_max)
    return float(np.sum(lags * w) / denom)


def calc_nr_peaks(peaks_df: pd.DataFrame) -> int:
    """Number of transition-level features in the peak group."""
    return int(peaks_df.shape[0])


def calc_log_sn_score(traces: Dict[str, np.ndarray]) -> float:
    """Estimate log S/N score by computing per-transition S/N and returning mean log.

    Signal is taken as the max intensity for the trace. Noise is estimated as
    the standard deviation of the trace after removing the top 10%% of values
    (simple baseline estimator).
    """
    sn_list = []
    for v in traces.values():
        v = np.asarray(v, dtype=float)
        if v.size == 0:
            continue
        signal = np.max(v)
        threshold = np.percentile(v, 90)
        baseline = v[v <= threshold]
        if baseline.size < 2:
            noise = np.std(v)
        else:
            noise = np.std(baseline)
        sn = signal / (noise + 1e-12)
        sn_list.append(np.log(sn + 1e-12))
    if len(sn_list) == 0:
        return 0.0
    return float(np.mean(sn_list))


def ranked_mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    """Compute mutual information between two 1D arrays using discrete ranks.

    The implementation follows the description in the user's notes: compute
    rank vectors (dense ranks), count joint states and evaluate
    mean(log(numJoint/(numFirst*numSecond))) + log(N) normalized by log(2).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size != y.size or x.size == 0:
        return 0.0
    rx = rankdata(x, method="dense").astype(int)
    ry = rankdata(y, method="dense").astype(int)
    nx = rx.max() + 1
    ny = ry.max() + 1
    # joint counts
    joint = np.zeros((nx, ny), dtype=float)
    for a, b in zip(rx, ry):
        joint[a, b] += 1.0
    N = float(x.size)
    row_sums = joint.sum(axis=1)
    col_sums = joint.sum(axis=0)
    eps = 1e-12
    total = 0.0
    for i in range(nx):
        for j in range(ny):
            nij = joint[i, j]
            if nij <= 0:
                continue
            term = np.log((nij + eps) / (row_sums[i] * col_sums[j] + eps))
            total += (nij / N) * term
    mi = (total + np.log(N + eps)) / np.log(2 + eps)
    return float(mi)


def build_mi_matrix(traces: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
    arr = np.vstack([traces[a].astype(float) for a in traces.keys()])
    n = arr.shape[0]
    mi = np.zeros((n, n), dtype=float)
    annotations = list(traces.keys())
    for i in range(n):
        for j in range(n):
            mi[i, j] = ranked_mutual_information(arr[i], arr[j])
    return mi, annotations


def calc_mi_score(mi_matrix: np.ndarray) -> float:
    return float(np.mean(mi_matrix))


def calc_mi_weighted_score(mi_matrix: np.ndarray, lib_intensities: np.ndarray) -> float:
    L = np.asarray(lib_intensities, dtype=float)
    w = np.outer(L, L)
    denom = np.sum(w)
    if denom == 0:
        return float(np.mean(mi_matrix))
    return float(np.sum(mi_matrix * w) / denom)
