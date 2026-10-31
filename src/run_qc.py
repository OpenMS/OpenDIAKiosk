"""Per-run quality control from an OpenSwath OSW file (Skyline and Spectronaut style).

One row per identified precursor and run (the rank-1 target peak at or below a
q-value cutoff) is enough for four run-level checks:

- RT calibration: library (iRT) retention time against measured retention
  time, per run, with the residual after calibration (Skyline's Retention
  Times > Regression, Spectronaut's iRT calibration),
- fragment and precursor mass error per run (Skyline's Mass Errors),
- peak width per run (Spectronaut's run QC),
- detections per run and how many runs each precursor is found in
  (Skyline's Detections plots).

Streamlit-free.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Above this many points per run the RT scatter shows a random sample
MAX_POINTS_PER_RUN = 5000


def _tables(osw_path: str | Path) -> set[str]:
    with sqlite3.connect(str(osw_path)) as conn:
        return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _columns(osw_path: str | Path, table: str) -> set[str]:
    with sqlite3.connect(str(osw_path)) as conn:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def run_name(filename: str) -> str:
    """``/data/run_01.mzML.gz`` -> ``run_01``."""
    name = Path(str(filename).replace("\\", "/")).name
    for ext in (".gz", ".mzML", ".mzml", ".mzXML", ".sqMass", ".d", ".raw"):
        if name.endswith(ext):
            name = name[: -len(ext)]
    return name or str(filename)


def identified_peaks(osw_path: str | Path, max_qvalue: float = 0.01) -> pd.DataFrame:
    """Rank-1 target peaks at or below ``max_qvalue``, one row per precursor and run.

    Without SCORE_MS2 (an unscored file) every precursor's largest peak in a run
    is used instead, and ``scored`` is False on all rows. Columns: run,
    precursor, measured RT, library RT, normalized RT, peak width, fragment and
    precursor mass error (mean absolute ppm, NaN when not scored), scored.
    """
    tables = _tables(osw_path)
    scored = "SCORE_MS2" in tables
    ms2 = _columns(osw_path, "FEATURE_MS2") if "FEATURE_MS2" in tables else set()
    ms1 = _columns(osw_path, "FEATURE_MS1") if "FEATURE_MS1" in tables else set()
    select = [
        "r.FILENAME AS filename",
        "f.PRECURSOR_ID AS precursor",
        "f.EXP_RT AS rt",
        "p.LIBRARY_RT AS library_rt",
        "f.NORM_RT AS norm_rt",
        "f.RIGHT_WIDTH - f.LEFT_WIDTH AS width",
        "m2.VAR_MASSDEV_SCORE AS ms2_ppm" if "VAR_MASSDEV_SCORE" in ms2 else "NULL AS ms2_ppm",
        "m1.VAR_MASSDEV_SCORE AS ms1_ppm" if "VAR_MASSDEV_SCORE" in ms1 else "NULL AS ms1_ppm",
        "m2.AREA_INTENSITY AS area" if "AREA_INTENSITY" in ms2 else "NULL AS area",
    ]
    joins = [
        "JOIN RUN r ON r.ID = f.RUN_ID",
        "JOIN PRECURSOR p ON p.ID = f.PRECURSOR_ID",
    ]
    if ms2:
        joins.append("LEFT JOIN FEATURE_MS2 m2 ON m2.FEATURE_ID = f.ID")
    if ms1:
        joins.append("LEFT JOIN FEATURE_MS1 m1 ON m1.FEATURE_ID = f.ID")
    where = ["COALESCE(p.DECOY, 0) = 0"]
    params: list = []
    if scored:
        joins.append("JOIN SCORE_MS2 s ON s.FEATURE_ID = f.ID")
        where += ["s.RANK = 1", "s.QVALUE <= ?"]
        params.append(float(max_qvalue))
    query = f"SELECT {', '.join(select)} FROM FEATURE f {' '.join(joins)} WHERE {' AND '.join(where)}"
    with sqlite3.connect(str(osw_path)) as conn:
        df = pd.read_sql_query(query, conn, params=params)
    if not scored and not df.empty:
        df = (
            df.assign(_area=pd.to_numeric(df["area"], errors="coerce").fillna(-np.inf))
            .sort_values("_area", ascending=False)
            .drop_duplicates(["filename", "precursor"])
            .drop(columns="_area")
        )
    df["run"] = df["filename"].map(run_name)
    if df["run"].nunique() != df["filename"].nunique():
        df["run"] = df["filename"]
    df["scored"] = scored
    for col in ["rt", "library_rt", "norm_rt", "width", "ms2_ppm", "ms1_ppm", "area"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.drop(columns=["filename"]).sort_values(["run", "rt"]).reset_index(drop=True)


def run_summary(peaks: pd.DataFrame) -> pd.DataFrame:
    """One row per run: identifications, medians of peak width and mass errors, RT fit."""
    rows = []
    for run, group in peaks.groupby("run", sort=True):
        fit = rt_fit(group)
        rows.append(
            {
                "run": run,
                "precursors": int(group["precursor"].nunique()),
                "median peak width (s)": group["width"].median(),
                "median fragment error (ppm)": group["ms2_ppm"].median(),
                "median precursor error (ppm)": group["ms1_ppm"].median(),
                "RT fit R²": fit["r2"] if fit else np.nan,
                "RT residual IQR": fit["residual_iqr"] if fit else np.nan,
            }
        )
    return pd.DataFrame(rows)


def rt_fit(group: pd.DataFrame) -> dict | None:
    """Linear fit of measured RT on library RT, and the spread of the residuals.

    Residuals are ``normalized RT - library RT``, i.e. in library units after
    OpenSwath's own calibration. Returns None with fewer than three points.
    """
    data = group[["library_rt", "rt"]].dropna()
    if len(data) < 3 or data["library_rt"].nunique() < 2:
        return None
    slope, intercept = np.polyfit(data["library_rt"], data["rt"], 1)
    predicted = slope * data["library_rt"] + intercept
    ss_res = float(((data["rt"] - predicted) ** 2).sum())
    ss_tot = float(((data["rt"] - data["rt"].mean()) ** 2).sum())
    residual = (group["norm_rt"] - group["library_rt"]).dropna()
    q1, q3 = (residual.quantile([0.25, 0.75]) if len(residual) else (np.nan, np.nan))
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        "residual_iqr": float(q3 - q1),
    }


def detections(peaks: pd.DataFrame) -> pd.DataFrame:
    """How many precursors are found in at least n of the runs, for n = 1..runs."""
    runs = peaks["run"].nunique()
    if runs == 0:
        return pd.DataFrame(columns=["at least n runs", "precursors"])
    found_in = peaks.groupby("precursor")["run"].nunique()
    return pd.DataFrame(
        {
            "at least n runs": np.arange(1, runs + 1),
            "precursors": [int((found_in >= n).sum()) for n in range(1, runs + 1)],
        }
    )


def rt_calibration_figure(peaks: pd.DataFrame, seed: int = 0) -> go.Figure:
    """Measured RT against library RT, one colour and fitted line per run."""
    fig = go.Figure()
    rng = np.random.default_rng(seed)
    palette = ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A",
               "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"]
    for index, (run, group) in enumerate(peaks.groupby("run", sort=True)):
        color = palette[index % len(palette)]
        data = group.dropna(subset=["library_rt", "rt"])
        if len(data) > MAX_POINTS_PER_RUN:
            data = data.iloc[np.sort(rng.choice(len(data), MAX_POINTS_PER_RUN, replace=False))]
        fig.add_trace(
            go.Scattergl(
                x=data["library_rt"], y=data["rt"], mode="markers", name=run, legendgroup=run,
                marker={"size": 4, "opacity": 0.5, "color": color},
                hovertemplate=run + "<br>library %{x:.1f}<br>measured %{y:.1f} s<extra></extra>",
            )
        )
        fit = rt_fit(group)
        if fit:
            xs = np.array([data["library_rt"].min(), data["library_rt"].max()])
            fig.add_trace(
                go.Scatter(
                    x=xs, y=fit["slope"] * xs + fit["intercept"], mode="lines", legendgroup=run,
                    showlegend=False, line={"color": color, "width": 2},
                    hovertemplate=f"{run}: R² {fit['r2']:.3f}<extra></extra>",
                )
            )
    fig.update_layout(
        title="RT calibration", xaxis_title="library retention time (iRT)",
        yaxis_title="measured retention time (s)", height=420, legend_title="run",
    )
    return fig


def per_run_box_figure(peaks: pd.DataFrame, column: str, title: str, axis: str) -> go.Figure:
    """Box plot of one per-peak value for every run."""
    fig = go.Figure()
    for run, group in peaks.groupby("run", sort=True):
        values = group[column].dropna()
        if values.empty:
            continue
        fig.add_trace(go.Box(y=values, name=run, boxpoints=False, showlegend=False))
    fig.update_layout(title=title, yaxis_title=axis, height=380)
    return fig


def rt_residual_figure(peaks: pd.DataFrame) -> go.Figure:
    """Residual RT after OpenSwath's calibration, per run (library units)."""
    data = peaks.assign(residual=peaks["norm_rt"] - peaks["library_rt"])
    fig = per_run_box_figure(data, "residual", "RT residual after calibration", "normalized − library RT (iRT)")
    fig.add_hline(y=0, line={"dash": "dot", "width": 1, "color": "gray"})
    return fig


def detections_figure(peaks: pd.DataFrame) -> go.Figure:
    """Precursors per run (bars) next to how many are found in at least n runs."""
    from plotly.subplots import make_subplots

    per_run = peaks.groupby("run")["precursor"].nunique().sort_index()
    cumulative = detections(peaks)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Precursors per run", "Found in at least n runs"))
    fig.add_trace(
        go.Bar(x=per_run.index, y=per_run.values, showlegend=False,
               hovertemplate="%{x}: %{y:,} precursors<extra></extra>"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(x=cumulative["at least n runs"], y=cumulative["precursors"], showlegend=False,
               hovertemplate="in ≥ %{x} runs: %{y:,} precursors<extra></extra>"),
        row=1, col=2,
    )
    fig.update_xaxes(title_text="runs", dtick=1, row=1, col=2)
    fig.update_yaxes(title_text="precursors", row=1, col=1)
    fig.update_layout(height=380, title="Detections")
    return fig
