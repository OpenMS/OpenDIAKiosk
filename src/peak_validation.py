"""Skyline-style peak validation for one precursor across runs.

Reads the chosen peak group of a precursor in every run from an OpenSwath OSW
file and turns it into the views Skyline uses to judge a peak:

- the integrated area shaded under each transition (Skyline's chromatogram graph),
- library match: library relative intensities against the measured transition
  areas, with the dot product (dotp),
- peak areas per run, stacked by transition (Peak Areas > Replicate Comparison),
- apex retention time and peak boundaries per run (Retention Times > Replicate
  Comparison).

Streamlit-free, so it can be tested and reused outside the page.
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _read_sql(osw_path: str | Path, query: str, params: Iterable = ()) -> pd.DataFrame:
    with sqlite3.connect(str(osw_path)) as conn:
        return pd.read_sql_query(query, conn, params=list(params))


def _tables(osw_path: str | Path) -> set[str]:
    df = _read_sql(osw_path, "SELECT name FROM sqlite_master WHERE type='table'")
    return set(df["name"].astype(str))


def _columns(osw_path: str | Path, table: str) -> set[str]:
    with sqlite3.connect(str(osw_path)) as conn:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def short_run_names(paths: Iterable[str]) -> dict[str, str]:
    """Run names for axes and legends: the file stem without the suffix all runs share.

    ``results/Sample_A_rep1_results.xic`` and ``results/Sample_B_results.xic``
    become ``Sample_A_rep1`` and ``Sample_B``. Names stay unique.
    """
    paths = list(paths)
    stems = [Path(p).name.split(".")[0] for p in paths]
    if len(stems) > 1:
        suffix = ""
        for chars in zip(*(stem[::-1] for stem in stems)):
            if len(set(chars)) != 1:
                break
            suffix += chars[0]
        suffix = suffix[::-1]
        # Cut only at a separator, so "A1_results"/"B1_results" keep their "1"
        cut = min((suffix.find(sep) for sep in "_-. " if sep in suffix), default=-1)
        suffix = suffix[cut:] if cut >= 0 else ""
        if suffix and all(len(stem) > len(suffix) for stem in stems):
            trimmed = [stem[: -len(suffix)].rstrip("_-. ") or stem for stem in stems]
            if len(set(trimmed)) == len(trimmed):
                stems = trimmed
    if len(set(stems)) != len(stems):
        stems = [str(p) for p in paths]
    return dict(zip(paths, stems))


def precursor_transitions(osw_path: str | Path, precursor_id: int) -> pd.DataFrame:
    """Detecting transitions of a precursor with their library intensities.

    Columns: TRANSITION_ID, ANNOTATION, PRODUCT_MZ, PRODUCT_CHARGE,
    LIBRARY_INTENSITY. ANNOTATION falls back to type + ordinal + charge
    (``y7^2``) when the library has none, which is how the XIC files label them.
    """
    columns = _columns(osw_path, "TRANSITION")
    annotation = "t.ANNOTATION" if "ANNOTATION" in columns else "NULL"
    df = _read_sql(
        osw_path,
        f"""
        SELECT t.ID AS TRANSITION_ID, {annotation} AS ANNOTATION, t.TYPE, t.ORDINAL,
               t.PRODUCT_MZ, t.CHARGE AS PRODUCT_CHARGE, t.LIBRARY_INTENSITY
        FROM TRANSITION t
        JOIN TRANSITION_PRECURSOR_MAPPING m ON m.TRANSITION_ID = t.ID
        WHERE m.PRECURSOR_ID = ? AND COALESCE(t.DETECTING, 1) = 1
        ORDER BY t.PRODUCT_MZ
        """,
        (int(precursor_id),),
    )
    if df.empty:
        return df.drop(columns=["TYPE", "ORDINAL"], errors="ignore")
    fallback = df.apply(_ion_label, axis=1)
    df["ANNOTATION"] = df["ANNOTATION"].where(df["ANNOTATION"].notna() & (df["ANNOTATION"] != ""), fallback)
    return df.drop(columns=["TYPE", "ORDINAL"])


def _ion_label(row: pd.Series) -> str:
    ion = f"{row.get('TYPE') or ''}{'' if pd.isna(row.get('ORDINAL')) else int(row['ORDINAL'])}"
    charge = row.get("PRODUCT_CHARGE")
    if charge is not None and not pd.isna(charge) and int(charge) > 1:
        ion = f"{ion}^{int(charge)}"
    return ion or str(row["TRANSITION_ID"])


def chosen_peaks(osw_path: str | Path, precursor_id: int, run_ids: Optional[list[int]] = None) -> pd.DataFrame:
    """The peak group picked for a precursor in each run.

    The pick is rank 1 in SCORE_MS2 when the file has been scored, otherwise the
    largest MS2 area (the same rule the chromatogram overlay uses). Columns:
    RUN_ID, FEATURE_ID, EXP_RT, LEFT_WIDTH, RIGHT_WIDTH, EXPECTED_RT, AREA,
    QVALUE, OSW_LIBRARY_DOTPROD.
    """
    tables = _tables(osw_path)
    has_score = "SCORE_MS2" in tables
    has_ms2 = "FEATURE_MS2" in tables
    ms2_columns = _columns(osw_path, "FEATURE_MS2") if has_ms2 else set()
    select = [
        "f.RUN_ID", "f.ID AS FEATURE_ID", "f.EXP_RT", "f.LEFT_WIDTH", "f.RIGHT_WIDTH",
        "f.EXP_RT - f.DELTA_RT AS EXPECTED_RT",
        "m.AREA_INTENSITY AS AREA" if has_ms2 else "NULL AS AREA",
        "s.QVALUE AS QVALUE" if has_score else "NULL AS QVALUE",
        "s.RANK AS RANK" if has_score else "NULL AS RANK",
        "m.VAR_LIBRARY_DOTPROD AS OSW_LIBRARY_DOTPROD" if "VAR_LIBRARY_DOTPROD" in ms2_columns
        else "NULL AS OSW_LIBRARY_DOTPROD",
    ]
    joins = []
    if has_ms2:
        joins.append("LEFT JOIN FEATURE_MS2 m ON m.FEATURE_ID = f.ID")
    if has_score:
        joins.append("LEFT JOIN SCORE_MS2 s ON s.FEATURE_ID = f.ID")
    where, params = "f.PRECURSOR_ID = ?", [int(precursor_id)]
    if run_ids:
        where += f" AND f.RUN_ID IN ({','.join('?' * len(run_ids))})"
        params += [int(r) for r in run_ids]
    df = _read_sql(
        osw_path,
        f"SELECT {', '.join(select)} FROM FEATURE f {' '.join(joins)} WHERE {where}",
        params,
    )
    if df.empty:
        return df.drop(columns=["RANK"])
    df["_rank"] = pd.to_numeric(df["RANK"], errors="coerce").fillna(np.inf)
    df["_area"] = pd.to_numeric(df["AREA"], errors="coerce").fillna(-np.inf)
    df = df.sort_values(["RUN_ID", "_rank", "_area"], ascending=[True, True, False])
    return df.drop_duplicates("RUN_ID").drop(columns=["RANK", "_rank", "_area"]).reset_index(drop=True)


def transition_areas(osw_path: str | Path, feature_ids: list[int]) -> pd.DataFrame:
    """Per-transition integrated areas (FEATURE_TRANSITION) of the given peak groups."""
    if not feature_ids or "FEATURE_TRANSITION" not in _tables(osw_path):
        return pd.DataFrame(columns=["FEATURE_ID", "TRANSITION_ID", "AREA_INTENSITY"])
    return _read_sql(
        osw_path,
        f"""
        SELECT FEATURE_ID, TRANSITION_ID, AREA_INTENSITY FROM FEATURE_TRANSITION
        WHERE FEATURE_ID IN ({','.join('?' * len(feature_ids))})
        """,
        [int(f) for f in feature_ids],
    )


def dotp(observed: Iterable[float], library: Iterable[float]) -> float:
    """Skyline's library dot product: cosine of the square-root intensities.

    1 means the measured transitions have exactly the library's relative
    intensities. Returns NaN when either side has no signal.
    """
    obs = np.sqrt(np.clip(np.asarray(list(observed), dtype=float), 0, None))
    lib = np.sqrt(np.clip(np.asarray(list(library), dtype=float), 0, None))
    obs, lib = np.nan_to_num(obs), np.nan_to_num(lib)
    norm = np.linalg.norm(obs) * np.linalg.norm(lib)
    return float(obs @ lib / norm) if norm > 0 else math.nan


def peak_validation_tables(
    osw_path: str | Path,
    precursor_id: int,
    run_labels: dict[int, str],
    transition_labels: Optional[dict[int, str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Everything the validation views need for one precursor.

    ``run_labels`` maps OSW RUN.ID to the name shown for the run, in display order.
    ``transition_labels`` maps TRANSITION.ID to the name the chromatograms use
    (the XIC annotation), so bars and traces carry the same names.
    Returns ``(runs, areas)``:

    - ``runs``: one row per run with a chosen peak: run, apex/left/right/expected
      RT, peak width, total area, q-value and dotp.
    - ``areas``: one row per run and transition: run, annotation, product m/z,
      area, its share of the run's total, and the library intensity.
    """
    transitions = precursor_transitions(osw_path, precursor_id)
    if transition_labels:
        named = transitions["TRANSITION_ID"].map(transition_labels)
        transitions["ANNOTATION"] = named.where(named.notna(), transitions["ANNOTATION"])
    peaks = chosen_peaks(osw_path, precursor_id, list(run_labels))
    if peaks.empty:
        return pd.DataFrame(), pd.DataFrame()
    peaks["run"] = peaks["RUN_ID"].map(run_labels)
    peaks = peaks.dropna(subset=["run"])
    order = {label: i for i, label in enumerate(run_labels.values())}
    peaks = peaks.sort_values("run", key=lambda s: s.map(order)).reset_index(drop=True)

    areas = transition_areas(osw_path, peaks["FEATURE_ID"].tolist())
    areas = areas.merge(peaks[["FEATURE_ID", "run"]], on="FEATURE_ID").merge(
        transitions, on="TRANSITION_ID", how="inner"
    )
    areas["AREA_INTENSITY"] = pd.to_numeric(areas["AREA_INTENSITY"], errors="coerce").fillna(0.0)
    totals = areas.groupby("run")["AREA_INTENSITY"].transform("sum")
    areas["share"] = np.where(totals > 0, 100 * areas["AREA_INTENSITY"] / totals, 0.0)
    areas = areas.sort_values(["run", "PRODUCT_MZ"], key=lambda s: s.map(order) if s.name == "run" else s)

    dotps = {
        run: dotp(group["AREA_INTENSITY"], group["LIBRARY_INTENSITY"])
        for run, group in areas.groupby("run", sort=False)
    }
    runs = pd.DataFrame(
        {
            "run": peaks["run"],
            "apex RT (s)": peaks["EXP_RT"],
            "start (s)": peaks["LEFT_WIDTH"],
            "end (s)": peaks["RIGHT_WIDTH"],
            "width (s)": peaks["RIGHT_WIDTH"] - peaks["LEFT_WIDTH"],
            "expected RT (s)": peaks["EXPECTED_RT"],
            "area": pd.to_numeric(peaks["AREA"], errors="coerce"),
            "q-value": pd.to_numeric(peaks["QVALUE"], errors="coerce"),
            "dotp": peaks["run"].map(dotps),
            "feature": peaks["FEATURE_ID"],
        }
    )
    areas = areas.rename(
        columns={
            "ANNOTATION": "transition",
            "PRODUCT_MZ": "product m/z",
            "AREA_INTENSITY": "area",
            "LIBRARY_INTENSITY": "library intensity",
        }
    )[["run", "transition", "product m/z", "area", "share", "library intensity", "TRANSITION_ID"]]
    return runs.reset_index(drop=True), areas.reset_index(drop=True)


def library_match_frame(areas: pd.DataFrame) -> pd.DataFrame:
    """Rows for an Insight MirrorPlot: measured areas per run on top, library below.

    Both sides are scaled to 100 % of their largest transition, so the shapes
    compare directly. ``side`` is the run name for measured rows and
    ``"library"`` for the library rows.
    """
    if areas.empty:
        return pd.DataFrame(columns=["side", "mz", "relative", "label", "peak_id", "matched"])

    def scaled(values: pd.Series) -> pd.Series:
        top = values.max()
        return 100 * values / top if top and top > 0 else values * 0.0

    measured = areas.assign(relative=areas.groupby("run")["area"].transform(scaled))
    measured = measured[["run", "product m/z", "relative", "transition"]].rename(columns={"run": "side"})
    library = areas.drop_duplicates("TRANSITION_ID")
    library = library.assign(side="library", relative=scaled(library["library intensity"].astype(float)))
    library = library[["side", "product m/z", "relative", "transition"]]
    frame = pd.concat([measured, library], ignore_index=True)
    frame = frame.rename(columns={"product m/z": "mz", "transition": "label"})
    frame["peak_id"] = np.arange(len(frame))
    frame["matched"] = True  # MirrorPlot labels highlighted peaks only
    return frame


def shade_peak(fig: go.Figure, start: float, end: float, apex: Optional[float] = None) -> go.Figure:
    """Fill the area under every transition trace between ``start`` and ``end``.

    Each fill takes its trace's colour and legend group, so hiding a transition
    in the legend hides its area too. A dotted line marks the apex.
    """
    palette = list(fig.layout.colorway or []) or [
        "#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A",
        "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
    ]
    fills = []
    for index, trace in enumerate(fig.data):
        if trace.type != "scatter" or trace.x is None or trace.y is None:
            continue
        x = np.asarray(trace.x, dtype=float)
        y = np.asarray(trace.y, dtype=float)
        inside = (x >= start) & (x <= end)
        if inside.sum() < 2:
            continue
        color = (trace.line.color if trace.line is not None else None) or palette[index % len(palette)]
        if trace.line is not None and trace.line.color is None:
            trace.line.color = color
        group = trace.legendgroup or trace.name
        trace.legendgroup = group
        fills.append(
            go.Scatter(
                x=x[inside], y=y[inside], mode="lines", line={"width": 0, "color": color},
                fill="tozeroy", fillcolor=_with_alpha(color, 0.3), legendgroup=group,
                showlegend=False, hoverinfo="skip", name=f"{trace.name} (integrated)",
            )
        )
    for fill in fills:
        fig.add_trace(fill)
    if apex is not None and not pd.isna(apex):
        fig.add_vline(x=float(apex), line={"dash": "dot", "width": 1, "color": "gray"})
    return fig


def _with_alpha(color: str, alpha: float) -> str:
    """``#rrggbb`` or ``rgb(r,g,b)`` as an ``rgba`` string."""
    color = color.strip()
    if color.startswith("#") and len(color) == 7:
        r, g, b = (int(color[i : i + 2], 16) for i in (1, 3, 5))
        return f"rgba({r},{g},{b},{alpha})"
    if color.startswith("rgb(") and color.endswith(")"):
        return f"rgba({color[4:-1]},{alpha})"
    return color


def peak_areas_figure(areas: pd.DataFrame, normalized: bool = False) -> go.Figure:
    """Peak Areas > Replicate Comparison: a bar per run, stacked by transition."""
    value = "share" if normalized else "area"
    fig = go.Figure()
    for transition, group in areas.groupby("transition", sort=False):
        fig.add_trace(
            go.Bar(
                x=group["run"], y=group[value], name=str(transition),
                hovertemplate=(
                    "%{x}<br>" + str(transition) + ": %{y:,.4g}" + (" %%" if normalized else "") + "<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        barmode="stack",
        title="Peak areas by run",
        yaxis_title="share of peak area (%)" if normalized else "peak area",
        xaxis_title=None,
        legend_title="transition",
        height=380,
    )
    return fig


def retention_times_figure(runs: pd.DataFrame) -> go.Figure:
    """Retention Times > Replicate Comparison: peak start to end per run, apex marked."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=runs["run"], y=runs["width (s)"], base=runs["start (s)"], name="peak",
            marker={"color": "rgba(99,110,250,0.35)", "line": {"width": 1, "color": "rgb(99,110,250)"}},
            customdata=runs[["start (s)", "end (s)"]].to_numpy(),
            hovertemplate="%{x}<br>%{customdata[0]:.1f} to %{customdata[1]:.1f} s<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=runs["run"], y=runs["apex RT (s)"], mode="markers", name="apex",
            marker={"symbol": "line-ew-open", "size": 34, "line": {"width": 4}},
            hovertemplate="%{x}<br>apex %{y:.1f} s<extra></extra>",
        )
    )
    if runs["expected RT (s)"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=runs["run"], y=runs["expected RT (s)"], mode="markers", name="expected (library)",
                marker={"symbol": "x-thin-open", "size": 12, "line": {"width": 2}},
                hovertemplate="%{x}<br>expected %{y:.1f} s<extra></extra>",
            )
        )
    fig.update_layout(title="Retention time by run", yaxis_title="retention time (s)", height=380)
    return fig
