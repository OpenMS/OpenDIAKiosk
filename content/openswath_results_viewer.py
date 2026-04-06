"""
content/openswath_results_viewer.py
OpenSwath Results Viewer

Interactive review page for OpenSwath OSW feature files and PyProphet TSV/matrix
exports saved in the current workspace.
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.common.common import page_setup


page_setup()


LEVEL_CONFIG: dict[str, dict[str, object]] = {
    "precursor": {
        "title": "Precursor Matrix",
        "entity_label": "precursors",
        "entity_candidates": ["transition_group_id", "FullPeptideName", "Sequence"],
        "meta_columns": [
            "transition_group_id",
            "Sequence",
            "FullPeptideName",
            "Charge",
            "ProteinName",
        ],
        "color": "#0F766E",
    },
    "peptide": {
        "title": "Peptide Matrix",
        "entity_label": "peptides",
        "entity_candidates": ["FullPeptideName", "Sequence"],
        "meta_columns": ["Sequence", "FullPeptideName"],
        "color": "#C2410C",
    },
    "protein": {
        "title": "Protein Matrix",
        "entity_label": "proteins",
        "entity_candidates": ["ProteinName"],
        "meta_columns": ["ProteinName"],
        "color": "#4338CA",
    },
}


workspace_dir = Path(st.session_state.get("workspace", ".")).resolve()
workflow_results_dir = workspace_dir / "openswath-workflow" / "results"


def _path_label(path_str: str | None) -> str:
    if path_str is None:
        return "None"
    path = Path(path_str)
    try:
        rel = path.relative_to(workspace_dir)
        return str(rel)
    except ValueError:
        return str(path)


def _file_mtime_ns(path_str: str | None) -> int:
    if not path_str:
        return 0
    path = Path(path_str)
    if not path.exists():
        return 0
    return path.stat().st_mtime_ns


@st.cache_data(show_spinner=False)
def _discover_workspace_results(workspace_dir_str: str) -> dict[str, list[str]]:
    workspace = Path(workspace_dir_str)
    found: dict[str, list[str]] = {
        "osw": [],
        "long_tsv": [],
        "precursor": [],
        "peptide": [],
        "protein": [],
    }
    if not workspace.exists():
        return found

    def _as_sorted_unique(paths: list[Path]) -> list[str]:
        unique = {str(path.resolve()) for path in paths if path.is_file()}
        return sorted(unique, key=lambda item: item.lower())

    found["osw"] = _as_sorted_unique(list(workspace.rglob("*.osw")))
    found["long_tsv"] = _as_sorted_unique(list(workspace.rglob("openswath_results.tsv")))
    found["precursor"] = _as_sorted_unique(
        list(workspace.rglob("openswath_results.precursor.tsv"))
    )
    found["peptide"] = _as_sorted_unique(
        list(workspace.rglob("openswath_results.peptide.tsv"))
    )
    found["protein"] = _as_sorted_unique(
        list(workspace.rglob("openswath_results.protein.tsv"))
    )
    return found


def _select_result_file(
    label: str,
    options: list[str],
    preferred_path: Path,
    key: str,
) -> str | None:
    select_options: list[str | None] = [None] + options
    preferred = str(preferred_path.resolve()) if preferred_path.exists() else None
    if preferred in select_options:
        index = select_options.index(preferred)
    elif len(select_options) > 1:
        index = 1
    else:
        index = 0
    return st.selectbox(
        label,
        options=select_options,
        index=index,
        format_func=_path_label,
        key=key,
    )


@st.cache_data(show_spinner=False)
def _load_rank1_score_data(osw_path_str: str, mtime_ns: int) -> tuple[bool, pd.DataFrame]:
    del mtime_ns
    with sqlite3.connect(osw_path_str) as conn:
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table'",
            conn,
        )
        if "SCORE_MS2" not in set(tables["name"].astype(str)):
            return False, pd.DataFrame()

        query = """
        SELECT
            SCORE_MS2.*,
            FEATURE.PRECURSOR_ID,
            FEATURE.EXP_RT,
            FEATURE.DELTA_RT,
            PRECURSOR.PRECURSOR_MZ,
            PRECURSOR.CHARGE,
            PRECURSOR.DECOY
        FROM SCORE_MS2
        INNER JOIN FEATURE ON FEATURE.ID = SCORE_MS2.FEATURE_ID
        INNER JOIN PRECURSOR ON PRECURSOR.ID = FEATURE.PRECURSOR_ID
        WHERE SCORE_MS2.RANK = 1
        """
        df = pd.read_sql_query(query, conn)

    for column in [
        "SCORE",
        "RANK",
        "PVALUE",
        "QVALUE",
        "PEP",
        "PRECURSOR_ID",
        "EXP_RT",
        "DELTA_RT",
        "PRECURSOR_MZ",
        "CHARGE",
        "DECOY",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return True, df


@st.cache_data(show_spinner=False)
def _load_tsv_dataframe(tsv_path_str: str, mtime_ns: int) -> pd.DataFrame:
    del mtime_ns
    return pd.read_csv(tsv_path_str, sep="\t")


def _build_run_mapping(run_names: list[str]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for index, run_name in enumerate(run_names, start=1):
        rows.append(
            {
                "run_label": f"run_{index}",
                "run_name": str(run_name),
                "run_basename": Path(str(run_name)).name or str(run_name),
            }
        )
    return pd.DataFrame(rows)


def _build_presence_summary(
    run_sets: list[dict[str, object]],
) -> tuple[pd.DataFrame, int]:
    union_ids: set[str] = set()
    for item in run_sets:
        union_ids |= set(item["ids"])

    union_count = len(union_ids)
    rows: list[dict[str, object]] = []
    for item in run_sets:
        ids = set(item["ids"])
        run_count = len(ids)
        rows.append(
            {
                "run_label": item["run_label"],
                "run_name": item["run_name"],
                "run_basename": item["run_basename"],
                "run_count": run_count,
                "additional_in_union": max(union_count - run_count, 0),
                "total_union": union_count,
            }
        )
    return pd.DataFrame(rows), union_count


def _prepare_long_results(df: pd.DataFrame) -> dict[str, object]:
    working = df.copy()
    if "decoy" in working.columns:
        decoy_series = pd.to_numeric(working["decoy"], errors="coerce").fillna(0)
        working = working.loc[decoy_series == 0].copy()
    if "peak_group_rank" in working.columns:
        rank_series = pd.to_numeric(working["peak_group_rank"], errors="coerce").fillna(1)
        working = working.loc[rank_series == 1].copy()

    entity_col = next(
        (
            column
            for column in ["transition_group_id", "id", "FullPeptideName", "Sequence"]
            if column in working.columns
        ),
        None,
    )
    run_col = "filename" if "filename" in working.columns else "run_id"
    if entity_col is None or run_col not in working.columns:
        return {
            "run_map": pd.DataFrame(),
            "summary": pd.DataFrame(),
            "run_sets": [],
            "entity_label": "precursors",
        }

    working = working.loc[working[run_col].notna() & working[entity_col].notna()].copy()
    working[run_col] = working[run_col].astype(str)
    working[entity_col] = working[entity_col].astype(str)
    working = working.drop_duplicates(subset=[run_col, entity_col])

    run_names = working[run_col].drop_duplicates().tolist()
    run_map = _build_run_mapping(run_names)
    run_label_map = dict(zip(run_map["run_name"], run_map["run_label"]))
    working["run_label"] = working[run_col].map(run_label_map)

    run_sets: list[dict[str, object]] = []
    for row in run_map.itertuples(index=False):
        ids = set(
            working.loc[working[run_col] == row.run_name, entity_col]
            .dropna()
            .astype(str)
            .tolist()
        )
        run_sets.append(
            {
                "run_label": row.run_label,
                "run_name": row.run_name,
                "run_basename": row.run_basename,
                "ids": ids,
            }
        )

    summary_df, _ = _build_presence_summary(run_sets)
    return {
        "run_map": run_map,
        "summary": summary_df,
        "run_sets": run_sets,
        "entity_label": "precursors",
    }


def _infer_matrix_run_columns(df: pd.DataFrame, level: str) -> list[str]:
    config = LEVEL_CONFIG[level]
    meta_columns = set(config["meta_columns"])
    run_columns: list[str] = []
    for column in df.columns:
        if column in meta_columns:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        if numeric.notna().any():
            run_columns.append(column)
    return run_columns


def _prepare_matrix_results(df: pd.DataFrame, level: str) -> dict[str, object]:
    config = LEVEL_CONFIG[level]
    entity_col = next(
        (column for column in config["entity_candidates"] if column in df.columns),
        None,
    )
    if entity_col is None:
        return {
            "run_map": pd.DataFrame(),
            "summary": pd.DataFrame(),
            "run_sets": [],
            "quant_df": pd.DataFrame(),
            "cv_df": pd.DataFrame(),
            "entity_label": config["entity_label"],
        }

    run_columns = _infer_matrix_run_columns(df, level)
    if not run_columns:
        return {
            "run_map": pd.DataFrame(),
            "summary": pd.DataFrame(),
            "run_sets": [],
            "quant_df": pd.DataFrame(),
            "cv_df": pd.DataFrame(),
            "entity_label": config["entity_label"],
        }

    working = df.copy()
    working[entity_col] = working[entity_col].astype(str)
    run_values = working[run_columns].apply(pd.to_numeric, errors="coerce")
    run_values = run_values.where(run_values > 0)

    run_map = _build_run_mapping(run_columns)
    run_sets: list[dict[str, object]] = []
    quant_frames: list[pd.DataFrame] = []
    for row in run_map.itertuples(index=False):
        values = run_values[row.run_name]
        present_mask = values.notna()
        ids = set(working.loc[present_mask, entity_col].dropna().astype(str).tolist())
        run_sets.append(
            {
                "run_label": row.run_label,
                "run_name": row.run_name,
                "run_basename": row.run_basename,
                "ids": ids,
            }
        )

        if present_mask.any():
            positive_values = values.loc[present_mask].astype(float)
            quant_frames.append(
                pd.DataFrame(
                    {
                        "run_label": row.run_label,
                        "run_name": row.run_name,
                        "run_basename": row.run_basename,
                        "intensity": positive_values.values,
                        "log2_intensity": np.log2(positive_values.values),
                    }
                )
            )

    summary_df, _ = _build_presence_summary(run_sets)

    cv_rows: list[dict[str, object]] = []
    for idx, value_row in run_values.iterrows():
        values = value_row.dropna().astype(float).values
        if len(values) < 2:
            continue
        sigma_ln = float(np.std(np.log(values), ddof=1))
        gcv_percent = 100.0 * math.sqrt(max(math.exp(sigma_ln**2) - 1.0, 0.0))
        cv_rows.append(
            {
                "entity_id": working.iloc[idx][entity_col],
                "gcv_percent": gcv_percent,
            }
        )

    quant_df = pd.concat(quant_frames, ignore_index=True) if quant_frames else pd.DataFrame()
    cv_df = pd.DataFrame(cv_rows)
    return {
        "run_map": run_map,
        "summary": summary_df,
        "run_sets": run_sets,
        "quant_df": quant_df,
        "cv_df": cv_df,
        "entity_label": config["entity_label"],
    }


def _score_histogram_figure(score_df: pd.DataFrame) -> tuple[go.Figure, float | None, float | None]:
    plotting = score_df.loc[score_df["SCORE"].notna()].copy()
    plotting["target_class"] = (
        plotting["DECOY"].fillna(0).astype(int).map({0: "Target", 1: "Decoy"})
    )
    plotting["target_class"] = plotting["target_class"].fillna("Unknown")

    cutoff_score: float | None = None
    cutoff_qvalue: float | None = None
    target_candidates = plotting.loc[
        (plotting["target_class"] == "Target") & plotting["QVALUE"].notna()
    ].copy()
    if not target_candidates.empty:
        target_candidates["qvalue_delta"] = (target_candidates["QVALUE"] - 0.01).abs()
        cutoff_row = target_candidates.sort_values(
            ["qvalue_delta", "QVALUE", "SCORE"],
            ascending=[True, True, False],
        ).iloc[0]
        cutoff_score = float(cutoff_row["SCORE"])
        cutoff_qvalue = float(cutoff_row["QVALUE"])

    fig = px.histogram(
        plotting,
        x="SCORE",
        color="target_class",
        nbins=70,
        opacity=0.65,
        barmode="overlay",
        color_discrete_map={"Target": "#0F766E", "Decoy": "#C2410C", "Unknown": "#64748B"},
    )
    fig.update_layout(
        template="plotly_white",
        title="Rank-1 MS2 score distribution",
        xaxis_title="MS2 score",
        yaxis_title="Count",
        legend_title_text="Class",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    if cutoff_score is not None:
        fig.add_vline(x=cutoff_score, line_dash="dash", line_color="#111827", line_width=2)
        fig.add_annotation(
            x=cutoff_score,
            y=1,
            yref="paper",
            text=f"Closest target q-value to 1%: score={cutoff_score:.3f}, q={cutoff_qvalue:.4f}",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#CBD5E1",
            borderwidth=1,
        )
    return fig, cutoff_score, cutoff_qvalue


def _union_bar_figure(
    summary_df: pd.DataFrame,
    entity_label: str,
    color: str,
    title: str,
) -> go.Figure:
    fig = go.Figure()
    customdata = summary_df[
        ["run_basename", "run_name", "run_count", "total_union", "additional_in_union"]
    ].to_numpy()
    fig.add_trace(
        go.Bar(
            x=summary_df["run_label"],
            y=summary_df["run_count"],
            name="Detected in run",
            marker=dict(color=color),
            customdata=customdata,
            hovertemplate=(
                "Run=%{x}<br>"
                "File=%{customdata[0]}<br>"
                "Full path=%{customdata[1]}<br>"
                f"Detected {entity_label}=%{{customdata[2]}}<br>"
                f"Union {entity_label}=%{{customdata[3]}}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Bar(
            x=summary_df["run_label"],
            y=summary_df["additional_in_union"],
            name="Additional in union",
            marker=dict(
                color=color,
                pattern=dict(shape="/", fgcolor=color, size=8, solidity=0.25),
                line=dict(color=color),
            ),
            opacity=0.22,
            customdata=customdata,
            hovertemplate=(
                "Run=%{x}<br>"
                "File=%{customdata[0]}<br>"
                f"Additional {entity_label} in union=%{{customdata[4]}}<br>"
                f"Union {entity_label}=%{{customdata[3]}}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        template="plotly_white",
        barmode="stack",
        title=title,
        xaxis_title="Run",
        yaxis_title=f"# unique {entity_label}",
        legend_title_text="Summary",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    for row in summary_df.itertuples(index=False):
        fig.add_annotation(
            x=row.run_label,
            y=row.total_union,
            text=str(int(row.total_union)),
            showarrow=False,
            yshift=12,
            font=dict(size=14, color="#111827"),
        )
        fig.add_annotation(
            x=row.run_label,
            y=max(row.run_count / 2.0, 0),
            text=str(int(row.run_count)),
            showarrow=False,
            font=dict(size=13, color="#FFFFFF"),
        )
    return fig


def _jaccard_heatmap_figure(
    run_sets: list[dict[str, object]],
    entity_label: str,
    title: str,
) -> go.Figure:
    labels = [str(item["run_label"]) for item in run_sets]
    z_values: list[list[float]] = []
    customdata: list[list[list[object]]] = []
    for row_item in run_sets:
        row_scores: list[float] = []
        row_customdata: list[list[object]] = []
        row_ids = set(row_item["ids"])
        for col_item in run_sets:
            col_ids = set(col_item["ids"])
            union_size = len(row_ids | col_ids)
            intersection_size = len(row_ids & col_ids)
            score = intersection_size / union_size if union_size else 0.0
            row_scores.append(score)
            row_customdata.append(
                [
                    row_item["run_name"],
                    col_item["run_name"],
                    intersection_size,
                    union_size,
                ]
            )
        z_values.append(row_scores)
        customdata.append(row_customdata)

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=labels,
            y=labels,
            zmin=0,
            zmax=1,
            colorscale="Blues",
            text=[[f"{value:.2f}" for value in row] for row in z_values],
            texttemplate="%{text}",
            textfont=dict(size=12),
            customdata=customdata,
            hovertemplate=(
                "Run A=%{y}<br>"
                "Run B=%{x}<br>"
                "Jaccard=%{z:.3f}<br>"
                f"Shared {entity_label}=%{{customdata[2]}}<br>"
                f"Union {entity_label}=%{{customdata[3]}}<br>"
                "Run A file=%{customdata[0]}<br>"
                "Run B file=%{customdata[1]}<extra></extra>"
            ),
            colorbar=dict(title="Jaccard"),
        )
    )
    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title="Run",
        yaxis_title="Run",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def _log_intensity_violin_figure(
    quant_df: pd.DataFrame,
    entity_label: str,
    color: str,
    title: str,
) -> go.Figure:
    fig = px.violin(
        quant_df,
        x="run_label",
        y="log2_intensity",
        color="run_label",
        box=True,
        points=False,
        category_orders={"run_label": quant_df["run_label"].drop_duplicates().tolist()},
        custom_data=["run_basename", "run_name", "intensity"],
    )
    fig.update_traces(
        hovertemplate=(
            "Run=%{x}<br>"
            "File=%{customdata[0]}<br>"
            "Full path=%{customdata[1]}<br>"
            "Intensity=%{customdata[2]:.3g}<br>"
            "log2 intensity=%{y:.2f}<extra></extra>"
        )
    )
    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title="Run",
        yaxis_title=f"log2 intensity ({entity_label})",
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_traces(marker_line_color=color)
    return fig


def _gcv_distribution_figure(cv_df: pd.DataFrame, entity_label: str, color: str, title: str) -> go.Figure:
    fig = px.histogram(
        cv_df,
        x="gcv_percent",
        nbins=60,
        marginal="box",
        color_discrete_sequence=[color],
    )
    median_gcv = cv_df["gcv_percent"].median() if not cv_df.empty else None
    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title=f"Geometric CV (%) across runs for {entity_label}",
        yaxis_title="Count",
        bargap=0.05,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    if median_gcv is not None and not math.isnan(float(median_gcv)):
        fig.add_vline(x=float(median_gcv), line_dash="dash", line_color="#111827")
        fig.add_annotation(
            x=float(median_gcv),
            y=1,
            yref="paper",
            text=f"Median gCV = {median_gcv:.1f}%",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#CBD5E1",
            borderwidth=1,
        )
    return fig


def _render_run_mapping(run_map: pd.DataFrame) -> None:
    if run_map.empty:
        return
    with st.expander("Run label mapping", expanded=False):
        st.dataframe(
            run_map.rename(
                columns={
                    "run_label": "Run",
                    "run_basename": "Filename",
                    "run_name": "Full path / source",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


st.title("📈 OpenSwath Results Viewer")
st.markdown(
    "Review saved OpenSwath OSW score distributions and PyProphet result exports "
    "from the current workspace."
)

if not workspace_dir.exists():
    st.error(f"Workspace not found: `{workspace_dir}`")
    st.stop()

discovered_files = _discover_workspace_results(str(workspace_dir))

st.subheader("0. Results Files")
st.caption(f"Workspace: `{workspace_dir}`")

col_a, col_b = st.columns(2)
with col_a:
    selected_osw = _select_result_file(
        "OSW File",
        discovered_files["osw"],
        workflow_results_dir / "openswath_results.osw",
        key="openswath_results_viewer_osw",
    )
    selected_long_tsv = _select_result_file(
        "Long Results TSV",
        discovered_files["long_tsv"],
        workflow_results_dir / "openswath_results.tsv",
        key="openswath_results_viewer_long_tsv",
    )
with col_b:
    selected_precursor_matrix = _select_result_file(
        "Precursor Matrix TSV",
        discovered_files["precursor"],
        workflow_results_dir / "openswath_results.precursor.tsv",
        key="openswath_results_viewer_precursor",
    )
    selected_peptide_matrix = _select_result_file(
        "Peptide Matrix TSV",
        discovered_files["peptide"],
        workflow_results_dir / "openswath_results.peptide.tsv",
        key="openswath_results_viewer_peptide",
    )
    selected_protein_matrix = _select_result_file(
        "Protein Matrix TSV",
        discovered_files["protein"],
        workflow_results_dir / "openswath_results.protein.tsv",
        key="openswath_results_viewer_protein",
    )

if not any(
    [
        selected_osw,
        selected_long_tsv,
        selected_precursor_matrix,
        selected_peptide_matrix,
        selected_protein_matrix,
    ]
):
    st.warning("No OpenSwath result files were found in the current workspace.")

st.markdown("---")
st.subheader("1. Scores and Quality Control")

if selected_osw:
    score_table_exists, score_df = _load_rank1_score_data(
        selected_osw,
        _file_mtime_ns(selected_osw),
    )
    if not score_table_exists:
        st.info("The selected OSW file does not contain a `SCORE_MS2` table.")
    elif score_df.empty:
        st.info("No rank-1 `SCORE_MS2` rows were found in the selected OSW file.")
    else:
        cutoff_fig, cutoff_score, cutoff_qvalue = _score_histogram_figure(score_df)
        target_count = int((score_df["DECOY"].fillna(0) == 0).sum())
        decoy_count = int((score_df["DECOY"].fillna(0) == 1).sum())
        metric_a, metric_b, metric_c = st.columns(3)
        metric_a.metric("Rank-1 targets", f"{target_count:,}")
        metric_b.metric("Rank-1 decoys", f"{decoy_count:,}")
        metric_c.metric(
            "Closest 1% cutoff",
            "n/a" if cutoff_score is None else f"{cutoff_score:.3f}",
            None if cutoff_qvalue is None else f"q={cutoff_qvalue:.4f}",
        )
        st.plotly_chart(cutoff_fig, use_container_width=True)
else:
    st.info("Select an OSW file above to inspect feature score distributions.")

st.markdown("---")
st.subheader("2. Identification and Quantification Results")

selected_matrix_files: list[tuple[str, str]] = []
if selected_precursor_matrix:
    selected_matrix_files.append(("precursor", selected_precursor_matrix))
if selected_peptide_matrix:
    selected_matrix_files.append(("peptide", selected_peptide_matrix))
if selected_protein_matrix:
    selected_matrix_files.append(("protein", selected_protein_matrix))

tab_specs: list[tuple[str, str | None]] = [(level, path) for level, path in selected_matrix_files]
if selected_long_tsv and not any(level == "precursor" for level, _ in selected_matrix_files):
    tab_specs.append(("long_tsv", selected_long_tsv))

if not tab_specs and not selected_long_tsv:
    st.info("Select one or more result TSV files above to review identifications and quantification.")
elif tab_specs:
    tab_titles = []
    for level, _path in tab_specs:
        if level == "long_tsv":
            tab_titles.append("Long TSV Fallback")
        else:
            tab_titles.append(str(LEVEL_CONFIG[level]["title"]))
    tabs = st.tabs(tab_titles)

    for tab, (level, path_str) in zip(tabs, tab_specs):
        with tab:
            if level == "long_tsv":
                long_df = _load_tsv_dataframe(path_str, _file_mtime_ns(path_str))
                prepared = _prepare_long_results(long_df)
                summary_df = prepared["summary"]
                run_sets = prepared["run_sets"]
                run_map = prepared["run_map"]
                if summary_df.empty or not run_sets:
                    st.info("The selected long TSV does not contain usable run and precursor columns.")
                    continue

                _render_run_mapping(run_map)
                mean_run_count = float(summary_df["run_count"].mean()) if not summary_df.empty else 0.0
                union_count = int(summary_df["total_union"].max()) if not summary_df.empty else 0
                metric_a, metric_b, metric_c = st.columns(3)
                metric_a.metric("Runs", f"{len(summary_df):,}")
                metric_b.metric("Mean precursors per run", f"{mean_run_count:,.1f}")
                metric_c.metric("Union precursors", f"{union_count:,}")

                plot_col_a, plot_col_b = st.columns(2)
                plot_col_a.plotly_chart(
                    _union_bar_figure(
                        summary_df,
                        entity_label="precursors",
                        color="#0F766E",
                        title="Per-run and union precursor IDs",
                    ),
                    use_container_width=True,
                )
                plot_col_b.plotly_chart(
                    _jaccard_heatmap_figure(
                        run_sets,
                        entity_label="precursors",
                        title="Jaccard similarity of precursor IDs across runs",
                    ),
                    use_container_width=True,
                )
                st.caption(
                    "This fallback view uses the long OpenSwath/PyProphet TSV because no precursor matrix was selected."
                )
                continue

            matrix_df = _load_tsv_dataframe(path_str, _file_mtime_ns(path_str))
            prepared = _prepare_matrix_results(matrix_df, level)
            summary_df = prepared["summary"]
            run_sets = prepared["run_sets"]
            run_map = prepared["run_map"]
            quant_df = prepared["quant_df"]
            cv_df = prepared["cv_df"]
            entity_label = str(prepared["entity_label"])
            color = str(LEVEL_CONFIG[level]["color"])

            if summary_df.empty or not run_sets:
                st.info(f"The selected {level} matrix does not contain usable run columns.")
                continue

            _render_run_mapping(run_map)

            mean_run_count = float(summary_df["run_count"].mean()) if not summary_df.empty else 0.0
            union_count = int(summary_df["total_union"].max()) if not summary_df.empty else 0
            median_gcv = float(cv_df["gcv_percent"].median()) if not cv_df.empty else None

            metric_a, metric_b, metric_c = st.columns(3)
            metric_a.metric("Runs", f"{len(summary_df):,}")
            metric_b.metric(f"Mean {entity_label} per run", f"{mean_run_count:,.1f}")
            metric_c.metric(
                "Median gCV",
                "n/a" if median_gcv is None else f"{median_gcv:.1f}%",
                f"Union {entity_label}: {union_count:,}",
            )

            plot_col_a, plot_col_b = st.columns(2)
            plot_col_a.plotly_chart(
                _union_bar_figure(
                    summary_df,
                    entity_label=entity_label,
                    color=color,
                    title=f"Per-run and union {entity_label}",
                ),
                use_container_width=True,
            )
            plot_col_b.plotly_chart(
                _jaccard_heatmap_figure(
                    run_sets,
                    entity_label=entity_label,
                    title=f"Jaccard similarity of {entity_label} across runs",
                ),
                use_container_width=True,
            )

            quant_col_a, quant_col_b = st.columns(2)
            if quant_df.empty:
                quant_col_a.info("No positive intensities were available for the violin plot.")
            else:
                quant_col_a.plotly_chart(
                    _log_intensity_violin_figure(
                        quant_df,
                        entity_label=entity_label,
                        color=color,
                        title=f"log2 intensity distribution for {entity_label}",
                    ),
                    use_container_width=True,
                )

            if cv_df.empty:
                quant_col_b.info("At least two quantified runs are required to calculate geometric CV.")
            else:
                quant_col_b.plotly_chart(
                    _gcv_distribution_figure(
                        cv_df,
                        entity_label=entity_label,
                        color=color,
                        title=f"Geometric CV distribution for {entity_label}",
                    ),
                    use_container_width=True,
                )
else:
    long_df = _load_tsv_dataframe(selected_long_tsv, _file_mtime_ns(selected_long_tsv))
    prepared = _prepare_long_results(long_df)
    summary_df = prepared["summary"]
    run_sets = prepared["run_sets"]
    run_map = prepared["run_map"]
    if summary_df.empty or not run_sets:
        st.info("The selected long TSV does not contain usable run and precursor columns.")
    else:
        _render_run_mapping(run_map)
        mean_run_count = float(summary_df["run_count"].mean()) if not summary_df.empty else 0.0
        union_count = int(summary_df["total_union"].max()) if not summary_df.empty else 0
        metric_a, metric_b, metric_c = st.columns(3)
        metric_a.metric("Runs", f"{len(summary_df):,}")
        metric_b.metric("Mean precursors per run", f"{mean_run_count:,.1f}")
        metric_c.metric("Union precursors", f"{union_count:,}")

        plot_col_a, plot_col_b = st.columns(2)
        plot_col_a.plotly_chart(
            _union_bar_figure(
                summary_df,
                entity_label="precursors",
                color="#0F766E",
                title="Per-run and union precursor IDs",
            ),
            use_container_width=True,
        )
        plot_col_b.plotly_chart(
            _jaccard_heatmap_figure(
                run_sets,
                entity_label="precursors",
                title="Jaccard similarity of precursor IDs across runs",
            ),
            use_container_width=True,
        )
