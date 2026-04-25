"""
content/openswath_results_comparison.py
OpenSwath Results Comparison

Compare multiple OpenSwath experiment bundles across:
- target libraries from PQP files
- precursor / peptide / protein quantification matrices

Experiment bundles can come from standard OpenSwath workflow outputs in peer
workspaces or from ad hoc uploads saved into the current workspace.
"""

from __future__ import annotations

import json
import math
import shutil
import sqlite3
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.common.common import page_setup
from src.common.standalone_report import (
    ReportFigure,
    ReportSection,
    ReportTable,
    render_report_downloads,
)


page_setup()


LEVEL_ORDER = ["precursor", "peptide", "protein"]
LEVEL_CONFIG: dict[str, dict[str, object]] = {
    "precursor": {
        "title": "Precursor",
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
        "title": "Peptide",
        "entity_label": "peptides",
        "entity_candidates": ["FullPeptideName", "Sequence"],
        "meta_columns": ["Sequence", "FullPeptideName"],
        "color": "#C2410C",
    },
    "protein": {
        "title": "Protein",
        "entity_label": "proteins",
        "entity_candidates": ["ProteinName"],
        "meta_columns": ["ProteinName"],
        "color": "#4338CA",
    },
}

VENN_FILL_COLORS = [
    "rgba(15,118,110,0.34)",
    "rgba(194,65,12,0.34)",
    "rgba(67,56,202,0.34)",
]
VENN_LINE_COLORS = ["#0F766E", "#C2410C", "#4338CA"]

workspace_dir = Path(st.session_state.get("workspace", ".")).resolve()
comparison_upload_root = (
    workspace_dir / "openswath-workflow" / "results" / "comparison_uploads"
)
standard_results_dir_name = Path("openswath-workflow", "results")
comparison_report_sections: list[ReportSection] = []


def _path_label(path_str: str | None) -> str:
    if not path_str:
        return "Missing"
    path = Path(path_str)
    try:
        return str(path.relative_to(workspace_dir))
    except ValueError:
        return str(path)


def _file_mtime_ns(path_str: str | None) -> int:
    if not path_str:
        return 0
    path = Path(path_str)
    if not path.exists():
        return 0
    return path.stat().st_mtime_ns


def _workspaces_root() -> Path:
    settings = st.session_state.get("settings", {})
    location = st.session_state.get("location")
    workspaces_dir = settings.get("workspaces_dir")
    repository_name = settings.get("repository-name", "OpenDIAKiosk")
    if workspaces_dir and location == "local":
        return Path(workspaces_dir, f"workspaces-{repository_name}").resolve()
    return workspace_dir.parent.resolve()


def _bundle_file_count(bundle: dict[str, Any]) -> int:
    core_keys = ("pqp", "osw", "precursor", "peptide", "protein")
    return sum(1 for key in core_keys if bundle["files"].get(key))


def _standard_workspace_files(candidate_workspace: Path) -> dict[str, str | None]:
    results_dir = candidate_workspace / standard_results_dir_name
    files: dict[str, str | None] = {
        "pqp": None,
        "osw": None,
        "precursor": None,
        "peptide": None,
        "protein": None,
        "long_tsv": None,
    }

    pqp_path = results_dir / "osdg" / "openswath_targets_and_decoys.pqp"
    osw_path = results_dir / "openswath_results.osw"
    long_path = results_dir / "openswath_results.tsv"
    precursor_path = results_dir / "openswath_results.precursor.tsv"
    peptide_path = results_dir / "openswath_results.peptide.tsv"
    protein_path = results_dir / "openswath_results.protein.tsv"

    for key, path in [
        ("pqp", pqp_path),
        ("osw", osw_path),
        ("long_tsv", long_path),
        ("precursor", precursor_path),
        ("peptide", peptide_path),
        ("protein", protein_path),
    ]:
        if path.exists():
            files[key] = str(path.resolve())

    return files


def _build_workspace_bundle(candidate_workspace: Path) -> dict[str, Any] | None:
    if not candidate_workspace.exists() or not candidate_workspace.is_dir():
        return None

    files = _standard_workspace_files(candidate_workspace)
    if not files["pqp"] and not any(files[level] for level in LEVEL_ORDER):
        return None

    bundle = {
        "id": f"workspace::{candidate_workspace.name}",
        "source_type": "workspace",
        "default_name": candidate_workspace.name,
        "source_label": f"Workspace: {candidate_workspace.name}",
        "workspace_name": candidate_workspace.name,
        "bundle_root": str(candidate_workspace.resolve()),
        "files": files,
        "metadata_path": None,
    }
    return bundle


def _discover_workspace_bundles() -> dict[str, dict[str, Any]]:
    root = _workspaces_root()
    if not root.exists():
        return {}

    bundles: dict[str, dict[str, Any]] = {}
    for candidate_workspace in sorted(root.iterdir(), key=lambda item: item.name.lower()):
        bundle = _build_workspace_bundle(candidate_workspace)
        if bundle is not None:
            bundles[bundle["id"]] = bundle
    return bundles


def _load_metadata(metadata_path: Path) -> dict[str, Any]:
    if not metadata_path.exists():
        return {}
    try:
        with open(metadata_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_metadata(metadata_path: Path, payload: dict[str, Any]) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _save_uploaded_file(uploaded_file, target_path: Path) -> str | None:
    if uploaded_file is None:
        return None
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "wb") as handle:
        handle.write(uploaded_file.getbuffer())
    return str(target_path.resolve())


def _save_uploaded_bundle(
    display_name: str,
    pqp_file,
    osw_file,
    precursor_file,
    peptide_file,
    protein_file,
) -> None:
    bundle_id = uuid.uuid4().hex[:12]
    bundle_dir = comparison_upload_root / bundle_id
    bundle_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "pqp": _save_uploaded_file(pqp_file, bundle_dir / "library.pqp"),
        "osw": _save_uploaded_file(osw_file, bundle_dir / "results.osw"),
        "precursor": _save_uploaded_file(
            precursor_file,
            bundle_dir / "results.precursor.tsv",
        ),
        "peptide": _save_uploaded_file(
            peptide_file,
            bundle_dir / "results.peptide.tsv",
        ),
        "protein": _save_uploaded_file(
            protein_file,
            bundle_dir / "results.protein.tsv",
        ),
        "long_tsv": None,
    }

    metadata = {
        "id": bundle_id,
        "display_name": display_name.strip() or f"uploaded_{bundle_id[:8]}",
        "created_in_workspace": workspace_dir.name,
        "files": {
            key: (Path(value).name if value else None) for key, value in files.items()
        },
        "original_names": {
            "pqp": None if pqp_file is None else pqp_file.name,
            "osw": None if osw_file is None else osw_file.name,
            "precursor": None if precursor_file is None else precursor_file.name,
            "peptide": None if peptide_file is None else peptide_file.name,
            "protein": None if protein_file is None else protein_file.name,
        },
    }
    _write_metadata(bundle_dir / "metadata.json", metadata)


def _discover_uploaded_bundles() -> dict[str, dict[str, Any]]:
    if not comparison_upload_root.exists():
        return {}

    bundles: dict[str, dict[str, Any]] = {}
    for bundle_dir in sorted(
        [path for path in comparison_upload_root.iterdir() if path.is_dir()],
        key=lambda item: item.name.lower(),
    ):
        metadata_path = bundle_dir / "metadata.json"
        metadata = _load_metadata(metadata_path)
        files = {
            "pqp": str((bundle_dir / "library.pqp").resolve())
            if (bundle_dir / "library.pqp").exists()
            else None,
            "osw": str((bundle_dir / "results.osw").resolve())
            if (bundle_dir / "results.osw").exists()
            else None,
            "precursor": str((bundle_dir / "results.precursor.tsv").resolve())
            if (bundle_dir / "results.precursor.tsv").exists()
            else None,
            "peptide": str((bundle_dir / "results.peptide.tsv").resolve())
            if (bundle_dir / "results.peptide.tsv").exists()
            else None,
            "protein": str((bundle_dir / "results.protein.tsv").resolve())
            if (bundle_dir / "results.protein.tsv").exists()
            else None,
            "long_tsv": None,
        }
        if not files["pqp"] and not any(files[level] for level in LEVEL_ORDER):
            continue

        display_name = str(metadata.get("display_name") or f"uploaded_{bundle_dir.name[:8]}")
        bundle = {
            "id": f"upload::{bundle_dir.name}",
            "source_type": "upload",
            "default_name": display_name,
            "source_label": f"Uploaded bundle: {display_name}",
            "workspace_name": workspace_dir.name,
            "bundle_root": str(bundle_dir.resolve()),
            "files": files,
            "metadata_path": str(metadata_path.resolve()),
        }
        bundles[bundle["id"]] = bundle
    return bundles


def _selected_alias(bundle: dict[str, Any]) -> str:
    key = f"openswath_compare_alias::{bundle['id']}"
    value = st.session_state.get(key, bundle["default_name"])
    text = str(value).strip()
    return text or str(bundle["default_name"])


def _resolve_selected_experiments(
    selected_ids: list[str],
    catalog: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    name_counter: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    for bundle_id in selected_ids:
        bundle = dict(catalog[bundle_id])
        requested_name = _selected_alias(bundle)
        name_counter[requested_name] += 1
        display_name = requested_name
        if name_counter[requested_name] > 1:
            display_name = f"{requested_name} ({name_counter[requested_name]})"
        bundle["requested_name"] = requested_name
        bundle["display_name"] = display_name
        selected.append(bundle)
    return selected


def _selected_summary_rows(selected_experiments: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for experiment in selected_experiments:
        files = experiment["files"]
        rows.append(
            {
                "Experiment": experiment["display_name"],
                "Source": experiment["source_label"],
                "Library PQP": "Yes" if files["pqp"] else "No",
                "OSW": "Yes" if files["osw"] else "No",
                "Precursor matrix": "Yes" if files["precursor"] else "No",
                "Peptide matrix": "Yes" if files["peptide"] else "No",
                "Protein matrix": "Yes" if files["protein"] else "No",
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def _load_library_targets(
    pqp_path_str: str,
    mtime_ns: int,
) -> dict[str, list[str]]:
    del mtime_ns
    query = """
    SELECT DISTINCT
        PRECURSOR.CHARGE AS PRECURSOR_CHARGE,
        PEPTIDE.MODIFIED_SEQUENCE AS MODIFIED_SEQUENCE,
        PROTEIN.PROTEIN_ACCESSION AS PROTEIN_ACCESSION
    FROM PRECURSOR
    INNER JOIN PRECURSOR_PEPTIDE_MAPPING
        ON PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID = PRECURSOR.ID
    INNER JOIN PEPTIDE
        ON PEPTIDE.ID = PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID
    LEFT JOIN PEPTIDE_PROTEIN_MAPPING
        ON PEPTIDE_PROTEIN_MAPPING.PEPTIDE_ID = PEPTIDE.ID
    LEFT JOIN PROTEIN
        ON PROTEIN.ID = PEPTIDE_PROTEIN_MAPPING.PROTEIN_ID
    WHERE COALESCE(PRECURSOR.DECOY, 0) = 0
      AND COALESCE(PEPTIDE.DECOY, 0) = 0
      AND (PROTEIN.ID IS NULL OR COALESCE(PROTEIN.DECOY, 0) = 0)
    """

    with sqlite3.connect(pqp_path_str) as conn:
        df = pd.read_sql_query(query, conn)

    if df.empty:
        return {"precursor": [], "peptide": [], "protein": []}

    df["MODIFIED_SEQUENCE"] = df["MODIFIED_SEQUENCE"].fillna("").astype(str)
    df["PRECURSOR_CHARGE"] = pd.to_numeric(df["PRECURSOR_CHARGE"], errors="coerce")
    precursor_ids = {
        f"{row.MODIFIED_SEQUENCE}/{int(row.PRECURSOR_CHARGE)}"
        for row in df.itertuples(index=False)
        if row.MODIFIED_SEQUENCE and pd.notna(row.PRECURSOR_CHARGE)
    }
    peptide_ids = {
        sequence for sequence in df["MODIFIED_SEQUENCE"].tolist() if sequence
    }
    protein_ids = {
        str(accession)
        for accession in df["PROTEIN_ACCESSION"].dropna().astype(str).tolist()
        if str(accession).strip()
    }

    return {
        "precursor": sorted(precursor_ids),
        "peptide": sorted(peptide_ids),
        "protein": sorted(protein_ids),
    }


@st.cache_data(show_spinner=False)
def _load_tsv_dataframe(tsv_path_str: str, mtime_ns: int) -> pd.DataFrame:
    del mtime_ns
    return pd.read_csv(tsv_path_str, sep="\t")


def _infer_matrix_run_columns(df: pd.DataFrame, level: str) -> list[str]:
    config = LEVEL_CONFIG[level]
    meta_columns = set(str(column) for column in config["meta_columns"])
    run_columns: list[str] = []
    for column in df.columns:
        if str(column) in meta_columns:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        if numeric.notna().any():
            run_columns.append(str(column))
    return run_columns


def _prepare_matrix_experiment(df: pd.DataFrame, level: str) -> dict[str, Any]:
    config = LEVEL_CONFIG[level]
    entity_col = next(
        (
            column
            for column in config["entity_candidates"]
            if str(column) in df.columns
        ),
        None,
    )
    if entity_col is None:
        return {
            "entity_ids": set(),
            "run_columns": [],
            "quant_df": pd.DataFrame(),
            "cv_df": pd.DataFrame(),
            "mean_df": pd.DataFrame(),
            "entity_label": config["entity_label"],
        }

    run_columns = _infer_matrix_run_columns(df, level)
    if not run_columns:
        return {
            "entity_ids": set(),
            "run_columns": [],
            "quant_df": pd.DataFrame(),
            "cv_df": pd.DataFrame(),
            "mean_df": pd.DataFrame(),
            "entity_label": config["entity_label"],
        }

    working = df[[entity_col] + run_columns].copy()
    working[entity_col] = working[entity_col].astype(str)
    numeric = working[run_columns].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.where(numeric > 0)
    numeric.index = working[entity_col]
    aggregated = numeric.groupby(level=0, sort=False).max()

    present_mask = aggregated.notna().any(axis=1)
    entity_ids = set(aggregated.index[present_mask].astype(str).tolist())

    stacked = aggregated.stack(dropna=True).reset_index()
    if stacked.empty:
        quant_df = pd.DataFrame(columns=["entity_id", "run_name", "run_basename", "intensity", "log2_intensity"])
    else:
        stacked.columns = ["entity_id", "run_name", "intensity"]
        stacked["run_basename"] = stacked["run_name"].map(lambda value: Path(str(value)).name)
        stacked["log2_intensity"] = np.log2(stacked["intensity"].astype(float))
        quant_df = stacked

    cv_rows: list[dict[str, Any]] = []
    mean_rows: list[dict[str, Any]] = []
    for entity_id, value_row in aggregated.iterrows():
        values = value_row.dropna().astype(float).values
        if len(values) == 0:
            continue

        log2_values = np.log2(values)
        mean_rows.append(
            {
                "entity_id": str(entity_id),
                "positive_run_count": int(len(values)),
                "mean_log2_intensity": float(np.mean(log2_values)),
            }
        )

        if len(values) < 2:
            continue

        sigma_ln = float(np.std(np.log(values), ddof=1))
        gcv_percent = 100.0 * math.sqrt(max(math.exp(sigma_ln**2) - 1.0, 0.0))
        cv_rows.append(
            {
                "entity_id": str(entity_id),
                "gcv_percent": gcv_percent,
            }
        )

    return {
        "entity_ids": entity_ids,
        "run_columns": run_columns,
        "quant_df": quant_df,
        "cv_df": pd.DataFrame(cv_rows),
        "mean_df": pd.DataFrame(mean_rows),
        "entity_label": config["entity_label"],
    }


def _build_presence_summary(set_map: dict[str, set[str]]) -> tuple[pd.DataFrame, int]:
    if not set_map:
        return pd.DataFrame(), 0
    union_ids = set().union(*set_map.values())
    union_count = len(union_ids)
    rows = []
    for experiment_name, entity_ids in set_map.items():
        rows.append(
            {
                "experiment_name": experiment_name,
                "entity_count": len(entity_ids),
                "additional_in_union": max(union_count - len(entity_ids), 0),
                "total_union": union_count,
            }
        )
    return pd.DataFrame(rows), union_count


def _build_library_example_analyte_table(
    set_map: dict[str, set[str]],
    max_examples_per_group: int = 8,
) -> pd.DataFrame:
    if not set_map:
        return pd.DataFrame()

    experiment_names = list(set_map.keys())
    all_sets = [set(ids) for ids in set_map.values()]
    shared_all = set.intersection(*all_sets) if all_sets else set()

    rows: list[dict[str, str]] = []
    for analyte_id in sorted(shared_all)[:max_examples_per_group]:
        rows.append(
            {
                "Category": "Shared by all libraries",
                "Experiment": "All selected libraries",
                "Analyte": analyte_id,
            }
        )

    for experiment_name in experiment_names:
        other_ids: set[str] = set()
        for other_name, entity_ids in set_map.items():
            if other_name != experiment_name:
                other_ids |= entity_ids
        unique_ids = set_map[experiment_name] - other_ids
        for analyte_id in sorted(unique_ids)[:max_examples_per_group]:
            rows.append(
                {
                    "Category": "Unique to one library",
                    "Experiment": experiment_name,
                    "Analyte": analyte_id,
                }
            )

    return pd.DataFrame(rows)


def _union_bar_figure(
    summary_df: pd.DataFrame,
    entity_label: str,
    color: str,
    title: str,
) -> go.Figure:
    fig = go.Figure()
    customdata = summary_df[
        ["experiment_name", "entity_count", "total_union", "additional_in_union"]
    ].to_numpy()
    fig.add_trace(
        go.Bar(
            x=summary_df["experiment_name"],
            y=summary_df["entity_count"],
            name="Present in experiment",
            marker=dict(color=color),
            customdata=customdata,
            hovertemplate=(
                "Experiment=%{customdata[0]}<br>"
                f"Detected {entity_label}=%{{customdata[1]}}<br>"
                f"Union {entity_label}=%{{customdata[2]}}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Bar(
            x=summary_df["experiment_name"],
            y=summary_df["additional_in_union"],
            name="Additional in union",
            marker=dict(
                color=color,
                pattern=dict(shape="/", fgcolor=color, size=8, solidity=0.22),
                line=dict(color=color),
            ),
            opacity=0.22,
            customdata=customdata,
            hovertemplate=(
                "Experiment=%{customdata[0]}<br>"
                f"Additional {entity_label} in union=%{{customdata[3]}}<br>"
                f"Union {entity_label}=%{{customdata[2]}}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        template="plotly_white",
        title=title,
        barmode="stack",
        xaxis_title="Experiment",
        yaxis_title=f"# unique {entity_label}",
        legend_title_text="Summary",
        margin=dict(l=20, r=20, t=60, b=80),
    )
    fig.update_xaxes(tickangle=-25)
    for row in summary_df.itertuples(index=False):
        fig.add_annotation(
            x=row.experiment_name,
            y=row.total_union,
            text=str(int(row.total_union)),
            showarrow=False,
            yshift=12,
            font=dict(size=14, color="#111827"),
        )
        fig.add_annotation(
            x=row.experiment_name,
            y=max(row.entity_count / 2.0, 0),
            text=str(int(row.entity_count)),
            showarrow=False,
            font=dict(size=12, color="#FFFFFF"),
        )
    return fig


def _jaccard_heatmap_figure(
    set_map: dict[str, set[str]],
    entity_label: str,
    title: str,
) -> go.Figure:
    labels = list(set_map.keys())
    z_values: list[list[float]] = []
    customdata: list[list[list[object]]] = []
    for row_name in labels:
        row_ids = set_map[row_name]
        row_scores: list[float] = []
        row_customdata: list[list[object]] = []
        for col_name in labels:
            col_ids = set_map[col_name]
            union_size = len(row_ids | col_ids)
            intersection_size = len(row_ids & col_ids)
            score = intersection_size / union_size if union_size else 0.0
            row_scores.append(score)
            row_customdata.append([row_name, col_name, intersection_size, union_size])
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
                "Experiment A=%{y}<br>"
                "Experiment B=%{x}<br>"
                "Jaccard=%{z:.3f}<br>"
                f"Shared {entity_label}=%{{customdata[2]}}<br>"
                f"Union {entity_label}=%{{customdata[3]}}<extra></extra>"
            ),
            colorbar=dict(title="Jaccard"),
        )
    )
    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title="Experiment",
        yaxis_title="Experiment",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def _build_baseline_overlap_summary(
    set_map: dict[str, set[str]],
    baseline_name: str,
) -> pd.DataFrame:
    baseline_ids = set_map.get(baseline_name, set())
    rows: list[dict[str, Any]] = []
    for experiment_name, entity_ids in set_map.items():
        if experiment_name == baseline_name:
            continue
        shared = len(baseline_ids & entity_ids)
        baseline_only = len(baseline_ids - entity_ids)
        experiment_only = len(entity_ids - baseline_ids)
        union_size = len(baseline_ids | entity_ids)
        jaccard = shared / union_size if union_size else np.nan
        baseline_overlap_pct = (
            100.0 * shared / len(baseline_ids) if baseline_ids else np.nan
        )
        rows.append(
            {
                "experiment_name": experiment_name,
                "shared": shared,
                "baseline_only": baseline_only,
                "experiment_only": experiment_only,
                "jaccard": jaccard,
                "baseline_overlap_pct": baseline_overlap_pct,
            }
        )
    return pd.DataFrame(rows)


def _build_unique_identification_origin_summary(
    set_map: dict[str, set[str]],
    library_set_map: dict[str, set[str]],
) -> pd.DataFrame:
    if not set_map or set(set_map.keys()) != set(library_set_map.keys()):
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for experiment_name, entity_ids in set_map.items():
        other_identified_ids: set[str] = set()
        other_library_ids: set[str] = set()
        for other_name, other_ids in set_map.items():
            if other_name == experiment_name:
                continue
            other_identified_ids |= other_ids
            other_library_ids |= library_set_map[other_name]

        unique_identified_ids = entity_ids - other_identified_ids
        library_unique_count = sum(
            1 for entity_id in unique_identified_ids if entity_id not in other_library_ids
        )
        library_shared_missed_count = len(unique_identified_ids) - library_unique_count
        unique_identified_count = len(unique_identified_ids)
        rows.append(
            {
                "experiment_name": experiment_name,
                "unique_identified_count": unique_identified_count,
                "library_unique_count": library_unique_count,
                "library_shared_missed_count": library_shared_missed_count,
                "library_unique_pct": (
                    100.0 * library_unique_count / unique_identified_count
                    if unique_identified_count
                    else 0.0
                ),
                "library_shared_missed_pct": (
                    100.0 * library_shared_missed_count / unique_identified_count
                    if unique_identified_count
                    else 0.0
                ),
            }
        )

    return pd.DataFrame(rows)


def _unique_identification_origin_figure(
    summary_df: pd.DataFrame,
    entity_label: str,
    title: str,
) -> go.Figure:
    fig = go.Figure()
    if summary_df.empty:
        fig.update_layout(
            template="plotly_white",
            title=title,
            xaxis_title="Experiment",
            yaxis_title=f"# uniquely identified {entity_label}",
        )
        return fig

    customdata = summary_df[
        [
            "experiment_name",
            "unique_identified_count",
            "library_unique_count",
            "library_shared_missed_count",
            "library_unique_pct",
            "library_shared_missed_pct",
        ]
    ].to_numpy()
    fig.add_trace(
        go.Bar(
            x=summary_df["experiment_name"],
            y=summary_df["library_unique_count"],
            name="Absent from all other libraries",
            marker=dict(color="#94A3B8"),
            customdata=customdata,
            hovertemplate=(
                "Experiment=%{customdata[0]}<br>"
                "Unique identified=%{customdata[1]}<br>"
                "Absent from all other libraries=%{customdata[2]}<br>"
                "Share of unique IDs=%{customdata[4]:.1f}%<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Bar(
            x=summary_df["experiment_name"],
            y=summary_df["library_shared_missed_count"],
            name="Present in other libraries, not identified elsewhere",
            marker=dict(color="#F59E0B"),
            customdata=customdata,
            hovertemplate=(
                "Experiment=%{customdata[0]}<br>"
                "Unique identified=%{customdata[1]}<br>"
                "Present in other libraries, not identified elsewhere=%{customdata[3]}<br>"
                "Share of unique IDs=%{customdata[5]:.1f}%<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        template="plotly_white",
        title=title,
        barmode="stack",
        xaxis_title="Experiment",
        yaxis_title=f"# uniquely identified {entity_label}",
        legend_title_text="Unique ID origin",
        margin=dict(l=20, r=20, t=60, b=80),
    )
    fig.update_xaxes(tickangle=-25)
    for row in summary_df.itertuples(index=False):
        fig.add_annotation(
            x=row.experiment_name,
            y=row.unique_identified_count,
            text=str(int(row.unique_identified_count)),
            showarrow=False,
            yshift=12,
            font=dict(size=13, color="#111827"),
        )
    return fig


def _baseline_overlap_figure(
    overlap_df: pd.DataFrame,
    baseline_name: str,
    entity_label: str,
    color: str,
    title: str,
) -> go.Figure:
    fig = go.Figure()
    if overlap_df.empty:
        fig.update_layout(
            template="plotly_white",
            title=title,
            xaxis_title="Experiment",
            yaxis_title=f"# {entity_label}",
        )
        return fig

    customdata = overlap_df[
        ["experiment_name", "shared", "baseline_only", "experiment_only", "jaccard", "baseline_overlap_pct"]
    ].to_numpy()
    fig.add_trace(
        go.Bar(
            x=overlap_df["experiment_name"],
            y=overlap_df["shared"],
            name=f"Shared with {baseline_name}",
            marker=dict(color=color),
            customdata=customdata,
            hovertemplate=(
                "Experiment=%{customdata[0]}<br>"
                "Shared=%{customdata[1]}<br>"
                "Jaccard=%{customdata[4]:.3f}<br>"
                "Baseline overlap=%{customdata[5]:.1f}%<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Bar(
            x=overlap_df["experiment_name"],
            y=overlap_df["baseline_only"],
            name=f"Only in {baseline_name}",
            marker=dict(color="#CBD5E1"),
            customdata=customdata,
            hovertemplate=(
                "Experiment=%{customdata[0]}<br>"
                f"Only in {baseline_name}=%{{customdata[2]}}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Bar(
            x=overlap_df["experiment_name"],
            y=overlap_df["experiment_only"],
            name="Only in experiment",
            marker=dict(color="#F59E0B"),
            customdata=customdata,
            hovertemplate=(
                "Experiment=%{customdata[0]}<br>"
                "Only in experiment=%{customdata[3]}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        template="plotly_white",
        title=title,
        barmode="stack",
        xaxis_title="Experiment",
        yaxis_title=f"# {entity_label}",
        legend_title_text="Baseline comparison",
        margin=dict(l=20, r=20, t=60, b=80),
    )
    fig.update_xaxes(tickangle=-25)
    for row in overlap_df.itertuples(index=False):
        fig.add_annotation(
            x=row.experiment_name,
            y=row.shared + row.baseline_only + row.experiment_only,
            text=f"J={row.jaccard:.2f}",
            showarrow=False,
            yshift=12,
            font=dict(size=12, color="#111827"),
        )
    return fig


def _venn_2_figure(set_map: dict[str, set[str]], entity_label: str, title: str) -> go.Figure:
    names = list(set_map.keys())
    first_ids = set_map[names[0]]
    second_ids = set_map[names[1]]
    counts = {
        "first_only": len(first_ids - second_ids),
        "second_only": len(second_ids - first_ids),
        "shared": len(first_ids & second_ids),
    }

    fig = go.Figure()
    for shape in [
        dict(
            type="circle",
            xref="x",
            yref="y",
            x0=1.2,
            y0=1.3,
            x1=5.3,
            y1=5.4,
            fillcolor=VENN_FILL_COLORS[0],
            line=dict(color=VENN_LINE_COLORS[0], width=3),
        ),
        dict(
            type="circle",
            xref="x",
            yref="y",
            x0=3.7,
            y0=1.3,
            x1=7.8,
            y1=5.4,
            fillcolor=VENN_FILL_COLORS[1],
            line=dict(color=VENN_LINE_COLORS[1], width=3),
        ),
    ]:
        fig.add_shape(shape)

    annotations = [
        (names[0], 2.2, 5.95, 13, VENN_LINE_COLORS[0]),
        (names[1], 6.8, 5.95, 13, VENN_LINE_COLORS[1]),
        (str(counts["first_only"]), 2.45, 3.35, 22, "#111827"),
        (str(counts["shared"]), 4.5, 3.35, 22, "#111827"),
        (str(counts["second_only"]), 6.55, 3.35, 22, "#111827"),
    ]
    for text, x, y, size, color in annotations:
        fig.add_annotation(
            x=x,
            y=y,
            text=text,
            showarrow=False,
            font=dict(size=size, color=color),
        )

    fig.update_layout(
        template="plotly_white",
        title=title,
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=False,
    )
    fig.update_xaxes(visible=False, range=[0, 9])
    fig.update_yaxes(visible=False, range=[0.5, 6.4], scaleanchor="x", scaleratio=1)
    return fig


def _venn_3_figure(set_map: dict[str, set[str]], entity_label: str, title: str) -> go.Figure:
    names = list(set_map.keys())
    set_a, set_b, set_c = set_map[names[0]], set_map[names[1]], set_map[names[2]]
    counts = {
        "a_only": len(set_a - set_b - set_c),
        "b_only": len(set_b - set_a - set_c),
        "c_only": len(set_c - set_a - set_b),
        "ab_only": len((set_a & set_b) - set_c),
        "ac_only": len((set_a & set_c) - set_b),
        "bc_only": len((set_b & set_c) - set_a),
        "abc": len(set_a & set_b & set_c),
    }

    fig = go.Figure()
    for shape in [
        dict(
            type="circle",
            xref="x",
            yref="y",
            x0=1.3,
            y0=2.1,
            x1=5.4,
            y1=6.2,
            fillcolor=VENN_FILL_COLORS[0],
            line=dict(color=VENN_LINE_COLORS[0], width=3),
        ),
        dict(
            type="circle",
            xref="x",
            yref="y",
            x0=4.1,
            y0=2.1,
            x1=8.2,
            y1=6.2,
            fillcolor=VENN_FILL_COLORS[1],
            line=dict(color=VENN_LINE_COLORS[1], width=3),
        ),
        dict(
            type="circle",
            xref="x",
            yref="y",
            x0=2.7,
            y0=0.3,
            x1=6.8,
            y1=4.4,
            fillcolor=VENN_FILL_COLORS[2],
            line=dict(color=VENN_LINE_COLORS[2], width=3),
        ),
    ]:
        fig.add_shape(shape)

    annotations = [
        (names[0], 2.6, 6.7, 13, VENN_LINE_COLORS[0]),
        (names[1], 6.8, 6.7, 13, VENN_LINE_COLORS[1]),
        (names[2], 4.7, 0.0, 13, VENN_LINE_COLORS[2]),
        (str(counts["a_only"]), 2.4, 4.85, 18, "#111827"),
        (str(counts["b_only"]), 7.15, 4.85, 18, "#111827"),
        (str(counts["c_only"]), 4.75, 1.0, 18, "#111827"),
        (str(counts["ab_only"]), 4.75, 5.0, 18, "#111827"),
        (str(counts["ac_only"]), 3.6, 3.05, 18, "#111827"),
        (str(counts["bc_only"]), 5.9, 3.05, 18, "#111827"),
        (str(counts["abc"]), 4.75, 3.95, 19, "#111827"),
    ]
    for text, x, y, size, color in annotations:
        fig.add_annotation(
            x=x,
            y=y,
            text=text,
            showarrow=False,
            font=dict(size=size, color=color),
        )

    fig.update_layout(
        template="plotly_white",
        title=title,
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=False,
    )
    fig.update_xaxes(visible=False, range=[0.5, 9.0])
    fig.update_yaxes(visible=False, range=[-0.3, 7.1], scaleanchor="x", scaleratio=1)
    return fig


def _upset_figure(set_map: dict[str, set[str]], entity_label: str, title: str) -> tuple[go.Figure, int]:
    labels = list(set_map.keys())
    universe = set().union(*set_map.values()) if set_map else set()
    membership_counts: Counter[tuple[bool, ...]] = Counter()
    for entity_id in universe:
        membership = tuple(entity_id in set_map[label] for label in labels)
        if any(membership):
            membership_counts[membership] += 1

    rows: list[dict[str, Any]] = []
    for membership, count in membership_counts.items():
        row = {
            "count": int(count),
            "degree": int(sum(membership)),
            "label": " + ".join(
                label for label, active in zip(labels, membership) if active
            ),
        }
        for label, active in zip(labels, membership):
            row[label] = bool(active)
        rows.append(row)

    upset_df = pd.DataFrame(rows)
    if upset_df.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", title=title)
        return fig, 0

    upset_df = upset_df.sort_values(
        ["count", "degree", "label"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    max_combinations = 20
    displayed_df = upset_df.head(max_combinations).copy()
    x_positions = list(range(len(displayed_df)))

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.72, 0.28],
    )
    fig.add_trace(
        go.Bar(
            x=x_positions,
            y=displayed_df["count"],
            marker=dict(color="#0F172A"),
            customdata=displayed_df[["label", "count"]].to_numpy(),
            hovertemplate=(
                "Intersection=%{customdata[0]}<br>"
                "Count=%{customdata[1]}<extra></extra>"
            ),
            text=displayed_df["count"],
            textposition="outside",
            name="Intersection size",
        ),
        row=1,
        col=1,
    )

    for label in labels:
        fig.add_trace(
            go.Scatter(
                x=x_positions,
                y=[label] * len(x_positions),
                mode="markers",
                marker=dict(color="#CBD5E1", size=10),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    for index, row in displayed_df.iterrows():
        active_labels = [label for label in labels if bool(row[label])]
        if len(active_labels) > 1:
            fig.add_trace(
                go.Scatter(
                    x=[index] * len(active_labels),
                    y=active_labels,
                    mode="lines",
                    line=dict(color="#0F172A", width=2),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
        fig.add_trace(
            go.Scatter(
                x=[index] * len(active_labels),
                y=active_labels,
                mode="markers",
                marker=dict(color="#0F172A", size=11),
                hovertemplate=(
                    "Intersection="
                    + row["label"]
                    + "<br>Experiment=%{y}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        template="plotly_white",
        title=title,
        height=520,
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=False,
    )
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=2, col=1)
    fig.update_yaxes(title_text="# entities", row=1, col=1)
    fig.update_yaxes(
        title_text="Experiment",
        row=2,
        col=1,
        categoryorder="array",
        categoryarray=list(reversed(labels)),
    )
    return fig, len(upset_df)


def _set_overlap_figure(
    set_map: dict[str, set[str]],
    entity_label: str,
    title: str,
) -> tuple[go.Figure, str | None]:
    if len(set_map) == 2:
        return _venn_2_figure(set_map, entity_label, title), (
            "Circle areas are illustrative; counts are exact."
        )
    if len(set_map) == 3:
        return _venn_3_figure(set_map, entity_label, title), (
            "Circle areas are illustrative; counts are exact."
        )
    upset_fig, total_combinations = _upset_figure(set_map, entity_label, title)
    note = None
    if total_combinations > 20:
        note = f"Showing the top 20 of {total_combinations} non-empty intersections."
    return upset_fig, note


def _log_intensity_violin_figure(
    quant_df: pd.DataFrame,
    entity_label: str,
    title: str,
) -> go.Figure:
    order = quant_df["experiment_name"].drop_duplicates().tolist()
    fig = px.violin(
        quant_df,
        x="experiment_name",
        y="log2_intensity",
        color="experiment_name",
        box=True,
        points=False,
        category_orders={"experiment_name": order},
        custom_data=["run_basename", "run_name", "entity_id", "intensity"],
    )
    fig.update_traces(
        hovertemplate=(
            "Experiment=%{x}<br>"
            "Run=%{customdata[0]}<br>"
            "Run path=%{customdata[1]}<br>"
            "Entity=%{customdata[2]}<br>"
            "Intensity=%{customdata[3]:.3g}<br>"
            "log2 intensity=%{y:.2f}<extra></extra>"
        )
    )
    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title="Experiment",
        yaxis_title=f"log2 intensity ({entity_label})",
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=80),
    )
    fig.update_xaxes(tickangle=-25)
    return fig


def _gcv_violin_figure(
    cv_df: pd.DataFrame,
    entity_label: str,
    title: str,
) -> go.Figure:
    order = cv_df["experiment_name"].drop_duplicates().tolist()
    fig = px.violin(
        cv_df,
        x="experiment_name",
        y="gcv_percent",
        color="experiment_name",
        box=True,
        points=False,
        category_orders={"experiment_name": order},
        custom_data=["entity_id"],
    )
    fig.update_traces(
        hovertemplate=(
            "Experiment=%{x}<br>"
            "Entity=%{customdata[0]}<br>"
            "Geometric CV=%{y:.2f}%<extra></extra>"
        )
    )
    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title="Experiment",
        yaxis_title=f"Geometric CV (%) across runs for {entity_label}",
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=80),
    )
    fig.update_xaxes(tickangle=-25)
    return fig


def _build_baseline_intensity_summary(
    mean_map: dict[str, pd.DataFrame],
    baseline_name: str,
) -> pd.DataFrame:
    baseline_df = mean_map.get(baseline_name, pd.DataFrame()).copy()
    if baseline_df.empty:
        return pd.DataFrame()

    baseline_ids = set(baseline_df["entity_id"].astype(str).tolist())
    baseline_df["entity_id"] = baseline_df["entity_id"].astype(str)

    rows: list[dict[str, Any]] = []
    for experiment_name, mean_df in mean_map.items():
        if experiment_name == baseline_name or mean_df.empty:
            continue

        current_df = mean_df.copy()
        current_df["entity_id"] = current_df["entity_id"].astype(str)
        current_ids = set(current_df["entity_id"].tolist())
        merged = baseline_df.merge(
            current_df,
            on="entity_id",
            how="inner",
            suffixes=("_baseline", "_experiment"),
        )
        pearson_r = np.nan
        median_abs_delta = np.nan
        if len(merged) >= 2:
            pearson_r = merged["mean_log2_intensity_baseline"].corr(
                merged["mean_log2_intensity_experiment"]
            )
        if not merged.empty:
            delta = (
                merged["mean_log2_intensity_experiment"]
                - merged["mean_log2_intensity_baseline"]
            ).abs()
            median_abs_delta = float(delta.median())

        rows.append(
            {
                "experiment_name": experiment_name,
                "shared_quantified": int(len(merged)),
                "baseline_only": int(len(baseline_ids - current_ids)),
                "experiment_only": int(len(current_ids - baseline_ids)),
                "pearson_r": pearson_r,
                "median_abs_log2_delta": median_abs_delta,
            }
        )

    return pd.DataFrame(rows)


def _baseline_intensity_scatter_figure(
    baseline_df: pd.DataFrame,
    compare_df: pd.DataFrame,
    baseline_name: str,
    compare_name: str,
    entity_label: str,
    title: str,
) -> go.Figure:
    baseline_working = baseline_df.copy()
    compare_working = compare_df.copy()
    baseline_working["entity_id"] = baseline_working["entity_id"].astype(str)
    compare_working["entity_id"] = compare_working["entity_id"].astype(str)

    merged = baseline_working.merge(
        compare_working,
        on="entity_id",
        how="inner",
        suffixes=("_baseline", "_compare"),
    )
    if merged.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", title=title)
        return fig

    merged = merged.rename(
        columns={
            "mean_log2_intensity_baseline": "baseline_log2",
            "mean_log2_intensity_compare": "compare_log2",
        }
    )
    merged["delta_log2"] = merged["compare_log2"] - merged["baseline_log2"]
    pearson_r = (
        merged["baseline_log2"].corr(merged["compare_log2"])
        if len(merged) >= 2
        else np.nan
    )

    fig = px.scatter(
        merged,
        x="baseline_log2",
        y="compare_log2",
        color="delta_log2",
        color_continuous_scale="RdBu",
        custom_data=["entity_id", "delta_log2"],
        opacity=0.65,
    )
    min_axis = float(min(merged["baseline_log2"].min(), merged["compare_log2"].min()))
    max_axis = float(max(merged["baseline_log2"].max(), merged["compare_log2"].max()))
    fig.add_trace(
        go.Scatter(
            x=[min_axis, max_axis],
            y=[min_axis, max_axis],
            mode="lines",
            line=dict(color="#111827", dash="dash"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.update_traces(
        hovertemplate=(
            "Entity=%{customdata[0]}<br>"
            f"{baseline_name}=%{{x:.2f}}<br>"
            f"{compare_name}=%{{y:.2f}}<br>"
            "Delta=%{customdata[1]:+.2f}<extra></extra>"
        )
    )
    fig.update_layout(
        template="plotly_white",
        title=title
        + (
            ""
            if pd.isna(pearson_r)
            else f" (Pearson r={float(pearson_r):.3f}, n={len(merged):,})"
        ),
        xaxis_title=f"{baseline_name} mean log2 intensity ({entity_label})",
        yaxis_title=f"{compare_name} mean log2 intensity ({entity_label})",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def _comparison_summary_bar_payload(
    summary_df: pd.DataFrame,
    entity_label: str,
    color: str,
) -> dict[str, object]:
    return {
        "dataframe": summary_df.copy(),
        "x_col": "experiment_name",
        "series": [
            {
                "column": "entity_count",
                "label": "Present in experiment",
                "color": color,
            },
            {
                "column": "additional_in_union",
                "label": "Additional in union",
                "color": color,
                "alpha": 0.22,
                "hatch": "///",
                "edgecolor": color,
            },
        ],
        "xlabel": "Experiment",
        "ylabel": f"# unique {entity_label}",
        "xtick_rotation": 25,
        "top_annotation_col": "total_union",
        "top_annotation_format": "{:,.0f}",
        "inner_annotation_col": "entity_count",
        "inner_annotation_format": "{:,.0f}",
    }


def _comparison_jaccard_payload(set_map: dict[str, set[str]]) -> dict[str, object]:
    return {
        "set_map": {str(name): set(values) for name, values in set_map.items()},
        "label_order": list(set_map.keys()),
        "xlabel": "Experiment",
        "ylabel": "Experiment",
    }


def _baseline_overlap_payload(
    overlap_df: pd.DataFrame,
    baseline_name: str,
    entity_label: str,
    color: str,
) -> dict[str, object]:
    annotation_texts = [
        "J=n/a" if pd.isna(value) else f"J={float(value):.2f}"
        for value in overlap_df["jaccard"]
    ]
    return {
        "dataframe": overlap_df.copy(),
        "x_col": "experiment_name",
        "series": [
            {
                "column": "shared",
                "label": f"Shared with {baseline_name}",
                "color": color,
            },
            {
                "column": "baseline_only",
                "label": f"Only in {baseline_name}",
                "color": "#CBD5E1",
                "edgecolor": "#CBD5E1",
            },
            {
                "column": "experiment_only",
                "label": "Only in experiment",
                "color": "#F59E0B",
                "edgecolor": "#F59E0B",
            },
        ],
        "xlabel": "Experiment",
        "ylabel": f"# {entity_label}",
        "xtick_rotation": 25,
        "annotation_texts": annotation_texts,
    }


def _unique_origin_payload(
    summary_df: pd.DataFrame,
    entity_label: str,
) -> dict[str, object]:
    return {
        "dataframe": summary_df.copy(),
        "x_col": "experiment_name",
        "series": [
            {
                "column": "library_unique_count",
                "label": "Absent from all other libraries",
                "color": "#94A3B8",
                "edgecolor": "#94A3B8",
            },
            {
                "column": "library_shared_missed_count",
                "label": "Present in other libraries, not identified elsewhere",
                "color": "#F59E0B",
                "edgecolor": "#F59E0B",
            },
        ],
        "xlabel": "Experiment",
        "ylabel": f"# uniquely identified {entity_label}",
        "xtick_rotation": 25,
        "top_annotation_col": "unique_identified_count",
        "top_annotation_format": "{:,.0f}",
    }


def _comparison_violin_payload(
    dataframe: pd.DataFrame,
    value_col: str,
    yaxis_title: str,
) -> dict[str, object]:
    return {
        "dataframe": dataframe.copy(),
        "group_col": "experiment_name",
        "value_col": value_col,
        "order": dataframe["experiment_name"].drop_duplicates().astype(str).tolist(),
        "xlabel": "Experiment",
        "ylabel": yaxis_title,
        "xtick_rotation": 25,
    }


def _baseline_intensity_scatter_payload(
    baseline_df: pd.DataFrame,
    compare_df: pd.DataFrame,
    baseline_name: str,
    compare_name: str,
    entity_label: str,
) -> dict[str, object] | None:
    baseline_working = baseline_df.copy()
    compare_working = compare_df.copy()
    baseline_working["entity_id"] = baseline_working["entity_id"].astype(str)
    compare_working["entity_id"] = compare_working["entity_id"].astype(str)

    merged = baseline_working.merge(
        compare_working,
        on="entity_id",
        how="inner",
        suffixes=("_baseline", "_compare"),
    )
    if merged.empty:
        return None

    merged = merged.rename(
        columns={
            "mean_log2_intensity_baseline": "baseline_log2",
            "mean_log2_intensity_compare": "compare_log2",
        }
    )
    merged["delta_log2"] = merged["compare_log2"] - merged["baseline_log2"]
    return {
        "dataframe": merged[["baseline_log2", "compare_log2", "delta_log2"]].copy(),
        "x_col": "baseline_log2",
        "y_col": "compare_log2",
        "color_col": "delta_log2",
        "xlabel": f"{baseline_name} mean log2 intensity ({entity_label})",
        "ylabel": f"{compare_name} mean log2 intensity ({entity_label})",
        "colorbar_label": "Delta log2",
        "identity_line": True,
        "cmap": "RdBu_r",
    }


st.title("🧪 OpenSwath Results Comparison")
st.markdown(
    """
Compare multiple OpenSwath result bundles side by side. Each bundle can include:
- the target+decoy **library PQP**
- the **OSW** SQLite result file
- exported **precursor / peptide / protein quantification matrices**

The comparison views below currently use the **target PQP** and the exported
quantification matrices. OSW files are tracked in each bundle summary so the
page can be extended with score-level comparisons later.
"""
)

with st.expander("How comparison bundles are discovered", expanded=False):
    st.markdown(
        """
        - **Existing workspaces** are discovered from sibling workspaces using the standard OpenSwath workflow output paths:
          `openswath-workflow/results/osdg/openswath_targets_and_decoys.pqp`,
          `openswath-workflow/results/openswath_results.osw`,
          and the three exported matrix TSVs.
        - **Uploaded bundles** are saved into the current workspace under
          `openswath-workflow/results/comparison_uploads/`.
        - Library comparisons use **targets only** (`DECOY = 0`).
        - Library precursor identities are compared as **modified sequence + charge**.
        """
    )

if not workspace_dir.exists():
    st.error(f"Workspace not found: `{workspace_dir}`")
    st.stop()

workspace_catalog = _discover_workspace_bundles()
uploaded_catalog = _discover_uploaded_bundles()
combined_catalog = {**workspace_catalog, **uploaded_catalog}

selection_col, upload_col = st.columns([1.2, 1.0])

with selection_col:
    st.subheader("1. Experiment Bundles")
    st.caption(f"Current workspace: `{workspace_dir.name}`")

    workspace_options = list(workspace_catalog.keys())
    uploaded_options = list(uploaded_catalog.keys())

    selected_workspace_ids = st.multiselect(
        "Existing workspace results",
        options=workspace_options,
        format_func=lambda bundle_id: (
            f"{workspace_catalog[bundle_id]['default_name']}"
            f"  ({_bundle_file_count(workspace_catalog[bundle_id])}/5 files)"
        ),
        help="Uses standard OpenSwath workflow output files from sibling workspaces.",
    )
    selected_uploaded_ids = st.multiselect(
        "Uploaded comparison bundles",
        options=uploaded_options,
        format_func=lambda bundle_id: (
            f"{uploaded_catalog[bundle_id]['default_name']}"
            f"  ({_bundle_file_count(uploaded_catalog[bundle_id])}/5 files)"
        ),
        help="Bundles uploaded from this page and saved in the current workspace.",
    )

with upload_col:
    st.subheader("2. Upload Bundle")
    with st.form("openswath_comparison_upload_bundle", clear_on_submit=True):
        upload_display_name = st.text_input(
            "Experiment name",
            placeholder="e.g. predicted library filtered",
            help="Used as the default comparison label for this uploaded bundle.",
        )
        upload_pqp = st.file_uploader(
            "Library PQP",
            type=["pqp"],
            accept_multiple_files=False,
            key="openswath_comparison_upload_pqp",
        )
        upload_osw = st.file_uploader(
            "OSW result",
            type=["osw", "sqlite", "db"],
            accept_multiple_files=False,
            key="openswath_comparison_upload_osw",
        )
        matrix_col_a, matrix_col_b, matrix_col_c = st.columns(3)
        with matrix_col_a:
            upload_precursor = st.file_uploader(
                "Precursor matrix TSV",
                type=["tsv"],
                accept_multiple_files=False,
                key="openswath_comparison_upload_precursor",
            )
        with matrix_col_b:
            upload_peptide = st.file_uploader(
                "Peptide matrix TSV",
                type=["tsv"],
                accept_multiple_files=False,
                key="openswath_comparison_upload_peptide",
            )
        with matrix_col_c:
            upload_protein = st.file_uploader(
                "Protein matrix TSV",
                type=["tsv"],
                accept_multiple_files=False,
                key="openswath_comparison_upload_protein",
            )
        submit_upload = st.form_submit_button("Save uploaded bundle", type="primary")

    if submit_upload:
        has_any_file = any(
            [
                upload_pqp,
                upload_osw,
                upload_precursor,
                upload_peptide,
                upload_protein,
            ]
        )
        if not has_any_file:
            st.warning("Select at least one file before saving a comparison bundle.")
        else:
            _save_uploaded_bundle(
                display_name=upload_display_name,
                pqp_file=upload_pqp,
                osw_file=upload_osw,
                precursor_file=upload_precursor,
                peptide_file=upload_peptide,
                protein_file=upload_protein,
            )
            st.success("Saved uploaded comparison bundle.")
            st.rerun()

    if uploaded_catalog:
        remove_uploaded_ids = st.multiselect(
            "Remove uploaded bundles",
            options=list(uploaded_catalog.keys()),
            format_func=lambda bundle_id: uploaded_catalog[bundle_id]["default_name"],
            key="openswath_comparison_remove_uploaded",
        )
        if st.button(
            "Remove selected uploaded bundles",
            disabled=not remove_uploaded_ids,
            key="openswath_comparison_remove_uploaded_button",
        ):
            for bundle_id in remove_uploaded_ids:
                bundle = uploaded_catalog[bundle_id]
                shutil.rmtree(bundle["bundle_root"], ignore_errors=True)
            st.success("Removed selected uploaded bundles.")
            st.rerun()

selected_ids = selected_workspace_ids + [
    bundle_id for bundle_id in selected_uploaded_ids if bundle_id not in selected_workspace_ids
]

if not selected_ids:
    st.info("Select at least two experiment bundles to start comparing them.")
    st.stop()

st.markdown("---")
st.subheader("3. Experiment Labels")

for bundle_id in selected_ids:
    bundle = combined_catalog[bundle_id]
    alias_key = f"openswath_compare_alias::{bundle_id}"
    if alias_key not in st.session_state:
        st.session_state[alias_key] = bundle["default_name"]
    st.text_input(
        f"Display name for {bundle['source_label']}",
        key=alias_key,
        help="If left unchanged, the default workspace or uploaded bundle name is used.",
    )

selected_experiments = _resolve_selected_experiments(selected_ids, combined_catalog)

st.markdown("##### Selected experiment summary")
selected_summary_df = _selected_summary_rows(selected_experiments)
st.dataframe(
    selected_summary_df,
    use_container_width=True,
    hide_index=True,
)
comparison_report_sections.append(
    ReportSection(
        title="Selected Experiment Bundles",
        description="Experiment bundles included in the current comparison session.",
        tables=[
            ReportTable(
                title="Selected experiment summary",
                dataframe=selected_summary_df,
                max_rows=None,
            )
        ],
    )
)

if len(selected_experiments) < 2:
    st.info("Select at least two experiment bundles to generate comparison plots.")
    st.stop()

baseline_id = st.selectbox(
    "Baseline comparison run / experiment",
    options=selected_ids,
    format_func=lambda bundle_id: next(
        experiment["display_name"]
        for experiment in selected_experiments
        if experiment["id"] == bundle_id
    ),
    key="openswath_comparison_baseline_id",
)
baseline_display_name = next(
    experiment["display_name"]
    for experiment in selected_experiments
    if experiment["id"] == baseline_id
)

st.markdown("---")
st.subheader("4. Library Targets")

library_target_map: dict[str, dict[str, set[str]]] = {}
library_experiments = [
    experiment for experiment in selected_experiments if experiment["files"]["pqp"]
]
if len(library_experiments) < 2:
    st.info("At least two selected experiments need a PQP file for the library comparison section.")
else:
    for experiment in library_experiments:
        loaded = _load_library_targets(
            experiment["files"]["pqp"],
            _file_mtime_ns(experiment["files"]["pqp"]),
        )
        library_target_map[experiment["display_name"]] = {
            level: set(loaded[level]) for level in LEVEL_ORDER
        }

    library_baseline_name = baseline_display_name
    if library_baseline_name not in library_target_map:
        library_baseline_name = next(iter(library_target_map.keys()))

    st.caption(
        "Library comparisons use target entries only (`DECOY = 0`). "
        "Precursor identities are compared as modified sequence + charge."
    )
    if library_baseline_name != baseline_display_name:
        st.caption(
            f"Selected baseline `{baseline_display_name}` has no PQP in the active comparison set. "
            f"Using `{library_baseline_name}` for library-baseline summaries."
        )

    library_tabs = st.tabs(
        [str(LEVEL_CONFIG[level]["title"]) for level in LEVEL_ORDER]
    )
    for tab, level in zip(library_tabs, LEVEL_ORDER):
        with tab:
            set_map = {
                experiment_name: library_target_map[experiment_name][level]
                for experiment_name in library_target_map
            }
            entity_label = str(LEVEL_CONFIG[level]["entity_label"])
            color = str(LEVEL_CONFIG[level]["color"])
            summary_df, union_count = _build_presence_summary(set_map)
            overlap_fig, overlap_note = _set_overlap_figure(
                set_map,
                entity_label=entity_label,
                title=f"Target overlap across experiments ({entity_label})",
            )
            overlap_summary_df = _build_baseline_overlap_summary(
                set_map,
                baseline_name=library_baseline_name,
            )

            metric_a, metric_b, metric_c = st.columns(3)
            metric_a.metric("Experiments", f"{len(set_map):,}")
            metric_b.metric(
                f"Mean {entity_label} per experiment",
                f"{summary_df['entity_count'].mean():,.1f}",
            )
            metric_c.metric("Union", f"{union_count:,}")

            overlap_plot = overlap_fig
            union_plot = _union_bar_figure(
                summary_df,
                entity_label=entity_label,
                color=color,
                title=f"Union coverage across experiments ({entity_label})",
            )
            plot_col_a, plot_col_b = st.columns(2)
            plot_col_a.plotly_chart(overlap_plot, use_container_width=True)
            plot_col_b.plotly_chart(
                union_plot,
                use_container_width=True,
            )
            if overlap_note:
                plot_col_a.caption(overlap_note)

            jaccard_plot = _jaccard_heatmap_figure(
                set_map,
                entity_label=entity_label,
                title=f"Jaccard similarity across experiments ({entity_label})",
            )
            detail_col_a, detail_col_b = st.columns(2)
            detail_col_a.plotly_chart(
                jaccard_plot,
                use_container_width=True,
            )
            baseline_overlap_plot = None
            if overlap_summary_df.empty:
                detail_col_b.info("A baseline comparison plot needs at least one non-baseline experiment.")
            else:
                baseline_overlap_plot = _baseline_overlap_figure(
                    overlap_summary_df,
                    baseline_name=library_baseline_name,
                    entity_label=entity_label,
                    color=color,
                    title=f"Baseline overlap vs {library_baseline_name}",
                )
                detail_col_b.plotly_chart(
                    baseline_overlap_plot,
                    use_container_width=True,
                )
                detail_col_b.dataframe(
                    overlap_summary_df.rename(
                        columns={
                            "experiment_name": "Experiment",
                            "shared": "Shared",
                            "baseline_only": f"Only in {library_baseline_name}",
                            "experiment_only": "Only in experiment",
                            "jaccard": "Jaccard",
                            "baseline_overlap_pct": f"% of {library_baseline_name}",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            st.markdown("##### Example shared and unique analytes")
            example_df = _build_library_example_analyte_table(set_map)
            if example_df.empty:
                st.info(
                    f"No shared-by-all or experiment-unique {entity_label} examples were found at this level."
                )
            else:
                st.caption(
                    "Shared examples are present in every selected library. "
                    "Unique examples are present in exactly one selected library."
                )
                st.dataframe(
                    example_df,
                    use_container_width=True,
                    hide_index=True,
                )

            report_figures = [
                ReportFigure(
                    title=f"Target overlap across experiments ({entity_label})",
                    figure=overlap_plot,
                    caption=overlap_note,
                    pdf_kind="set_overlap",
                    pdf_payload={
                        "set_map": {
                            str(name): set(values)
                            for name, values in set_map.items()
                        },
                        "max_intersections": 20,
                    },
                ),
                ReportFigure(
                    title=f"Union coverage across experiments ({entity_label})",
                    figure=union_plot,
                    pdf_kind="stacked_bar",
                    pdf_payload=_comparison_summary_bar_payload(
                        summary_df,
                        entity_label,
                        color,
                    ),
                ),
                ReportFigure(
                    title=f"Jaccard similarity across experiments ({entity_label})",
                    figure=jaccard_plot,
                    pdf_kind="jaccard_heatmap",
                    pdf_payload=_comparison_jaccard_payload(set_map),
                ),
            ]
            if baseline_overlap_plot is not None:
                report_figures.append(
                    ReportFigure(
                        title=f"Baseline overlap vs {library_baseline_name}",
                        figure=baseline_overlap_plot,
                        pdf_kind="stacked_bar",
                        pdf_payload=_baseline_overlap_payload(
                            overlap_summary_df,
                            library_baseline_name,
                            entity_label,
                            color,
                        ),
                    )
                )

            library_summary_table = pd.DataFrame(
                [
                    {
                        "Experiments": len(set_map),
                        f"Mean {entity_label} per experiment": f"{summary_df['entity_count'].mean():,.1f}",
                        "Union": union_count,
                        "Baseline": library_baseline_name,
                    }
                ]
            )
            report_tables = [
                ReportTable(
                    title="Library comparison summary",
                    dataframe=library_summary_table,
                    max_rows=None,
                ),
                ReportTable(
                    title="Shared and unique analyte examples",
                    dataframe=example_df,
                    caption=(
                        "Shared examples are present in every selected library. "
                        "Unique examples are present in exactly one selected library."
                    )
                    if not example_df.empty
                    else "No shared-by-all or unique analyte examples were found at this level.",
                    max_rows=None,
                ),
            ]
            if not overlap_summary_df.empty:
                report_tables.append(
                    ReportTable(
                        title="Baseline overlap summary",
                        dataframe=overlap_summary_df.rename(
                            columns={
                                "experiment_name": "Experiment",
                                "shared": "Shared",
                                "baseline_only": f"Only in {library_baseline_name}",
                                "experiment_only": "Only in experiment",
                                "jaccard": "Jaccard",
                                "baseline_overlap_pct": f"% of {library_baseline_name}",
                            }
                        ),
                        max_rows=None,
                    )
                )
            comparison_report_sections.append(
                ReportSection(
                    title=f"Library Targets: {LEVEL_CONFIG[level]['title']}",
                    description=(
                        "Library target overlap across the selected experiment bundles. "
                        "Only target entries (`DECOY = 0`) are included."
                    ),
                    figures=report_figures,
                    tables=report_tables,
                )
            )

st.markdown("---")
st.subheader("5. Identification and Quantification")
st.caption(
    "Identification overlap is computed from the union of non-zero matrix entries within each experiment. "
    "Quantification plots use all positive matrix intensities across runs in each experiment."
)

result_tabs = st.tabs([str(LEVEL_CONFIG[level]["title"]) for level in LEVEL_ORDER])

for tab, level in zip(result_tabs, LEVEL_ORDER):
    with tab:
        valid_experiments = [
            experiment for experiment in selected_experiments if experiment["files"][level]
        ]
        if len(valid_experiments) < 2:
            st.info(
                f"At least two selected experiments need a {level} matrix to compare this level."
            )
            continue

        prepared_map: dict[str, dict[str, Any]] = {}
        for experiment in valid_experiments:
            df = _load_tsv_dataframe(
                experiment["files"][level],
                _file_mtime_ns(experiment["files"][level]),
            )
            prepared = _prepare_matrix_experiment(df, level)
            if prepared["entity_ids"] or not prepared["quant_df"].empty:
                prepared_map[experiment["display_name"]] = prepared

        if len(prepared_map) < 2:
            st.info(f"Could not derive at least two usable {level} matrices from the selected experiments.")
            continue

        entity_label = str(LEVEL_CONFIG[level]["entity_label"])
        color = str(LEVEL_CONFIG[level]["color"])
        set_map = {
            experiment_name: prepared["entity_ids"]
            for experiment_name, prepared in prepared_map.items()
        }
        summary_df, union_count = _build_presence_summary(set_map)
        overlap_fig, overlap_note = _set_overlap_figure(
            set_map,
            entity_label=entity_label,
            title=f"Identification overlap across experiments ({entity_label})",
        )

        quant_frames: list[pd.DataFrame] = []
        cv_frames: list[pd.DataFrame] = []
        mean_map: dict[str, pd.DataFrame] = {}
        run_count_rows: list[dict[str, Any]] = []
        for experiment_name, prepared in prepared_map.items():
            run_count_rows.append(
                {"experiment_name": experiment_name, "run_count": len(prepared["run_columns"])}
            )

            if not prepared["quant_df"].empty:
                quant_frame = prepared["quant_df"].copy()
                quant_frame["experiment_name"] = experiment_name
                quant_frames.append(quant_frame)

            if not prepared["cv_df"].empty:
                cv_frame = prepared["cv_df"].copy()
                cv_frame["experiment_name"] = experiment_name
                cv_frames.append(cv_frame)

            mean_map[experiment_name] = prepared["mean_df"].copy()

        quant_df = (
            pd.concat(quant_frames, ignore_index=True)
            if quant_frames
            else pd.DataFrame()
        )
        cv_df = (
            pd.concat(cv_frames, ignore_index=True)
            if cv_frames
            else pd.DataFrame()
        )
        run_count_df = pd.DataFrame(run_count_rows)

        level_baseline_name = baseline_display_name
        if level_baseline_name not in prepared_map:
            level_baseline_name = next(iter(prepared_map.keys()))
        if level_baseline_name != baseline_display_name:
            st.caption(
                f"Selected baseline `{baseline_display_name}` does not have a usable {level} matrix here. "
                f"Using `{level_baseline_name}` for baseline summaries in this tab."
            )

        baseline_overlap_df = _build_baseline_overlap_summary(
            set_map,
            baseline_name=level_baseline_name,
        )
        intensity_summary_df = _build_baseline_intensity_summary(
            mean_map,
            baseline_name=level_baseline_name,
        )

        metric_a, metric_b, metric_c, metric_d = st.columns(4)
        metric_a.metric("Experiments", f"{len(prepared_map):,}")
        metric_b.metric(
            f"Mean {entity_label} per experiment",
            f"{summary_df['entity_count'].mean():,.1f}",
        )
        metric_c.metric("Union", f"{union_count:,}")
        metric_d.metric(
            "Median runs / experiment",
            f"{run_count_df['run_count'].median():.1f}",
        )

        overlap_plot = overlap_fig
        union_plot = _union_bar_figure(
            summary_df,
            entity_label=entity_label,
            color=color,
            title=f"Union coverage across experiments ({entity_label})",
        )
        plot_col_a, plot_col_b = st.columns(2)
        plot_col_a.plotly_chart(overlap_plot, use_container_width=True)
        plot_col_b.plotly_chart(
            union_plot,
            use_container_width=True,
        )
        if overlap_note:
            plot_col_a.caption(overlap_note)

        jaccard_plot = _jaccard_heatmap_figure(
            set_map,
            entity_label=entity_label,
            title=f"Jaccard similarity across experiments ({entity_label})",
        )
        plot_col_c, plot_col_d = st.columns(2)
        plot_col_c.plotly_chart(
            jaccard_plot,
            use_container_width=True,
        )
        baseline_overlap_plot = None
        if baseline_overlap_df.empty:
            plot_col_d.info("A baseline overlap view needs at least one non-baseline experiment.")
        else:
            baseline_overlap_plot = _baseline_overlap_figure(
                baseline_overlap_df,
                baseline_name=level_baseline_name,
                entity_label=entity_label,
                color=color,
                title=f"Identification overlap vs {level_baseline_name}",
            )
            plot_col_d.plotly_chart(
                baseline_overlap_plot,
                use_container_width=True,
            )

        unique_origin_library_map = {
            experiment_name: library_target_map[experiment_name][level]
            for experiment_name in prepared_map
            if experiment_name in library_target_map
        }
        unique_origin_col_a, unique_origin_col_b = st.columns(2)
        if set(unique_origin_library_map.keys()) != set(prepared_map.keys()):
            missing_library_names = [
                experiment_name
                for experiment_name in prepared_map
                if experiment_name not in unique_origin_library_map
            ]
            unique_origin_col_a.info(
                "Unique-identification attribution needs a PQP library for every compared experiment at this level."
            )
            if missing_library_names:
                unique_origin_col_b.caption(
                    "Missing library support for: "
                    + ", ".join(f"`{name}`" for name in missing_library_names)
                )
            unique_origin_df = pd.DataFrame()
            unique_origin_plot = None
        else:
            unique_origin_df = _build_unique_identification_origin_summary(
                set_map,
                unique_origin_library_map,
            )
            unique_origin_plot = _unique_identification_origin_figure(
                unique_origin_df,
                entity_label=entity_label,
                title=f"Origin of uniquely identified {entity_label}",
            )
            unique_origin_col_a.plotly_chart(
                unique_origin_plot,
                use_container_width=True,
            )
            unique_origin_col_b.dataframe(
                unique_origin_df.rename(
                    columns={
                        "experiment_name": "Experiment",
                        "unique_identified_count": "Unique identified",
                        "library_unique_count": "Absent from other libraries",
                        "library_shared_missed_count": "Present in other libraries",
                        "library_unique_pct": "% absent from other libraries",
                        "library_shared_missed_pct": "% present in other libraries",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
            unique_origin_col_b.caption(
                "Unique identified entities are those found in only one experiment result set. "
                "The breakdown separates library-exclusive analytes from analytes that were available "
                "in other libraries but not identified there."
            )

        quant_col_a, quant_col_b = st.columns(2)
        quant_plot = None
        if quant_df.empty:
            quant_col_a.info("No positive intensities were available for the log-intensity violin plot.")
        else:
            quant_plot = _log_intensity_violin_figure(
                quant_df,
                entity_label=entity_label,
                title=f"log2 intensity distribution across experiments ({entity_label})",
            )
            quant_col_a.plotly_chart(
                quant_plot,
                use_container_width=True,
            )

        gcv_plot = None
        if cv_df.empty:
            quant_col_b.info("At least two quantified runs per entity are required to calculate geometric CV.")
        else:
            gcv_plot = _gcv_violin_figure(
                cv_df,
                entity_label=entity_label,
                title=f"Geometric CV distribution across experiments ({entity_label})",
            )
            quant_col_b.plotly_chart(
                gcv_plot,
                use_container_width=True,
            )

        summary_col_a, summary_col_b = st.columns(2)
        if intensity_summary_df.empty:
            summary_col_a.info("No baseline intensity summary could be calculated for this level.")
        else:
            display_df = intensity_summary_df.rename(
                columns={
                    "experiment_name": "Experiment",
                    "shared_quantified": "Shared quantified",
                    "baseline_only": f"Only in {level_baseline_name}",
                    "experiment_only": "Only in experiment",
                    "pearson_r": "Pearson r",
                    "median_abs_log2_delta": "Median |delta log2|",
                }
            )
            summary_col_a.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
            )

        compare_candidates = [
            experiment_name
            for experiment_name in prepared_map.keys()
            if experiment_name != level_baseline_name
        ]
        if not compare_candidates:
            summary_col_b.info("A baseline scatter plot needs at least one non-baseline experiment.")
            scatter_plot = None
        else:
            compare_experiment_name = st.selectbox(
                f"Scatter comparison against baseline ({LEVEL_CONFIG[level]['title']})",
                options=compare_candidates,
                key=f"openswath_comparison_scatter::{level}",
            )
            baseline_mean_df = mean_map.get(level_baseline_name, pd.DataFrame())
            compare_mean_df = mean_map.get(compare_experiment_name, pd.DataFrame())
            if baseline_mean_df.empty or compare_mean_df.empty:
                summary_col_b.info("No shared quantified entities were available for the baseline scatter plot.")
                scatter_plot = None
            else:
                scatter_plot = _baseline_intensity_scatter_figure(
                    baseline_df=baseline_mean_df,
                    compare_df=compare_mean_df,
                    baseline_name=level_baseline_name,
                    compare_name=compare_experiment_name,
                    entity_label=entity_label,
                    title=f"Baseline mean intensity comparison ({entity_label})",
                )
                summary_col_b.plotly_chart(
                    scatter_plot,
                    use_container_width=True,
                )

        report_figures = [
            ReportFigure(
                title=f"Identification overlap across experiments ({entity_label})",
                figure=overlap_plot,
                caption=overlap_note,
                pdf_kind="set_overlap",
                pdf_payload={
                    "set_map": {
                        str(name): set(values)
                        for name, values in set_map.items()
                    },
                    "max_intersections": 20,
                },
            ),
            ReportFigure(
                title=f"Union coverage across experiments ({entity_label})",
                figure=union_plot,
                pdf_kind="stacked_bar",
                pdf_payload=_comparison_summary_bar_payload(
                    summary_df,
                    entity_label,
                    color,
                ),
            ),
            ReportFigure(
                title=f"Jaccard similarity across experiments ({entity_label})",
                figure=jaccard_plot,
                pdf_kind="jaccard_heatmap",
                pdf_payload=_comparison_jaccard_payload(set_map),
            ),
        ]
        if baseline_overlap_plot is not None:
            report_figures.append(
                ReportFigure(
                    title=f"Identification overlap vs {level_baseline_name}",
                    figure=baseline_overlap_plot,
                    pdf_kind="stacked_bar",
                    pdf_payload=_baseline_overlap_payload(
                        baseline_overlap_df,
                        level_baseline_name,
                        entity_label,
                        color,
                    ),
                )
            )
        if unique_origin_plot is not None:
            report_figures.append(
                ReportFigure(
                    title=f"Origin of uniquely identified {entity_label}",
                    figure=unique_origin_plot,
                    pdf_kind="stacked_bar",
                    pdf_payload=_unique_origin_payload(unique_origin_df, entity_label),
                )
            )
        if quant_plot is not None:
            report_figures.append(
                ReportFigure(
                    title=f"log2 intensity distribution across experiments ({entity_label})",
                    figure=quant_plot,
                    pdf_kind="violin",
                    pdf_payload=_comparison_violin_payload(
                        quant_df,
                        "log2_intensity",
                        f"log2 intensity ({entity_label})",
                    ),
                )
            )
        if gcv_plot is not None:
            report_figures.append(
                ReportFigure(
                    title=f"Geometric CV distribution across experiments ({entity_label})",
                    figure=gcv_plot,
                    pdf_kind="violin",
                    pdf_payload=_comparison_violin_payload(
                        cv_df,
                        "gcv_percent",
                        f"Geometric CV (%) across runs for {entity_label}",
                    ),
                )
            )
        if scatter_plot is not None:
            scatter_payload = _baseline_intensity_scatter_payload(
                baseline_mean_df,
                compare_mean_df,
                level_baseline_name,
                compare_experiment_name,
                entity_label,
            )
            report_figures.append(
                ReportFigure(
                    title=f"Baseline mean intensity comparison ({entity_label})",
                    figure=scatter_plot,
                    pdf_kind="scatter",
                    pdf_payload={} if scatter_payload is None else scatter_payload,
                )
            )

        summary_metrics_table = pd.DataFrame(
            [
                {
                    "Experiments": len(prepared_map),
                    f"Mean {entity_label} per experiment": f"{summary_df['entity_count'].mean():,.1f}",
                    "Union": union_count,
                    "Median runs / experiment": f"{run_count_df['run_count'].median():.1f}",
                    "Baseline": level_baseline_name,
                }
            ]
        )
        report_tables = [
            ReportTable(
                title="Result comparison summary",
                dataframe=summary_metrics_table,
                max_rows=None,
            )
        ]
        if not baseline_overlap_df.empty:
            report_tables.append(
                ReportTable(
                    title="Baseline identification overlap summary",
                    dataframe=baseline_overlap_df.rename(
                        columns={
                            "experiment_name": "Experiment",
                            "shared": "Shared",
                            "baseline_only": f"Only in {level_baseline_name}",
                            "experiment_only": "Only in experiment",
                            "jaccard": "Jaccard",
                            "baseline_overlap_pct": f"% of {level_baseline_name}",
                        }
                    ),
                    max_rows=None,
                )
            )
        if not unique_origin_df.empty:
            report_tables.append(
                ReportTable(
                    title="Unique identification origin summary",
                    dataframe=unique_origin_df.rename(
                        columns={
                            "experiment_name": "Experiment",
                            "unique_identified_count": "Unique identified",
                            "library_unique_count": "Absent from other libraries",
                            "library_shared_missed_count": "Present in other libraries",
                            "library_unique_pct": "% absent from other libraries",
                            "library_shared_missed_pct": "% present in other libraries",
                        }
                    ),
                    max_rows=None,
                )
            )
        if not intensity_summary_df.empty:
            report_tables.append(
                ReportTable(
                    title="Baseline intensity summary",
                    dataframe=intensity_summary_df.rename(
                        columns={
                            "experiment_name": "Experiment",
                            "shared_quantified": "Shared quantified",
                            "baseline_only": f"Only in {level_baseline_name}",
                            "experiment_only": "Only in experiment",
                            "pearson_r": "Pearson r",
                            "median_abs_log2_delta": "Median |delta log2|",
                        }
                    ),
                    max_rows=None,
                )
            )

        comparison_report_sections.append(
            ReportSection(
                title=f"Identification and Quantification: {LEVEL_CONFIG[level]['title']}",
                description=(
                    "Identification overlap and quantitative consistency across the selected experiment bundles."
                ),
                figures=report_figures,
                tables=report_tables,
            )
        )

comparison_report_metadata = {
    "Workspace": workspace_dir.name,
    "Baseline experiment": baseline_display_name,
    "Selected experiments": ", ".join(
        experiment["display_name"] for experiment in selected_experiments
    ),
}
comparison_report_basename = (
    f"{workspace_dir.name.replace(' ', '_')}_openswath_results_comparison_report"
)
render_report_downloads(
    report_key="openswath_results_comparison_report",
    title="OpenSwath Results Comparison Report",
    basename=comparison_report_basename,
    sections=comparison_report_sections,
    subtitle="Standalone export from the OpenSwath Results Comparison page.",
    metadata=comparison_report_metadata,
)
