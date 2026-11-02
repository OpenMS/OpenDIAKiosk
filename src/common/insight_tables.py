"""Streamed result tables built on openms-insight.

``st.dataframe`` ships the whole table to the browser, which stalls on
full-scale runs. An openms-insight ``Table`` caches the data on disk once and
sends one page at a time; its download button exports every row and column.
"""
import hashlib
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import polars as pl
import streamlit as st
from openms_insight import Table

ROW_FIELD = "row"
# Folder in the workspace holding the caches. Pages that list the workspace's
# parquet files (the XIC viewer) skip it.
INSIGHT_CACHE_DIRNAME = "insight_tables"
# Rows sampled when fingerprinting a dataframe that has no source files.
_FINGERPRINT_SAMPLE_ROWS = 2000


def _fingerprint_dataframe(df: pd.DataFrame) -> str:
    """Cheap content hash of a dataframe, for tables computed in the session.

    Hashing every cell on every Streamlit rerun would defeat the purpose for
    large tables, so this combines the shape/schema, the column sums and an
    evenly spaced sample of rows (always including the first and last row).
    """
    digest = hashlib.sha256()
    digest.update(repr((df.shape, [str(c) for c in df.columns], [str(t) for t in df.dtypes])).encode())
    digest.update(df.select_dtypes(include="number").sum().to_numpy(dtype="float64").tobytes())
    if len(df):
        step = max(len(df) // _FINGERPRINT_SAMPLE_ROWS, 1)
        rows = df.iloc[::step].astype(str)
        digest.update(pd.util.hash_pandas_object(rows, index=False).values.tobytes())
        digest.update(df.iloc[[-1]].astype(str).to_csv(index=False).encode())
    return digest.hexdigest()


def plot_cache(name: str, df: pd.DataFrame) -> dict:
    """``cache_id``/``cache_path`` for an insight plot of a table computed in the session.

    Keying the cache on the table's content rebuilds it when an upstream step
    changes the data, and reuses it on every other rerun. Rebuilding on each
    rerun (``regenerate_cache=True``) instead hands the browser new data every
    time the component asks for it, and the plot never leaves "Loading...".
    """
    return {
        "cache_id": f"{name}_{_fingerprint_dataframe(df)[:16]}",
        "cache_path": str(_default_cache_dir()),
    }


def _default_cache_dir() -> Path:
    """Cache location for tables that are not tied to a workflow result directory."""
    workspace = st.session_state.get("workspace")
    base = Path(workspace) if workspace else Path(tempfile.gettempdir(), "opendiakiosk")
    return base / INSIGHT_CACHE_DIRNAME


def _cache_key(source_files, extra: str) -> str:
    """Hash the source files' identity and mtimes, so new results get a new cache."""
    parts = [extra]
    for path in source_files:
        path = Path(path)
        mtime = path.stat().st_mtime_ns if path.exists() else 0
        parts.append(f"{path}:{mtime}")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:10]


def _to_polars(df: pd.DataFrame, column_overrides: dict | None = None) -> tuple[pl.DataFrame, list[dict]]:
    """Convert to polars with Tabulator-safe field names and column definitions.

    Tabulator reads a "." in a field name as a nested lookup, so such columns get
    a safe field name and keep their original name as the column title.
    """
    # A named index usually holds the IDs, so keep it as a column
    df = df.reset_index(drop=all(n is None for n in df.index.names))
    renames, column_definitions, used = {}, [], {ROW_FIELD}
    for column in df.columns:
        field = str(column).replace(".", "_")
        while field in used:
            field = f"{field}_"
        used.add(field)
        renames[column] = field
        col_def = {"field": field, "title": str(column), "headerTooltip": True}
        if pd.api.types.is_numeric_dtype(df[column]):
            col_def["sorter"] = "number"
            col_def["hozAlign"] = "right"
        col_def.update((column_overrides or {}).get(column, {}))
        column_definitions.append(col_def)
    df = df.rename(columns=renames)
    for column in df.columns:
        if df[column].dtype == object:
            df[column] = df[column].map(lambda v: None if v is None or v != v else str(v))
    pl_df = pl.from_pandas(df).with_row_index(ROW_FIELD)
    return pl_df, column_definitions


def show_insight_table(
    df: pd.DataFrame,
    name: str,
    cache_dir=None,
    source_files=(),
    extra_key: str = "",
    title: str | None = None,
    height: int = 500,
    page_size: int = 100,
    state_manager=None,
    column_overrides: dict | None = None,
    **table_kwargs,
) -> None:
    """Render ``df`` as a server-side paginated openms-insight table.

    The cache is keyed on ``source_files`` (path and mtime) and ``extra_key``;
    when either changes (a rerun, new group assignments) a new cache replaces the
    old one, so the page never shows a previous run's rows. Without
    ``source_files`` (a table computed in the session, e.g. a downstream
    analysis step) it is keyed on a fingerprint of the table's content instead.

    Insight components request a rerun the first time they render, so don't
    draw one only inside ``if st.button(...):``; keep the result in session
    state and render it outside the button block.

    Args:
        df: Table to show.
        name: Stable name for this table; must be unique per page.
        cache_dir: Directory holding openms-insight caches. Defaults to
            ``<workspace>/insight_tables``.
        source_files: Files the table is derived from.
        extra_key: Anything else the table depends on (e.g. parameters).
        title: Title, also used as the CSV download's file name.
        height: Component height in pixels.
        page_size: Rows per page.
        state_manager: Shared ``StateManager`` for tables linked through
            ``filters``/``interactivity``; defaults to openms-insight's own.
        column_overrides: Extra Tabulator column settings per column name,
            e.g. a ``formatter``. openms-insight stores floats as float32, so
            a column of rounded values needs one to display without noise.
        **table_kwargs: Passed on to ``Table`` (e.g. ``initial_sort``).
    """
    cache_dir = Path(cache_dir) if cache_dir else _default_cache_dir()
    if not source_files:
        extra_key = f"{extra_key}|{_fingerprint_dataframe(df)}"
    cache_id = f"{name}_{_cache_key(source_files, extra_key)}"

    if not (cache_dir / cache_id).is_dir():
        # Drop this table's caches from earlier results before building the new one
        for stale in cache_dir.glob(f"{name}_" + "?" * 10):
            shutil.rmtree(stale, ignore_errors=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        pl_df, column_definitions = _to_polars(df, column_overrides)
        Table(
            cache_id=cache_id,
            data=pl_df.lazy(),
            cache_path=str(cache_dir),
            column_definitions=column_definitions,
            index_field=ROW_FIELD,
            title=title or name,
            page_size=page_size,
            **table_kwargs,
        )

    table = Table(cache_id=cache_id, cache_path=str(cache_dir))
    table(key=f"insight_table_{name}", height=height, state_manager=state_manager)
