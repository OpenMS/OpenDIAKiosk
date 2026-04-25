from __future__ import annotations

import base64
import hashlib
import html
import importlib.util
from dataclasses import dataclass, field, replace
from datetime import datetime
from io import BytesIO
from pathlib import Path
from textwrap import wrap
from typing import Any, Iterable

import numpy as np
import pandas as pd
import streamlit as st
from plotly.graph_objects import Figure


@dataclass
class ReportBranding:
    app_name: str = "OpenDIAKiosk"
    version: str | None = None
    generated_on: str | None = None
    logo_data_uri: str | None = None
    logo_path: str | None = None

    @property
    def product_label(self) -> str:
        if self.version:
            return f"{self.app_name} v{self.version}"
        return self.app_name


@dataclass
class ReportFigure:
    title: str
    figure: Figure
    caption: str | None = None
    pdf_kind: str | None = None
    pdf_payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportTable:
    title: str
    dataframe: pd.DataFrame
    caption: str | None = None
    max_rows: int | None = 50


@dataclass
class ReportSection:
    title: str
    description: str | None = None
    figures: list[ReportFigure] = field(default_factory=list)
    tables: list[ReportTable] = field(default_factory=list)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_logo_path() -> Path | None:
    for relative_path in [
        Path("assets/OpenDIAKiosk_logo_portrait.png"),
        Path("assets/OpenDIAKiosk_logo.png"),
        Path("assets/OpenDIAKiosk_logo_portrait.svg"),
        Path("assets/OpenDIAKiosk_logo.svg"),
    ]:
        candidate = _repo_root() / relative_path
        if candidate.exists():
            return candidate
    return None


def _image_data_uri(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None

    suffix_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".svg": "image/svg+xml",
    }
    mime_type = suffix_map.get(path.suffix.lower())
    if mime_type is None:
        return None

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _resolve_branding() -> ReportBranding:
    settings = st.session_state.get("settings", {})
    logo_path = _default_logo_path()
    return ReportBranding(
        app_name=str(settings.get("app-name", "OpenDIAKiosk")),
        version=(
            None
            if settings.get("version") in (None, "")
            else str(settings.get("version"))
        ),
        logo_data_uri=_image_data_uri(logo_path),
        logo_path=None if logo_path is None else str(logo_path),
    )


def _format_generated_timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def _clean_table_dataframe(dataframe: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    cleaned = dataframe.copy()
    if max_rows is not None and len(cleaned) > max_rows:
        cleaned = cleaned.head(max_rows).copy()
    return cleaned.fillna("")


def _table_to_html(table: ReportTable) -> str:
    dataframe = _clean_table_dataframe(table.dataframe, table.max_rows)
    title = html.escape(table.title)
    caption = (
        f'<p class="report-note">{html.escape(table.caption)}</p>'
        if table.caption
        else ""
    )
    truncated_note = ""
    if table.max_rows is not None and len(table.dataframe) > table.max_rows:
        truncated_note = (
            f'<p class="report-note">Showing the first {table.max_rows:,} of '
            f"{len(table.dataframe):,} rows.</p>"
        )
    table_html = dataframe.to_html(
        index=False,
        escape=True,
        border=0,
        classes="report-table",
    )
    return (
        '<div class="report-card">'
        f"<h4>{title}</h4>"
        f"{caption}"
        f"{truncated_note}"
        f"{table_html}"
        "</div>"
    )


def build_html_report(
    title: str,
    sections: Iterable[ReportSection],
    subtitle: str | None = None,
    metadata: dict[str, str] | None = None,
    branding: ReportBranding | None = None,
) -> str:
    escaped_title = html.escape(title)
    metadata = metadata or {}
    branding = branding or ReportBranding()

    metadata_items = "".join(
        (
            '<div class="metadata-item">'
            f'<span class="metadata-key">{html.escape(str(key))}</span>'
            f'<span class="metadata-value">{html.escape(str(value))}</span>'
            "</div>"
        )
        for key, value in metadata.items()
        if value not in (None, "")
    )
    metadata_block = (
        f'<div class="metadata-grid">{metadata_items}</div>' if metadata_items else ""
    )

    logo_block = ""
    if branding.logo_data_uri:
        logo_block = (
            '<div class="brand-logo-shell">'
            f'<img class="brand-logo" src="{branding.logo_data_uri}" alt="{html.escape(branding.app_name)} logo" />'
            "</div>"
        )

    generated_block = ""
    if branding.generated_on:
        generated_block = (
            '<div class="brand-meta">'
            f"<span>Generated on: {html.escape(branding.generated_on)}</span>"
            "</div>"
        )

    subtitle_block = (
        f'<p class="report-subtitle">{html.escape(subtitle)}</p>'
        if subtitle
        else ""
    )

    body_parts: list[str] = []
    include_plotly_js = True
    for section_index, section in enumerate(sections):
        cards: list[str] = []
        for table in section.tables:
            cards.append(_table_to_html(table))

        for figure_index, report_figure in enumerate(section.figures):
            figure_html = report_figure.figure.to_html(
                full_html=False,
                include_plotlyjs=include_plotly_js,
                config={"displayModeBar": False, "responsive": True},
                default_width="100%",
                default_height="520px",
                div_id=f"report-figure-{section_index}-{figure_index}",
            )
            include_plotly_js = False
            caption = (
                f'<p class="report-note">{html.escape(report_figure.caption)}</p>'
                if report_figure.caption
                else ""
            )
            cards.append(
                '<div class="report-card">'
                f"<h4>{html.escape(report_figure.title)}</h4>"
                f"{caption}"
                f"{figure_html}"
                "</div>"
            )

        if not cards:
            continue

        description = (
            f'<p class="section-description">{html.escape(section.description)}</p>'
            if section.description
            else ""
        )
        body_parts.append(
            '<section class="report-section">'
            f"<h2>{html.escape(section.title)}</h2>"
            f"{description}"
            f'{"".join(cards)}'
            "</section>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{escaped_title}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #0f172a;
      --muted: #475569;
      --line: #dbe1ea;
      --card: #ffffff;
      --paper: #f8fafc;
      --header-a: #2c74bb;
      --header-b: #1f5fa7;
    }}
    body {{
      margin: 0;
      padding: 0;
      background: var(--paper);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    .report-shell {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 24px 24px 56px;
    }}
    .report-header {{
      background: linear-gradient(135deg, var(--header-a) 0%, var(--header-b) 100%);
      border-radius: 18px;
      padding: 24px 26px 26px;
      margin-bottom: 24px;
      box-shadow: 0 14px 26px rgba(15, 23, 42, 0.16);
      color: #ffffff;
    }}
    .brand-banner {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 18px;
    }}
    .brand-lockup {{
      display: flex;
      align-items: center;
      gap: 18px;
      min-width: 0;
    }}
    .brand-logo-shell {{
      flex: 0 0 auto;
      background: rgba(255, 255, 255, 0.96);
      border-radius: 14px;
      padding: 10px 14px;
      box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.4);
    }}
    .brand-logo {{
      display: block;
      width: auto;
      height: 58px;
      max-width: 180px;
      object-fit: contain;
    }}
    .brand-copy {{
      min-width: 0;
    }}
    .brand-product {{
      font-size: 1.8rem;
      font-weight: 700;
      line-height: 1.1;
      margin: 0;
    }}
    .brand-meta {{
      margin-top: 4px;
      font-size: 0.96rem;
      color: rgba(255, 255, 255, 0.88);
    }}
    .report-header h1 {{
      margin: 0 0 6px;
      font-size: 1.7rem;
      line-height: 1.15;
      color: #ffffff;
    }}
    .report-subtitle {{
      margin: 0 0 16px;
      color: rgba(255, 255, 255, 0.88);
      font-size: 1rem;
    }}
    .metadata-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 10px 18px;
      margin-top: 12px;
    }}
    .metadata-item {{
      background: rgba(255, 255, 255, 0.12);
      border: 1px solid rgba(255, 255, 255, 0.22);
      border-radius: 12px;
      padding: 10px 12px;
      backdrop-filter: blur(2px);
    }}
    .metadata-key {{
      display: block;
      font-size: 0.78rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: rgba(255, 255, 255, 0.74);
      margin-bottom: 4px;
    }}
    .metadata-value {{
      display: block;
      word-break: break-word;
      color: #ffffff;
    }}
    .report-section {{
      margin: 28px 0 34px;
    }}
    .report-section h2 {{
      margin: 0 0 8px;
      font-size: 1.4rem;
    }}
    .section-description {{
      margin: 0 0 14px;
      color: var(--muted);
    }}
    .report-card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 18px;
      margin: 0 0 18px;
      box-shadow: 0 8px 18px rgba(15, 23, 42, 0.04);
      break-inside: avoid;
    }}
    .report-card h4 {{
      margin: 0 0 8px;
      font-size: 1rem;
    }}
    .report-note {{
      margin: 0 0 10px;
      color: var(--muted);
      font-size: 0.92rem;
    }}
    .report-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.92rem;
    }}
    .report-table th,
    .report-table td {{
      border: 1px solid var(--line);
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
    }}
    .report-table th {{
      background: #f1f5f9;
      font-weight: 700;
    }}
    @media (max-width: 720px) {{
      .brand-banner {{
        flex-direction: column;
        align-items: flex-start;
      }}
      .brand-product {{
        font-size: 1.45rem;
      }}
    }}
    @media print {{
      body {{
        background: #fff;
      }}
      .report-shell {{
        max-width: none;
        padding: 0;
      }}
      .report-card,
      .report-header {{
        box-shadow: none;
        break-inside: avoid;
      }}
    }}
  </style>
</head>
<body>
  <main class="report-shell">
    <header class="report-header">
      <div class="brand-banner">
        <div class="brand-lockup">
          {logo_block}
          <div class="brand-copy">
            <div class="brand-product">{html.escape(branding.product_label)}</div>
            {generated_block}
          </div>
        </div>
      </div>
      <h1>{escaped_title}</h1>
      {subtitle_block}
      {metadata_block}
    </header>
    {''.join(body_parts)}
  </main>
</body>
</html>
"""


def pdf_export_available() -> tuple[bool, str | None]:
    if importlib.util.find_spec("matplotlib") is None:
        return (
            False,
            "Direct PDF export requires `matplotlib` in the runtime environment.",
        )
    return True, None


def _wrap_text_lines(text: str, width: int = 92) -> list[str]:
    lines: list[str] = []
    for paragraph in str(text).splitlines() or [""]:
        wrapped = wrap(paragraph, width=width) or [""]
        lines.extend(wrapped)
    return lines


def _add_cover_page(
    pdf,
    title: str,
    subtitle: str | None = None,
    metadata: dict[str, str] | None = None,
    branding: ReportBranding | None = None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig = plt.figure(figsize=(11, 8.5))
    canvas = fig.add_axes([0, 0, 1, 1])
    canvas.axis("off")

    header_patch = FancyBboxPatch(
        (0.04, 0.865),
        0.92,
        0.11,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=0,
        facecolor="#2C74BB",
        transform=canvas.transAxes,
    )
    canvas.add_patch(header_patch)

    if branding and branding.logo_path:
        logo_path = Path(branding.logo_path)
        if logo_path.exists():
            logo_box = FancyBboxPatch(
                (0.055, 0.885),
                0.16,
                0.072,
                boxstyle="round,pad=0.008,rounding_size=0.015",
                linewidth=0,
                facecolor="#FFFFFF",
                transform=canvas.transAxes,
            )
            canvas.add_patch(logo_box)
            try:
                logo_ax = fig.add_axes([0.064, 0.892, 0.145, 0.056])
                logo_ax.imshow(plt.imread(str(logo_path)))
                logo_ax.axis("off")
            except Exception:
                pass

    product_label = branding.product_label if branding else "OpenDIAKiosk"
    canvas.text(
        0.25,
        0.933,
        product_label,
        transform=canvas.transAxes,
        fontsize=19,
        fontweight="bold",
        color="#FFFFFF",
        va="center",
    )
    if branding and branding.generated_on:
        canvas.text(
            0.25,
            0.898,
            f"Generated on: {branding.generated_on}",
            transform=canvas.transAxes,
            fontsize=10,
            color="#E2E8F0",
            va="center",
        )

    canvas.text(
        0.06,
        0.81,
        title,
        transform=canvas.transAxes,
        fontsize=21,
        fontweight="bold",
        color="#0F172A",
        va="top",
    )

    y_cursor = 0.765
    if subtitle:
        canvas.text(
            0.06,
            y_cursor,
            subtitle,
            transform=canvas.transAxes,
            fontsize=11,
            color="#475569",
            va="top",
        )
        y_cursor -= 0.05

    lines = [
        f"{key}: {value}"
        for key, value in (metadata or {}).items()
        if value not in (None, "")
    ]
    if not lines:
        lines = ["No additional metadata."]

    for raw_line in lines:
        for line in _wrap_text_lines(raw_line, width=92):
            canvas.text(
                0.06,
                y_cursor,
                line,
                transform=canvas.transAxes,
                fontsize=10.5,
                color="#0F172A",
                va="top",
            )
            y_cursor -= 0.032

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _add_text_page(
    pdf,
    title: str,
    lines: list[str],
    subtitle: str | None = None,
) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    fig.text(0.06, 0.95, title, ha="left", va="top", fontsize=20, fontweight="bold")
    y = 0.91
    if subtitle:
        fig.text(0.06, y, subtitle, ha="left", va="top", fontsize=11, color="#475569")
        y -= 0.05

    for raw_line in lines:
        wrapped_lines = _wrap_text_lines(raw_line)
        for line in wrapped_lines:
            fig.text(0.06, y, line, ha="left", va="top", fontsize=10.5, color="#0F172A")
            y -= 0.03
            if y < 0.07:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                fig = plt.figure(figsize=(11, 8.5))
                ax = fig.add_axes([0, 0, 1, 1])
                ax.axis("off")
                y = 0.95
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _blank_plot_figure(title: str, message: str):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        fontsize=12,
        color="#475569",
        transform=ax.transAxes,
    )
    ax.set_title(title, loc="left", pad=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    return fig


def _build_jaccard_matrix(
    set_map: dict[str, set[str]],
    label_order: list[str] | None = None,
) -> tuple[list[str], np.ndarray]:
    labels = label_order or list(set_map.keys())
    matrix = np.zeros((len(labels), len(labels)), dtype=float)
    for row_index, row_name in enumerate(labels):
        row_ids = set_map.get(row_name, set())
        for col_index, col_name in enumerate(labels):
            col_ids = set_map.get(col_name, set())
            union_size = len(row_ids | col_ids)
            shared_size = len(row_ids & col_ids)
            matrix[row_index, col_index] = shared_size / union_size if union_size else 0.0
    return labels, matrix


def _render_histogram_figure(report_figure: ReportFigure):
    import matplotlib.pyplot as plt

    payload = report_figure.pdf_payload
    dataframe = payload.get("dataframe", pd.DataFrame())
    if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
        return _blank_plot_figure(report_figure.title, "No data were available for this figure.")

    value_col = str(payload.get("value_col", ""))
    if value_col not in dataframe.columns:
        return _blank_plot_figure(report_figure.title, "The expected value column was not available.")

    plotting = dataframe.copy()
    plotting[value_col] = pd.to_numeric(plotting[value_col], errors="coerce")
    plotting = plotting.loc[plotting[value_col].notna()].copy()
    if plotting.empty:
        return _blank_plot_figure(report_figure.title, "No numeric values were available for this figure.")

    bins = int(payload.get("bins", 50))
    group_col = payload.get("group_col")
    color_map = payload.get("color_map", {})
    xlabel = str(payload.get("xlabel", value_col))
    ylabel = str(payload.get("ylabel", "Count"))

    fig, ax = plt.subplots(figsize=(11, 6.2))
    bin_edges = np.histogram_bin_edges(plotting[value_col].to_numpy(), bins=bins)

    if group_col and str(group_col) in plotting.columns:
        order = payload.get("order") or plotting[str(group_col)].dropna().astype(str).unique().tolist()
        for label in order:
            values = pd.to_numeric(
                plotting.loc[plotting[str(group_col)].astype(str) == str(label), value_col],
                errors="coerce",
            ).dropna()
            if values.empty:
                continue
            ax.hist(
                values.to_numpy(),
                bins=bin_edges,
                alpha=0.58,
                color=color_map.get(label),
                label=str(label),
            )
        if ax.has_data():
            ax.legend(frameon=False)
    else:
        ax.hist(plotting[value_col].to_numpy(), bins=bin_edges, color="#2C74BB", alpha=0.8)

    vline = payload.get("vline")
    if vline is not None:
        try:
            ax.axvline(float(vline), color="#111827", linestyle="--", linewidth=1.8)
        except (TypeError, ValueError):
            pass

    annotation_text = payload.get("annotation_text")
    if annotation_text:
        ax.text(
            0.985,
            0.96,
            str(annotation_text),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9.5,
            color="#0F172A",
            bbox={"facecolor": "#FFFFFF", "alpha": 0.92, "edgecolor": "#CBD5E1"},
        )

    ax.set_title(report_figure.title, loc="left", pad=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    return fig


def _render_stacked_bar_figure(report_figure: ReportFigure):
    import matplotlib.pyplot as plt

    payload = report_figure.pdf_payload
    dataframe = payload.get("dataframe", pd.DataFrame())
    if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
        return _blank_plot_figure(report_figure.title, "No data were available for this figure.")

    x_col = str(payload.get("x_col", ""))
    series = payload.get("series", [])
    if x_col not in dataframe.columns or not series:
        return _blank_plot_figure(report_figure.title, "The expected bar-chart columns were not available.")

    x_labels = dataframe[x_col].astype(str).tolist()
    x_positions = np.arange(len(x_labels))
    fig_width = max(8.5, len(x_labels) * 0.82)
    fig, ax = plt.subplots(figsize=(fig_width, 6.4))

    bottom = np.zeros(len(dataframe), dtype=float)
    first_values: np.ndarray | None = None
    for index, series_spec in enumerate(series):
        column = str(series_spec.get("column", ""))
        if column not in dataframe.columns:
            continue
        values = pd.to_numeric(dataframe[column], errors="coerce").fillna(0).to_numpy(dtype=float)
        if index == 0:
            first_values = values.copy()
        ax.bar(
            x_positions,
            values,
            bottom=bottom,
            label=series_spec.get("label"),
            color=series_spec.get("color"),
            alpha=float(series_spec.get("alpha", 1.0)),
            hatch=series_spec.get("hatch"),
            edgecolor=series_spec.get("edgecolor", series_spec.get("color", "#0F172A")),
            linewidth=0.9,
        )
        bottom += values

    annotation_texts = payload.get("annotation_texts")
    if annotation_texts:
        ymax = float(bottom.max()) if len(bottom) else 0.0
        offset = max(ymax * 0.03, 1.0)
        for xpos, yval, text in zip(x_positions, bottom, annotation_texts):
            ax.text(
                xpos,
                yval + offset,
                str(text),
                ha="center",
                va="bottom",
                fontsize=10,
                color="#111827",
            )

    top_annotation_col = payload.get("top_annotation_col")
    if top_annotation_col and str(top_annotation_col) in dataframe.columns:
        ymax = float(bottom.max()) if len(bottom) else 0.0
        offset = max(ymax * 0.03, 1.0)
        format_string = str(payload.get("top_annotation_format", "{:,.0f}"))
        top_values = pd.to_numeric(dataframe[str(top_annotation_col)], errors="coerce")
        for xpos, yval, raw_value in zip(x_positions, bottom, top_values):
            if pd.isna(raw_value):
                continue
            ax.text(
                xpos,
                yval + offset,
                format_string.format(float(raw_value)),
                ha="center",
                va="bottom",
                fontsize=10,
                color="#111827",
            )

    inner_annotation_col = payload.get("inner_annotation_col")
    if first_values is not None and inner_annotation_col and str(inner_annotation_col) in dataframe.columns:
        format_string = str(payload.get("inner_annotation_format", "{:,.0f}"))
        inner_values = pd.to_numeric(dataframe[str(inner_annotation_col)], errors="coerce")
        for xpos, bar_height, raw_value in zip(x_positions, first_values, inner_values):
            if pd.isna(raw_value) or bar_height <= 0:
                continue
            ax.text(
                xpos,
                bar_height / 2.0,
                format_string.format(float(raw_value)),
                ha="center",
                va="center",
                fontsize=9.5,
                color="#FFFFFF",
                fontweight="bold",
            )

    ax.set_title(report_figure.title, loc="left", pad=12, fontweight="bold")
    ax.set_xlabel(str(payload.get("xlabel", "")))
    ax.set_ylabel(str(payload.get("ylabel", "")))
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        x_labels,
        rotation=float(payload.get("xtick_rotation", 0)),
        ha="right" if float(payload.get("xtick_rotation", 0)) else "center",
    )
    ax.grid(axis="y", alpha=0.18)
    if any(series_spec.get("label") for series_spec in series):
        ax.legend(frameon=False)
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    return fig


def _render_jaccard_heatmap_figure(report_figure: ReportFigure):
    import matplotlib.pyplot as plt

    payload = report_figure.pdf_payload
    raw_set_map = payload.get("set_map", {})
    if not raw_set_map:
        return _blank_plot_figure(report_figure.title, "No overlap data were available for this figure.")

    set_map = {str(label): set(values) for label, values in raw_set_map.items()}
    labels, matrix = _build_jaccard_matrix(
        set_map,
        label_order=payload.get("label_order"),
    )
    fig_width = max(6.5, len(labels) * 0.95)
    fig_height = max(5.5, len(labels) * 0.82)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)

    for row_index in range(len(labels)):
        for col_index in range(len(labels)):
            ax.text(
                col_index,
                row_index,
                f"{matrix[row_index, col_index]:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                color="#0F172A",
            )

    ax.set_title(report_figure.title, loc="left", pad=12, fontweight="bold")
    ax.set_xlabel(str(payload.get("xlabel", "")))
    ax.set_ylabel(str(payload.get("ylabel", "")))
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Jaccard")
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    return fig


def _render_violin_figure(report_figure: ReportFigure):
    import matplotlib.pyplot as plt

    payload = report_figure.pdf_payload
    dataframe = payload.get("dataframe", pd.DataFrame())
    if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
        return _blank_plot_figure(report_figure.title, "No data were available for this figure.")

    group_col = str(payload.get("group_col", ""))
    value_col = str(payload.get("value_col", ""))
    if group_col not in dataframe.columns or value_col not in dataframe.columns:
        return _blank_plot_figure(report_figure.title, "The expected violin-plot columns were not available.")

    order = payload.get("order") or dataframe[group_col].dropna().astype(str).unique().tolist()
    datasets: list[np.ndarray] = []
    labels: list[str] = []
    for label in order:
        values = pd.to_numeric(
            dataframe.loc[dataframe[group_col].astype(str) == str(label), value_col],
            errors="coerce",
        ).dropna()
        if values.empty:
            continue
        labels.append(str(label))
        datasets.append(values.to_numpy(dtype=float))

    if not datasets:
        return _blank_plot_figure(report_figure.title, "No numeric values were available for this figure.")

    fig_width = max(8.5, len(labels) * 0.82)
    fig, ax = plt.subplots(figsize=(fig_width, 6.4))
    positions = np.arange(1, len(labels) + 1)
    violin_parts = ax.violinplot(
        datasets,
        positions=positions,
        showmeans=False,
        showextrema=False,
        showmedians=False,
    )

    color_map = payload.get("color_map", {})
    palette = list(plt.get_cmap("tab20").colors)
    for index, body in enumerate(violin_parts["bodies"]):
        color = color_map.get(labels[index], palette[index % len(palette)])
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.74)

    box = ax.boxplot(
        datasets,
        positions=positions,
        widths=0.16,
        patch_artist=True,
        showfliers=False,
    )
    for patch in box["boxes"]:
        patch.set_facecolor("#FFFFFF")
        patch.set_alpha(0.92)
        patch.set_edgecolor("#334155")
    for median in box["medians"]:
        median.set_color("#111827")
        median.set_linewidth(1.4)

    ax.set_title(report_figure.title, loc="left", pad=12, fontweight="bold")
    ax.set_xlabel(str(payload.get("xlabel", "")))
    ax.set_ylabel(str(payload.get("ylabel", value_col)))
    ax.set_xticks(positions)
    rotation = float(payload.get("xtick_rotation", 0))
    ax.set_xticklabels(
        labels,
        rotation=rotation,
        ha="right" if rotation else "center",
    )
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    return fig


def _render_scatter_figure(report_figure: ReportFigure):
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    payload = report_figure.pdf_payload
    dataframe = payload.get("dataframe", pd.DataFrame())
    if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
        return _blank_plot_figure(report_figure.title, "No data were available for this figure.")

    x_col = str(payload.get("x_col", ""))
    y_col = str(payload.get("y_col", ""))
    if x_col not in dataframe.columns or y_col not in dataframe.columns:
        return _blank_plot_figure(report_figure.title, "The expected scatter-plot columns were not available.")

    plotting = dataframe.copy()
    plotting[x_col] = pd.to_numeric(plotting[x_col], errors="coerce")
    plotting[y_col] = pd.to_numeric(plotting[y_col], errors="coerce")
    plotting = plotting.loc[plotting[x_col].notna() & plotting[y_col].notna()].copy()
    if plotting.empty:
        return _blank_plot_figure(report_figure.title, "No numeric values were available for this figure.")

    fig, ax = plt.subplots(figsize=(8.8, 6.6))
    color_col = payload.get("color_col")
    if color_col and str(color_col) in plotting.columns:
        plotting[str(color_col)] = pd.to_numeric(plotting[str(color_col)], errors="coerce")
        plotting = plotting.loc[plotting[str(color_col)].notna()].copy()

    if color_col and str(color_col) in plotting.columns and not plotting.empty:
        color_values = plotting[str(color_col)].to_numpy(dtype=float)
        norm = None
        if np.nanmin(color_values) < 0 < np.nanmax(color_values):
            norm = TwoSlopeNorm(vmin=float(np.nanmin(color_values)), vcenter=0.0, vmax=float(np.nanmax(color_values)))
        scatter = ax.scatter(
            plotting[x_col],
            plotting[y_col],
            c=color_values,
            cmap=str(payload.get("cmap", "RdBu_r")),
            norm=norm,
            alpha=float(payload.get("alpha", 0.65)),
            s=float(payload.get("marker_size", 18)),
            linewidths=0,
        )
        colorbar_label = payload.get("colorbar_label")
        if colorbar_label:
            fig.colorbar(scatter, ax=ax, label=str(colorbar_label))
    else:
        ax.scatter(
            plotting[x_col],
            plotting[y_col],
            color=str(payload.get("color", "#2C74BB")),
            alpha=float(payload.get("alpha", 0.65)),
            s=float(payload.get("marker_size", 18)),
            linewidths=0,
        )

    if payload.get("identity_line", False):
        min_axis = float(min(plotting[x_col].min(), plotting[y_col].min()))
        max_axis = float(max(plotting[x_col].max(), plotting[y_col].max()))
        ax.plot([min_axis, max_axis], [min_axis, max_axis], linestyle="--", color="#111827", linewidth=1.3)

    ax.set_title(report_figure.title, loc="left", pad=12, fontweight="bold")
    ax.set_xlabel(str(payload.get("xlabel", x_col)))
    ax.set_ylabel(str(payload.get("ylabel", y_col)))
    ax.grid(alpha=0.18)
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    return fig


def _build_membership_dataframe(set_map: dict[str, set[str]]) -> pd.DataFrame:
    labels = list(set_map.keys())
    universe = set().union(*set_map.values()) if set_map else set()
    rows: list[dict[str, Any]] = []
    for entity_id in universe:
        membership = tuple(entity_id in set_map[label] for label in labels)
        if not any(membership):
            continue
        row = {
            "count": 1,
            "degree": int(sum(membership)),
            "label": " + ".join(label for label, active in zip(labels, membership) if active),
        }
        for label, active in zip(labels, membership):
            row[label] = bool(active)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    membership_df = (
        pd.DataFrame(rows)
        .groupby(["label", "degree"] + labels, dropna=False, as_index=False)["count"]
        .sum()
        .sort_values(["count", "degree", "label"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    return membership_df


def _render_venn_2_figure(report_figure: ReportFigure, set_map: dict[str, set[str]]):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    names = list(set_map.keys())
    first_ids = set_map[names[0]]
    second_ids = set_map[names[1]]
    counts = {
        "first_only": len(first_ids - second_ids),
        "second_only": len(second_ids - first_ids),
        "shared": len(first_ids & second_ids),
    }

    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    ax.add_patch(Circle((0.42, 0.5), 0.26, facecolor="#0F766E", edgecolor="#0F766E", alpha=0.32, linewidth=2.2))
    ax.add_patch(Circle((0.58, 0.5), 0.26, facecolor="#C2410C", edgecolor="#C2410C", alpha=0.32, linewidth=2.2))
    ax.text(0.31, 0.79, names[0], ha="center", va="center", fontsize=12, color="#0F766E", fontweight="bold")
    ax.text(0.69, 0.79, names[1], ha="center", va="center", fontsize=12, color="#C2410C", fontweight="bold")
    ax.text(0.33, 0.50, f"{counts['first_only']:,}", ha="center", va="center", fontsize=18, color="#111827")
    ax.text(0.50, 0.50, f"{counts['shared']:,}", ha="center", va="center", fontsize=18, color="#111827")
    ax.text(0.67, 0.50, f"{counts['second_only']:,}", ha="center", va="center", fontsize=18, color="#111827")
    ax.set_title(report_figure.title, loc="left", pad=12, fontweight="bold")
    ax.set_xlim(0.08, 0.92)
    ax.set_ylim(0.14, 0.9)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    return fig


def _render_venn_3_figure(report_figure: ReportFigure, set_map: dict[str, set[str]]):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

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

    fig, ax = plt.subplots(figsize=(8.8, 6.8))
    circles = [
        ((0.40, 0.58), "#0F766E"),
        ((0.60, 0.58), "#C2410C"),
        ((0.50, 0.39), "#4338CA"),
    ]
    for center, color in circles:
        ax.add_patch(Circle(center, 0.24, facecolor=color, edgecolor=color, alpha=0.28, linewidth=2.2))

    ax.text(0.29, 0.84, names[0], ha="center", va="center", fontsize=12, color="#0F766E", fontweight="bold")
    ax.text(0.71, 0.84, names[1], ha="center", va="center", fontsize=12, color="#C2410C", fontweight="bold")
    ax.text(0.50, 0.12, names[2], ha="center", va="center", fontsize=12, color="#4338CA", fontweight="bold")

    labels = [
        (0.30, 0.59, counts["a_only"]),
        (0.70, 0.59, counts["b_only"]),
        (0.50, 0.28, counts["c_only"]),
        (0.50, 0.63, counts["ab_only"]),
        (0.40, 0.43, counts["ac_only"]),
        (0.60, 0.43, counts["bc_only"]),
        (0.50, 0.49, counts["abc"]),
    ]
    for xpos, ypos, value in labels:
        ax.text(xpos, ypos, f"{value:,}", ha="center", va="center", fontsize=16, color="#111827")

    ax.set_title(report_figure.title, loc="left", pad=12, fontweight="bold")
    ax.set_xlim(0.1, 0.9)
    ax.set_ylim(0.08, 0.9)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    return fig


def _render_upset_figure(report_figure: ReportFigure, set_map: dict[str, set[str]]):
    import matplotlib.pyplot as plt

    membership_df = _build_membership_dataframe(set_map)
    if membership_df.empty:
        return _blank_plot_figure(report_figure.title, "No intersections were available for this figure.")

    labels = list(set_map.keys())
    max_intersections = int(report_figure.pdf_payload.get("max_intersections", 20))
    displayed = membership_df.head(max_intersections).copy()
    x_positions = np.arange(len(displayed))

    fig = plt.figure(figsize=(max(10.5, len(displayed) * 0.6), 7.4))
    grid = fig.add_gridspec(2, 1, height_ratios=[3.3, 1.5], hspace=0.05)
    ax_bar = fig.add_subplot(grid[0, 0])
    ax_matrix = fig.add_subplot(grid[1, 0], sharex=ax_bar)

    counts = displayed["count"].to_numpy(dtype=float)
    ax_bar.bar(x_positions, counts, color="#0F172A")
    for xpos, count in zip(x_positions, counts):
        ax_bar.text(xpos, count, f"{int(count):,}", ha="center", va="bottom", fontsize=9)
    ax_bar.set_ylabel("# entities")
    ax_bar.set_title(report_figure.title, loc="left", pad=12, fontweight="bold")
    ax_bar.grid(axis="y", alpha=0.16)
    ax_bar.set_xticks([])

    reversed_labels = list(reversed(labels))
    y_map = {label: index for index, label in enumerate(reversed_labels)}
    for ypos, label in enumerate(reversed_labels):
        ax_matrix.scatter(x_positions, [ypos] * len(x_positions), color="#CBD5E1", s=38, zorder=1)

    for xpos, row in displayed.iterrows():
        active_labels = [label for label in labels if bool(row[label])]
        active_positions = [y_map[label] for label in active_labels]
        if len(active_positions) > 1:
            ax_matrix.plot(
                [xpos, xpos],
                [min(active_positions), max(active_positions)],
                color="#0F172A",
                linewidth=1.8,
                zorder=2,
            )
        ax_matrix.scatter([xpos] * len(active_positions), active_positions, color="#0F172A", s=52, zorder=3)

    ax_matrix.set_yticks(range(len(reversed_labels)))
    ax_matrix.set_yticklabels(reversed_labels)
    ax_matrix.set_xticks([])
    ax_matrix.set_xlabel("")
    ax_matrix.spines["top"].set_visible(False)
    ax_matrix.spines["right"].set_visible(False)

    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    return fig


def _render_set_overlap_figure(report_figure: ReportFigure):
    payload = report_figure.pdf_payload
    raw_set_map = payload.get("set_map", {})
    if not raw_set_map:
        return _blank_plot_figure(report_figure.title, "No overlap data were available for this figure.")

    set_map = {str(label): set(values) for label, values in raw_set_map.items()}
    if len(set_map) == 2:
        return _render_venn_2_figure(report_figure, set_map)
    if len(set_map) == 3:
        return _render_venn_3_figure(report_figure, set_map)
    return _render_upset_figure(report_figure, set_map)


def _build_pdf_plot(report_figure: ReportFigure):
    kind = report_figure.pdf_kind
    if kind == "histogram":
        return _render_histogram_figure(report_figure)
    if kind == "stacked_bar":
        return _render_stacked_bar_figure(report_figure)
    if kind == "jaccard_heatmap":
        return _render_jaccard_heatmap_figure(report_figure)
    if kind == "violin":
        return _render_violin_figure(report_figure)
    if kind == "scatter":
        return _render_scatter_figure(report_figure)
    if kind == "set_overlap":
        return _render_set_overlap_figure(report_figure)
    return None


def _add_figure_page(
    pdf,
    section_title: str,
    report_figure: ReportFigure,
) -> None:
    import matplotlib.pyplot as plt

    rendered = _build_pdf_plot(report_figure)
    if rendered is None:
        _add_text_page(
            pdf,
            f"{section_title}: {report_figure.title}",
            [
                "No static PDF renderer was configured for this figure.",
                "The standalone HTML export still contains the interactive Plotly version.",
            ],
            subtitle=report_figure.caption,
        )
        return

    rendered.text(
        0.02,
        0.985,
        section_title,
        ha="left",
        va="top",
        fontsize=9.5,
        color="#64748B",
    )
    if report_figure.caption:
        rendered.text(
            0.02,
            0.014,
            report_figure.caption,
            ha="left",
            va="bottom",
            fontsize=9.2,
            color="#475569",
        )
    pdf.savefig(rendered, bbox_inches="tight")
    plt.close(rendered)


def _add_table_pages(
    pdf,
    section_title: str,
    report_table: ReportTable,
) -> None:
    import matplotlib.pyplot as plt

    dataframe = _clean_table_dataframe(report_table.dataframe, report_table.max_rows)
    if dataframe.empty:
        _add_text_page(
            pdf,
            f"{section_title}: {report_table.title}",
            [report_table.caption or "No rows were available for this table."],
        )
        return

    display_df = dataframe.astype(str)
    rows_per_page = 24
    total_pages = max((len(display_df) - 1) // rows_per_page + 1, 1)
    caption_lines: list[str] = []
    if report_table.caption:
        caption_lines.extend(_wrap_text_lines(report_table.caption, width=88))
    if report_table.max_rows is not None and len(report_table.dataframe) > report_table.max_rows:
        caption_lines.append(
            f"Showing the first {report_table.max_rows:,} of {len(report_table.dataframe):,} rows."
        )

    for page_index in range(total_pages):
        page_df = display_df.iloc[
            page_index * rows_per_page : (page_index + 1) * rows_per_page
        ]
        fig, ax = plt.subplots(figsize=(11.7, 8.3))
        ax.axis("off")
        fig.suptitle(
            f"{section_title}: {report_table.title} (page {page_index + 1}/{total_pages})",
            x=0.02,
            y=0.98,
            ha="left",
            fontsize=14,
            fontweight="bold",
        )

        if caption_lines:
            fig.text(0.02, 0.92, "\n".join(caption_lines), ha="left", va="top", fontsize=9, color="#475569")
            table_bbox = [0.02, 0.06, 0.96, 0.76]
        else:
            table_bbox = [0.02, 0.06, 0.96, 0.84]

        table = ax.table(
            cellText=page_df.values.tolist(),
            colLabels=page_df.columns.tolist(),
            loc="center",
            cellLoc="left",
            colLoc="left",
            bbox=table_bbox,
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.2)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#E2E8F0")
            cell.set_edgecolor("#CBD5E1")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def build_pdf_report(
    title: str,
    sections: Iterable[ReportSection],
    subtitle: str | None = None,
    metadata: dict[str, str] | None = None,
    branding: ReportBranding | None = None,
) -> tuple[bytes | None, str | None]:
    available, reason = pdf_export_available()
    if not available:
        return None, reason

    try:
        from matplotlib.backends.backend_pdf import PdfPages

        buffer = BytesIO()
        with PdfPages(buffer) as pdf:
            _add_cover_page(
                pdf,
                title=title,
                subtitle=subtitle,
                metadata=metadata,
                branding=branding,
            )

            for section in sections:
                if section.description:
                    _add_text_page(pdf, section.title, [section.description])
                for report_table in section.tables:
                    _add_table_pages(pdf, section.title, report_table)
                for report_figure in section.figures:
                    _add_figure_page(pdf, section.title, report_figure)

        return buffer.getvalue(), None
    except Exception as exc:
        return None, str(exc)


def render_report_downloads(
    report_key: str,
    title: str,
    basename: str,
    sections: list[ReportSection],
    subtitle: str | None = None,
    metadata: dict[str, str] | None = None,
) -> None:
    if not sections:
        return

    base_branding = _resolve_branding()
    signature_key = f"{report_key}::signature"
    generated_on_key = f"{report_key}::generated_on"
    pdf_bytes_key = f"{report_key}::pdf_bytes"
    pdf_ready_key = f"{report_key}::pdf_ready"

    signature_source = build_html_report(
        title=title,
        sections=sections,
        subtitle=subtitle,
        metadata=metadata,
        branding=replace(base_branding, generated_on="__GENERATED_AT__"),
    )
    signature = hashlib.sha1(signature_source.encode("utf-8")).hexdigest()

    if st.session_state.get(signature_key) != signature:
        st.session_state[signature_key] = signature
        st.session_state[generated_on_key] = _format_generated_timestamp()
        st.session_state.pop(pdf_bytes_key, None)
        st.session_state[pdf_ready_key] = False

    branding = replace(
        base_branding,
        generated_on=st.session_state.get(generated_on_key),
    )
    html_report = build_html_report(
        title=title,
        sections=sections,
        subtitle=subtitle,
        metadata=metadata,
        branding=branding,
    )

    st.markdown("---")
    st.subheader("Download Report")
    st.caption(
        "The standalone HTML report can be shared directly and opened without the Streamlit app. "
        "The PDF export uses static matplotlib figures, so it does not depend on Chrome or a browser renderer."
    )

    col_html, col_pdf = st.columns(2)
    col_html.download_button(
        "Download HTML report",
        data=html_report.encode("utf-8"),
        file_name=f"{basename}.html",
        mime="text/html",
        use_container_width=True,
    )

    pdf_bytes = st.session_state.get(pdf_bytes_key)
    pdf_ready = bool(st.session_state.get(pdf_ready_key, False))
    pdf_available, pdf_reason = pdf_export_available()
    if not pdf_available:
        col_pdf.info(pdf_reason)
        return

    if pdf_ready and pdf_bytes:
        col_pdf.download_button(
            "Download PDF report",
            data=pdf_bytes,
            file_name=f"{basename}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
        return

    if col_pdf.button("Prepare PDF report", key=f"{report_key}::prepare_pdf", use_container_width=True):
        with st.spinner("Rendering PDF report…"):
            pdf_bytes, pdf_error = build_pdf_report(
                title=title,
                sections=sections,
                subtitle=subtitle,
                metadata=metadata,
                branding=branding,
            )
        if pdf_bytes is None:
            st.session_state.pop(pdf_bytes_key, None)
            st.session_state[pdf_ready_key] = False
            col_pdf.error(
                pdf_error
                or "Direct PDF export failed. The HTML report is still available."
            )
        else:
            st.session_state[pdf_bytes_key] = pdf_bytes
            st.session_state[pdf_ready_key] = True
            st.rerun()
