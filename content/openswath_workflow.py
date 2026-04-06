"""
content/openswath_workflow.py
OpenSwath Workflow — Run Page

Shows a rendered HTML pipeline diagram, I/O mapping, pre-flight checks,
and the WorkflowManager-based execution section (run / stop / log).

The configuration of all parameters happens on the **OpenSwath Configuration**
page; this page only runs the workflow and shows results.

Key design:
  * ``use_easypqp`` is read from ``workspace/params.json:osag_input_mode``
    (saved by the configuration page) — NOT re-derived from file presence.
  * Execution is delegated to ``OpenSwathWorkflow(WorkflowManager)``, which
    uses ``executor.run_topp()`` for OpenMS TOPP tools and
    ``executor.run_command()`` for click-based CLIs (easypqp, pyprophet).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from src.common.common import page_setup
from src.workflow.OpenSwathWorkflow import OpenSwathWorkflow

# -----------------------------------------------------------------------------
# Page setup

page_setup()

workspace_dir = Path(st.session_state.get("workspace", "."))

# Instantiate the workflow manager once per page load.
# WorkflowManager creates its own workflow_dir, logger, executor, etc.
wf = OpenSwathWorkflow()

# -----------------------------------------------------------------------------
# Read workspace-level params (saved by the configuration page)
# This is the SINGLE source of truth for what the user configured.

_ws_params_file = workspace_dir / "params.json"
_ws_params: dict = {}
if _ws_params_file.exists():
    try:
        with open(_ws_params_file, encoding="utf-8") as _f:
            _ws_params = json.load(_f)
    except Exception:
        pass

# -- Derive pipeline shape from saved configuration ----------------------------
osag_mode = _ws_params.get("osag_input_mode", "Use existing transition list(s)")
use_easypqp = osag_mode == "Generate from FASTA (predict transitions)"
osag_fasta = _ws_params.get("osag_fasta", "")


# Workspace file lists (for display / pre-flight only — not re-deriving logic)
def _mzml_files() -> list[str]:
    mzml_dir = workspace_dir / "mzML-files"
    names: list[str] = []
    if mzml_dir.exists():
        names = [
            p.name
            for p in mzml_dir.iterdir()
            if p.is_file() and "external_files.txt" not in p.name
        ]
        ext = mzml_dir / "external_files.txt"
        if ext.exists():
            names += [
                Path(l.strip()).name for l in ext.read_text().splitlines() if l.strip()
            ]
    return names


def _fasta_files() -> list[str]:
    d = workspace_dir / "input-files" / "fasta"
    return [p.name for p in d.iterdir() if p.is_file()] if d.exists() else []


def _lib_files() -> list[str]:
    d = workspace_dir / "input-files" / "libraries"
    return [p.name for p in d.iterdir() if p.is_file()] if d.exists() else []


mzml_list = _mzml_files()
fasta_list = _fasta_files()
lib_list = _lib_files()

# Fixed filenames
EPQP_OUT = "easypqp_insilico_library.tsv"
OSAG_OUT = "openswathassay_targets.tsv"
OSDG_OUT = "openswath_targets_and_decoys.pqp"
OSW_OUT = "openswath_results.osw"
PY_OUT = "openswath_results.tsv"

# -----------------------------------------------------------------------------
# Pipeline step definitions (for diagram / IO table)

STEPS: list[dict] = []

if use_easypqp:
    STEPS.append(
        {
            "id": "easypqp",
            "label": "EasyPQP",
            "subtitle": "In-silico library",
            "inputs": fasta_list[:2] or ["(FASTA)"],
            "output": EPQP_OUT,
            "color": "#7C3AED",
            "optional": True,
        }
    )

_osag_input = EPQP_OUT if use_easypqp else (lib_list[0] if lib_list else "library.tsv")
STEPS += [
    {
        "id": "osag",
        "label": "OpenSwathAssay\nGenerator",
        "subtitle": "Assay refinement",
        "inputs": [_osag_input],
        "output": OSAG_OUT,
        "color": "#0369A1",
        "optional": False,
    },
    {
        "id": "osdg",
        "label": "OpenSwathDecoy\nGenerator",
        "subtitle": "Decoy generation",
        "inputs": [OSAG_OUT],
        "output": OSDG_OUT,
        "color": "#0369A1",
        "optional": False,
    },
    {
        "id": "osw",
        "label": "OpenSwath\nWorkflow",
        "subtitle": "DIA extraction",
        "inputs": [OSDG_OUT] + mzml_list[:2] + (["…"] if len(mzml_list) > 2 else []),
        "output": OSW_OUT,
        "color": "#065F46",
        "optional": False,
    },
    {
        "id": "pyprophet",
        "label": "PyProphet",
        "subtitle": "Score / Infer / Export",
        "inputs": [OSW_OUT],
        "output": PY_OUT,
        "color": "#92400E",
        "optional": False,
    },
]

# -----------------------------------------------------------------------------
# Pipeline diagram (HTML/CSS, rendered inline — no Mermaid plugin needed)
# Status icons derived from actual output file existence (not session_state),
# so they are accurate even after a page reload.

_results_dir = wf.workflow_dir / "results"


def _file_status(step_id: str) -> str:
    """Map step → output file → existence → status icon."""
    file_map = {
        "easypqp": _results_dir / "insilico" / EPQP_OUT,
        "osag": _results_dir / "osag" / OSAG_OUT,
        "osdg": _results_dir / "osdg" / OSDG_OUT,
        "osw": _results_dir / OSW_OUT,
        "pyprophet": _results_dir / PY_OUT,
    }
    wf_status = wf.get_workflow_status()
    if wf_status["running"]:
        return "🔄"
    p = file_map.get(step_id)
    return "✅" if (p and p.exists()) else "⬜"


def _build_pipeline_html(steps: list[dict]) -> str:
    node_parts: list[str] = []
    for i, step in enumerate(steps):
        sid = step["id"]
        color = step["color"]
        label = step["label"].replace("\n", "<br>")
        sub = step["subtitle"]
        opt = " <em>(optional)</em>" if step["optional"] else ""
        icon = _file_status(sid)
        out = step["output"]

        inp_html = "".join(
            f'<span style="font-size:10px;background:#1E293B;color:#94A3B8;'
            f'border-radius:4px;padding:1px 5px;margin:1px 0;display:inline-block;">{inp}</span>'
            for inp in step["inputs"]
        )
        arrow = (
            '<div style="display:flex;align-items:center;padding:0 4px;'
            'color:#64748B;font-size:22px;flex-shrink:0;">&#8594;</div>'
            if i < len(steps) - 1
            else ""
        )

        node_parts.append(f"""
        <div style="display:flex;align-items:stretch;flex-shrink:0;">
          <div style="background:#1E293B;border:2px solid {color};border-radius:10px;
                      padding:12px 14px;min-width:155px;max-width:195px;
                      display:flex;flex-direction:column;gap:6px;position:relative;">
            <div style="position:absolute;top:6px;right:8px;font-size:15px;">{icon}</div>
            <div style="font-weight:700;font-size:13px;color:{color};line-height:1.3;">{label}</div>
            <div style="font-size:10px;color:#94A3B8;">{sub}{opt}</div>
            <div style="font-size:10px;color:#64748B;margin-top:2px;">
              <span style="color:#475569;">in:</span><br>{inp_html}
            </div>
            <div style="margin-top:4px;">
              <span style="font-size:10px;color:#475569;">out:</span><br>
              <span style="font-size:10px;background:#0F172A;color:#34D399;
                           border-radius:4px;padding:1px 6px;font-family:monospace;">{out}</span>
            </div>
          </div>
          {arrow}
        </div>
        """)

    mzml_label = (
        ", ".join(mzml_list[:3]) + ("…" if len(mzml_list) > 3 else "")
        if mzml_list
        else "(no mzML — upload via File Upload)"
    )

    return f"""
    <div style="font-family:'Inter',sans-serif;background:#0F172A;
                border-radius:12px;padding:20px;overflow-x:auto;">
      <div style="display:flex;align-items:center;flex-wrap:nowrap;gap:0;">
        {"".join(node_parts)}
      </div>
      <div style="margin-top:12px;padding-left:8px;border-left:2px dashed #334155;
                  color:#64748B;font-size:11px;">
        <span style="color:#0EA5E9;">&#11015; mzML inputs into OpenSwathWorkflow:</span>
        <span style="font-family:monospace;color:#94A3B8;margin-left:6px;">{mzml_label}</span>
      </div>
      <div style="margin-top:14px;display:flex;gap:16px;font-size:11px;color:#64748B;">
        <span>⬜ Not yet produced</span>
        <span>🔄 Running</span>
        <span>✅ Output exists</span>
      </div>
    </div>
    """


# -----------------------------------------------------------------------------
# PAGE HEADER

st.title("🚀 OpenSwath Workflow — Run")
st.markdown(
    "This page runs the full OpenSWATH pipeline based on the parameters saved "
    "on the **OpenSwath Configuration** page.  "
    "Configure parameters there first, then return here to run and monitor execution."
)

# Warn if params haven't been saved yet
if not _ws_params:
    st.warning(
        "⚠️ No saved configuration found (`workspace/params.json` is empty or missing). "
        "Go to the **OpenSwath Configuration** page, configure your parameters, "
        "and click **Save all parameters** before running.",
        icon="⚠️",
    )

# Show mode badge
mode_badge = (
    "🔬 EasyPQP (generate from FASTA)" if use_easypqp else "📂 Existing transition list"
)
st.caption(f"Configured assay source: **{mode_badge}**")

# -----------------------------------------------------------------------------
# SECTION 1 — Pipeline Diagram

st.markdown("---")
st.subheader("📊 Pipeline Overview")
st.caption(
    "Output file icons update automatically: ✅ = file present on disk, "
    "🔄 = workflow currently running, ⬜ = not yet generated."
)
components.html(_build_pipeline_html(STEPS), height=360, scrolling=True)

# -----------------------------------------------------------------------------
# SECTION 2 — I/O Mapping

st.markdown("---")
st.subheader("🗂️ I/O Mapping")

_io_rows: list[dict] = []
if use_easypqp:
    _io_rows.append(
        {
            "Step": "① EasyPQP (in-silico)",
            "Inputs": osag_fasta or (fasta_list[0] if fasta_list else "(no FASTA)"),
            "Output": f"`{EPQP_OUT}`",
            "Location": f"`results/insilico/{EPQP_OUT}`",
            "Optional": "✅",
        }
    )
_io_rows += [
    {
        "Step": "② OpenSwathAssayGenerator",
        "Inputs": _osag_input,
        "Output": f"`{OSAG_OUT}`",
        "Location": f"`results/osag/{OSAG_OUT}`",
        "Optional": "",
    },
    {
        "Step": "③ OpenSwathDecoyGenerator",
        "Inputs": OSAG_OUT,
        "Output": f"`{OSDG_OUT}`",
        "Location": f"`results/osdg/{OSDG_OUT}`",
        "Optional": "",
    },
    {
        "Step": "④ OpenSwathWorkflow",
        "Inputs": f"{OSDG_OUT}  +  {len(mzml_list)} mzML file(s)",
        "Output": f"`{OSW_OUT}`",
        "Location": f"`results/{OSW_OUT}`",
        "Optional": "",
    },
    {
        "Step": "⑤ PyProphet",
        "Inputs": OSW_OUT,
        "Output": f"`{PY_OUT}`",
        "Location": f"`results/{PY_OUT}`",
        "Optional": "",
    },
]

_hdr = st.columns([2.4, 2.8, 2.5, 2.8, 0.7])
for col, txt in zip(_hdr, ["Step", "Input(s)", "Output", "Workspace path", "Optional"]):
    col.markdown(f"**{txt}**")
for row in _io_rows:
    c1, c2, c3, c4, c5 = st.columns([2.4, 2.8, 2.5, 2.8, 0.7])
    c1.markdown(f"**{row['Step']}**")
    c2.caption(row["Inputs"])
    c3.markdown(row["Output"])
    c4.caption(row["Location"])
    c5.markdown(row["Optional"])

with st.expander("📁 Workspace files", expanded=False):
    wc1, wc2, wc3 = st.columns(3)
    with wc1:
        st.markdown(f"**mzML** ({len(mzml_list)})")
        for f in mzml_list:
            st.caption(f"• {f}")
        if not mzml_list:
            st.warning("None — upload via File Upload.")
    with wc2:
        st.markdown(f"**FASTA** ({len(fasta_list)})")
        for f in fasta_list:
            st.caption(f"• {f}")
        if not fasta_list:
            st.info("None — EasyPQP step will be skipped.")
    with wc3:
        st.markdown(f"**Libraries** ({len(lib_list)})")
        for f in lib_list:
            st.caption(f"• {f}")
        if not lib_list and not fasta_list:
            st.warning("No library source available.")

# -----------------------------------------------------------------------------
# SECTION 3 — Pre-flight Checks

st.markdown("---")
st.subheader("✅ Pre-flight Checks")

_checks: list[tuple[str, str]] = []
_ok = True

# Config saved
if _ws_params:
    _checks.append(("✅", "Configuration saved (`params.json` found)."))
else:
    _checks.append(
        ("❌", "No saved configuration — go to **OpenSwath Configuration** and save.")
    )
    _ok = False

# mzML
if mzml_list:
    _checks.append(("✅", f"{len(mzml_list)} mzML file(s) in workspace."))
else:
    _checks.append(("❌", "No mzML files — add via **File Upload**."))
    _ok = False

# Library source
if use_easypqp and fasta_list:
    _checks.append(
        ("✅", f"EasyPQP mode: FASTA `{osag_fasta or fasta_list[0]}` found.")
    )
elif use_easypqp and not fasta_list:
    _checks.append(("❌", "EasyPQP mode selected but no FASTA in workspace."))
    _ok = False
elif not use_easypqp and lib_list:
    _checks.append(("✅", f"Using existing library: `{lib_list[0]}`."))
elif not use_easypqp and not lib_list:
    _checks.append(
        ("❌", "No transition library in workspace and EasyPQP is not selected.")
    )
    _ok = False

# EasyPQP executable
if use_easypqp:
    ep_exe = shutil.which("easypqp")
    if ep_exe:
        _checks.append(("✅", f"`easypqp` found at `{ep_exe}`."))
    else:
        _checks.append(
            ("❌", "`easypqp` not found in PATH — install with `pip install easypqp`.")
        )
        _ok = False

# TOPP tools
_shared_ini = workspace_dir / "ini"
for tool in ["OpenSwathAssayGenerator", "OpenSwathDecoyGenerator", "OpenSwathWorkflow"]:
    ini_ok = (_shared_ini / f"{tool}.ini").exists()
    exe_ok = bool(shutil.which(tool))
    if ini_ok:
        _checks.append(("✅", f"`{tool}.ini` present in workspace."))
    else:
        _checks.append(
            ("⚠️", f"`{tool}.ini` missing — run configuration page to copy it.")
        )
    if exe_ok:
        _checks.append(("✅", f"`{tool}` found at `{shutil.which(tool)}`."))
    else:
        _checks.append(("❌", f"`{tool}` not found in PATH — install OpenMS."))
        _ok = False

# PyProphet
py_exe = shutil.which("pyprophet")
if py_exe:
    _checks.append(("✅", f"`pyprophet` found at `{py_exe}`."))
else:
    _checks.append(
        ("❌", "`pyprophet` not found — install with `pip install pyprophet`.")
    )
    _ok = False

for icon, msg in _checks:
    st.markdown(f"{icon} {msg}")

if not _ok:
    st.warning(
        "Some required inputs or executables are missing. "
        "Resolve the issues above before running."
    )

# -----------------------------------------------------------------------------
# SECTION 4 — Execution (delegated to WorkflowManager / StreamlitUI)

st.markdown("---")
st.subheader("▶ Run Workflow")

st.markdown(
    """
The workflow runs in a **background process** so you can navigate away and return.
Log output is streamed below as the steps execute.

Use **Rerun Behavior** below to either start from scratch or reuse outputs from
previous successful steps and resume from a selected point in the pipeline.
Click **Stop Workflow** to terminate the running process.
"""
)

wf.show_execution_section()

# -----------------------------------------------------------------------------
# SECTION 5 — Results

st.markdown("---")
st.subheader("📦 Results")
wf.show_results_section()
