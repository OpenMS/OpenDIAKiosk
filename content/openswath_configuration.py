"""
content/openswath_configuration.py
OpenSwath Workflow Configuration Page

Pipeline order (left to right):
  1. [Optional] EasyPQP in-silico library  ← FASTA from workspace
  2. [Optional] OpenSwathAssayGenerator     ← library TSV/PQP
  3.            OpenSwathDecoyGenerator      ← assay output
  4.            OpenSwathWorkflow            ← decoy library + mzML files
  5.            PyProphet                    ← openswath_results.osw

INI-based tools  → StreamlitUI.input_TOPP()
JSON-based tools → custom widgets driven by the JSON schema
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import streamlit as st

from src.common.common import page_setup, save_params
from src.workflow.ParameterManager import ParameterManager
from src.workflow.StreamlitUI import StreamlitUI
from src.workflow.PyProphet import PyProphetCLI

# -----------------------------------------------------------------------------
# Page setup

params = page_setup()

workspace_dir = Path(st.session_state.get("workspace", "."))
pm = ParameterManager(workspace_dir)
ui = StreamlitUI(workspace_dir, logger=None, executor=None, parameter_manager=pm)
py = PyProphetCLI(pm, workspace_dir, executor=None, logger=None)

# Convenience prefixes
TPFX = pm.topp_param_prefix  # e.g. "<ws>-TOPP-"
PPFX = pm.param_prefix  # e.g. "<ws>-param-"

# -----------------------------------------------------------------------------
# Asset / ini directories

INI_DIR = pm.ini_dir
INI_DIR.mkdir(parents=True, exist_ok=True)

ASSET_OSW = Path("assets", "common-tool-descriptors", "openswathworkflow")
ASSET_OSAG = Path("assets", "common-tool-descriptors", "openswathassaygenerator")
ASSET_OSDG = Path("assets", "common-tool-descriptors", "openswathdecoygenerator")
ASSET_EPQP = Path(
    "assets", "common-tool-descriptors", "easypqp_insilico", "easypqp_insilico.json"
)
ASSET_PY = Path("assets", "common-tool-descriptors", "pyprophet")

OSW_XIC_OUT = "openswath_results_chromatograms.xic"
OSW_XIM_OUT = "openswath_results_mobilograms.xim"
OSW_DEBUG_IM_OUT = "_debug_calibration_im.txt"
OSW_DEBUG_MZ_OUT = "_debug_calibration_mz.txt"
OSW_DEBUG_IRT_TRAFO_OUT = "_debug_calibration_irt.trafoXML"
OSW_DEBUG_IRT_MZML_OUT = "_debug_calibration_irt_chrom.mzML"
OSW_CACHE_DIR = workspace_dir / "openswath-workflow-temp"
PY_MATRIX_LEVELS = ["precursor", "peptide", "protein"]
PY_MATRIX_OUTS = {
    "precursor": "openswath_results.precursor.tsv",
    "peptide": "openswath_results.peptide.tsv",
    "protein": "openswath_results.protein.tsv",
}
PY_INFER_CONTEXTS = ["global", "experiment-wide", "run-specific"]


def _copy_ini_once(asset_dir: Path, dest_name: str) -> Path | None:
    """
    Copy the first *.ini found in *asset_dir* into INI_DIR as *dest_name*
    (only if it does not already exist there).  Returns the dest path or None.
    """
    candidates = sorted(asset_dir.glob("*.ini")) if asset_dir.exists() else []
    if not candidates:
        return None
    src = candidates[-1]  # prefer latest (sorted alphabetically)
    dest = INI_DIR / dest_name
    if not dest.exists():
        try:
            shutil.copy2(src, dest)
        except Exception:
            return None
    return dest


def _load_json_asset(path: Path) -> dict:
    """Load a JSON asset, return empty dict on failure."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_json_params(section_key: str, data: dict) -> None:
    """Merge *data* into params.json under *section_key*."""
    current = pm.get_parameters_from_json()
    current[section_key] = data
    with open(pm.params_file, "w", encoding="utf-8") as f:
        json.dump(current, f, indent=4)


def _saved_params() -> dict:
    return pm.get_parameters_from_json()


def _saved_flat_param(name: str, default=None):
    return _saved_params().get(name, default)


def _saved_tool_param(tool: str, name: str, default=None):
    return _saved_params().get(tool, {}).get(name, default)


def _seed_choice_state(widget_key: str, options: list[str], saved_value=None) -> None:
    """Restore a saved select/radio choice when the current widget state is missing."""
    if not options:
        return
    current_value = st.session_state.get(widget_key)
    if current_value in options:
        return
    if saved_value in options:
        st.session_state[widget_key] = saved_value
    else:
        st.session_state[widget_key] = options[0]


def _seed_multiselect_state(
    widget_key: str, options: list[str], saved_values: list[str] | None = None
) -> None:
    """Restore a saved multiselect choice when the current widget state is missing."""
    if widget_key in st.session_state:
        current = st.session_state.get(widget_key, [])
        st.session_state[widget_key] = [value for value in current if value in options]
        return
    saved_values = saved_values or []
    st.session_state[widget_key] = [value for value in saved_values if value in options]


def _seed_checkbox_state(widget_key: str, saved_value: bool) -> None:
    """Restore a saved checkbox state when the current widget state is missing."""
    if widget_key not in st.session_state:
        st.session_state[widget_key] = bool(saved_value)


def _normalize_context_selection(saved_values) -> list[str]:
    """Normalize saved infer contexts while preserving the supported order."""
    if isinstance(saved_values, str):
        saved_values = [saved_values]
    if not isinstance(saved_values, list):
        saved_values = []
    normalized = [value for value in PY_INFER_CONTEXTS if value in saved_values]
    return normalized or ["global"]


# -----------------------------------------------------------------------------
# Workspace file helpers


def _mzml_files() -> list[str]:
    mzml_dir = workspace_dir / "mzML-files"
    names: list[str] = []
    if mzml_dir.exists():
        names = [
            p.name
            for p in mzml_dir.iterdir()
            if p.is_file() and "external_files.txt" not in p.name
        ]
        ext_file = mzml_dir / "external_files.txt"
        if ext_file.exists():
            names += [
                Path(l.strip()).name
                for l in ext_file.read_text().splitlines()
                if l.strip()
            ]
    return names


def _fasta_files() -> list[str]:
    d = workspace_dir / "input-files" / "fasta"
    return [p.name for p in d.iterdir() if p.is_file()] if d.exists() else []


def _lib_files() -> list[str]:
    d = workspace_dir / "input-files" / "libraries"
    return [p.name for p in d.iterdir() if p.is_file()] if d.exists() else []


# -----------------------------------------------------------------------------
# Advanced toggle (shared across all sections)

# The global advanced toggle is initialized in page_setup()/render_sidebar().

# -----------------------------------------------------------------------------
# PAGE HEADER

st.title("⚙️ OpenSwath Configuration")
st.markdown(
    """
Configure and inspect parameters for the full **OpenSWATH / PyProphet** pipeline.
Steps are laid out in execution order — each section feeds its output into the next.
"""
)

with st.expander("ℹ️ Pipeline overview", expanded=False):
    st.markdown(
        """
        ```
        [FASTA in workspace]
              │
              ▼ (if FASTA present)
        ① EasyPQP insilico-library
              │  predicted library TSV
              ▼ (if library present — FASTA or uploaded)
        ② OpenSwathAssayGenerator   (INI-based)
              │  refined assay TSV/PQP
              ▼
        ③ OpenSwathDecoyGenerator   (INI-based)
              │  target+decoy transition list
              ▼                           mzML files (from File Upload)
        ④ OpenSwathWorkflow         (INI-based) ◄---------------------
              │  openswath_results.osw
              ▼
        ⑤ PyProphet score → infer peptide → infer protein → export TSV
        ```
        Inputs are **derived automatically** from upstream outputs where possible.
        Switch to **Advanced** mode (sidebar) to expose all INI parameters.
        """
    )

# -----------------------------------------------------------------------------
# SECTION 0 — Workspace snapshot (read-only, always visible)

st.markdown("---")
st.subheader("📂 Workspace Files")
st.caption("Files detected from the File Upload page — read-only here.")

mzml_list = _mzml_files()
fasta_list = _fasta_files()
lib_list = _lib_files()

col_mz, col_fa, col_lib = st.columns(3)
with col_mz:
    st.markdown(f"**mzML files** ({len(mzml_list)})")
    if mzml_list:
        for f in mzml_list:
            st.caption(f"• {f}")
    else:
        st.info("No mzML files — add via File Upload.")
with col_fa:
    st.markdown(f"**FASTA files** ({len(fasta_list)})")
    if fasta_list:
        for f in fasta_list:
            st.caption(f"• {f}")
    else:
        st.info("No FASTA — upload to enable insilico library generation.")
with col_lib:
    st.markdown(f"**Libraries / transition lists** ({len(lib_list)})")
    if lib_list:
        for f in lib_list:
            st.caption(f"• {f}")
    else:
        st.info("No libraries — will be generated or upload manually.")

# --- Assay source selection (choose FASTA -> predict transitions, or use existing lists)
if fasta_list and lib_list:
    mode_options = [
        "Generate from FASTA (predict transitions)",
        "Use existing transition list(s)",
    ]
    _seed_choice_state(
        "osag_input_mode",
        mode_options,
        _saved_flat_param("osag_input_mode", mode_options[0]),
    )
    mode = st.radio(
        "Assay source",
        mode_options,
        key="osag_input_mode",
    )
    if mode == "Generate from FASTA (predict transitions)":
        _seed_choice_state(
            "osag_workspace_fasta",
            fasta_list,
            _saved_flat_param(
                "osag_fasta",
                _saved_flat_param("easypqp:database:fasta", fasta_list[0]),
            ),
        )
        sel = st.selectbox(
            "Select FASTA to use for prediction",
            options=fasta_list,
            key="osag_workspace_fasta",
        )
        # write into easypqp FASTA param so downstream UI picks it up
        st.session_state[f"{PPFX}easypqp:database:fasta"] = sel
        # ensure easypqp output filename is set
        st.session_state[f"{PPFX}easypqp:output_and_runtime:output_file"] = (
            "easypqp_insilico_library.tsv"
        )
    else:
        st.caption("Using existing transition list(s) uploaded to the workspace.")
elif fasta_list and not lib_list:
    # Only FASTA present — default to generate-from-FASTA mode
    st.info(
        "Only FASTA files present — will generate predicted transition list from FASTA."
    )
    st.session_state["osag_input_mode"] = "Generate from FASTA (predict transitions)"
    _seed_choice_state(
        "osag_workspace_fasta_only",
        fasta_list,
        _saved_flat_param(
            "osag_fasta",
            _saved_flat_param("easypqp:database:fasta", fasta_list[0]),
        ),
    )
    sel = st.selectbox(
        "Select FASTA to use for prediction",
        options=fasta_list,
        key="osag_workspace_fasta_only",
    )
    st.session_state[f"{PPFX}easypqp:database:fasta"] = sel
    st.session_state[f"{PPFX}easypqp:output_and_runtime:output_file"] = (
        "easypqp_insilico_library.tsv"
    )
elif lib_list and not fasta_list:
    st.info("No FASTA files present — using existing transition list(s) in workspace.")
    st.session_state["osag_input_mode"] = "Use existing transition list(s)"

# -----------------------------------------------------------------------------
# SECTION 1 — Spectral Library Parameters

st.markdown("---")
st.subheader("Spectral Library Parameters")
st.markdown(
    "Configure how the spectral library is generated or refined before DIA analysis."
)

# -- 1a: EasyPQP in-silico (only when a FASTA is present and assay source requires it)
# Render EasyPQP only if FASTA present and user did not choose to use existing transition lists
if (
    fasta_list
    and st.session_state.get(
        "osag_input_mode", "Generate from FASTA (predict transitions)"
    )
    != "Use existing transition list(s)"
):
    with st.expander("🔬 EasyPQP In-Silico Library Generation", expanded=True):
        st.caption(
            "Generate a predicted spectral library from the FASTA file(s) in your workspace. "
            "Uses deep-learning models for RT, ion mobility, and MS2 intensity prediction."
        )

        epqp_cfg = _load_json_asset(ASSET_EPQP)
        # Prefer structured schema when available
        structured_asset = ASSET_EPQP.parent / "easypqp_insilico_structured_config.json"
        struct_cfg = (
            _load_json_asset(structured_asset) if structured_asset.exists() else {}
        )

        if struct_cfg:
            # Ensure a sensible initial FASTA selection is present in session state
            fasta_key = f"{PPFX}easypqp:database:fasta"
            if fasta_key not in st.session_state:
                st.session_state[fasta_key] = _saved_flat_param(
                    "easypqp:database:fasta", fasta_list[0] if fasta_list else ""
                )

            # Provide a FASTA dropdown (derived input) instead of the free-text field
            if fasta_list:
                _seed_choice_state(
                    "easypqp_fasta_sel",
                    fasta_list,
                    _saved_flat_param(
                        "easypqp:database:fasta",
                        _saved_flat_param("osag_fasta", fasta_list[0]),
                    ),
                )
                sel = st.selectbox(
                    "Input FASTA (workspace)",
                    options=fasta_list,
                    key="easypqp_fasta_sel",
                )
                st.session_state[fasta_key] = sel
            else:
                st.info(
                    "No FASTA files in workspace — upload to enable in-silico library generation."
                )

            # Interactive editor for static_mods (dict: residue -> mass)
            def _render_static(full_key, opt_def):
                editor_key = f"{full_key}_editor_static"

                # initialize editor state
                if editor_key not in st.session_state:
                    short_key = full_key.replace(PPFX, "", 1)
                    init = (
                        st.session_state.get(
                            full_key,
                            _saved_flat_param(
                                short_key,
                                opt_def.get("value") or opt_def.get("default"),
                            ),
                        )
                        or {}
                    )
                    if isinstance(init, str):
                        try:
                            init = json.loads(init)
                        except Exception:
                            init = {}
                    if isinstance(init, dict):
                        items = [(k, float(v)) for k, v in init.items()]
                    else:
                        items = []
                    st.session_state[editor_key] = items
                    st.session_state[full_key] = json.dumps({k: v for k, v in items})

                st.markdown("**Static Modifications**")
                cols = st.columns([2, 2, 1])
                for i, (res, mass) in enumerate(list(st.session_state[editor_key])):
                    col_a, col_b, col_c = st.columns([1, 2, 0.5])
                    with col_a:
                        new_res = st.text_input(
                            f"Residue {i}",
                            value=res,
                            max_chars=3,
                            key=f"{editor_key}_res_{i}",
                        )
                    with col_b:
                        new_mass = st.number_input(
                            f"Mass {i}",
                            value=float(mass),
                            format="%.4f",
                            key=f"{editor_key}_mass_{i}",
                        )
                    with col_c:
                        if st.button("✕", key=f"{editor_key}_rm_{i}"):
                            st.session_state[editor_key].pop(i)
                            # update main JSON key and rerun
                            st.session_state[full_key] = json.dumps(
                                {k: v for k, v in st.session_state[editor_key]}
                            )
                            st.rerun()
                    # commit edits back to editor list
                    st.session_state[editor_key][i] = (new_res, float(new_mass))

                add_col1, add_col2 = st.columns([3, 1])
                with add_col2:
                    if st.button("➕ Add Static Mod", key=f"{editor_key}_add"):
                        st.session_state[editor_key].append(("", 0.0))
                        st.rerun()

                # Preset buttons for common static modifications
                existing_static_residues = {r for r, _ in st.session_state[editor_key]}
                preset_static = [
                    ("C", 57.0215, "Carbamidomethyl-C"),
                ]

                # Render presets in the left area if available
                avail = [
                    p for p in preset_static if p[0] not in existing_static_residues
                ]
                if avail:
                    preset_cols = st.columns(min(len(avail), 2))
                    for idx, (residue, mass, label) in enumerate(avail):
                        with preset_cols[idx % len(preset_cols)]:
                            if st.button(
                                f"➕ {label}",
                                key=f"{editor_key}_preset_static_{residue}",
                            ):
                                st.session_state[editor_key].append((residue, mass))
                                st.session_state[full_key] = json.dumps(
                                    {k: v for k, v in st.session_state[editor_key]}
                                )
                                st.rerun()

                # sync JSON value
                st.session_state[full_key] = json.dumps(
                    {k: v for k, v in st.session_state[editor_key]}
                )

            # Interactive editor for variable_mods (dict: position -> list[float])
            def _render_variable(full_key, opt_def):
                editor_key = f"{full_key}_editor_var"

                if editor_key not in st.session_state:
                    short_key = full_key.replace(PPFX, "", 1)
                    init = (
                        st.session_state.get(
                            full_key,
                            _saved_flat_param(
                                short_key,
                                opt_def.get("value") or opt_def.get("default"),
                            ),
                        )
                        or {}
                    )
                    if isinstance(init, str):
                        try:
                            init = json.loads(init)
                        except Exception:
                            init = {}
                    if isinstance(init, dict):
                        items = [(k, [float(x) for x in v]) for k, v in init.items()]
                    else:
                        items = []
                    st.session_state[editor_key] = items
                    st.session_state[full_key] = json.dumps({k: v for k, v in items})

                st.markdown("**Variable Modifications**")
                for i, (pos, masses) in enumerate(list(st.session_state[editor_key])):
                    col_a, col_b, col_c = st.columns([1, 3, 0.5])
                    with col_a:
                        new_pos = st.text_input(
                            f"Marker {i}",
                            value=pos,
                            max_chars=3,
                            key=f"{editor_key}_pos_{i}",
                        )
                    with col_b:
                        masses_str = ", ".join(str(m) for m in masses)
                        new_masses_str = st.text_input(
                            f"Masses {i}",
                            value=masses_str,
                            key=f"{editor_key}_masses_{i}",
                        )
                        try:
                            new_masses = [
                                float(m.strip())
                                for m in new_masses_str.split(",")
                                if m.strip()
                            ]
                        except ValueError:
                            new_masses = masses
                    with col_c:
                        if st.button("✕", key=f"{editor_key}_rm_{i}"):
                            st.session_state[editor_key].pop(i)
                            st.session_state[full_key] = json.dumps(
                                {k: v for k, v in st.session_state[editor_key]}
                            )
                            st.rerun()
                    st.session_state[editor_key][i] = (new_pos, new_masses)

                add_col1, add_col2 = st.columns([3, 1])
                with add_col2:
                    if st.button("➕ Add Var Mod", key=f"{editor_key}_add"):
                        st.session_state[editor_key].append(("", [0.0]))
                        st.rerun()

                # Preset buttons for common variable modifications
                existing_var_positions = {p for p, _ in st.session_state[editor_key]}
                presets_var = [
                    ("[", [42.0], "Acetyl (N-term)"),
                    ("M", [15.9949], "Oxidation-M"),
                    ("S", [79.9663], "Phospho-S"),
                    ("T", [79.9663], "Phospho-T"),
                    ("Y", [79.9663], "Phospho-Y"),
                ]

                available_presets = [
                    p for p in presets_var if p[0] not in existing_var_positions
                ]
                if available_presets:
                    preset_cols = st.columns(min(len(available_presets), 3))
                    for idx, (position, masses, label) in enumerate(available_presets):
                        with preset_cols[idx % len(preset_cols)]:
                            if st.button(
                                f"➕ {label}",
                                key=f"{editor_key}_preset_var_{position}_{masses[0]}",
                            ):
                                st.session_state[editor_key].append((position, masses))
                                st.session_state[full_key] = json.dumps(
                                    {k: v for k, v in st.session_state[editor_key]}
                                )
                                st.rerun()

                st.session_state[full_key] = json.dumps(
                    {k: v for k, v in st.session_state[editor_key]}
                )

            custom = {
                "database.static_mods": _render_static,
                "database.variable_mods": _render_variable,
            }

            # Remove the rendered fasta and output_file widgets and render the rest
            cfg_to_render = dict(struct_cfg)
            # remove fasta from schema to avoid duplicate widget keys
            try:
                cfg_to_render["sections"]["database"]["options"].pop("fasta", None)
            except Exception:
                pass
            # remove output_file so we can display a fixed output filename
            try:
                cfg_to_render["sections"]["output_and_runtime"]["options"].pop(
                    "output_file", None
                )
            except Exception:
                pass

            ui.render_structured_config(
                cfg_to_render,
                key_prefix="easypqp",
                custom_renderers=custom,
                default_open_sections=[],
            )

            # Display derived, uneditable output filename and set parameter
            easypqp_out_key = f"{PPFX}easypqp:output_and_runtime:output_file"
            easypqp_out_name = "easypqp_insilico_library.tsv"
            st.text_input(
                "Output library file",
                value=easypqp_out_name,
                disabled=True,
                key="easypqp_out_display",
            )
            st.session_state[easypqp_out_key] = easypqp_out_name
        else:
            st.info("No structured EasyPQP schema found — falling back to legacy UI.")

else:
    st.info(
        "No FASTA file found in workspace — EasyPQP insilico library generation is unavailable. "
        "Upload a FASTA via the **File Upload** page to enable this step.",
        icon="ℹ️",
    )

# -- 1b: OpenSwathAssayGenerator (INI-based) -----------------------------------
with st.expander(
    "⚙️ OpenSwathAssayGenerator — Assay Refinement",
    expanded=bool(lib_list or fasta_list),
):
    st.caption(
        "Refine a spectral library / transition list: filter m/z ranges, set "
        "fragment ion types, and produce an optimised assay file for OpenSWATH."
    )

    osag_ini = _copy_ini_once(ASSET_OSAG, "OpenSwathAssayGenerator.ini")
    if osag_ini is None:
        st.warning(
            "OpenSwathAssayGenerator INI not found in "
            "`assets/common-tool-descriptors/openswathassaygenerator/`. "
            "Place the INI file there to enable auto-generated parameter widgets."
        )
    else:
        # Derive available inputs based on the Assay source choice made earlier
        osag_inputs: list[str] = []
        mode = st.session_state.get("osag_input_mode", None)
        if mode is None:
            # fallback defaults
            if fasta_list:
                mode = "Generate from FASTA (predict transitions)"
            elif lib_list:
                mode = "Use existing transition list(s)"
            st.session_state["osag_input_mode"] = mode

        if mode == "Generate from FASTA (predict transitions)":
            # Use the fixed EasyPQP output as OSAG input
            easypqp_out_name = "easypqp_insilico_library.tsv"
            st.session_state[f"{PPFX}easypqp:output_and_runtime:output_file"] = (
                easypqp_out_name
            )
            osag_inputs = [easypqp_out_name]
        else:
            # Use existing transition lists from workspace (optionally include easypqp output if present)
            osag_inputs = list(lib_list)
            epqp_out = (
                pm.get_parameters_from_json().get("easypqp", {}).get("output_file")
            )
            if epqp_out and epqp_out not in osag_inputs:
                osag_inputs.insert(0, epqp_out)

        if osag_inputs:
            if len(osag_inputs) == 1:
                # derived single input -> display uneditable
                st.text_input(
                    "Input transition list",
                    value=osag_inputs[0],
                    disabled=True,
                    key="osag_in_display",
                )
                st.session_state[f"{TPFX}OpenSwathAssayGenerator:1:in"] = osag_inputs[0]
            else:
                _seed_choice_state(
                    "osag_in_sel",
                    osag_inputs,
                    _saved_tool_param("OpenSwathAssayGenerator", "in", osag_inputs[0]),
                )
                osag_in = st.selectbox(
                    "Input transition list",
                    options=osag_inputs,
                    key="osag_in_sel",
                    help="Source library — generated by EasyPQP or uploaded to workspace.",
                )
                st.session_state[f"{TPFX}OpenSwathAssayGenerator:1:in"] = osag_in
        else:
            st.info(
                "No transition list available. Run EasyPQP first or upload a library file."
            )

        osag_out_name = "openswathassay_targets.tsv"
        st.text_input(
            "Output assay file",
            value=osag_out_name,
            disabled=True,
            key="osag_out_display",
        )
        st.session_state[f"{TPFX}OpenSwathAssayGenerator:1:out"] = osag_out_name

        st.markdown("**Parameters**")
        ui.input_TOPP(
            "OpenSwathAssayGenerator",
            num_cols=3,
            display_tool_name=False,
            display_subsections=True,
            exclude_parameters=["in", "out", "in_type", "out_type"],
        )

# --- OpenSwathDecoyGenerator (top-level under Spectral Library Parameters)
with st.expander("⚙️ OpenSwathDecoyGenerator Parameters", expanded=True):
    st.markdown(
        "Append decoy sequences to the assay library. The output feeds directly into OpenSwathWorkflow's `-tr` parameter."
    )

    osdg_ini = _copy_ini_once(ASSET_OSDG, "OpenSwathDecoyGenerator.ini")
    if osdg_ini is None:
        st.warning(
            "OpenSwathDecoyGenerator INI not found in `assets/common-tool-descriptors/openswathdecoygenerator/`."
        )
    else:
        # Derive input: prefer OSAG output, then libs
        decoy_in_opts: list[str] = []
        osag_out_val = st.session_state.get(f"{TPFX}OpenSwathAssayGenerator:1:out", "")
        if osag_out_val:
            decoy_in_opts.append(osag_out_val)
        decoy_in_opts += [l for l in lib_list if l not in decoy_in_opts]

        if decoy_in_opts:
            if len(decoy_in_opts) == 1:
                st.text_input(
                    "Input transition list",
                    value=decoy_in_opts[0],
                    disabled=True,
                    key="osdg_in_display",
                )
                st.session_state[f"{TPFX}OpenSwathDecoyGenerator:1:in"] = decoy_in_opts[
                    0
                ]
            else:
                _seed_choice_state(
                    "osdg_in_sel",
                    decoy_in_opts,
                    _saved_tool_param(
                        "OpenSwathDecoyGenerator", "in", decoy_in_opts[0]
                    ),
                )
                decoy_in = st.selectbox(
                    "Input transition list",
                    options=decoy_in_opts,
                    key="osdg_in_sel",
                    help="Source assay file — normally the OpenSwathAssayGenerator output.",
                )
                st.session_state[f"{TPFX}OpenSwathDecoyGenerator:1:in"] = decoy_in
        else:
            st.info("No assay file available. Configure OpenSwathAssayGenerator first.")

        decoy_out_name = "openswath_targets_and_decoys.pqp"
        st.text_input(
            "Output file (target + decoy library)",
            value=decoy_out_name,
            disabled=True,
            key="osdg_out_display",
        )
        st.session_state[f"{TPFX}OpenSwathDecoyGenerator:1:out"] = decoy_out_name

        st.markdown("**Decoy Generator Parameters**")
        ui.input_TOPP(
            "OpenSwathDecoyGenerator",
            num_cols=3,
            display_tool_name=False,
            display_subsections=True,
            exclude_parameters=["in", "out", "in_type", "out_type"],
        )

# -----------------------------------------------------------------------------
# SECTION 2 — OpenSwathWorkflow (INI-based)

st.markdown("---")
st.subheader("OpenSwathWorkflow")
st.markdown(
    "Main DIA extraction engine. mzML files are taken from the File Upload page; "
    "the transition library is derived from the decoy-generation step above."
)

# Copy OpenSWATH INI into workspace if needed
osw_inis = sorted(ASSET_OSW.glob("*.ini")) if ASSET_OSW.exists() else []
if not osw_inis:
    st.error(
        "No OpenSwathWorkflow INI found in "
        "`assets/common-tool-descriptors/openswathworkflow/`. "
        "Add the release INI to enable parameter configuration."
    )
else:
    selected_ini_name = next(
        # (n.name for n in osw_inis if "release" in n.name.lower()), osw_inis[-1].name
        (n.name for n in osw_inis if "dev" in n.name.lower()),
        osw_inis[-1].name,
    )
    dest_osw = INI_DIR / "OpenSwathWorkflow.ini"
    if not dest_osw.exists():
        try:
            shutil.copy2(ASSET_OSW / selected_ini_name, dest_osw)
        except Exception as e:
            st.error(f"Could not copy INI: {e}")

    st.caption(f"Using descriptor: `{selected_ini_name}`")

    # -- Derived inputs --------------------------------------------------------
    col_in1, col_in2 = st.columns(2)

    with col_in1:
        st.markdown("**Input mzML files** (from File Upload)")
        if mzml_list:
            for m in mzml_list:
                st.caption(f"• {m}")
            mzml_space = " ".join(mzml_list)
            st.text_input(
                "Input mzML files",
                value=mzml_space,
                disabled=True,
                key="osw_mzmls_display",
            )
            st.session_state[f"{TPFX}OpenSwathWorkflow:1:in"] = mzml_space
        else:
            st.warning("No mzML files in workspace — add via **File Upload**.")

    with col_in2:
        st.markdown("**Transition library** (`-tr`)")
        # Auto-derive from decoy generator output, then OSAG, then libs
        tr_candidates: list[str] = []
        dec_out_val = st.session_state.get(f"{TPFX}OpenSwathDecoyGenerator:1:out", "")
        if dec_out_val:
            tr_candidates.append(dec_out_val)
        osag_out_val2 = st.session_state.get(f"{TPFX}OpenSwathAssayGenerator:1:out", "")
        if osag_out_val2 and osag_out_val2 not in tr_candidates:
            tr_candidates.append(osag_out_val2)
        tr_candidates += [l for l in lib_list if l not in tr_candidates]

        if tr_candidates:
            if len(tr_candidates) == 1:
                st.text_input(
                    "Transition library (-tr)",
                    value=tr_candidates[0],
                    disabled=True,
                    key="osw_tr_display",
                )
                st.session_state[f"{TPFX}OpenSwathWorkflow:1:tr"] = tr_candidates[0]
                st.caption(f"→ `-tr {tr_candidates[0]}`")
            else:
                _seed_choice_state(
                    "osw_tr_sel",
                    tr_candidates,
                    _saved_tool_param("OpenSwathWorkflow", "tr", tr_candidates[0]),
                )
                tr_file = st.selectbox(
                    "Select transition library",
                    options=tr_candidates,
                    key="osw_tr_sel",
                    help="Normally the OpenSwathDecoyGenerator output. You can override here if needed.",
                )
                st.session_state[f"{TPFX}OpenSwathWorkflow:1:tr"] = tr_file
                st.caption(f"→ `-tr {tr_file}`")
        else:
            st.info("No transition list available — configure upstream steps first.")

    # -- Fixed output ----------------------------------------------------------
    st.markdown("**Output**")
    out_key = f"{TPFX}OpenSwathWorkflow:1:out_features"
    st.session_state[out_key] = "openswath_results.osw"
    st.text_input(
        "Output features file (fixed)",
        value="openswath_results.osw",
        disabled=True,
        help="Result OSW file — passed to PyProphet in the next step.",
    )

    st.markdown("**Optional OpenSwath Outputs**")
    aux_col1, aux_col2 = st.columns(2)

    save_xics_key = "osw_save_xics"
    save_xims_key = "osw_save_xims"
    chrom_key = f"{TPFX}OpenSwathWorkflow:1:out_chrom"
    mobil_key = f"{TPFX}OpenSwathWorkflow:1:out_mobilogram"

    if save_xics_key not in st.session_state:
        saved_chrom = _saved_tool_param("OpenSwathWorkflow", "out_chrom", None)
        st.session_state[save_xics_key] = (
            bool(saved_chrom) if saved_chrom is not None else True
        )
    if save_xims_key not in st.session_state:
        saved_mobil = _saved_tool_param("OpenSwathWorkflow", "out_mobilogram", None)
        st.session_state[save_xims_key] = (
            bool(saved_mobil) if saved_mobil is not None else False
        )

    with aux_col1:
        save_xics = st.checkbox(
            "Save XICs",
            key=save_xics_key,
            help="Write extracted ion chromatograms as a fixed Parquet `.xic` output.",
        )
        st.text_input(
            "Chromatogram output",
            value=OSW_XIC_OUT if save_xics else "(disabled)",
            disabled=True,
            key="osw_xic_output_display",
        )
    with aux_col2:
        save_xims = st.checkbox(
            "Save XIMs",
            key=save_xims_key,
            help="Write extracted ion mobilograms as a fixed Parquet `.xim` output.",
        )
        st.text_input(
            "Mobilogram output",
            value=OSW_XIM_OUT if save_xims else "(disabled)",
            disabled=True,
            key="osw_xim_output_display",
        )

    st.session_state[chrom_key] = OSW_XIC_OUT if save_xics else ""
    st.session_state[mobil_key] = OSW_XIM_OUT if save_xims else ""

    st.markdown("**Calibration Debug Outputs**")
    debug_toggle_key = "osw_save_calibration_debug"
    debug_irt_mzml_key = f"{TPFX}OpenSwathWorkflow:1:Debugging:irt_mzml"
    debug_irt_trafo_key = f"{TPFX}OpenSwathWorkflow:1:Debugging:irt_trafo"
    debug_im_key = f"{TPFX}OpenSwathWorkflow:1:Calibration:MassIMCorrection:debug_im_file"
    debug_mz_key = f"{TPFX}OpenSwathWorkflow:1:Calibration:MassIMCorrection:debug_mz_file"

    saved_debug_enabled = any(
        [
            _saved_tool_param("OpenSwathWorkflow", "Debugging:irt_mzml", ""),
            _saved_tool_param("OpenSwathWorkflow", "Debugging:irt_trafo", ""),
            _saved_tool_param(
                "OpenSwathWorkflow",
                "Calibration:MassIMCorrection:debug_im_file",
                "",
            ),
            _saved_tool_param(
                "OpenSwathWorkflow",
                "Calibration:MassIMCorrection:debug_mz_file",
                "",
            ),
        ]
    )
    if debug_toggle_key not in st.session_state:
        st.session_state[debug_toggle_key] = bool(saved_debug_enabled)

    save_debug_outputs = st.checkbox(
        "Save calibration debug files",
        key=debug_toggle_key,
        help="Write fixed debug outputs for ion mobility calibration, m/z calibration, and iRT calibration artifacts.",
    )
    debug_display_col1, debug_display_col2 = st.columns(2)
    with debug_display_col1:
        st.text_input(
            "Ion mobility debug file",
            value=OSW_DEBUG_IM_OUT if save_debug_outputs else "(disabled)",
            disabled=True,
            key="osw_debug_im_display",
        )
        st.text_input(
            "m/z debug file",
            value=OSW_DEBUG_MZ_OUT if save_debug_outputs else "(disabled)",
            disabled=True,
            key="osw_debug_mz_display",
        )
    with debug_display_col2:
        st.text_input(
            "iRT transform debug file",
            value=OSW_DEBUG_IRT_TRAFO_OUT if save_debug_outputs else "(disabled)",
            disabled=True,
            key="osw_debug_trafo_display",
        )
        st.text_input(
            "iRT chromatogram debug file",
            value=OSW_DEBUG_IRT_MZML_OUT if save_debug_outputs else "(disabled)",
            disabled=True,
            key="osw_debug_irt_mzml_display",
        )

    st.session_state[debug_irt_mzml_key] = (
        OSW_DEBUG_IRT_MZML_OUT if save_debug_outputs else ""
    )
    st.session_state[debug_irt_trafo_key] = (
        OSW_DEBUG_IRT_TRAFO_OUT if save_debug_outputs else ""
    )
    st.session_state[debug_im_key] = OSW_DEBUG_IM_OUT if save_debug_outputs else ""
    st.session_state[debug_mz_key] = OSW_DEBUG_MZ_OUT if save_debug_outputs else ""

    st.markdown("**Input Read Mode**")
    read_options = [
        "normal",
        "cache",
        "workingInMemory",
        "cacheWorkingInMemory",
    ]
    read_option_help = {
        "normal": "No on-disk caching. Streams the input directly and is the lowest-overhead option for a single pass.",
        "cache": "Creates cached files on disk first and reads from those cached files. Useful for random access or repeated passes.",
        "workingInMemory": "Loads the regular OpenSWATH access objects into RAM for faster repeated access. Does not create disk cache files.",
        "cacheWorkingInMemory": "Creates disk cache files first, then loads that cached representation into memory.",
    }
    read_option_widget_key = "osw_read_options"
    temp_dir_key = f"{TPFX}OpenSwathWorkflow:1:tempDirectory"
    saved_read_option = _saved_tool_param("OpenSwathWorkflow", "readOptions", "normal")
    _seed_choice_state(read_option_widget_key, read_options, saved_read_option)

    read_mode_col, read_info_col = st.columns([1.2, 1.8])
    with read_mode_col:
        selected_read_option = st.selectbox(
            "Read option",
            options=read_options,
            key=read_option_widget_key,
            help="Controls whether OpenSWATH streams input directly, caches to disk, or loads data into memory.",
        )
    with read_info_col:
        st.caption(read_option_help[selected_read_option])

    st.session_state[f"{TPFX}OpenSwathWorkflow:1:readOptions"] = selected_read_option
    if selected_read_option in {"cache", "cacheWorkingInMemory"}:
        OSW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        st.session_state[temp_dir_key] = str(OSW_CACHE_DIR.resolve())
        st.text_input(
            "Workspace cache directory",
            value=str(OSW_CACHE_DIR.resolve()),
            disabled=True,
            key="osw_cache_dir_display",
            help="Automatically created for cache-based OpenSWATH read modes.",
        )
    else:
        st.session_state[temp_dir_key] = "/tmp"
        st.caption(
            "No dedicated workspace cache directory is needed for this read mode."
        )

    # -- INI-driven parameter widgets ------------------------------------------
    st.markdown("**OpenSwathWorkflow Parameters**")
    ui.input_TOPP(
        "OpenSwathWorkflow",
        num_cols=3,
        display_tool_name=False,
        display_subsections=True,
        exclude_parameters=[
            "in",
            "tr",
            "out_features",
            "out_chrom",
            "out_mobilogram",
            "irt_mzml",
            "irt_trafo",
            "debug_im_file",
            "debug_mz_file",
            "readOptions",
            "tempDirectory",
            "tr_type",
            "out_features_type",
            "out_qc",
        ],
    )

# -----------------------------------------------------------------------------
# SECTION 4 — PyProphet (JSON-driven)

st.markdown("---")
st.subheader("PyProphet")
st.markdown(
    "Semi-supervised scoring, peptide/protein inference and TSV export. "
    "The input is the `openswath_results.osw` file produced above."
)

# Prefer structured JSON configs in assets/common-tool-descriptors/pyprophet
if ASSET_PY.exists():
    cfg_files = sorted(ASSET_PY.glob("*.json"))
else:
    cfg_files = []

if cfg_files:
    st.caption(
        "PyProphet structured configuration templates detected — use the sections below to configure each subcommand."
    )
    # Load each config and render with the structured renderer
    for cfg_path in cfg_files:
        cfg = _load_json_asset(cfg_path)
        if not cfg:
            continue
        cmd_key = cfg_path.stem
        title = cfg.get("meta", {}).get("command", cmd_key)
        saved_cmd_cfg_path = workspace_dir / "tools-configs" / "pyprophet" / f"{cmd_key}.json"
        saved_cmd_cfg = (
            _load_json_asset(saved_cmd_cfg_path) if saved_cmd_cfg_path.exists() else {}
        )
        with st.expander(f"⚙️ {title}", expanded=False):
            # Prepare config to render; hide experimental alignment and derive IO
            cfg_to_render = dict(cfg)
            try:
                cmd_name = cfg.get("meta", {}).get("command", "")
            except Exception:
                cmd_name = ""

            # Hide alignment section for export commands
            if "export" in cmd_name.lower() or "export" in cmd_key.lower():
                sections = cfg_to_render.get("sections", {})
                if "alignment" in sections:
                    sections.pop("alignment", None)
                    cfg_to_render["sections"] = sections

            infer_context_values: list[str] = []
            if cmd_key in {
                "pyprophet_infer_peptide_config",
                "pyprophet_infer_protein_config",
            }:
                sections = cfg_to_render.get("sections", {})
                context_section = sections.get("inference_context", {})
                context_options = context_section.get("options", {})
                context_options.pop("context", None)
                if context_options:
                    context_section["options"] = context_options
                    sections["inference_context"] = context_section
                else:
                    sections.pop("inference_context", None)
                cfg_to_render["sections"] = sections

            # Determine derived IO values (don't render 'in'/'out' widgets)
            derived_in = None
            derived_out = None
            low_cmd = cmd_key.lower()
            if (
                "score" in low_cmd
                or "infer" in low_cmd
                or "score" in cmd_name.lower()
                or "infer" in cmd_name.lower()
            ):
                derived_in = "openswath_results.osw"
            if "export" in low_cmd or "export" in cmd_name.lower():
                derived_in = "openswath_results.osw"
                derived_out = "openswath_results.tsv"

            if (derived_in is not None) or (derived_out is not None):
                # remove io options so renderer won't create editable widgets
                try:
                    io_opts = (
                        cfg_to_render.get("sections", {})
                        .get("io", {})
                        .get("options", {})
                    )
                    io_opts.pop("in", None)
                    io_opts.pop("out", None)
                    if "io" in cfg_to_render.get("sections", {}):
                        cfg_to_render["sections"]["io"]["options"] = io_opts
                except Exception:
                    pass

            ui.render_structured_config(
                cfg_to_render,
                key_prefix=f"pyprophet:{cmd_key}",
                default_open_sections=["io"],
            )

            # Display and set derived IO parameters as uneditable values
            try:
                if derived_in is not None:
                    full_in = f"{PPFX}pyprophet:{cmd_key}:io:in"
                    st.text_input(
                        "Input file",
                        value=derived_in,
                        disabled=True,
                        key=f"py_in_display_{cmd_key}",
                    )
                    st.session_state[full_in] = derived_in
                if derived_out is not None:
                    full_out = f"{PPFX}pyprophet:{cmd_key}:io:out"
                    st.text_input(
                        "Output file",
                        value=derived_out,
                        disabled=True,
                        key=f"py_out_display_{cmd_key}",
                    )
                    st.session_state[full_out] = derived_out
            except Exception:
                pass

            if cmd_key in {
                "pyprophet_infer_peptide_config",
                "pyprophet_infer_protein_config",
            }:
                infer_type = "peptide" if "peptide" in cmd_key else "protein"
                saved_contexts = _normalize_context_selection(
                    saved_cmd_cfg.get(
                        "contexts",
                        saved_cmd_cfg.get("context", "global"),
                    )
                )
                st.caption(
                    "Run this inference separately for each selected context."
                )
                ctx_cols = st.columns(len(PY_INFER_CONTEXTS))
                for col, context in zip(ctx_cols, PY_INFER_CONTEXTS):
                    widget_key = (
                        f"pyprophet_{infer_type}_context_"
                        f"{context.replace('-', '_')}"
                    )
                    _seed_checkbox_state(widget_key, context in saved_contexts)
                    if col.checkbox(context, key=widget_key):
                        infer_context_values.append(context)
                if not infer_context_values:
                    st.warning(
                        "At least one inference context should be selected. "
                        "If left empty, the workflow will fall back to `global`."
                    )

            # Auto-save per-command tool config to workspace/tools-configs/pyprophet/{cmd_key}.json
            try:
                tools_dir = workspace_dir / "tools-configs" / "pyprophet"
                tools_dir.mkdir(parents=True, exist_ok=True)
                out_dict = {}
                for section_name, section_def in cfg_to_render.get(
                    "sections", {}
                ).items():
                    opts = section_def.get("options", {})
                    for opt_name, opt_def in opts.items():
                        full = f"{PPFX}pyprophet:{cmd_key}:{section_name}:{opt_name}"
                        v = st.session_state.get(full, None)
                        if isinstance(v, str):
                            try:
                                parsed = json.loads(v)
                                out_dict[opt_name] = parsed
                            except Exception:
                                out_dict[opt_name] = v
                        else:
                            out_dict[opt_name] = v
                if cmd_key in {
                    "pyprophet_infer_peptide_config",
                    "pyprophet_infer_protein_config",
                }:
                    out_dict.pop("context", None)
                    out_dict["contexts"] = infer_context_values or ["global"]
                out_path = tools_dir / f"{cmd_key}.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(out_dict, f, indent=2)
            except Exception as e:
                st.warning(f"Could not auto-save PyProphet config: {e}")
else:
    st.info("No structured PyProphet schemas found — falling back to built-in UI.")
    try:
        py.ui()
    except Exception:
        st.warning("PyProphet UI unavailable.")

matrix_cfg_path = (
    workspace_dir
    / "tools-configs"
    / "pyprophet"
    / "pyprophet_export_matrix_config.json"
)
matrix_saved = _load_json_asset(matrix_cfg_path) if matrix_cfg_path.exists() else {}

with st.expander("⚙️ pyprophet export matrix", expanded=False):
    st.caption(
        "Export quantification matrices at selected precursor, peptide, and protein levels. "
        "Each selected level writes a fixed output file into the workflow results directory."
    )

    _seed_multiselect_state(
        "pyprophet_matrix_levels",
        PY_MATRIX_LEVELS,
        matrix_saved.get("levels", PY_MATRIX_LEVELS),
    )
    matrix_levels = st.multiselect(
        "Matrix export levels",
        options=PY_MATRIX_LEVELS,
        key="pyprophet_matrix_levels",
        help="Choose which matrix quantification levels to export after scoring and inference.",
    )

    for level in PY_MATRIX_LEVELS:
        if level in matrix_levels:
            st.caption(f"`{level}` → `{PY_MATRIX_OUTS[level]}`")

    matrix_col1, matrix_col2, matrix_col3 = st.columns(3)
    matrix_col1.caption("Outputs are written as fixed TSV files.")
    matrix_transition_quant = matrix_col2.checkbox(
        "Transition quantification",
        value=matrix_saved.get("transition_quantification", True),
        key="pyprophet_matrix_transition_quantification",
        help="Report aggregated transition-level quantification.",
    )
    matrix_use_alignment = matrix_col3.checkbox(
        "Use alignment",
        value=matrix_saved.get("use_alignment", True),
        key="pyprophet_matrix_use_alignment",
        help="Recover peaks with good alignment scores if alignment data is present.",
    )

    matrix_col1, matrix_col2, matrix_col3 = st.columns(3)
    matrix_col1.number_input(
        "Max transition PEP",
        value=float(matrix_saved.get("max_transition_pep", 0.7)),
        step=0.05,
        key="pyprophet_matrix_max_transition_pep",
    )
    matrix_col2.selectbox(
        "IPF mode",
        options=["peptidoform", "augmented", "disable"],
        index=["peptidoform", "augmented", "disable"].index(
            matrix_saved.get("ipf", "peptidoform")
        ),
        key="pyprophet_matrix_ipf",
    )
    matrix_col3.number_input(
        "IPF max peptidoform PEP",
        value=float(matrix_saved.get("ipf_max_peptidoform_pep", 0.4)),
        step=0.05,
        key="pyprophet_matrix_ipf_max_peptidoform_pep",
    )

    matrix_col1, matrix_col2, matrix_col3 = st.columns(3)
    matrix_col1.number_input(
        "Max run-specific peakgroup q-value",
        value=float(matrix_saved.get("max_rs_peakgroup_qvalue", 0.05)),
        step=0.01,
        key="pyprophet_matrix_max_rs_peakgroup_qvalue",
    )
    matrix_col2.number_input(
        "Max global peptide q-value",
        value=float(matrix_saved.get("max_global_peptide_qvalue", 0.01)),
        step=0.01,
        key="pyprophet_matrix_max_global_peptide_qvalue",
    )
    matrix_col3.number_input(
        "Max global protein q-value",
        value=float(matrix_saved.get("max_global_protein_qvalue", 0.01)),
        step=0.01,
        key="pyprophet_matrix_max_global_protein_qvalue",
    )

    matrix_col1, matrix_col2, matrix_col3 = st.columns(3)
    matrix_col1.number_input(
        "Max alignment PEP",
        value=float(matrix_saved.get("max_alignment_pep", 0.7)),
        step=0.05,
        key="pyprophet_matrix_max_alignment_pep",
    )
    matrix_col2.number_input(
        "Top N features",
        min_value=1,
        value=int(matrix_saved.get("top_n", 3)),
        step=1,
        key="pyprophet_matrix_top_n",
    )
    matrix_col3.checkbox(
        "Consistent top features",
        value=matrix_saved.get("consistent_top", True),
        key="pyprophet_matrix_consistent_top",
        help="Use the same top features across all runs.",
    )

    st.selectbox(
        "Normalization",
        options=["none", "median", "medianmedian", "quantile"],
        index=["none", "median", "medianmedian", "quantile"].index(
            matrix_saved.get("normalization", "none")
        ),
        key="pyprophet_matrix_normalization",
    )

    try:
        matrix_cfg_path.parent.mkdir(parents=True, exist_ok=True)
        with open(matrix_cfg_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "levels": st.session_state.get("pyprophet_matrix_levels", []),
                    "csv": False,
                    "transition_quantification": matrix_transition_quant,
                    "max_transition_pep": st.session_state.get(
                        "pyprophet_matrix_max_transition_pep", 0.7
                    ),
                    "ipf": st.session_state.get("pyprophet_matrix_ipf", "peptidoform"),
                    "ipf_max_peptidoform_pep": st.session_state.get(
                        "pyprophet_matrix_ipf_max_peptidoform_pep", 0.4
                    ),
                    "max_rs_peakgroup_qvalue": st.session_state.get(
                        "pyprophet_matrix_max_rs_peakgroup_qvalue", 0.05
                    ),
                    "max_global_peptide_qvalue": st.session_state.get(
                        "pyprophet_matrix_max_global_peptide_qvalue", 0.01
                    ),
                    "max_global_protein_qvalue": st.session_state.get(
                        "pyprophet_matrix_max_global_protein_qvalue", 0.01
                    ),
                    "use_alignment": matrix_use_alignment,
                    "max_alignment_pep": st.session_state.get(
                        "pyprophet_matrix_max_alignment_pep", 0.7
                    ),
                    "top_n": st.session_state.get("pyprophet_matrix_top_n", 3),
                    "consistent_top": st.session_state.get(
                        "pyprophet_matrix_consistent_top", True
                    ),
                    "normalization": st.session_state.get(
                        "pyprophet_matrix_normalization", "none"
                    ),
                },
                f,
                indent=2,
            )
    except Exception as e:
        st.warning(f"Could not auto-save pyprophet export matrix config: {e}")

# -----------------------------------------------------------------------------
# SAVE button

st.markdown("---")
save_col, _ = st.columns([1, 3])
with save_col:
    if st.button("💾 Save all parameters to workspace", type="primary"):
        pm.save_parameters()

        # Also persist the non-TOPP workflow-level keys that the workflow
        # execution page needs to read (session_state only is not enough
        # because the WorkflowManager runs in a subprocess with no session_state).
        _extra = pm.get_parameters_from_json()
        _extra["osag_input_mode"] = st.session_state.get(
            "osag_input_mode", "Use existing transition list(s)"
        )
        # Capture the FASTA selected for EasyPQP under a stable key
        _extra["osag_fasta"] = st.session_state.get(
            "osag_workspace_fasta",
            st.session_state.get("osag_workspace_fasta_only", ""),
        )
        # Also copy easypqp section if the user configured it via the
        # structured UI (params may be stored under PPFX keys)
        _ep = {}
        for _k, _v in st.session_state.items():
            if _k.startswith(f"{PPFX}easypqp:"):
                _short = _k.replace(f"{PPFX}easypqp:", "")
                # flatten nested keys to top-level for easier lookup
                _ep[_short.split(":")[-1]] = _v
        if _ep:
            _extra.setdefault("easypqp", {}).update(_ep)

        with open(pm.params_file, "w", encoding="utf-8") as _fh:
            import json as _json

            _json.dump(_extra, _fh, indent=4)

        save_params(pm.get_parameters_from_json())
        # Also auto-save EasyPQP structured config (if present) so workflow
        # execution can pick it up via --config. Merge flattened UI keys into
        # the canonical EasyPQP schema (use the shipped asset as a template)
        try:
            tools_dir = workspace_dir / "tools-configs" / "easypqp"
            tools_dir.mkdir(parents=True, exist_ok=True)
            flat = _extra.get("easypqp") or {}
            if flat:
                # Load asset template if available, else start from minimal structure
                template = _load_json_asset(ASSET_EPQP) or {}

                def _set(target: dict, path: tuple, value):
                    d = target
                    for p in path[:-1]:
                        if p not in d or not isinstance(d[p], dict):
                            d[p] = {}
                        d = d[p]
                    d[path[-1]] = value

                merged = dict(template) if isinstance(template, dict) else {}

                # mapping from flat key -> path in merged config
                key_map = {
                    "fasta": ("database", "fasta"),
                    "generate_decoys": ("database", "generate_decoys"),
                    "decoy_tag": ("database", "decoy_tag"),
                    "missed_cleavages": ("database", "enzyme", "missed_cleavages"),
                    "min_len": ("database", "enzyme", "min_len"),
                    "max_len": ("database", "enzyme", "max_len"),
                    "cleave_at": ("database", "enzyme", "cleave_at"),
                    "restrict": ("database", "enzyme", "restrict"),
                    "c_terminal": ("database", "enzyme", "c_terminal"),
                    "semi_enzymatic": ("database", "enzyme", "semi_enzymatic"),
                    "max_variable_mods": ("database", "max_variable_mods"),
                    "peptide_min_mass": ("database", "peptide_min_mass"),
                    "peptide_max_mass": ("database", "peptide_max_mass"),
                    "static_mods": ("database", "static_mods"),
                    "variable_mods": ("database", "variable_mods"),
                    "precursor_charge": ("insilico_settings", "precursor_charge"),
                    "max_fragment_charge": ("insilico_settings", "max_fragment_charge"),
                    "min_transitions": ("insilico_settings", "min_transitions"),
                    "max_transitions": ("insilico_settings", "max_transitions"),
                    "fragmentation_model": ("insilico_settings", "fragmentation_model"),
                    "allowed_fragment_types": (
                        "insilico_settings",
                        "allowed_fragment_types",
                    ),
                    "rt_scale": ("insilico_settings", "rt_scale"),
                    "unimod_annotation": ("insilico_settings", "unimod_annotation"),
                    "max_delta_unimod": ("insilico_settings", "max_delta_unimod"),
                    "enable_unannotated": ("insilico_settings", "enable_unannotated"),
                    "unimod_xml_path": ("insilico_settings", "unimod_xml_path"),
                    "device": ("dl_feature_generators", "device"),
                    "instrument": ("dl_feature_generators", "instrument"),
                    "nce": ("dl_feature_generators", "nce"),
                    "batch_size": ("dl_feature_generators", "batch_size"),
                    "peptide_chunking": ("peptide_chunking",),
                    "output_file": ("output_file",),
                    "write_report": ("write_report",),
                    "parquet_output": ("parquet_output",),
                    "threads": ("threads",),
                }

                # detect fine-tune related keys to avoid mapping collisions
                ft_keys = {
                    "fine_tune",
                    "train_data_path",
                    "save_model",
                    "learning_rate",
                    "epochs",
                    "batch_size",
                }

                def _template_has_null(tmpl: dict, path: tuple) -> bool:
                    d = tmpl
                    try:
                        for p in path:
                            d = d[p]
                        return d is None
                    except Exception:
                        return False

                for k, v in flat.items():
                    if k in ("static_mods", "variable_mods") and isinstance(v, str):
                        try:
                            parsed = json.loads(v)
                            _set(merged, key_map.get(k, (k,)), parsed)
                        except Exception:
                            # leave as-is (template default) on parse failure
                            pass
                    elif k == "precursor_charge":
                        # accept newline/comma separated or JSON lists
                        if isinstance(v, str):
                            parts = [
                                p.strip()
                                for p in v.replace("\n", ",").split(",")
                                if p.strip()
                            ]
                            try:
                                vals = [int(x) for x in parts]
                            except Exception:
                                vals = parts
                            _set(merged, key_map.get(k, (k,)), vals)
                    elif k == "allowed_fragment_types" and isinstance(v, str):
                        _set(
                            merged,
                            key_map.get(k, (k,)),
                            [t.strip() for t in v.splitlines() if t.strip()],
                        )
                    else:
                        # route fine-tune specific keys into dl_feature_generators.fine_tune_config
                        if k in ft_keys and any(
                            x in flat
                            for x in (
                                "fine_tune",
                                "epochs",
                                "learning_rate",
                                "save_model",
                                "train_data_path",
                            )
                        ):
                            ft_path = ("dl_feature_generators", "fine_tune_config", k)
                            # coerce types
                            val = v
                            if isinstance(v, str) and v == "":
                                # if template expects null for this field, set None
                                if _template_has_null(template, ft_path):
                                    val = None
                            else:
                                # try JSON parse
                                if isinstance(v, str):
                                    try:
                                        val = json.loads(v)
                                    except Exception:
                                        pass
                            _set(merged, ft_path, val)
                            continue

                        path = key_map.get(k)
                        if path:
                            val = v
                            # preserve dict/list as-is
                            if isinstance(v, (dict, list)):
                                _set(merged, path, v)
                                continue
                            # try parse JSON strings
                            if isinstance(v, str):
                                try:
                                    parsed = json.loads(v)
                                    val = parsed
                                except Exception:
                                    # keep string
                                    val = v

                            # convert empty strings to None when template had null
                            if val == "" and _template_has_null(template, path):
                                val = None

                            # convert numeric strings to int/float if possible
                            if isinstance(val, str):
                                if val.isdigit():
                                    try:
                                        val = int(val)
                                    except Exception:
                                        pass
                                else:
                                    try:
                                        f = float(val)
                                        val = f
                                    except Exception:
                                        pass

                            _set(merged, path, val)

                easypqp_path = tools_dir / "easypqp_insilico.json"
                with open(easypqp_path, "w", encoding="utf-8") as _efh:
                    _json.dump(merged, _efh, indent=2)
                st.info(f"Saved EasyPQP config to workspace: {easypqp_path}")
        except Exception as _e:
            st.warning(f"Could not auto-save EasyPQP config: {_e}")
        st.success("All parameters saved to workspace `params.json`.")
