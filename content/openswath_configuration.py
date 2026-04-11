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

import copy
import json
import shutil
from pathlib import Path

import pyopenms as poms
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
saved_params_cache = pm.get_parameters_from_json()

# Convenience prefixes
TPFX = pm.topp_param_prefix  # e.g. "<ws>-TOPP-"
PPFX = pm.param_prefix  # e.g. "<ws>-param-"
OSW_DRAFT_KEY = f"{workspace_dir.resolve()}::OpenSwathWorkflow::draft_values"

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


def _ensure_osw_ini_for_workspace() -> tuple[Path | None, str | None]:
    """
    Ensure the OpenSwathWorkflow workspace INI matches the installed binary.

    Falls back to an existing workspace copy, then to the bundled asset if the
    binary is unavailable.
    """
    dest = INI_DIR / "OpenSwathWorkflow.ini"
    state_prefix = f"{workspace_dir.resolve()}::OpenSwathWorkflow"
    signature_key = f"{state_prefix}::descriptor_signature"
    source_key = f"{state_prefix}::descriptor_source"
    saved_tool_params = _saved_params().get("OpenSwathWorkflow", {})
    current_signature = json.dumps(saved_tool_params, sort_keys=True, default=str)

    if (not dest.exists()) or (
        st.session_state.get(signature_key) != current_signature
    ):
        if pm.refresh_ini_from_binary("OpenSwathWorkflow", saved_tool_params):
            st.session_state[source_key] = "binary"
        elif dest.exists():
            st.session_state[source_key] = "workspace"
        else:
            candidates = sorted(ASSET_OSW.glob("*.ini")) if ASSET_OSW.exists() else []
            if candidates:
                src = next(
                    (p for p in candidates if "dev" in p.name.lower()),
                    candidates[-1],
                )
                try:
                    shutil.copy2(src, dest)
                    st.session_state[source_key] = f"asset:{src.name}"
                except Exception:
                    st.session_state[source_key] = None
            else:
                st.session_state[source_key] = None
        st.session_state[signature_key] = current_signature

    source = st.session_state.get(source_key)
    if dest.exists():
        return dest, source
    return None, source


def _collect_current_osw_values() -> dict[str, object]:
    values: dict[str, object] = {}
    prefix = f"{TPFX}OpenSwathWorkflow:1:"
    for key, value in st.session_state.items():
        if not key.startswith(prefix):
            continue
        if key.endswith("_display"):
            continue
        short_key = key.split(":1:", 1)[1]
        values[short_key] = value
    return values


def _osw_draft_values() -> dict[str, object]:
    draft = st.session_state.get(OSW_DRAFT_KEY, {})
    return draft.copy() if isinstance(draft, dict) else {}


def _collect_saved_and_current_osw_values() -> dict[str, object]:
    saved_tool_params = pm.get_parameters_from_json().get("OpenSwathWorkflow", {})
    merged_values = (
        saved_tool_params.copy() if isinstance(saved_tool_params, dict) else {}
    )
    merged_values.update(_osw_draft_values())
    merged_values.update(_collect_current_osw_values())
    return merged_values


def _store_current_osw_draft() -> None:
    current_values = _collect_current_osw_values()
    if current_values:
        st.session_state[OSW_DRAFT_KEY] = _collect_saved_and_current_osw_values()


def _read_ini_short_values(
    tool: str, ini_path: Path, short_keys: list[str]
) -> dict[str, object]:
    values: dict[str, object] = {}
    if not ini_path.exists():
        return values

    try:
        param = poms.Param()
        poms.ParamXMLFile().load(str(ini_path), param)
        ini_keys = {
            k.decode() if isinstance(k, (bytes, bytearray)) else str(k): k
            for k in param.keys()
        }
        for short_key in short_keys:
            full_key = f"{tool}:1:{short_key}"
            if full_key not in ini_keys:
                continue
            value = param.getValue(ini_keys[full_key])
            if isinstance(value, bytes):
                value = value.decode()
            values[short_key] = value
    except Exception:
        return {}

    return values


def _sync_osw_ini_from_current_state() -> tuple[bool, Path, Path | None, int, dict[str, object]]:
    """
    Rebuild the workspace OpenSwathWorkflow INI from the installed binary and the
    current OpenSwathWorkflow widget state, then mirror it into
    openswath-workflow/ini/.
    """
    merged_tool_params = _collect_saved_and_current_osw_values()
    dest = INI_DIR / "OpenSwathWorkflow.ini"
    mirrored_dest = workspace_dir / "openswath-workflow" / "ini" / "OpenSwathWorkflow.ini"

    synced = pm.refresh_ini_from_binary("OpenSwathWorkflow", merged_tool_params)
    if not synced and dest.exists():
        try:
            param = poms.Param()
            poms.ParamXMLFile().load(str(dest), param)
            ini_keys = {
                k.decode() if isinstance(k, (bytes, bytearray)) else str(k)
                for k in param.keys()
            }
            for short_key, value in merged_tool_params.items():
                ini_key = f"OpenSwathWorkflow:1:{short_key}"
                if ini_key not in ini_keys:
                    continue
                ini_value = param.getValue(ini_key)
                param.setValue(ini_key, pm._coerce_topp_value(ini_value, value))
            poms.ParamXMLFile().store(str(dest), param)
            synced = True
        except Exception:
            synced = False

    if not synced or not dest.exists():
        return False, dest, None, len(merged_tool_params), {}

    try:
        mirrored_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dest, mirrored_dest)
    except Exception:
        mirrored_dest = None

    verified_values = _read_ini_short_values(
        "OpenSwathWorkflow",
        dest,
        [
            "readOptions",
            "tempDirectory",
            "out_chrom",
            "out_mobilogram",
            "Debugging:irt_mzml",
            "Debugging:irt_trafo",
            "Calibration:MassIMCorrection:debug_im_file",
            "Calibration:MassIMCorrection:debug_mz_file",
        ],
    )
    return True, dest, mirrored_dest, len(merged_tool_params), verified_values


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
    return saved_params_cache


def _saved_flat_param(name: str, default=None):
    return _saved_params().get(name, default)


def _saved_tool_param(tool: str, name: str, default=None):
    return _saved_params().get(tool, {}).get(name, default)


def _write_json_if_changed(path: Path, data: dict) -> None:
    existing = None
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = None

    if existing == data:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


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


def _seed_value_state(widget_key: str, saved_value) -> None:
    """Restore a generic widget value when the current widget state is missing."""
    if widget_key not in st.session_state:
        st.session_state[widget_key] = saved_value


def _rehydrate_osw_special_widgets() -> None:
    """
    Reset custom OpenSwathWorkflow helper widgets when the saved tool payload
    changes so they re-seed from workspace params instead of stale session state.
    """
    signature_key = f"{workspace_dir.resolve()}::OpenSwathWorkflow::special_widget_state"
    saved_tool_params = pm.get_parameters_from_json().get("OpenSwathWorkflow", {})
    current_signature = json.dumps(saved_tool_params, sort_keys=True, default=str)

    if st.session_state.get(signature_key) == current_signature:
        return

    for widget_key in [
        "osw_save_xics",
        "osw_save_xims",
        "osw_save_calibration_debug",
        "osw_read_options",
        "osw_xic_output_display",
        "osw_xim_output_display",
        "osw_debug_im_display",
        "osw_debug_mz_display",
        "osw_debug_trafo_display",
        "osw_debug_irt_mzml_display",
        "osw_cache_dir_display",
        "osw_tr_sel",
    ]:
        st.session_state.pop(widget_key, None)

    st.session_state[signature_key] = current_signature


def _normalize_context_selection(saved_values) -> list[str]:
    """Normalize saved infer contexts while preserving the supported order."""
    if isinstance(saved_values, str):
        saved_values = [saved_values]
    if not isinstance(saved_values, list):
        saved_values = []
    normalized = [value for value in PY_INFER_CONTEXTS if value in saved_values]
    return normalized or ["global"]


def _pyprophet_config_path(cmd_key: str) -> Path:
    return workspace_dir / "tools-configs" / "pyprophet" / f"{cmd_key}.json"


def _pyprophet_widget_keys(
    cmd_key: str,
    section_name: str,
    opt_name: str,
    nested_name: str | None = None,
) -> tuple[str, str]:
    short_key = f"pyprophet:{cmd_key}:{section_name}:{opt_name}"
    if nested_name is not None:
        short_key = f"{short_key}:{nested_name}"
    return short_key, f"{PPFX}{short_key}"


def _pyprophet_saved_or_default_value(
    saved_cfg: dict,
    opt_name: str,
    opt_def: dict,
    nested_name: str | None = None,
):
    if nested_name is not None:
        nested_saved = saved_cfg.get(opt_name)
        if isinstance(nested_saved, dict) and nested_name in nested_saved:
            return nested_saved[nested_name]
        return opt_def.get("value", opt_def.get("default"))
    if opt_name in saved_cfg:
        return saved_cfg.get(opt_name)
    return opt_def.get("value", opt_def.get("default"))


def _prepare_pyprophet_config(
    cmd_key: str,
    cfg: dict,
) -> tuple[dict, str, str | None, str | None]:
    cfg_to_render = copy.deepcopy(cfg)
    cmd_name = cfg.get("meta", {}).get("command", "")
    low_cmd_name = cmd_name.lower()
    low_cmd_key = cmd_key.lower()

    if "export" in low_cmd_name or "export" in low_cmd_key:
        sections = cfg_to_render.get("sections", {})
        if "alignment" in sections:
            sections.pop("alignment", None)
            cfg_to_render["sections"] = sections

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

    derived_in = None
    derived_out = None
    if "score" in low_cmd_name or "score" in low_cmd_key:
        derived_in = "openswath_results.osw"
    if "infer" in low_cmd_name or "infer" in low_cmd_key:
        derived_in = "openswath_results.osw"
    if "export" in low_cmd_name or "export" in low_cmd_key:
        derived_in = "openswath_results.osw"
        derived_out = "openswath_results.tsv"

    if (derived_in is not None) or (derived_out is not None):
        try:
            io_opts = cfg_to_render.get("sections", {}).get("io", {}).get("options", {})
            io_opts.pop("in", None)
            io_opts.pop("out", None)
            if "io" in cfg_to_render.get("sections", {}):
                cfg_to_render["sections"]["io"]["options"] = io_opts
        except Exception:
            pass

    return cfg_to_render, cmd_name, derived_in, derived_out


def _seed_pyprophet_structured_state(
    cmd_key: str,
    cfg: dict,
    saved_cfg: dict,
    derived_in: str | None = None,
    derived_out: str | None = None,
) -> None:
    for section_name, section_def in cfg.get("sections", {}).items():
        for opt_name, opt_def in section_def.get("options", {}).items():
            if isinstance(opt_def, dict) and "options" in opt_def:
                for nested_name, nested_def in opt_def.get("options", {}).items():
                    _, full_key = _pyprophet_widget_keys(
                        cmd_key,
                        section_name,
                        opt_name,
                        nested_name,
                    )
                    _seed_value_state(
                        full_key,
                        _pyprophet_saved_or_default_value(
                            saved_cfg, opt_name, nested_def, nested_name
                        ),
                    )
                continue

            _, full_key = _pyprophet_widget_keys(cmd_key, section_name, opt_name)
            _seed_value_state(
                full_key,
                _pyprophet_saved_or_default_value(saved_cfg, opt_name, opt_def),
            )

    if derived_in is not None:
        st.session_state[f"{PPFX}pyprophet:{cmd_key}:io:in"] = derived_in
    if derived_out is not None:
        st.session_state[f"{PPFX}pyprophet:{cmd_key}:io:out"] = derived_out


def _coerce_pyprophet_state_value(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def _collect_pyprophet_command_payload(
    cmd_key: str,
    cfg: dict,
    saved_cfg: dict,
    derived_in: str | None = None,
    derived_out: str | None = None,
    infer_context_values: list[str] | None = None,
) -> dict:
    payload: dict[str, object] = {}
    for section_name, section_def in cfg.get("sections", {}).items():
        for opt_name, opt_def in section_def.get("options", {}).items():
            if section_name == "io" and opt_name == "in" and derived_in is not None:
                payload[opt_name] = derived_in
                continue
            if section_name == "io" and opt_name == "out" and derived_out is not None:
                payload[opt_name] = derived_out
                continue

            if isinstance(opt_def, dict) and "options" in opt_def:
                nested_payload = {}
                for nested_name, nested_def in opt_def.get("options", {}).items():
                    _, full_key = _pyprophet_widget_keys(
                        cmd_key,
                        section_name,
                        opt_name,
                        nested_name,
                    )
                    value = st.session_state.get(
                        full_key,
                        _pyprophet_saved_or_default_value(
                            saved_cfg, opt_name, nested_def, nested_name
                        ),
                    )
                    nested_payload[nested_name] = _coerce_pyprophet_state_value(value)
                payload[opt_name] = nested_payload
                continue

            _, full_key = _pyprophet_widget_keys(cmd_key, section_name, opt_name)
            value = st.session_state.get(
                full_key,
                _pyprophet_saved_or_default_value(saved_cfg, opt_name, opt_def),
            )
            payload[opt_name] = _coerce_pyprophet_state_value(value)

    if infer_context_values is not None:
        payload.pop("context", None)
        payload["contexts"] = infer_context_values or ["global"]

    return payload


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
                    ("$K", [8.014199], "Acetyl-K"),
                    ("$R", [10.008269], "Acetyl-R"),
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
                autosave=False,
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
            autosave=False,
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
            autosave=False,
        )

# -----------------------------------------------------------------------------
# SECTION 2 — OpenSwathWorkflow (INI-based)

st.markdown("---")
st.subheader("OpenSwathWorkflow")
st.markdown(
    "Main DIA extraction engine. mzML files are taken from the File Upload page; "
    "the transition library is derived from the decoy-generation step above."
)

osw_ini, osw_ini_source = _ensure_osw_ini_for_workspace()
if not osw_ini:
    st.error(
        "Could not create `OpenSwathWorkflow.ini` from the installed "
        "`OpenSwathWorkflow` binary, and no fallback asset descriptor is available."
    )
else:
    if osw_ini_source == "binary":
        st.caption(
            "Using descriptor from installed `OpenSwathWorkflow` binary via "
            "`OpenSwathWorkflow -write_ini`."
        )
    elif osw_ini_source == "workspace":
        st.caption("Using existing workspace descriptor: `OpenSwathWorkflow.ini`")
    elif isinstance(osw_ini_source, str) and osw_ini_source.startswith("asset:"):
        st.caption(
            f"Using fallback asset descriptor: `{osw_ini_source.split(':', 1)[1]}`"
        )
    else:
        st.caption("Using workspace descriptor: `OpenSwathWorkflow.ini`")

    _rehydrate_osw_special_widgets()

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
    debug_im_key = (
        f"{TPFX}OpenSwathWorkflow:1:Calibration:MassIMCorrection:debug_im_file"
    )
    debug_mz_key = (
        f"{TPFX}OpenSwathWorkflow:1:Calibration:MassIMCorrection:debug_mz_file"
    )

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
    osw_param_view_key = "osw_param_view_mode"
    osw_param_view_options = ["Grouped", "Focused"]
    _seed_choice_state(osw_param_view_key, osw_param_view_options, "Grouped")
    osw_param_view_mode = st.radio(
        "Parameter view",
        options=osw_param_view_options,
        key=osw_param_view_key,
        horizontal=True,
        help=(
            "Grouped shows all top-level parameter sections with General open by default. "
            "Focused renders one top-level section at a time and is lighter on memory."
        ),
    )
    osw_draft_values = _osw_draft_values()
    ui.input_TOPP(
        "OpenSwathWorkflow",
        num_cols=3,
        display_tool_name=False,
        display_subsections=True,
        custom_defaults=osw_draft_values,
        custom_defaults_override_saved=bool(osw_draft_values),
        lazy_grouped_top_level_sections=(osw_param_view_mode == "Grouped"),
        lazy_top_level_sections=(osw_param_view_mode == "Focused"),
        lazy_top_level_label="OpenSwathWorkflow parameter group",
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
        autosave=False,
    )
    _store_current_osw_draft()

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

matrix_cfg_path = (
    workspace_dir
    / "tools-configs"
    / "pyprophet"
    / "pyprophet_export_matrix_config.json"
)
matrix_saved = _load_json_asset(matrix_cfg_path) if matrix_cfg_path.exists() else {}
pyprophet_command_payloads: dict[str, dict] = {}
pyprophet_matrix_payload: dict | None = None

if cfg_files:
    st.caption(
        "PyProphet structured configuration templates detected. "
        "Grouped renders each subcommand as its own panel; Focused renders one panel at a time."
    )

    pyprophet_order = {
        "pyprophet_score_config": 0,
        "pyprophet_infer_peptide_config": 1,
        "pyprophet_infer_protein_config": 2,
        "pyprophet_export_tsv_config": 3,
    }
    cfg_entries: list[tuple[str, str, dict]] = []
    saved_cmd_cfgs: dict[str, dict] = {}
    for cfg_path in cfg_files:
        cfg = _load_json_asset(cfg_path)
        if not cfg:
            continue
        cmd_key = cfg_path.stem
        title = cfg.get("meta", {}).get("command", cmd_key)
        cfg_entries.append((cmd_key, title, cfg))
        saved_cfg_path = _pyprophet_config_path(cmd_key)
        saved_cmd_cfgs[cmd_key] = (
            _load_json_asset(saved_cfg_path) if saved_cfg_path.exists() else {}
        )

    cfg_entries.sort(key=lambda entry: pyprophet_order.get(entry[0], 99))

    pyprophet_view_key = "pyprophet_param_view_mode"
    pyprophet_view_options = ["Grouped", "Focused"]
    _seed_choice_state(pyprophet_view_key, pyprophet_view_options, "Grouped")
    pyprophet_view_mode = st.radio(
        "PyProphet parameter view",
        options=pyprophet_view_options,
        key=pyprophet_view_key,
        horizontal=True,
        help=(
            "Grouped shows each PyProphet subcommand as its own panel. "
            "Focused renders one panel at a time and is lighter over remote connections."
        ),
    )

    def _current_infer_contexts(cmd_key: str, saved_cmd_cfg: dict) -> list[str]:
        infer_type = "peptide" if "peptide" in cmd_key else "protein"
        saved_contexts = _normalize_context_selection(
            saved_cmd_cfg.get("contexts", saved_cmd_cfg.get("context", "global"))
        )
        selected = []
        for context in PY_INFER_CONTEXTS:
            widget_key = f"pyprophet_{infer_type}_context_{context.replace('-', '_')}"
            if st.session_state.get(widget_key, context in saved_contexts):
                selected.append(context)
        return selected or ["global"]

    def _render_pyprophet_command_panel(
        cmd_key: str,
        title: str,
        cfg: dict,
        saved_cmd_cfg: dict,
        show_title: bool = True,
    ) -> None:
        cfg_to_render, _, derived_in, derived_out = _prepare_pyprophet_config(
            cmd_key, cfg
        )
        _seed_pyprophet_structured_state(
            cmd_key,
            cfg,
            saved_cmd_cfg,
            derived_in=derived_in,
            derived_out=derived_out,
        )

        if show_title:
            st.markdown(f"**⚙️ {title}**")
        ui.render_structured_config(
            cfg_to_render,
            key_prefix=f"pyprophet:{cmd_key}",
            default_open_sections=["io"],
            autosave=False,
        )

        if derived_in is not None:
            st.text_input(
                "Input file",
                value=derived_in,
                disabled=True,
                key=f"py_in_display_{cmd_key}",
            )
        if derived_out is not None:
            st.text_input(
                "Output file",
                value=derived_out,
                disabled=True,
                key=f"py_out_display_{cmd_key}",
            )

        infer_context_values = None
        if cmd_key in {
            "pyprophet_infer_peptide_config",
            "pyprophet_infer_protein_config",
        }:
            infer_type = "peptide" if "peptide" in cmd_key else "protein"
            saved_contexts = _normalize_context_selection(
                saved_cmd_cfg.get("contexts", saved_cmd_cfg.get("context", "global"))
            )
            st.caption("Run this inference separately for each selected context.")
            ctx_cols = st.columns(len(PY_INFER_CONTEXTS))
            infer_context_values = []
            for col, context in zip(ctx_cols, PY_INFER_CONTEXTS):
                widget_key = f"pyprophet_{infer_type}_context_{context.replace('-', '_')}"
                _seed_checkbox_state(widget_key, context in saved_contexts)
                if col.checkbox(context, key=widget_key):
                    infer_context_values.append(context)
            if not infer_context_values:
                st.warning(
                    "At least one inference context should be selected. "
                    "If left empty, the workflow will fall back to `global`."
                )

        pyprophet_command_payloads[cmd_key] = _collect_pyprophet_command_payload(
            cmd_key,
            cfg,
            saved_cmd_cfg,
            derived_in=derived_in,
            derived_out=derived_out,
            infer_context_values=infer_context_values,
        )

    def _seed_pyprophet_matrix_state(saved_cfg: dict) -> None:
        _seed_multiselect_state(
            "pyprophet_matrix_levels",
            PY_MATRIX_LEVELS,
            saved_cfg.get("levels", PY_MATRIX_LEVELS),
        )
        _seed_checkbox_state(
            "pyprophet_matrix_transition_quantification",
            saved_cfg.get("transition_quantification", True),
        )
        _seed_checkbox_state(
            "pyprophet_matrix_use_alignment",
            saved_cfg.get("use_alignment", True),
        )
        _seed_checkbox_state(
            "pyprophet_matrix_consistent_top",
            saved_cfg.get("consistent_top", True),
        )
        _seed_value_state(
            "pyprophet_matrix_max_transition_pep",
            float(saved_cfg.get("max_transition_pep", 0.7)),
        )
        _seed_choice_state(
            "pyprophet_matrix_ipf",
            ["peptidoform", "augmented", "disable"],
            saved_cfg.get("ipf", "peptidoform"),
        )
        _seed_value_state(
            "pyprophet_matrix_ipf_max_peptidoform_pep",
            float(saved_cfg.get("ipf_max_peptidoform_pep", 0.4)),
        )
        _seed_value_state(
            "pyprophet_matrix_max_rs_peakgroup_qvalue",
            float(saved_cfg.get("max_rs_peakgroup_qvalue", 0.05)),
        )
        _seed_value_state(
            "pyprophet_matrix_max_global_peptide_qvalue",
            float(saved_cfg.get("max_global_peptide_qvalue", 0.01)),
        )
        _seed_value_state(
            "pyprophet_matrix_max_global_protein_qvalue",
            float(saved_cfg.get("max_global_protein_qvalue", 0.01)),
        )
        _seed_value_state(
            "pyprophet_matrix_max_alignment_pep",
            float(saved_cfg.get("max_alignment_pep", 0.7)),
        )
        _seed_value_state(
            "pyprophet_matrix_top_n",
            int(saved_cfg.get("top_n", 3)),
        )
        _seed_choice_state(
            "pyprophet_matrix_normalization",
            ["none", "median", "medianmedian", "quantile"],
            saved_cfg.get("normalization", "none"),
        )

    def _render_pyprophet_matrix_panel(
        saved_cfg: dict,
        show_title: bool = True,
    ) -> None:
        _seed_pyprophet_matrix_state(saved_cfg)

        if show_title:
            st.markdown("**⚙️ pyprophet export matrix**")
        st.caption(
            "Export quantification matrices at selected precursor, peptide, and protein levels. "
            "Each selected level writes a fixed output file into the workflow results directory."
        )

        matrix_levels = st.multiselect(
            "Matrix export levels",
            options=PY_MATRIX_LEVELS,
            key="pyprophet_matrix_levels",
            help="Choose which matrix quantification levels to export after scoring and inference.",
        )

        for level in PY_MATRIX_LEVELS:
            if level in matrix_levels:
                st.caption(f"`{level}` -> `{PY_MATRIX_OUTS[level]}`")

        matrix_col1, matrix_col2, matrix_col3 = st.columns(3)
        matrix_col1.caption("Outputs are written as fixed TSV files.")
        matrix_col2.checkbox(
            "Transition quantification",
            value=bool(st.session_state["pyprophet_matrix_transition_quantification"]),
            key="pyprophet_matrix_transition_quantification",
            help="Report aggregated transition-level quantification.",
        )
        matrix_col3.checkbox(
            "Use alignment",
            value=bool(st.session_state["pyprophet_matrix_use_alignment"]),
            key="pyprophet_matrix_use_alignment",
            help="Recover peaks with good alignment scores if alignment data is present.",
        )

        matrix_col1, matrix_col2, matrix_col3 = st.columns(3)
        matrix_col1.number_input(
            "Max transition PEP",
            value=float(st.session_state["pyprophet_matrix_max_transition_pep"]),
            step=0.05,
            key="pyprophet_matrix_max_transition_pep",
        )
        matrix_col2.selectbox(
            "IPF mode",
            options=["peptidoform", "augmented", "disable"],
            index=["peptidoform", "augmented", "disable"].index(
                st.session_state["pyprophet_matrix_ipf"]
            ),
            key="pyprophet_matrix_ipf",
        )
        matrix_col3.number_input(
            "IPF max peptidoform PEP",
            value=float(st.session_state["pyprophet_matrix_ipf_max_peptidoform_pep"]),
            step=0.05,
            key="pyprophet_matrix_ipf_max_peptidoform_pep",
        )

        matrix_col1, matrix_col2, matrix_col3 = st.columns(3)
        matrix_col1.number_input(
            "Max run-specific peakgroup q-value",
            value=float(st.session_state["pyprophet_matrix_max_rs_peakgroup_qvalue"]),
            step=0.01,
            key="pyprophet_matrix_max_rs_peakgroup_qvalue",
        )
        matrix_col2.number_input(
            "Max global peptide q-value",
            value=float(st.session_state["pyprophet_matrix_max_global_peptide_qvalue"]),
            step=0.01,
            key="pyprophet_matrix_max_global_peptide_qvalue",
        )
        matrix_col3.number_input(
            "Max global protein q-value",
            value=float(st.session_state["pyprophet_matrix_max_global_protein_qvalue"]),
            step=0.01,
            key="pyprophet_matrix_max_global_protein_qvalue",
        )

        matrix_col1, matrix_col2, matrix_col3 = st.columns(3)
        matrix_col1.number_input(
            "Max alignment PEP",
            value=float(st.session_state["pyprophet_matrix_max_alignment_pep"]),
            step=0.05,
            key="pyprophet_matrix_max_alignment_pep",
        )
        matrix_col2.number_input(
            "Top N features",
            min_value=1,
            value=int(st.session_state["pyprophet_matrix_top_n"]),
            step=1,
            key="pyprophet_matrix_top_n",
        )
        matrix_col3.checkbox(
            "Consistent top features",
            value=bool(st.session_state["pyprophet_matrix_consistent_top"]),
            key="pyprophet_matrix_consistent_top",
            help="Use the same top features across all runs.",
        )

        st.selectbox(
            "Normalization",
            options=["none", "median", "medianmedian", "quantile"],
            index=["none", "median", "medianmedian", "quantile"].index(
                st.session_state["pyprophet_matrix_normalization"]
            ),
            key="pyprophet_matrix_normalization",
        )

    def _collect_pyprophet_matrix_payload(saved_cfg: dict) -> dict:
        levels = st.session_state.get(
            "pyprophet_matrix_levels",
            saved_cfg.get("levels", PY_MATRIX_LEVELS),
        )
        return {
            "levels": [level for level in levels if level in PY_MATRIX_LEVELS],
            "csv": False,
            "transition_quantification": st.session_state.get(
                "pyprophet_matrix_transition_quantification",
                saved_cfg.get("transition_quantification", True),
            ),
            "max_transition_pep": st.session_state.get(
                "pyprophet_matrix_max_transition_pep",
                saved_cfg.get("max_transition_pep", 0.7),
            ),
            "ipf": st.session_state.get(
                "pyprophet_matrix_ipf",
                saved_cfg.get("ipf", "peptidoform"),
            ),
            "ipf_max_peptidoform_pep": st.session_state.get(
                "pyprophet_matrix_ipf_max_peptidoform_pep",
                saved_cfg.get("ipf_max_peptidoform_pep", 0.4),
            ),
            "max_rs_peakgroup_qvalue": st.session_state.get(
                "pyprophet_matrix_max_rs_peakgroup_qvalue",
                saved_cfg.get("max_rs_peakgroup_qvalue", 0.05),
            ),
            "max_global_peptide_qvalue": st.session_state.get(
                "pyprophet_matrix_max_global_peptide_qvalue",
                saved_cfg.get("max_global_peptide_qvalue", 0.01),
            ),
            "max_global_protein_qvalue": st.session_state.get(
                "pyprophet_matrix_max_global_protein_qvalue",
                saved_cfg.get("max_global_protein_qvalue", 0.01),
            ),
            "use_alignment": st.session_state.get(
                "pyprophet_matrix_use_alignment",
                saved_cfg.get("use_alignment", True),
            ),
            "max_alignment_pep": st.session_state.get(
                "pyprophet_matrix_max_alignment_pep",
                saved_cfg.get("max_alignment_pep", 0.7),
            ),
            "top_n": st.session_state.get(
                "pyprophet_matrix_top_n",
                saved_cfg.get("top_n", 3),
            ),
            "consistent_top": st.session_state.get(
                "pyprophet_matrix_consistent_top",
                saved_cfg.get("consistent_top", True),
            ),
            "normalization": st.session_state.get(
                "pyprophet_matrix_normalization",
                saved_cfg.get("normalization", "none"),
            ),
        }

    if pyprophet_view_mode == "Focused":
        panel_options = [cmd_key for cmd_key, _, _ in cfg_entries] + ["__matrix__"]
        panel_labels = {
            cmd_key: title for cmd_key, title, _ in cfg_entries
        } | {"__matrix__": "pyprophet export matrix"}
        _seed_choice_state(
            "pyprophet_panel_selection",
            panel_options,
            panel_options[0] if panel_options else "__matrix__",
        )
        selected_pyprophet_panel = st.selectbox(
            "PyProphet panel",
            options=panel_options,
            key="pyprophet_panel_selection",
            format_func=lambda option: panel_labels.get(option, option),
            help="Only the selected PyProphet panel is rendered on this page load.",
        )

        if selected_pyprophet_panel == "__matrix__":
            _render_pyprophet_matrix_panel(matrix_saved, show_title=True)
        else:
            selected_entry = next(
                (
                    entry
                    for entry in cfg_entries
                    if entry[0] == selected_pyprophet_panel
                ),
                None,
            )
            if selected_entry is not None:
                cmd_key, title, cfg = selected_entry
                _render_pyprophet_command_panel(
                    cmd_key,
                    title,
                    cfg,
                    saved_cmd_cfgs.get(cmd_key, {}),
                    show_title=True,
                )
    else:
        for index, (cmd_key, title, cfg) in enumerate(cfg_entries):
            with st.expander(f"⚙️ {title}", expanded=index == 0):
                _render_pyprophet_command_panel(
                    cmd_key,
                    title,
                    cfg,
                    saved_cmd_cfgs.get(cmd_key, {}),
                    show_title=False,
                )
        with st.expander("⚙️ pyprophet export matrix", expanded=False):
            _render_pyprophet_matrix_panel(matrix_saved, show_title=False)

    for cmd_key, _, cfg in cfg_entries:
        if cmd_key in pyprophet_command_payloads:
            continue
        _, _, derived_in, derived_out = _prepare_pyprophet_config(cmd_key, cfg)
        infer_context_values = None
        if cmd_key in {
            "pyprophet_infer_peptide_config",
            "pyprophet_infer_protein_config",
        }:
            infer_context_values = _current_infer_contexts(
                cmd_key,
                saved_cmd_cfgs.get(cmd_key, {}),
            )
        pyprophet_command_payloads[cmd_key] = _collect_pyprophet_command_payload(
            cmd_key,
            cfg,
            saved_cmd_cfgs.get(cmd_key, {}),
            derived_in=derived_in,
            derived_out=derived_out,
            infer_context_values=infer_context_values,
        )

    pyprophet_matrix_payload = _collect_pyprophet_matrix_payload(matrix_saved)
else:
    st.info("No structured PyProphet schemas found — falling back to built-in UI.")
    try:
        py.ui()
    except Exception:
        st.warning("PyProphet UI unavailable.")

# -----------------------------------------------------------------------------
# SAVE button

st.markdown("---")
st.caption(
    "Changes stay live in this Streamlit session. Click **Save all parameters to workspace** before leaving this page or running the workflow."
)
save_col, _ = st.columns([1, 3])
with save_col:
    if st.button("💾 Save all parameters to workspace", type="primary"):
        pm.save_parameters()
        (
            osw_ini_synced,
            osw_ini_path,
            osw_mirrored_ini_path,
            osw_synced_count,
            osw_verified_values,
        ) = _sync_osw_ini_from_current_state()
        if not osw_ini_synced:
            st.warning(
                "Could not update `OpenSwathWorkflow.ini` from the current OpenSwathWorkflow UI state."
            )

        # Also persist the non-TOPP workflow-level keys that the workflow
        # execution page needs to read (session_state only is not enough
        # because the WorkflowManager runs in a subprocess with no session_state).
        _extra = pm.get_parameters_from_json()
        _osw_values = _collect_saved_and_current_osw_values()
        if _osw_values:
            _extra.setdefault("OpenSwathWorkflow", {}).update(_osw_values)
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

        try:
            pyprophet_dir = workspace_dir / "tools-configs" / "pyprophet"
            for cmd_key, payload in pyprophet_command_payloads.items():
                _write_json_if_changed(pyprophet_dir / f"{cmd_key}.json", payload)
            if pyprophet_matrix_payload is not None:
                _write_json_if_changed(matrix_cfg_path, pyprophet_matrix_payload)
            if pyprophet_command_payloads:
                st.info(
                    "Saved PyProphet subcommand configs to workspace: "
                    f"`{pyprophet_dir}`"
                )
            if pyprophet_matrix_payload is not None:
                matrix_levels = pyprophet_matrix_payload.get("levels", [])
                st.info(
                    "Saved PyProphet matrix export config: "
                    f"`{matrix_cfg_path}` "
                    f"(levels: {', '.join(matrix_levels) if matrix_levels else 'none'})"
                )
        except Exception as _e:
            st.warning(f"Could not save PyProphet config: {_e}")
        st.success(f"Saved workspace parameters to `{pm.params_file}`.")
        if osw_ini_synced:
            st.success(
                "Updated the shared `OpenSwathWorkflow.ini` used for saved workspace settings "
                f"({osw_synced_count} values considered): `{osw_ini_path}`"
            )
            if osw_mirrored_ini_path is not None:
                st.success(
                    "Updated the workflow run copy of `OpenSwathWorkflow.ini`: "
                    f"`{osw_mirrored_ini_path}`"
                )
            if osw_verified_values:
                verified_pairs = ", ".join(
                    f"`{key}={value}`"
                    for key, value in osw_verified_values.items()
                    if value not in ("", None)
                )
                if verified_pairs:
                    st.info(
                        "Verified saved values in `OpenSwathWorkflow.ini`: "
                        f"{verified_pairs}"
                    )
