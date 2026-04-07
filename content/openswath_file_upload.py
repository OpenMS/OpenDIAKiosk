import shutil
from pathlib import Path

import pandas as pd
import streamlit as st

from src.common.common import (
    TK_AVAILABLE,
    page_setup,
    save_params,
    show_table,
    tk_directory_dialog,
    v_space,
)
from src import fileupload


params = page_setup()

workspace_dir = Path(st.session_state.workspace)
mzML_dir = workspace_dir / "mzML-files"
fasta_dir = workspace_dir / "input-files" / "fasta"
lib_dir = workspace_dir / "input-files" / "libraries"
xic_dir = workspace_dir / "xic-files"

imported_results_root = workspace_dir / "openswath-workflow" / "results" / "imported"
result_target_dirs = {
    "osw": imported_results_root / "osw",
    "long_tsv": imported_results_root / "tsv" / "long",
    "precursor_matrix": imported_results_root / "tsv" / "precursor",
    "peptide_matrix": imported_results_root / "tsv" / "peptide",
    "protein_matrix": imported_results_root / "tsv" / "protein",
}


def _workspace_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(workspace_dir.resolve()))
    except ValueError:
        return str(path)


def _save_uploaded_file(uploaded_file, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / getattr(uploaded_file, "name", "uploaded.bin")
    with open(target_path, "wb") as fh:
        fh.write(uploaded_file.getbuffer())
    return target_path


def _save_optional_upload(uploaded_file, target_dir: Path, label: str) -> bool:
    if uploaded_file is None:
        return False
    target_path = _save_uploaded_file(uploaded_file, target_dir)
    st.success(f"Saved {label}: {target_path.name}")
    return True


def _list_workspace_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted([p for p in path.iterdir() if p.is_file()], key=lambda item: item.name.lower())


def _clear_xic_viewer_state() -> None:
    for key in [
        "xic_tmp_paths",
        "file_analytes",
        "shared_analytes",
        "files_loaded",
        "xic_run_metadata",
        "boundary_run_mapping",
    ]:
        if key in st.session_state:
            del st.session_state[key]


def _classify_imported_result(path: Path) -> str:
    parent_name = path.parent.name.lower()
    if path.suffix.lower() == ".osw":
        return "OSW"
    if parent_name == "long":
        return "Long TSV"
    if parent_name == "precursor":
        return "Precursor matrix"
    if parent_name == "peptide":
        return "Peptide matrix"
    if parent_name == "protein":
        return "Protein matrix"
    return path.suffix.lower().lstrip(".") or "file"


def _list_imported_results() -> list[Path]:
    if not imported_results_root.exists():
        return []
    return sorted(
        [p for p in imported_results_root.rglob("*") if p.is_file()],
        key=lambda item: _workspace_rel(item).lower(),
    )


def _path_matches_suffix(path: Path, suffixes: tuple[str, ...]) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix.lower()) for suffix in suffixes)


def _list_local_files(local_dir: Path, suffixes: tuple[str, ...]) -> list[Path]:
    if not local_dir.is_dir():
        return []
    return sorted(
        [
            path
            for path in local_dir.iterdir()
            if path.is_file() and _path_matches_suffix(path, suffixes)
        ],
        key=lambda item: item.name.lower(),
    )


def _looks_like_matrix_tsv(path: Path, level: str) -> bool:
    name = path.name.lower()
    return (
        name.endswith(f".{level}.tsv")
        or name.endswith(f"_{level}.tsv")
        or name.endswith(f"-{level}.tsv")
        or f".{level}." in name
        or f"_{level}_" in name
        or f"-{level}-" in name
    )


def _looks_like_any_matrix_tsv(path: Path) -> bool:
    return any(
        _looks_like_matrix_tsv(path, level)
        for level in ("precursor", "peptide", "protein")
    )


def _looks_like_long_results_tsv(path: Path) -> bool:
    name = path.name.lower()
    if _looks_like_any_matrix_tsv(path):
        return False
    return (
        name == "openswath_results.tsv"
        or name == "results.tsv"
        or name.endswith("_results.tsv")
        or name.endswith(".results.tsv")
        or "long" in name
    )


def _local_file_multiselect(
    label: str,
    paths: list[Path],
    default_paths: list[Path],
    key: str,
    help_text: str | None = None,
) -> list[Path]:
    path_map = {path.name: path for path in paths}
    selected = st.multiselect(
        label,
        options=list(path_map.keys()),
        default=[path.name for path in default_paths if path.name in path_map],
        key=key,
        help=help_text,
    )
    return [path_map[name] for name in selected]


def _copy_or_link_local_files(
    paths: list[Path],
    target_dir: Path,
    label: str,
    make_copy: bool,
) -> int:
    if not paths:
        return 0

    target_dir.mkdir(parents=True, exist_ok=True)
    added = 0
    skipped = 0

    for source_path in paths:
        target_path = target_dir / source_path.name
        source_resolved = source_path.resolve()

        if target_path.exists() or target_path.is_symlink():
            if target_path.is_dir():
                st.warning(f"Skipped {source_path.name}: target path is a directory.")
                skipped += 1
                continue
            try:
                if target_path.resolve() == source_resolved:
                    added += 1
                    continue
            except FileNotFoundError:
                pass
            target_path.unlink()

        if make_copy:
            shutil.copy2(source_path, target_path)
        else:
            try:
                target_path.symlink_to(source_resolved)
            except OSError:
                shutil.copy2(source_path, target_path)
                st.warning(f"Could not link {source_path.name}; copied it instead.")
        added += 1

    if added:
        action = "Copied" if make_copy else "Added"
        st.success(f"{action} {added} {label} file(s) to the workspace.")
    if skipped:
        st.info(f"Skipped {skipped} {label} file(s).")
    return added


if mzML_dir.exists() and not any(mzML_dir.iterdir()):
    fileupload.load_example_mzML_files()


st.title("File Upload")

tabs = ["File Upload"]
if st.session_state.location == "local":
    tabs.append("Files from local folder")

tabs = st.tabs(tabs)

with tabs[0]:
    with st.form("mzML-upload", clear_on_submit=True):
        st.markdown("##### Primary Inputs")
        files = st.file_uploader(
            "mzML files",
            type=["mzML", "mzML.gz"],
            accept_multiple_files=(st.session_state.location == "local"),
            help="Upload your mzML files here. You can also upload files later or use the local folder option.",
        )
        fasta_file = st.file_uploader(
            "Optional FASTA file",
            type=["fasta", "fa", "faa"],
            accept_multiple_files=False,
            help="Upload an optional FASTA file for sequence lookup.",
            key="fasta_upload",
        )
        lib_file = st.file_uploader(
            "Optional spectral library / transition list",
            type=["tsv", "traML", "pqp"],
            accept_multiple_files=False,
            help="Upload an optional transition list or spectral library (TSV/TraML/PQP).",
            key="lib_upload",
        )

        st.markdown("##### Existing OpenSwath Results")
        osw_result_file = st.file_uploader(
            "Optional OpenSwath OSW result",
            type=["osw", "sqlite", "db"],
            accept_multiple_files=False,
            help="Upload an existing OpenSwath OSW SQLite result file.",
            key="osw_result_upload",
        )
        long_tsv_file = st.file_uploader(
            "Optional long results TSV",
            type=["tsv"],
            accept_multiple_files=False,
            help="Upload an existing long-format OpenSwath/PyProphet TSV result file.",
            key="long_tsv_upload",
        )
        matrix_col1, matrix_col2, matrix_col3 = st.columns(3)
        with matrix_col1:
            precursor_matrix_file = st.file_uploader(
                "Precursor matrix TSV",
                type=["tsv"],
                accept_multiple_files=False,
                key="precursor_matrix_upload",
            )
        with matrix_col2:
            peptide_matrix_file = st.file_uploader(
                "Peptide matrix TSV",
                type=["tsv"],
                accept_multiple_files=False,
                key="peptide_matrix_upload",
            )
        with matrix_col3:
            protein_matrix_file = st.file_uploader(
                "Protein matrix TSV",
                type=["tsv"],
                accept_multiple_files=False,
                key="protein_matrix_upload",
            )
        xic_files = st.file_uploader(
            "Optional XIC files",
            type=["xic", "parquet"],
            accept_multiple_files=True,
            help="Upload one or more OpenMS XIC parquet files for the chromatogram viewer.",
            key="result_xic_upload",
        )

        cols = st.columns(3)
        if cols[1].form_submit_button("Add files to workspace", type="primary"):
            any_saved = False
            if files:
                fileupload.save_uploaded_mzML(files)
                any_saved = True

            if fasta_file is not None:
                fasta_path = _save_uploaded_file(fasta_file, fasta_dir)
                st.success(f"Saved FASTA to workspace: {fasta_path.name}")
                any_saved = True

            if lib_file is not None:
                lib_path = _save_uploaded_file(lib_file, lib_dir)
                st.success(f"Saved library to workspace: {lib_path.name}")
                any_saved = True

            any_saved = _save_optional_upload(
                osw_result_file,
                result_target_dirs["osw"],
                "OSW result",
            ) or any_saved
            any_saved = _save_optional_upload(
                long_tsv_file,
                result_target_dirs["long_tsv"],
                "long TSV result",
            ) or any_saved
            any_saved = _save_optional_upload(
                precursor_matrix_file,
                result_target_dirs["precursor_matrix"],
                "precursor matrix",
            ) or any_saved
            any_saved = _save_optional_upload(
                peptide_matrix_file,
                result_target_dirs["peptide_matrix"],
                "peptide matrix",
            ) or any_saved
            any_saved = _save_optional_upload(
                protein_matrix_file,
                result_target_dirs["protein_matrix"],
                "protein matrix",
            ) or any_saved

            if xic_files:
                fileupload.save_uploaded_xic(xic_files)
                any_saved = True

            if not any_saved:
                st.warning("Select files first.")

if st.session_state.location == "local":
    with tabs[1]:
        st_cols = st.columns([0.05, 0.95], gap="small")
        with st_cols[0]:
            st.write("\n")
            st.write("\n")
            dialog_button = st.button(
                "📁",
                key="local_browse",
                help="Browse for your local directory with MS data.",
                disabled=not TK_AVAILABLE,
            )
            if dialog_button:
                st.session_state["local_dir"] = tk_directory_dialog(
                    "Select directory with your MS data",
                    st.session_state["previous_dir"],
                )
                st.session_state["previous_dir"] = st.session_state["local_dir"]
        with st_cols[1]:
            local_files_dir = st.text_input(
                "path to folder with files",
                value=st.session_state["local_dir"],
            )
        local_files_dir = rf"{local_files_dir}"
        local_dir_path = Path(local_files_dir).expanduser()
        local_dir_is_valid = bool(local_files_dir) and local_dir_path.is_dir()

        use_copy = st.checkbox(
            "Make a copy of files",
            key="local_browse-copy_files",
            value=True,
            help="Create a copy of files in workspace.",
        )
        if not use_copy:
            st.warning(
                "**Warning**: You have deselected the `Make a copy of files` option. "
                "This **_assumes you know what you are doing_**. "
                "The workspace will link to the original files when possible. "
            )

        if local_files_dir and not local_dir_is_valid:
            st.warning("Select an existing local folder.")

        selected_local_groups: list[tuple[list[Path], Path, str]] = []
        if local_dir_is_valid:
            mzml_candidates = _list_local_files(local_dir_path, (".mzml", ".mzml.gz"))
            fasta_candidates = _list_local_files(
                local_dir_path,
                (".fasta", ".fa", ".faa"),
            )
            tsv_candidates = _list_local_files(local_dir_path, (".tsv",))
            lib_candidates = _list_local_files(
                local_dir_path,
                (".tsv", ".traml", ".pqp"),
            )
            osw_candidates = _list_local_files(
                local_dir_path,
                (".osw", ".sqlite", ".db"),
            )
            xic_candidates = _list_local_files(local_dir_path, (".xic", ".parquet"))

            long_tsv_defaults = [
                path for path in tsv_candidates if _looks_like_long_results_tsv(path)
            ]
            precursor_defaults = [
                path
                for path in tsv_candidates
                if _looks_like_matrix_tsv(path, "precursor")
            ]
            peptide_defaults = [
                path
                for path in tsv_candidates
                if _looks_like_matrix_tsv(path, "peptide")
            ]
            protein_defaults = [
                path
                for path in tsv_candidates
                if _looks_like_matrix_tsv(path, "protein")
            ]
            result_like_tsv = set(
                long_tsv_defaults
                + precursor_defaults
                + peptide_defaults
                + protein_defaults
            )
            lib_defaults = [
                path
                for path in lib_candidates
                if path.suffix.lower() in {".traml", ".pqp"} or path not in result_like_tsv
            ]

            st.markdown("##### Primary Inputs")
            mzml_selected = _local_file_multiselect(
                "mzML files",
                mzml_candidates,
                mzml_candidates,
                key="local_mzml_selection",
                help_text="Detected .mzML and .mzML.gz files.",
            )
            primary_col1, primary_col2 = st.columns(2)
            with primary_col1:
                fasta_selected = _local_file_multiselect(
                    "Optional FASTA files",
                    fasta_candidates,
                    fasta_candidates,
                    key="local_fasta_selection",
                    help_text="Detected .fasta, .fa, and .faa files.",
                )
            with primary_col2:
                lib_selected = _local_file_multiselect(
                    "Optional spectral library / transition lists",
                    lib_candidates,
                    lib_defaults,
                    key="local_library_selection",
                    help_text="Detected .tsv, .traML, and .pqp files. TSVs can be ambiguous; unselect result TSVs here if needed.",
                )

            st.markdown("##### Existing OpenSwath Results")
            result_col1, result_col2 = st.columns(2)
            with result_col1:
                osw_selected = _local_file_multiselect(
                    "Optional OpenSwath OSW results",
                    osw_candidates,
                    osw_candidates,
                    key="local_osw_result_selection",
                    help_text="Detected .osw, .sqlite, and .db files.",
                )
            with result_col2:
                long_tsv_selected = _local_file_multiselect(
                    "Optional long results TSVs",
                    tsv_candidates,
                    long_tsv_defaults,
                    key="local_long_tsv_selection",
                    help_text="Detected .tsv files. Defaults favor names like openswath_results.tsv or *_results.tsv.",
                )

            matrix_col1, matrix_col2, matrix_col3 = st.columns(3)
            with matrix_col1:
                precursor_selected = _local_file_multiselect(
                    "Precursor matrix TSVs",
                    tsv_candidates,
                    precursor_defaults,
                    key="local_precursor_matrix_selection",
                )
            with matrix_col2:
                peptide_selected = _local_file_multiselect(
                    "Peptide matrix TSVs",
                    tsv_candidates,
                    peptide_defaults,
                    key="local_peptide_matrix_selection",
                )
            with matrix_col3:
                protein_selected = _local_file_multiselect(
                    "Protein matrix TSVs",
                    tsv_candidates,
                    protein_defaults,
                    key="local_protein_matrix_selection",
                )

            xic_selected = _local_file_multiselect(
                "Optional XIC files",
                xic_candidates,
                xic_candidates,
                key="local_xic_selection",
                help_text="Detected .xic and .parquet files for the chromatogram viewer.",
            )

            selected_local_groups = [
                (mzml_selected, mzML_dir, "mzML"),
                (fasta_selected, fasta_dir, "FASTA"),
                (lib_selected, lib_dir, "library"),
                (osw_selected, result_target_dirs["osw"], "OSW result"),
                (long_tsv_selected, result_target_dirs["long_tsv"], "long TSV result"),
                (
                    precursor_selected,
                    result_target_dirs["precursor_matrix"],
                    "precursor matrix",
                ),
                (
                    peptide_selected,
                    result_target_dirs["peptide_matrix"],
                    "peptide matrix",
                ),
                (
                    protein_selected,
                    result_target_dirs["protein_matrix"],
                    "protein matrix",
                ),
                (xic_selected, xic_dir, "XIC"),
            ]

        cols = st.columns([0.65, 0.3, 0.4, 0.25], gap="small")
        copy_button = cols[1].button(
            "Add selected files to workspace",
            type="primary",
            disabled=not local_dir_is_valid,
        )
        if copy_button:
            total_added = 0
            xic_added = False
            for selected_paths, target_dir, label in selected_local_groups:
                added = _copy_or_link_local_files(
                    selected_paths,
                    target_dir,
                    label,
                    use_copy,
                )
                total_added += added
                xic_added = xic_added or (label == "XIC" and added > 0)

            if xic_added:
                _clear_xic_viewer_state()
            if not total_added:
                st.warning("Select files first.")


mzml_files = [
    f.name for f in _list_workspace_files(mzML_dir) if "external_files.txt" not in f.name
]
external_files = mzML_dir / "external_files.txt"
external_list: list[str] = []
if external_files.exists():
    with open(external_files, "r", encoding="utf-8") as f_handle:
        external_list = [f.strip() for f in f_handle.readlines() if f.strip()]
mzml_display = mzml_files + external_list

fasta_list = [p.name for p in _list_workspace_files(fasta_dir)]
lib_list = [p.name for p in _list_workspace_files(lib_dir)]
imported_results = _list_imported_results()
xic_list = _list_workspace_files(xic_dir)

has_workspace_files = any(
    [
        mzml_display,
        fasta_list,
        lib_list,
        imported_results,
        xic_list,
    ]
)

if has_workspace_files:
    v_space(2)

    col_mz, col_fa, col_lib = st.columns([3, 2, 3])

    with col_mz:
        st.markdown(f"##### mzML files ({len(mzml_display)})")
        if mzml_display:
            show_table(pd.DataFrame({"file name": mzml_display}))
        else:
            st.info("No mzML files in workspace")
        to_remove_mz = st.multiselect("Select mzML files to remove", options=mzml_display)
        rm_mz_c1, rm_mz_c2 = st.columns([1, 1])
        if rm_mz_c2.button("Remove selected mzML", disabled=not any(to_remove_mz)):
            params = fileupload.remove_selected_mzML_files(
                [Path(f).stem for f in to_remove_mz],
                params,
            )
            save_params(params)
            st.rerun()
        if rm_mz_c1.button("Remove all mzML", disabled=not any(mzml_display)):
            params = fileupload.remove_all_mzML_files(params)
            save_params(params)
            st.rerun()

    with col_fa:
        st.markdown(f"##### FASTA files ({len(fasta_list)})")
        if fasta_list:
            show_table(pd.DataFrame({"file name": fasta_list}))
            to_remove_fasta = st.multiselect(
                "Select FASTA to remove",
                options=sorted(fasta_list),
            )
            fa_c1, fa_c2 = st.columns(2)
            if fa_c2.button("Remove selected FASTA", disabled=not any(to_remove_fasta)):
                for fn in to_remove_fasta:
                    try:
                        (fasta_dir / fn).unlink()
                    except Exception:
                        st.warning(f"Could not remove {fn}")
                st.success("Selected FASTA files removed")
                st.rerun()
            if fa_c1.button("Remove all FASTA", disabled=not any(fasta_list)):
                for path in _list_workspace_files(fasta_dir):
                    try:
                        path.unlink()
                    except Exception:
                        pass
                st.success("All FASTA files removed")
                st.rerun()
        else:
            st.info("No FASTA files in workspace")

    with col_lib:
        st.markdown(f"##### Library / Transition lists ({len(lib_list)})")
        if lib_list:
            show_table(pd.DataFrame({"file name": lib_list}))
            to_remove_lib = st.multiselect(
                "Select libraries to remove",
                options=sorted(lib_list),
            )
            lb_c1, lb_c2 = st.columns(2)
            if lb_c2.button(
                "Remove selected libraries",
                disabled=not any(to_remove_lib),
            ):
                for fn in to_remove_lib:
                    try:
                        (lib_dir / fn).unlink()
                    except Exception:
                        st.warning(f"Could not remove {fn}")
                st.success("Selected library files removed")
                st.rerun()
            if lb_c1.button("Remove all libraries", disabled=not any(lib_list)):
                for path in _list_workspace_files(lib_dir):
                    try:
                        path.unlink()
                    except Exception:
                        pass
                st.success("All library files removed")
                st.rerun()
        else:
            st.info("No library files in workspace")

    st.markdown("---")
    result_col, xic_col = st.columns([3, 2])

    with result_col:
        st.markdown(f"##### Imported OpenSwath Results ({len(imported_results)})")
        if imported_results:
            imported_df = pd.DataFrame(
                {
                    "type": [_classify_imported_result(path) for path in imported_results],
                    "file name": [path.name for path in imported_results],
                    "workspace path": [_workspace_rel(path) for path in imported_results],
                }
            )
            show_table(imported_df)
            imported_labels = imported_df["workspace path"].tolist()
            imported_path_map = {
                _workspace_rel(path): path for path in imported_results
            }
            to_remove_results = st.multiselect(
                "Select imported results to remove",
                options=imported_labels,
            )
            rs_c1, rs_c2 = st.columns(2)
            if rs_c2.button(
                "Remove selected imported results",
                disabled=not any(to_remove_results),
            ):
                for label in to_remove_results:
                    try:
                        imported_path_map[label].unlink()
                    except Exception:
                        st.warning(f"Could not remove {label}")
                st.success("Selected imported result files removed")
                st.rerun()
            if rs_c1.button(
                "Remove all imported results",
                disabled=not any(imported_results),
            ):
                shutil.rmtree(imported_results_root, ignore_errors=True)
                st.success("All imported result files removed")
                st.rerun()
        else:
            st.info("No imported OpenSwath results in workspace")

    with xic_col:
        st.markdown(f"##### XIC files ({len(xic_list)})")
        if xic_list:
            xic_df = pd.DataFrame(
                {
                    "file name": [path.name for path in xic_list],
                    "workspace path": [_workspace_rel(path) for path in xic_list],
                }
            )
            show_table(xic_df)
            xic_path_map = {_workspace_rel(path): path for path in xic_list}
            to_remove_xic = st.multiselect(
                "Select XIC files to remove",
                options=list(xic_path_map.keys()),
            )
            xic_c1, xic_c2 = st.columns(2)
            if xic_c2.button("Remove selected XIC", disabled=not any(to_remove_xic)):
                for label in to_remove_xic:
                    try:
                        xic_path_map[label].unlink()
                    except Exception:
                        st.warning(f"Could not remove {label}")
                _clear_xic_viewer_state()
                st.success("Selected XIC files removed")
                st.rerun()
            if xic_c1.button("Remove all XIC", disabled=not any(xic_list)):
                fileupload.remove_all_xic_files()
                st.rerun()
        else:
            st.info("No XIC files in workspace")

save_params(params)
