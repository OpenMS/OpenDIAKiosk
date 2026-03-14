import json as _json
import shutil
import time
import traceback
from pathlib import Path

import streamlit as st

from src.common.common import page_setup, save_params
from src.workflow.OpenSwathDecoyGeneratorWorkflow import OpenSwathDecoyGeneratorWorkflow

params = page_setup()

# Early binary check to avoid silent workflow failures.
decoy_binary = shutil.which("OpenSwathDecoyGenerator")
if not decoy_binary:
    st.title("🎭 OpenSwath Decoy Generator")
    st.error(
        "OpenSwathDecoyGenerator could not be found on the system PATH. "
        "Install OpenMS via conda or pip, or build it from source, then reload this page."
    )
    st.markdown(
        """
Possible installation options:
- `conda install -c bioconda openms`
- `pip install openms`
- Build OpenMS from source and ensure `OpenSwathDecoyGenerator` is on your `PATH`
"""
    )
    st.stop()

st.title("🎭 OpenSwath Decoy Generator")
st.markdown(
    """
Generate decoy transitions from targeted libraries using
[OpenSwathDecoyGenerator](https://openms.de/doxygen/nightly/html/TOPP_OpenSwathDecoyGenerator.html).
Use this to create target-decoy libraries for downstream FDR estimation.
"""
)

st.markdown("---")
st.subheader("📥 Input Library File")

default_workspace = Path(st.session_state.get("workspace", ".")).resolve()
assay_results_dir = Path(
    default_workspace, "openswath-assay-generator", "results", "openswath"
).resolve()

existing_libraries = []
if assay_results_dir.exists():
    valid_extensions = {".traml", ".tsv", ".mrm", ".pqp", ".oswpq"}
    existing_libraries = sorted(
        [
            f.name
            for f in assay_results_dir.glob("*")
            if f.is_file() and f.suffix.lower() in valid_extensions
        ]
    )

if existing_libraries:
    st.write("**Option 1: Use existing OpenSwath Assay library**")
    selected_lib = st.selectbox(
        "Select from existing libraries",
        options=[""] + existing_libraries,
        index=0,
        key="select_existing_decoy_input",
    )
    if selected_lib:
        selected_lib_path = Path(assay_results_dir, selected_lib)
        size_mb = selected_lib_path.stat().st_size / (1024 * 1024)
        st.write(f"✅ Selected: {selected_lib} ({size_mb:.1f} MB)")
else:
    selected_lib = None

st.write("**Option 2: Upload new library file**")
input_file = st.file_uploader(
    "Or upload spectral library (TraML, TSV, MRM, PQP, or oswpq)",
    type=["traml", "tsv", "mrm", "pqp", "oswpq"],
    key="decoy_input_library_file",
)
if input_file is not None:
    st.write(f"✅ Loaded: {input_file.name}")

st.markdown("---")
st.subheader("📤 Output Configuration")
col1, col2 = st.columns(2)
with col1:
    output_file = st.text_input(
        "Output filename",
        value="openswath_decoys.tsv",
        help="Output path/name for generated decoy transitions.",
    )
with col2:
    output_type = st.selectbox(
        "Output format",
        options=["auto", "tsv", "TraML", "pqp", "oswpq"],
        index=0,
        help="'auto' infers output type from filename extension.",
    )

col1, col2 = st.columns(2)
with col1:
    input_type = st.selectbox(
        "Input format",
        options=["auto", "tsv", "mrm", "pqp", "TraML", "oswpq"],
        index=0,
        help="'auto' infers input type from file extension/content.",
    )
with col2:
    decoy_tag = st.text_input(
        "Decoy tag",
        value="DECOY_",
        help="Prefix added to decoy peptide/protein identifiers.",
    )

st.markdown("---")
st.subheader("1️⃣ Decoy Method")
col1, col2 = st.columns(2)
with col1:
    method = st.selectbox(
        "Decoy generation method",
        options=["shuffle", "pseudo-reverse", "reverse", "shift"],
        index=0,
    )
with col2:
    switch_kr = st.checkbox(
        "Switch terminal K/R",
        value=True,
        help="Switch K/R to alter precursor mass where appropriate.",
    )

col1, col2 = st.columns(2)
with col1:
    min_decoy_fraction = st.number_input(
        "Minimum decoy fraction",
        value=0.8,
        min_value=0.0,
        max_value=2.0,
        step=0.05,
    )
with col2:
    aim_decoy_fraction = st.number_input(
        "Target decoy fraction",
        value=1.0,
        min_value=0.0,
        max_value=2.0,
        step=0.05,
    )

with st.expander("⚙️ Advanced Decoy Parameters", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        shuffle_max_attempts = st.number_input(
            "Shuffle max attempts",
            value=30,
            min_value=1,
            step=1,
        )
    with col2:
        shuffle_identity_threshold = st.number_input(
            "Shuffle sequence identity threshold",
            value=0.5,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
        )

    col1, col2 = st.columns(2)
    with col1:
        shift_precursor_mz_shift = st.number_input(
            "Shift precursor m/z (Th)",
            value=0.0,
            step=0.1,
            help="Used by the 'shift' method.",
        )
    with col2:
        shift_product_mz_shift = st.number_input(
            "Shift product m/z (Th)",
            value=20.0,
            step=0.1,
            help="Used by the 'shift' method.",
        )

with st.expander("⚙️ Advanced Fragment Parameters", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        product_mz_threshold = st.number_input(
            "Product m/z threshold",
            value=0.025,
            step=0.001,
        )
    with col2:
        allowed_fragment_types = st.text_input(
            "Allowed fragment types",
            value="b,y",
        )

    col1, col2 = st.columns(2)
    with col1:
        allowed_fragment_charges = st.text_input(
            "Allowed fragment charges",
            value="1,2,3,4",
        )
    with col2:
        separate = st.checkbox(
            "Write decoys separately",
            value=False,
            help="If enabled, decoys are not appended to targets.",
        )

    col1, col2 = st.columns(2)
    with col1:
        enable_detection_specific_losses = st.checkbox(
            "Enable detection-specific neutral losses",
            value=False,
        )
    with col2:
        enable_detection_unspecific_losses = st.checkbox(
            "Enable detection-unspecific neutral losses",
            value=False,
        )

st.markdown("---")
st.subheader("2️⃣ Processing Options")
threads = st.number_input(
    "Number of Threads",
    value=0,
    min_value=0,
    help="Number of parallel threads. 0 uses tool defaults.",
)

st.markdown("---")

if st.button(
    "🚀 Run Decoy Generation in Workspace", type="primary", use_container_width=True
):
    try:
        st.info("🔄 Initializing decoy workflow...")

        wf = OpenSwathDecoyGeneratorWorkflow()
        wf_dir = Path(wf.workflow_dir).resolve()
        st.write(f"✅ Workflow created: {wf_dir}")

        input_dest_dir = Path(wf_dir, "input-files", "library")
        input_dest_dir.mkdir(parents=True, exist_ok=True)

        if input_file is None and not selected_lib:
            st.error("❌ Please upload a library or select an existing one.")
        else:
            if input_file is not None:
                input_name = getattr(input_file, "name", "library.tsv")
                input_path = Path(input_dest_dir, input_name)
                with open(input_path, "wb") as fh:
                    fh.write(input_file.getbuffer())
                st.write(f"✅ Input uploaded to: {input_path}")
            else:
                input_path = Path(assay_results_dir, selected_lib)
                st.write(f"✅ Using existing library: {input_path}")

            results_dir = Path(wf_dir, "results", "decoy").resolve()
            results_dir.mkdir(parents=True, exist_ok=True)
            output_filename = Path(output_file).name if output_file else "openswath_decoys.tsv"
            workspace_output_file = str(Path(results_dir, output_filename))

            params_to_write = {
                "input_library": str(input_path),
                "input_type": input_type,
                "output_file": workspace_output_file,
                "output_type": output_type,
                "method": method,
                "decoy_tag": decoy_tag,
                "switchKR": bool(switch_kr),
                "min_decoy_fraction": float(min_decoy_fraction),
                "aim_decoy_fraction": float(aim_decoy_fraction),
                "shuffle_max_attempts": int(shuffle_max_attempts),
                "shuffle_sequence_identity_threshold": float(shuffle_identity_threshold),
                "shift_precursor_mz_shift": float(shift_precursor_mz_shift),
                "shift_product_mz_shift": float(shift_product_mz_shift),
                "product_mz_threshold": float(product_mz_threshold),
                "allowed_fragment_types": str(allowed_fragment_types),
                "allowed_fragment_charges": str(allowed_fragment_charges),
                "enable_detection_specific_losses": bool(enable_detection_specific_losses),
                "enable_detection_unspecific_losses": bool(enable_detection_unspecific_losses),
                "separate": bool(separate),
                "threads": int(threads),
            }

            params_file = Path(wf.parameter_manager.params_file).resolve()
            params_file.parent.mkdir(parents=True, exist_ok=True)
            with open(params_file, "w", encoding="utf-8") as f:
                _json.dump(params_to_write, f, indent=2)

            wf.start_workflow()
            st.success("✅ Decoy workflow submitted. Processing in background...")

            with st.expander("📂 Workspace Info", expanded=False):
                st.write("**Workspace Location:**")
                st.code(str(wf_dir), language="bash")
                st.write("**Log Location:**")
                st.code(str(Path(wf_dir, "logs", "workflow.log")), language="bash")

            pid_dir = wf.executor.pid_dir
            progress_placeholder = st.empty()
            max_wait_time = 15 * 60
            start_time = time.time()
            poll_interval = 2

            while time.time() - start_time < max_wait_time:
                if not pid_dir.exists():
                    progress_placeholder.info("✅ Workflow completed! Refreshing to show results...")
                    time.sleep(1)
                    st.rerun()
                    break

                elapsed = int(time.time() - start_time)
                progress_placeholder.info(f"⏳ Processing... ({elapsed}s elapsed)")
                time.sleep(poll_interval)
            else:
                progress_placeholder.warning(
                    "⚠️ Workflow is taking longer than expected. "
                    "Check logs or refresh to see results."
                )

    except Exception as e:
        st.error(f"❌ Failed to start workflow: {e}")
        st.text(traceback.format_exc())

st.markdown("---")
st.subheader("📥 Download Results")

default_workspace = Path(st.session_state.get("workspace", ".")).resolve()
decoy_workspace = Path(default_workspace, "openswath-decoy-generator").resolve()
results_dir = Path(decoy_workspace, "results", "decoy").resolve()

if decoy_workspace.exists() and results_dir.exists():
    all_decoy_files = list(results_dir.glob("*"))
    decoy_files = [f for f in all_decoy_files if f.is_file() and f.exists()]

    if decoy_files:
        st.success("✅ Decoy results generated! Download below.")

        for decoy_file in decoy_files:
            try:
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.write(f"📄 {decoy_file.name}")
                with col2:
                    size_mb = decoy_file.stat().st_size / (1024 * 1024)
                    if size_mb < 1:
                        st.write(f"({decoy_file.stat().st_size / 1024:.1f} KB)")
                    else:
                        st.write(f"({size_mb:.1f} MB)")
                with col3:
                    with open(decoy_file, "rb") as f:
                        st.download_button(
                            "⬇️ Download",
                            data=f.read(),
                            file_name=decoy_file.name,
                            key=f"decoy_{decoy_file.name}",
                            use_container_width=True,
                        )
            except Exception as e:
                st.warning(f"⚠️ Could not access {decoy_file.name}: {e}")

        st.markdown("---")
        st.subheader("🗑️ Clear Results")

        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button(
                "🗑️ Clear Results",
                key="clear_decoy_results_btn",
                use_container_width=True,
            ):
                st.session_state.clear_decoy_results_confirm = True

        if st.session_state.get("clear_decoy_results_confirm", False):
            st.warning("⚠️ **WARNING: This will permanently delete the following files:**")
            for file in decoy_files:
                size_mb = file.stat().st_size / (1024 * 1024)
                if size_mb < 1:
                    size_str = f"{file.stat().st_size / 1024:.1f} KB"
                else:
                    size_str = f"{size_mb:.1f} MB"
                st.write(f"  - `{file.name}` ({size_str})")

            col1, col2, col3 = st.columns([2, 1, 1])
            with col2:
                if st.button("❌ Cancel", key="cancel_decoy_clear", use_container_width=True):
                    st.session_state.clear_decoy_results_confirm = False
                    st.rerun()
            with col3:
                if st.button(
                    "🗑️ Delete",
                    key="confirm_decoy_clear",
                    type="secondary",
                    use_container_width=True,
                ):
                    deleted_count = 0
                    for file in decoy_files:
                        try:
                            file.unlink()
                            deleted_count += 1
                        except Exception as e:
                            st.warning(f"Could not delete {file.name}: {e}")

                    st.session_state.clear_decoy_results_confirm = False
                    st.success(f"✅ Successfully deleted {deleted_count} file(s)!")
                    time.sleep(1)
                    st.rerun()
    else:
        st.info("⏳ Processing in progress... Refresh to check for new files.")
else:
    st.info("📂 No decoy workspace found. Run the workflow above to generate results.")

save_params(params)
