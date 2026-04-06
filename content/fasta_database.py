"""
content/fasta_database.py
FASTA Database Manager — UniProt Download & Filtering

Two tabs:
  1. Download from UniProt  — multi-species, reviewed/unreviewed, extra query
  2. Filter FASTA           — random subsample OR filter by accession list
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import streamlit as st

from src.common.common import page_setup
from src.workflow.UniProtFastaManager import (
    COMMON_SPECIES,
    REVIEW_OPTIONS,
    FilterResult,
    count_fasta_entries,
    download_uniprot_fasta,
    filter_fasta_by_accession,
    filter_fasta_random,
)

# -----------------------------------------------------------------------------
page_setup()


# Session state defaults
_ss = {
    "dl_result_text": None,  # str — last successful download FASTA
    "dl_filename": "",
    "filt_result_text": None,  # str — last filter result FASTA
    "filt_filename": "",
    "upload_fasta_text": None,  # str — uploaded FASTA content
    "upload_filename": "",
}
for k, v in _ss.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -----------------------------------------------------------------------------
# Page header

st.title("🧬 FASTA Database Manager")
st.markdown(
    """
Download **reviewed or unreviewed proteomes** directly from UniProt, or
**filter an existing FASTA file** by random subsampling or a specific list
of protein accessions.
"""
)

tab_download, tab_filter = st.tabs(["⬇️ Download from UniProt", "🔬 Filter FASTA"])

# -----------------------------------------------------------------------------
# TAB 1 — Download from UniProt

with tab_download:
    st.subheader("Download from UniProt")
    st.markdown(
        "Select one or more species and click **Download FASTA**. "
        "The FASTA is fetched via the UniProt REST API and can be saved to "
        "your workspace or downloaded directly."
    )

    # -- Species selection -----------------------------------------------------
    st.markdown("#### Species")
    species_list = list(COMMON_SPECIES.keys())
    default_human = [species_list[0]]  # Human

    selected_species = st.multiselect(
        "Select species",
        options=species_list,
        default=default_human,
        help="Hold Ctrl/Cmd to select multiple species. "
        "Results will be merged into a single FASTA.",
        key="dl_species",
    )

    # Custom taxonomy
    with st.expander("➕ Add custom taxonomy ID", expanded=False):
        st.caption(
            "Enter a comma-separated list of NCBI taxonomy IDs to include "
            "alongside the species selected above."
        )
        custom_taxon = st.text_input(
            "Custom taxonomy IDs (comma-separated)",
            value="",
            placeholder="e.g. 9606, 10090",
            key="dl_custom_taxon",
        )

    # -- Review status ---------------------------------------------------------
    st.markdown("#### Review status")
    col_rev1, col_rev2 = st.columns(2)
    with col_rev1:
        review_option = st.radio(
            "Protein review status",
            options=list(REVIEW_OPTIONS.keys()),
            index=0,
            key="dl_review",
            help=(
                "**Reviewed (Swiss-Prot)** — manually curated, high quality.\n\n"
                "**Unreviewed (TrEMBL)** — computationally analysed.\n\n"
                "**Both** — combine reviewed and unreviewed."
            ),
        )
    with col_rev2:
        include_isoforms = st.checkbox(
            "Include canonical isoforms",
            value=False,
            key="dl_isoforms",
            help="Download all annotated isoforms for each protein.",
        )

    # -- Extra filter ---------------------------------------------------------
    with st.expander("🔎 Additional UniProt query filters (optional)", expanded=False):
        st.caption(
            "You can add any valid UniProt query syntax here. "
            "Examples: `gene:TP53`, `length:[100 TO 500]`, `cc_subcellular_location:nucleus`"
        )
        extra_query = st.text_input(
            "Extra UniProt query",
            value="",
            placeholder="e.g. gene:TP53",
            key="dl_extra_query",
        )

    # -- Workspace save --------------------------------------------------------
    workspace_dir = Path(st.session_state.get("workspace", "."))
    fasta_workspace = workspace_dir / "input-files" / "fasta"

    with st.expander("💾 Workspace save options", expanded=False):
        save_to_ws = st.checkbox(
            "Also save to workspace (input-files/fasta/)",
            value=True,
            key="dl_save_ws",
        )
        custom_fname = st.text_input(
            "Override filename (leave empty to auto-generate)",
            value="",
            key="dl_custom_fname",
            placeholder="e.g. human_reviewed.fasta",
        )

    # -- Download button -------------------------------------------------------
    st.markdown("---")

    if not selected_species:
        st.info("Select at least one species to enable download.", icon="ℹ️")

    dl_btn = st.button(
        "⬇️ Download FASTA from UniProt",
        type="primary",
        disabled=not selected_species,
        key="dl_btn",
    )

    if dl_btn:
        # Resolve custom taxonomy
        extra_taxon_ids: list[int] = []
        if custom_taxon.strip():
            for raw in custom_taxon.split(","):
                raw = raw.strip()
                if raw.isdigit():
                    extra_taxon_ids.append(int(raw))
                elif raw:
                    st.warning(f"Ignoring invalid taxonomy ID: `{raw}`")

        prog_bar = st.progress(0.0)
        status = st.empty()

        def _progress(fraction, msg):
            if fraction is not None:
                prog_bar.progress(min(float(fraction), 1.0))
            status.caption(msg)

        with st.spinner("Connecting to UniProt…"):
            t0 = time.perf_counter()
            result = download_uniprot_fasta(
                species_names=selected_species,
                review_option=review_option,
                extra_query=extra_query,
                include_isoforms=include_isoforms,
                progress_cb=_progress,
            )
            elapsed = time.perf_counter() - t0

        prog_bar.empty()
        status.empty()

        if not result.success:
            st.error(f"Download failed: {result.error}")
        else:
            fname = custom_fname.strip() or result.filename
            if not fname.endswith(".fasta"):
                fname += ".fasta"

            st.session_state.dl_result_text = result.fasta_text
            st.session_state.dl_filename = fname
            st.success(
                f"✅ Downloaded **{result.n_entries:,}** sequences "
                f"in {elapsed:.1f} s  →  `{fname}`"
            )

            # Save to workspace
            if save_to_ws:
                fasta_workspace.mkdir(parents=True, exist_ok=True)
                out_path = fasta_workspace / fname
                out_path.write_text(result.fasta_text, encoding="utf-8")
                st.success(f"Saved to workspace: `{out_path}`")

    # -- Download button (always shown if result is available) --------------
    if st.session_state.dl_result_text:
        st.markdown("---")
        st.markdown(
            f"**Last download:** `{st.session_state.dl_filename}`  "
            f"({count_fasta_entries(st.session_state.dl_result_text):,} entries)"
        )
        st.download_button(
            label="💾 Save FASTA to disk",
            data=st.session_state.dl_result_text.encode("utf-8"),
            file_name=st.session_state.dl_filename,
            mime="text/plain",
            key="dl_save_btn",
        )
        with st.expander("👁️ Preview first 10 entries", expanded=False):
            lines = st.session_state.dl_result_text.splitlines()
            preview_lines: list[str] = []
            entry_count = 0
            for line in lines:
                if line.startswith(">"):
                    entry_count += 1
                    if entry_count > 10:
                        break
                preview_lines.append(line)
            st.code("\n".join(preview_lines), language="text")


# -----------------------------------------------------------------------------
# TAB 2 — Filter FASTA

with tab_filter:
    st.subheader("Filter an Existing FASTA File")
    st.markdown(
        "Upload a FASTA file (or use the last UniProt download) and filter it "
        "by **random subsampling** or **specific protein accessions**."
    )

    # -- Input source ---------------------------------------------------------
    st.markdown("#### Input FASTA source")

    source_opts = ["Upload a FASTA file"]
    if st.session_state.dl_result_text:
        source_opts.insert(0, "Use last UniProt download")

    # Check workspace for FASTA files
    fasta_workspace_files: list[str] = []
    _fasta_dir = Path(st.session_state.get("workspace", ".")) / "input-files" / "fasta"
    if _fasta_dir.exists():
        fasta_workspace_files = [p.name for p in _fasta_dir.iterdir() if p.is_file()]
    if fasta_workspace_files:
        source_opts.append("Use file from workspace")

    fasta_source = st.radio(
        "FASTA source",
        options=source_opts,
        index=0,
        key="filt_source",
        horizontal=True,
    )

    fasta_text_input: str | None = None
    fasta_source_name = "input.fasta"

    if fasta_source == "Use last UniProt download":
        fasta_text_input = st.session_state.dl_result_text
        fasta_source_name = st.session_state.dl_filename
        n_entries = count_fasta_entries(fasta_text_input)
        st.info(
            f"Using **{n_entries:,}** entries from `{fasta_source_name}`",
            icon="ℹ️",
        )

    elif fasta_source == "Upload a FASTA file":
        uploaded = st.file_uploader(
            "Upload FASTA file",
            type=["fasta", "fa", "faa", "txt"],
            key="filt_upload",
            help="Standard FASTA format. Large files (>200 MB) may be slow.",
        )
        if uploaded is not None:
            content = uploaded.read().decode("utf-8", errors="replace")
            st.session_state.upload_fasta_text = content
            st.session_state.upload_filename = uploaded.name
        if st.session_state.upload_fasta_text:
            fasta_text_input = st.session_state.upload_fasta_text
            fasta_source_name = st.session_state.upload_filename
            n_entries = count_fasta_entries(fasta_text_input)
            st.info(
                f"Loaded **{n_entries:,}** entries from `{fasta_source_name}`",
                icon="ℹ️",
            )

    elif fasta_source == "Use file from workspace":
        sel_ws = st.selectbox(
            "Select workspace FASTA",
            options=fasta_workspace_files,
            key="filt_ws_sel",
        )
        if sel_ws:
            ws_path = _fasta_dir / sel_ws
            fasta_text_input = ws_path.read_text(encoding="utf-8", errors="replace")
            fasta_source_name = sel_ws
            n_entries = count_fasta_entries(fasta_text_input)
            st.info(
                f"Loaded **{n_entries:,}** entries from `{fasta_source_name}`",
                icon="ℹ️",
            )

    # -- Filter mode -----------------------------------------------------------
    st.markdown("---")
    st.markdown("#### Filter mode")

    filter_mode = st.radio(
        "How to filter",
        options=["Random subsampling", "Filter by accession list"],
        key="filt_mode",
        horizontal=True,
    )

    # --- Mode A: Random subsampling ------------------------------------------
    if filter_mode == "Random subsampling":
        st.markdown(
            "Randomly select **N** proteins from the input FASTA. "
            "Set a seed to make the selection reproducible."
        )

        n_total = count_fasta_entries(fasta_text_input) if fasta_text_input else 0

        col_n, col_seed, col_pct = st.columns(3)
        with col_n:
            n_subsample = st.number_input(
                "Number of proteins to keep (N)",
                min_value=1,
                max_value=max(n_total, 1),
                value=min(100, max(n_total, 1)),
                step=10,
                key="filt_n",
                help="Must be ≤ total entries in the input FASTA.",
            )
        with col_pct:
            if n_total > 0:
                st.metric(
                    "Fraction kept",
                    f"{n_subsample / n_total:.1%}",
                    help=f"{n_subsample:,} of {n_total:,} entries",
                )
        with col_seed:
            use_seed = st.checkbox(
                "Use fixed random seed", value=True, key="filt_use_seed"
            )
            seed_val = st.number_input(
                "Random seed",
                value=42,
                min_value=0,
                key="filt_seed",
                disabled=not use_seed,
            )

        run_random = st.button(
            "🎲 Apply Random Subsampling",
            type="primary",
            disabled=(fasta_text_input is None),
            key="filt_run_random",
        )

        if run_random and fasta_text_input:
            with st.spinner("Subsampling…"):
                res: FilterResult = filter_fasta_random(
                    fasta_text=fasta_text_input,
                    n=int(n_subsample),
                    seed=int(seed_val) if use_seed else None,
                    source_filename=fasta_source_name,
                )
            if not res.success:
                st.error(f"Filter failed: {res.error}")
            else:
                st.session_state.filt_result_text = res.fasta_text
                st.session_state.filt_filename = res.filename
                st.success(
                    f"✅ Kept **{res.n_out:,}** of **{res.n_in:,}** entries "
                    f"→ `{res.filename}`"
                )

    # --- Mode B: Filter by accession list ------------------------------------
    else:
        st.markdown(
            "Provide a list of UniProt accession IDs (one per line, or comma-separated). "
            "Only proteins matching those accessions will be retained."
        )

        col_acc, col_up = st.columns([3, 2])
        with col_acc:
            acc_text = st.text_area(
                "Accession IDs",
                height=200,
                placeholder="P12345\nO00187\nQ9Y6K9\n...",
                key="filt_accs",
                help="One accession per line, or comma-separated.",
            )
        with col_up:
            acc_file = st.file_uploader(
                "Or upload a text file of accessions",
                type=["txt", "csv", "tsv"],
                key="filt_acc_file",
                help="One accession per line, or comma-separated values.",
            )
            if acc_file is not None:
                acc_text_from_file = acc_file.read().decode("utf-8", errors="replace")
                if (
                    "acc_text_from_file" not in st.session_state
                    or st.session_state.get("acc_file_name") != acc_file.name
                ):
                    st.session_state["acc_text_loaded"] = acc_text_from_file
                    st.session_state["acc_file_name"] = acc_file.name
                    st.info(
                        f"Loaded accessions from `{acc_file.name}`. "
                        "They have been pasted into the text area above — "
                        "click Apply to proceed."
                    )

        # Merge text-area + file
        combined_text = acc_text or ""
        if st.session_state.get("acc_text_loaded"):
            combined_text = (
                combined_text + "\n" + st.session_state["acc_text_loaded"]
            ).strip()

        # Parse accessions
        accs: list[str] = []
        for token in combined_text.replace(",", "\n").splitlines():
            token = token.strip()
            if token:
                accs.append(token)

        accs = list(dict.fromkeys(accs))  # deduplicate while preserving order

        if accs:
            st.caption(f"{len(accs):,} unique accession(s) ready to filter.")

        run_acc = st.button(
            "🔬 Apply Accession Filter",
            type="primary",
            disabled=(fasta_text_input is None or not accs),
            key="filt_run_acc",
        )

        if run_acc and fasta_text_input and accs:
            with st.spinner(f"Filtering {len(accs):,} accessions…"):
                res: FilterResult = filter_fasta_by_accession(
                    fasta_text=fasta_text_input,
                    accessions=accs,
                    source_filename=fasta_source_name,
                )

            if not res.success:
                st.error(f"Filter failed: {res.error}")
                if res.missing_accs:
                    st.warning(
                        f"None of the {len(res.missing_accs):,} requested "
                        "accessions were found in the FASTA."
                    )
            else:
                st.session_state.filt_result_text = res.fasta_text
                st.session_state.filt_filename = res.filename
                st.success(
                    f"✅ Matched **{res.n_out:,}** of **{len(accs):,}** "
                    f"requested accessions from {res.n_in:,} total entries "
                    f"→ `{res.filename}`"
                )
                if res.missing_accs:
                    with st.expander(
                        f"⚠️ {len(res.missing_accs):,} accessions not found",
                        expanded=True,
                    ):
                        st.caption(
                            "These accessions were not matched in the input FASTA:"
                        )
                        st.code("\n".join(res.missing_accs), language="text")

    # -- Filter result output --------------------------------------------------
    if st.session_state.filt_result_text:
        st.markdown("---")
        n_filt = count_fasta_entries(st.session_state.filt_result_text)
        st.markdown(
            f"**Filter result:** `{st.session_state.filt_filename}`  "
            f"({n_filt:,} entries)"
        )

        col_dl, col_ws = st.columns(2)
        with col_dl:
            st.download_button(
                label="💾 Download filtered FASTA",
                data=st.session_state.filt_result_text.encode("utf-8"),
                file_name=st.session_state.filt_filename,
                mime="text/plain",
                key="filt_dl_btn",
            )
        with col_ws:
            if st.button("📂 Save to workspace", key="filt_save_ws"):
                _fasta_dir.mkdir(parents=True, exist_ok=True)
                dest = _fasta_dir / st.session_state.filt_filename
                dest.write_text(st.session_state.filt_result_text, encoding="utf-8")
                st.success(f"Saved to workspace: `{dest}`")

        with st.expander("👁️ Preview first 10 entries", expanded=False):
            lines = st.session_state.filt_result_text.splitlines()
            preview: list[str] = []
            count = 0
            for line in lines:
                if line.startswith(">"):
                    count += 1
                    if count > 10:
                        break
                preview.append(line)
            st.code("\n".join(preview), language="text")
