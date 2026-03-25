import subprocess
from pathlib import Path
import json
import streamlit as st


class PyProphetCLI:
    """
    Full PyProphet CLI wrapper with comprehensive options and integration
    with the workflow runner (CommandExecutor).
    """

    def __init__(
        self, parameter_manager, workflow_dir: Path, executor=None, logger=None
    ):
        self.pm = parameter_manager
        self.workflow_dir = Path(workflow_dir)
        self.executor = executor  # CommandExecutor for running via workflow
        self.logger = logger

    def save_params_to_json(self, command: str, params: dict):
        """Save PyProphet command params to workspace params.json"""
        data = self.pm.get_parameters_from_json()
        if "pyprophet" not in data:
            data["pyprophet"] = {}
        data["pyprophet"][command] = params
        with open(self.pm.params_file, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=4)

    def build_score_command(self, params: dict) -> list:
        """Build pyprophet score command from params dict"""
        cmd = ["pyprophet", "score"]
        if params.get("in"):
            cmd += ["--in", params["in"]]
        if params.get("out"):
            cmd += ["--out", params["out"]]
        if (
            params.get("subsample_ratio") is not None
            and params["subsample_ratio"] != 1.0
        ):
            cmd += ["--subsample_ratio", str(params["subsample_ratio"])]
        if params.get("classifier"):
            cmd += ["--classifier", params["classifier"]]
        if params.get("autotune"):
            cmd += ["--autotune"]
        if params.get("apply_weights"):
            cmd += ["--apply_weights", params["apply_weights"]]
        if params.get("xeval_fraction") is not None and params["xeval_fraction"] != 0.5:
            cmd += ["--xeval_fraction", str(params["xeval_fraction"])]
        if params.get("xeval_num_iter") is not None and params["xeval_num_iter"] != 10:
            cmd += ["--xeval_num_iter", str(params["xeval_num_iter"])]
        if (
            params.get("ss_initial_fdr") is not None
            and params["ss_initial_fdr"] != 0.15
        ):
            cmd += ["--ss_initial_fdr", str(params["ss_initial_fdr"])]
        if (
            params.get("ss_iteration_fdr") is not None
            and params["ss_iteration_fdr"] != 0.05
        ):
            cmd += ["--ss_iteration_fdr", str(params["ss_iteration_fdr"])]
        if params.get("ss_num_iter") is not None and params["ss_num_iter"] != 10:
            cmd += ["--ss_num_iter", str(params["ss_num_iter"])]
        if params.get("ss_main_score") and params["ss_main_score"] != "auto":
            cmd += ["--ss_main_score", params["ss_main_score"]]
        if params.get("ss_score_filter"):
            cmd += ["--ss_score_filter", params["ss_score_filter"]]
        if params.get("ss_scale_features"):
            cmd += ["--ss_scale_features"]
        if params.get("group_id") and params["group_id"] != "group_id":
            cmd += ["--group_id", params["group_id"]]
        if params.get("parametric"):
            cmd += ["--parametric"]
        if params.get("pfdr"):
            cmd += ["--pfdr"]
        if params.get("pi0_lambda"):
            lambda_vals = params["pi0_lambda"]
            if isinstance(lambda_vals, str):
                lambda_vals = lambda_vals.split(",")
            cmd += ["--pi0_lambda"] + [str(v.strip()) for v in lambda_vals]
        if params.get("pi0_method"):
            cmd += ["--pi0_method", params["pi0_method"]]
        if params.get("pi0_smooth_df") is not None and params["pi0_smooth_df"] != 3:
            cmd += ["--pi0_smooth_df", str(params["pi0_smooth_df"])]
        if params.get("pi0_smooth_log_pi0"):
            cmd += ["--pi0_smooth_log_pi0"]
        if params.get("lfdr_truncate", True):
            cmd += ["--lfdr_truncate"]
        if params.get("lfdr_monotone", True):
            cmd += ["--lfdr_monotone"]
        if params.get("lfdr_transformation"):
            cmd += ["--lfdr_transformation", params["lfdr_transformation"]]
        if params.get("lfdr_adj") is not None and params["lfdr_adj"] != 1.5:
            cmd += ["--lfdr_adj", str(params["lfdr_adj"])]
        if params.get("lfdr_eps") is not None and params["lfdr_eps"] != 1e-8:
            cmd += ["--lfdr_eps", str(params["lfdr_eps"])]
        if params.get("level"):
            cmd += ["--level", params["level"]]
        if params.get("add_alignment_features"):
            cmd += ["--add_alignment_features"]
        if (
            params.get("ipf_max_peakgroup_rank") is not None
            and params["ipf_max_peakgroup_rank"] != 1
        ):
            cmd += ["--ipf_max_peakgroup_rank", str(params["ipf_max_peakgroup_rank"])]
        if (
            params.get("ipf_max_peakgroup_pep") is not None
            and params["ipf_max_peakgroup_pep"] != 0.7
        ):
            cmd += ["--ipf_max_peakgroup_pep", str(params["ipf_max_peakgroup_pep"])]
        if (
            params.get("ipf_max_transition_isotope_overlap") is not None
            and params["ipf_max_transition_isotope_overlap"] != 0.5
        ):
            cmd += [
                "--ipf_max_transition_isotope_overlap",
                str(params["ipf_max_transition_isotope_overlap"]),
            ]
        if (
            params.get("ipf_min_transition_sn") is not None
            and params["ipf_min_transition_sn"] != 0
        ):
            cmd += ["--ipf_min_transition_sn", str(params["ipf_min_transition_sn"])]
        if params.get("glyco"):
            cmd += ["--glyco"]
        if params.get("density_estimator"):
            cmd += ["--density_estimator", params["density_estimator"]]
        if params.get("grid_size") is not None and params["grid_size"] != 256:
            cmd += ["--grid_size", str(params["grid_size"])]
        if params.get("tric_chromprob"):
            cmd += ["--tric_chromprob"]
        if params.get("color_palette") and params["color_palette"] != "normal":
            cmd += ["--color_palette", params["color_palette"]]
        if params.get("main_score_selection_report"):
            cmd += ["--main_score_selection_report"]
        if params.get("test"):
            cmd += ["--test"]
        if params.get("profile"):
            cmd += ["--profile"]
        if params.get("threads") is not None and params["threads"] != 1:
            cmd += ["--threads", str(params["threads"])]
        return cmd

    def build_infer_command(self, level: str, params: dict) -> list:
        """Build pyprophet infer peptide/protein command"""
        cmd = ["pyprophet", "infer", level]
        if params.get("in"):
            cmd += ["--in", params["in"]]
        if params.get("out"):
            cmd += ["--out", params["out"]]
        if params.get("context"):
            cmd += ["--context", params["context"]]
        if params.get("parametric"):
            cmd += ["--parametric"]
        if params.get("pfdr"):
            cmd += ["--pfdr"]
        if params.get("pi0_lambda"):
            lambda_vals = params["pi0_lambda"]
            if isinstance(lambda_vals, str):
                lambda_vals = lambda_vals.split(",")
            cmd += ["--pi0_lambda"] + [str(v.strip()) for v in lambda_vals]
        if params.get("pi0_method"):
            cmd += ["--pi0_method", params["pi0_method"]]
        if params.get("pi0_smooth_df") is not None:
            cmd += ["--pi0_smooth_df", str(params["pi0_smooth_df"])]
        if params.get("lfdr_truncate", True):
            cmd += ["--lfdr_truncate"]
        if params.get("lfdr_monotone", True):
            cmd += ["--lfdr_monotone"]
        if params.get("lfdr_transformation"):
            cmd += ["--lfdr_transformation", params["lfdr_transformation"]]
        if params.get("lfdr_adj") is not None:
            cmd += ["--lfdr_adj", str(params["lfdr_adj"])]
        if params.get("lfdr_eps") is not None:
            cmd += ["--lfdr_eps", str(params["lfdr_eps"])]
        return cmd

    def build_export_command(self, params: dict) -> list:
        """Build pyprophet export tsv command"""
        cmd = ["pyprophet", "export", "tsv"]
        if params.get("in"):
            cmd += ["--in", params["in"]]
        if params.get("out"):
            cmd += ["--out", params["out"]]
        if params.get("format"):
            cmd += ["--format", params["format"]]
        if params.get("csv"):
            cmd += ["--csv"]
        if params.get("transition_quantification", True):
            cmd += ["--transition_quantification"]
        if params.get("max_transition_pep") is not None:
            cmd += ["--max_transition_pep", str(params["max_transition_pep"])]
        if params.get("ipf"):
            cmd += ["--ipf", params["ipf"]]
        if params.get("ipf_max_peptidoform_pep") is not None:
            cmd += ["--ipf_max_peptidoform_pep", str(params["ipf_max_peptidoform_pep"])]
        if params.get("max_rs_peakgroup_qvalue") is not None:
            cmd += ["--max_rs_peakgroup_qvalue", str(params["max_rs_peakgroup_qvalue"])]
        if params.get("max_global_peptide_qvalue") is not None:
            cmd += [
                "--max_global_peptide_qvalue",
                str(params["max_global_peptide_qvalue"]),
            ]
        if params.get("max_global_protein_qvalue") is not None:
            cmd += [
                "--max_global_protein_qvalue",
                str(params["max_global_protein_qvalue"]),
            ]
        if params.get("use_alignment", True):
            cmd += ["--use_alignment"]
        if params.get("max_alignment_pep") is not None:
            cmd += ["--max_alignment_pep", str(params["max_alignment_pep"])]
        return cmd

    def ui(self):
        """Render UI for all PyProphet commands (score, infer:peptide, infer:protein, export:tsv)"""
        st.markdown("### PyProphet Commands")
        st.markdown(
            "Configure PyProphet commands for your workflow. Each command can be configured independently."
        )

        # Create tabs for each command
        tabs = st.tabs(["Score", "Infer Peptide", "Infer Protein", "Export TSV"])

        # --- SCORE TAB ---
        with tabs[0]:
            st.markdown("#### pyprophet score")
            st.caption(
                "Semi-supervised learning and error-rate estimation for MS1, MS2 and transition-level data."
            )

            col1, col2 = st.columns(2)
            score_in = col1.file_uploader(
                "Input (.osw/.parquet/.tsv)",
                type=["osw", "parquet", "tsv"],
                key="pyprophet_score_in",
            )
            score_out = col2.text_input(
                "Output file", key="pyprophet_score_out", placeholder="output.osw"
            )

            with st.expander("Scoring Options", expanded=True):
                col1, col2, col3 = st.columns(3)
                classifier = col1.selectbox(
                    "Classifier",
                    ["LDA", "SVM", "XGBoost", "HistGradientBoosting"],
                    index=0,
                    key="pyprophet_score_classifier",
                )
                autotune = col2.checkbox(
                    "Autotune",
                    value=False,
                    key="pyprophet_score_autotune",
                    help="Autotune hyperparameters for classifier",
                )
                level = col3.selectbox(
                    "Level",
                    ["ms1", "ms2", "ms1ms2", "transition", "alignment"],
                    index=1,
                    key="pyprophet_score_level",
                )

                col1, col2, col3 = st.columns(3)
                subsample_ratio = col1.number_input(
                    "Subsample ratio",
                    min_value=0.0,
                    max_value=1.0,
                    value=1.0,
                    step=0.05,
                    key="pyprophet_score_subsample",
                )
                threads = col2.number_input(
                    "Threads",
                    min_value=-1,
                    max_value=128,
                    value=1,
                    step=1,
                    key="pyprophet_score_threads",
                    help="-1 = all CPUs",
                )
                xeval_fraction = col3.number_input(
                    "Cross-validation fraction",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                    key="pyprophet_score_xeval_frac",
                )

                col1, col2, col3 = st.columns(3)
                xeval_num_iter = col1.number_input(
                    "Cross-validation iterations",
                    min_value=1,
                    value=10,
                    step=1,
                    key="pyprophet_score_xeval_num_iter",
                    help="Number of iterations for cross-validation",
                )
                ss_initial_fdr = col2.number_input(
                    "Initial FDR cutoff",
                    value=0.15,
                    step=0.01,
                    key="pyprophet_score_ss_initial_fdr",
                )
                ss_iteration_fdr = col3.number_input(
                    "Iteration FDR cutoff",
                    value=0.05,
                    step=0.01,
                    key="pyprophet_score_ss_iteration_fdr",
                )

                col1, col2, col3 = st.columns(3)
                ss_num_iter = col1.number_input(
                    "Semi-supervised iterations",
                    min_value=1,
                    value=10,
                    step=1,
                    key="pyprophet_score_ss_num_iter",
                    help="Number of iterations for semi-supervised learning",
                )
                ss_main_score = col2.text_input(
                    "Main score (semi-supervised)",
                    value="auto",
                    key="pyprophet_score_ss_main_score",
                    help="Score to use for bootstrap selection (e.g., 'var_xcorr_shape' or 'auto')",
                )
                apply_weights = col3.selectbox(
                    "Apply weights",
                    ["none", "classifier", "crosslinker"],
                    key="pyprophet_score_apply_weights",
                    help="Apply weights to classifier or crosslinker",
                )

                ss_scale_features = st.checkbox(
                    "Scale features (semi-supervised)",
                    value=False,
                    key="pyprophet_score_ss_scale_features",
                    help="Scale/standardize features before semi-supervised learning",
                )
                ss_score_filter = st.text_area(
                    "Score filter (semi-supervised)",
                    value="",
                    height=2,
                    key="pyprophet_score_ss_score_filter",
                    placeholder="e.g., var_ms1_xcorr_coelution,var_library_corr",
                    help="Comma-separated scores or predefined profiles (e.g., 'metabolomics')",
                )

                group_id = st.text_input(
                    "Group identifier",
                    value="group_id",
                    key="pyprophet_score_group_id",
                    help="Group ID for statistics calculation",
                )

            with st.expander("Advanced Options (pi0, lfdr, IPF)"):
                col1, col2 = st.columns(2)
                parametric = col1.checkbox(
                    "Parametric p-values", value=False, key="pyprophet_score_parametric"
                )
                pfdr = col2.checkbox(
                    "Compute pFDR",
                    value=False,
                    key="pyprophet_score_pfdr",
                    help="Positive FDR instead of FDR",
                )

                col1, col2, col3 = st.columns(3)
                pi0_method = col1.selectbox(
                    "pi0 method",
                    ["bootstrap", "smoother"],
                    key="pyprophet_score_pi0_method",
                )
                pi0_smooth_df = col2.number_input(
                    "pi0 smooth degrees of freedom",
                    value=3,
                    key="pyprophet_score_pi0_smooth_df",
                )
                pi0_smooth_log = col3.checkbox(
                    "pi0 smooth log", value=False, key="pyprophet_score_pi0_smooth_log"
                )

                col1, col2, col3 = st.columns(3)
                lfdr_truncate = col1.checkbox(
                    "lfdr truncate",
                    value=True,
                    key="pyprophet_score_lfdr_truncate",
                    help="Set lfdr > 1 to 1",
                )
                lfdr_monotone = col2.checkbox(
                    "lfdr monotone",
                    value=True,
                    key="pyprophet_score_lfdr_monotone",
                    help="Ensure non-decreasing lfdr",
                )
                lfdr_transform = col3.selectbox(
                    "lfdr transformation",
                    ["probit", "logit"],
                    key="pyprophet_score_lfdr_transform",
                )

                col1, col2 = st.columns(2)
                lfdr_adj = col1.number_input(
                    "lfdr smoothing bandwidth",
                    value=1.5,
                    step=0.1,
                    key="pyprophet_score_lfdr_adj",
                )
                lfdr_eps = col2.number_input(
                    "lfdr eps (p-value tails)",
                    value=1e-8,
                    format="%.2e",
                    key="pyprophet_score_lfdr_eps",
                )

                col1, col2, col3 = st.columns(3)
                ipf_max_rank = col1.number_input(
                    "IPF max peak group rank",
                    value=1,
                    min_value=1,
                    key="pyprophet_score_ipf_max_rank",
                )
                ipf_max_pep = col2.number_input(
                    "IPF max PEP",
                    value=0.7,
                    step=0.05,
                    key="pyprophet_score_ipf_max_pep",
                )
                ipf_iso_overlap = col3.number_input(
                    "IPF isotope overlap",
                    value=0.5,
                    step=0.05,
                    key="pyprophet_score_ipf_iso_overlap",
                )

                add_alignment = st.checkbox(
                    "Add alignment features",
                    value=False,
                    key="pyprophet_score_add_alignment",
                )
                glyco = st.checkbox(
                    "Glycopeptide scoring", value=False, key="pyprophet_score_glyco"
                )

            with st.expander("Output & Reporting"):
                col1, col2 = st.columns(2)
                color_palette = col1.selectbox(
                    "Color palette",
                    ["normal", "protan", "deutran", "tritan"],
                    key="pyprophet_score_color_palette",
                )
                main_score_report = col2.checkbox(
                    "Main score selection report",
                    value=False,
                    key="pyprophet_score_main_score_report",
                    help="Generate report for main score selection process",
                )

                col1, col2 = st.columns(2)
                test_mode = col1.checkbox(
                    "Test mode (fixed seed)",
                    value=False,
                    key="pyprophet_score_test_mode",
                )
                profile_mode = col2.checkbox(
                    "Enable profiling",
                    value=False,
                    key="pyprophet_score_profile_mode",
                    help="Memory allocation tracking (requires memray)",
                )

            score_params = {
                "in": getattr(score_in, "name", "") if score_in is not None else "",
                "out": score_out,
                "classifier": classifier,
                "autotune": autotune,
                "level": level,
                "subsample_ratio": subsample_ratio,
                "threads": threads,
                "xeval_fraction": xeval_fraction,
                "xeval_num_iter": xeval_num_iter,
                "ss_initial_fdr": ss_initial_fdr,
                "ss_iteration_fdr": ss_iteration_fdr,
                "ss_num_iter": ss_num_iter,
                "ss_main_score": ss_main_score,
                "ss_score_filter": ss_score_filter,
                "ss_scale_features": ss_scale_features,
                "apply_weights": apply_weights,
                "group_id": group_id,
                "parametric": parametric,
                "pfdr": pfdr,
                "pi0_method": pi0_method,
                "pi0_smooth_df": pi0_smooth_df,
                "pi0_smooth_log_pi0": pi0_smooth_log,
                "lfdr_truncate": lfdr_truncate,
                "lfdr_monotone": lfdr_monotone,
                "lfdr_transformation": lfdr_transform,
                "lfdr_adj": lfdr_adj,
                "lfdr_eps": lfdr_eps,
                "ipf_max_peakgroup_rank": ipf_max_rank,
                "ipf_max_peakgroup_pep": ipf_max_pep,
                "ipf_max_transition_isotope_overlap": ipf_iso_overlap,
                "add_alignment_features": add_alignment,
                "glyco": glyco,
                "color_palette": color_palette,
                "main_score_selection_report": main_score_report,
                "test": test_mode,
                "profile": profile_mode,
            }

            cols = st.columns(3)
            if cols[0].button("Save score params", key="score_save"):
                self.save_params_to_json("score", score_params)
                st.success("✓ Saved score parameters")
            if cols[1].button("Preview command", key="score_preview"):
                cmd = self.build_score_command(score_params)
                st.code(" ".join(cmd), language="bash")
            if cols[2].button("Run score", key="score_run"):
                cmd = self.build_score_command(score_params)
                if self.executor and self.logger:
                    self.logger.log(f"Starting: {' '.join(cmd)}")
                    success = self.executor.run_command(cmd)
                    if success:
                        st.success("✓ Score completed successfully")
                    else:
                        st.error("✗ Score failed—check logs")
                else:
                    st.info("Run button available only during workflow execution")

        # --- INFER PEPTIDE TAB ---
        with tabs[1]:
            st.markdown("#### pyprophet infer peptide")
            st.caption(
                "Infer peptides and conduct error-rate estimation in different contexts."
            )

            col1, col2, col3 = st.columns(3)
            pep_in = col1.file_uploader(
                "Input (.osw/.parquet)",
                type=["osw", "parquet"],
                key="pyprophet_infer_pep_in",
            )
            pep_out = col2.text_input(
                "Output file", key="pyprophet_infer_pep_out", placeholder="output.osw"
            )
            pep_context = col3.selectbox(
                "Context",
                ["run-specific", "experiment-wide", "global"],
                key="pyprophet_infer_pep_context",
            )

            with st.expander("Advanced Options"):
                col1, col2 = st.columns(2)
                pep_parametric = col1.checkbox(
                    "Parametric", value=False, key="pyprophet_infer_pep_parametric"
                )
                pep_pfdr = col2.checkbox(
                    "pFDR", value=False, key="pyprophet_infer_pep_pfdr"
                )

            pep_params = {
                "in": getattr(pep_in, "name", "") if pep_in is not None else "",
                "out": pep_out,
                "context": pep_context,
                "parametric": pep_parametric,
                "pfdr": pep_pfdr,
            }

            cols = st.columns(3)
            if cols[0].button("Save infer:peptide params", key="pep_save"):
                self.save_params_to_json("infer:peptide", pep_params)
                st.success("✓ Saved infer:peptide parameters")
            if cols[1].button("Preview command", key="pep_preview"):
                cmd = self.build_infer_command("peptide", pep_params)
                st.code(" ".join(cmd), language="bash")
            if cols[2].button("Run infer:peptide", key="pep_run"):
                cmd = self.build_infer_command("peptide", pep_params)
                if self.executor and self.logger:
                    self.logger.log(f"Starting: {' '.join(cmd)}")
                    success = self.executor.run_command(cmd)
                    if success:
                        st.success("✓ Infer:peptide completed successfully")
                    else:
                        st.error("✗ Infer:peptide failed—check logs")
                else:
                    st.info("Run button available only during workflow execution")

        # --- INFER PROTEIN TAB ---
        with tabs[2]:
            st.markdown("#### pyprophet infer protein")
            st.caption(
                "Infer proteins and conduct error-rate estimation in different contexts."
            )

            col1, col2, col3 = st.columns(3)
            prot_in = col1.file_uploader(
                "Input (.osw/.parquet)",
                type=["osw", "parquet"],
                key="pyprophet_infer_prot_in",
            )
            prot_out = col2.text_input(
                "Output file", key="pyprophet_infer_prot_out", placeholder="output.osw"
            )
            prot_context = col3.selectbox(
                "Context",
                ["run-specific", "experiment-wide", "global"],
                key="pyprophet_infer_prot_context",
            )

            with st.expander("Advanced Options"):
                col1, col2 = st.columns(2)
                prot_parametric = col1.checkbox(
                    "Parametric", value=False, key="pyprophet_infer_prot_parametric"
                )
                prot_pfdr = col2.checkbox(
                    "pFDR", value=False, key="pyprophet_infer_prot_pfdr"
                )

            prot_params = {
                "in": getattr(prot_in, "name", "") if prot_in is not None else "",
                "out": prot_out,
                "context": prot_context,
                "parametric": prot_parametric,
                "pfdr": prot_pfdr,
            }

            cols = st.columns(3)
            if cols[0].button("Save infer:protein params", key="prot_save"):
                self.save_params_to_json("infer:protein", prot_params)
                st.success("✓ Saved infer:protein parameters")
            if cols[1].button("Preview command", key="prot_preview"):
                cmd = self.build_infer_command("protein", prot_params)
                st.code(" ".join(cmd), language="bash")
            if cols[2].button("Run infer:protein", key="prot_run"):
                cmd = self.build_infer_command("protein", prot_params)
                if self.executor and self.logger:
                    self.logger.log(f"Starting: {' '.join(cmd)}")
                    success = self.executor.run_command(cmd)
                    if success:
                        st.success("✓ Infer:protein completed successfully")
                    else:
                        st.error("✗ Infer:protein failed—check logs")
                else:
                    st.info("Run button available only during workflow execution")

        # --- EXPORT TSV TAB ---
        with tabs[3]:
            st.markdown("#### pyprophet export tsv")
            st.caption("Export Proteomics/Peptidoform TSV/CSV tables")

            col1, col2 = st.columns(2)
            exp_in = col1.file_uploader(
                "Input (.osw)", type=["osw"], key="pyprophet_export_in"
            )
            exp_out = col2.text_input(
                "Output TSV path", key="pyprophet_export_out", placeholder="output.tsv"
            )

            with st.expander("Export Options", expanded=True):
                col1, col2, col3 = st.columns(3)
                exp_format = col1.selectbox(
                    "Format",
                    ["legacy_split", "legacy_merged"],
                    key="pyprophet_export_format",
                )
                exp_csv = col2.checkbox(
                    "CSV instead of TSV", value=False, key="pyprophet_export_csv"
                )
                exp_ipf = col3.selectbox(
                    "IPF",
                    ["disable", "peptidoform", "augmented"],
                    key="pyprophet_export_ipf",
                )

                col1, col2 = st.columns(2)
                max_rs_qvalue = col1.number_input(
                    "Max run-specific q-value",
                    value=0.05,
                    step=0.01,
                    key="pyprophet_export_max_rs_qvalue",
                )
                max_pep_qvalue = col2.number_input(
                    "Max peptide q-value",
                    value=0.01,
                    step=0.01,
                    key="pyprophet_export_max_pep_qvalue",
                )

            exp_params = {
                "in": getattr(exp_in, "name", "") if exp_in is not None else "",
                "out": exp_out,
                "format": exp_format,
                "csv": exp_csv,
                "ipf": exp_ipf,
                "max_rs_peakgroup_qvalue": max_rs_qvalue,
                "max_global_peptide_qvalue": max_pep_qvalue,
            }

            cols = st.columns(3)
            if cols[0].button("Save export:tsv params", key="exp_save"):
                self.save_params_to_json("export:tsv", exp_params)
                st.success("✓ Saved export:tsv parameters")
            if cols[1].button("Preview command", key="exp_preview"):
                cmd = self.build_export_command(exp_params)
                st.code(" ".join(cmd), language="bash")
            if cols[2].button("Run export:tsv", key="exp_run"):
                cmd = self.build_export_command(exp_params)
                if self.executor and self.logger:
                    self.logger.log(f"Starting: {' '.join(cmd)}")
                    success = self.executor.run_command(cmd)
                    if success:
                        st.success("✓ Export:tsv completed successfully")
                    else:
                        st.error("✗ Export:tsv failed—check logs")
                else:
                    st.info("Run button available only during workflow execution")

    def get_commands_for_workflow(self) -> list:
        """
        Get all configured PyProphet commands for integration into workflow runner.
        Returns a list of command lists ready for execution.
        """
        data = self.pm.get_parameters_from_json()
        pyprophet_params = data.get("pyprophet", {})
        commands = []

        if "score" in pyprophet_params:
            cmd = self.build_score_command(pyprophet_params["score"])
            if pyprophet_params["score"].get("in"):
                commands.append(cmd)

        if "infer:peptide" in pyprophet_params:
            cmd = self.build_infer_command("peptide", pyprophet_params["infer:peptide"])
            if pyprophet_params["infer:peptide"].get("in"):
                commands.append(cmd)

        if "infer:protein" in pyprophet_params:
            cmd = self.build_infer_command("protein", pyprophet_params["infer:protein"])
            if pyprophet_params["infer:protein"].get("in"):
                commands.append(cmd)

        if "export:tsv" in pyprophet_params:
            cmd = self.build_export_command(pyprophet_params["export:tsv"])
            if pyprophet_params["export:tsv"].get("in"):
                commands.append(cmd)

        return commands
