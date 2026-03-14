import sys
import shutil
import streamlit as st
from pathlib import Path
from .WorkflowManager import WorkflowManager


class EasyPQPWorkflow(WorkflowManager):
    """Workflow wrapper for running EasyPQP insilico-library via the workflow system.

    This class provides upload/configure/execution/result fragments compatible
    with the existing `WorkflowManager` and `StreamlitUI` utilities.
    """

    def __init__(self) -> None:
        super().__init__("EasyPQP InSilico", st.session_state["workspace"])

    def upload(self) -> None:
        # Upload widgets: FASTA, optional UniMod XML, optional training TSV
        t = st.tabs(["Files"])
        with t[0]:
            self.ui.upload_widget(key="fasta", file_types="fasta", name="FASTA File")
            self.ui.upload_widget(
                key="unimod", file_types="xml", name="UniMod XML (optional)"
            )
            self.ui.upload_widget(
                key="train-data", file_types="tsv", name="Training Data (TSV, optional)"
            )

    @st.fragment
    def configure(self) -> None:
        # Select the uploaded files and set simple parameters
        self.ui.select_input_file("fasta", multiple=False)
        self.ui.select_input_file("unimod", multiple=False)
        self.ui.select_input_file("train-data", multiple=False)

        # Basic EasyPQP parameters saved via ParameterManager
        self.ui.input_widget(
            "output_file", "./easypqp_insilico_library.tsv", "Output File"
        )
        self.ui.input_widget("generate_decoys", False, "Generate Decoys")
        self.ui.input_widget("decoy_tag", "rev_", "Decoy Tag")
        self.ui.input_widget("precursor_charge", "2,3", "Precursor Charge(s)")
        self.ui.input_widget("max_fragment_charge", 2, "Max Fragment Charge")
        self.ui.input_widget("instrument", "QE", "Instrument")
        self.ui.input_widget("nce", 20, "NCE")
        self.ui.input_widget("batch_size", 10, "Batch Size")
        self.ui.input_widget("threads", 0, "Threads")
        self.ui.input_widget("write_report", True, "Write HTML Report")
        self.ui.input_widget("parquet_output", False, "Parquet Output")

    def execution(self) -> bool:
        """Execute EasyPQP insilico-library generation.

        Preferred method: uses config file if available
        Fallback: builds command from individual parameters
        """
        self.logger.log("=" * 80)
        self.logger.log("EASYPQP WORKFLOW EXECUTION STARTED")
        self.logger.log("=" * 80)

        # Find the easypqp executable
        # This should be in the venv's bin directory
        easypqp_cmd = shutil.which("easypqp")
        if not easypqp_cmd:
            self.logger.log("❌ ERROR: 'easypqp' command not found in PATH")
            self.logger.log(f"Searched in PATH. Current sys.prefix: {sys.prefix}")
            return False

        self.logger.log(f"Found easypqp at: {easypqp_cmd}")

        # Validate required params
        self.logger.log("Loading parameters from JSON...")
        params = self.parameter_manager.get_parameters_from_json()
        self.logger.log(f"Parameters loaded. Keys: {list(params.keys())}")

        # Prepare results directory
        results_dir = Path(self.workflow_dir, "results", "insilico")
        results_dir.mkdir(parents=True, exist_ok=True)
        self.logger.log(f"Results directory ready: {results_dir}")

        # Check if a config file was provided (preferred method)
        config_file = params.get("config_file")
        self.logger.log(f"Config file from params: {config_file}")

        if config_file:
            config_path = Path(config_file)
            self.logger.log(f"Config path exists check: {config_path.exists()}")
            self.logger.log(
                f"Config file content size: {config_path.stat().st_size if config_path.exists() else 'N/A'}"
            )

            if config_path.exists():
                self.logger.log(f"Using merged config file: {config_file}")
                output_file = params.get(
                    "output_file",
                    str(Path(results_dir, "easypqp_insilico_library.tsv")),
                )
                self.logger.log(f"Output file: {output_file}")

                # Build command using config file
                # Use the easypqp executable found above
                cmd = [
                    easypqp_cmd,
                    "insilico-library",
                    "--config",
                    str(config_file),
                    "--output_file",
                    str(output_file),
                ]

                # Run command via executor
                cmd_str = " ".join(cmd)
                self.logger.log(f"Full command: {cmd_str}")
                self.logger.log("Spawning subprocess...")
                success = self.executor.run_command(cmd)

                if not success:
                    self.logger.log("❌ EasyPQP execution failed (non-zero exit code).")
                    return False

                self.logger.log("✅ EasyPQP execution finished successfully.")
                return True
            else:
                self.logger.log(f"⚠️ Config file not found at: {config_file}")
        else:
            self.logger.log("⚠️ No config_file in parameters")

        # Fallback: build command from individual parameters
        fasta_param = params.get("fasta")
        if not fasta_param:
            self.logger.log("ERROR: No FASTA selected and no config file provided.")
            return False

        # Resolve fasta file path using FileManager
        in_fasta = self.file_manager.get_files(fasta_param)
        fasta_path = in_fasta[0]

        output_file = params.get(
            "output_file", str(Path(results_dir, "easypqp_insilico_library.tsv"))
        )

        # Build command
        # Use 'easypqp' CLI entry point (not python -m easypqp.main)
        cmd = [
            easypqp_cmd,
            "insilico-library",
            "--fasta",
            str(fasta_path),
            "--output_file",
            str(output_file),
        ]

        # Database parameters
        if params.get("generate_decoys"):
            cmd += ["--generate_decoys"]
        else:
            cmd += ["--no-generate_decoys"]

        cmd += ["--decoy_tag", str(params.get("decoy_tag", "rev_"))]

        # Precursor charge (handle comma-separated list)
        precursor_charge = params.get("precursor_charge", "2,3")
        cmd += ["--precursor_charge", str(precursor_charge)]

        # Fragment parameters
        cmd += ["--max_fragment_charge", str(params.get("max_fragment_charge", 2))]
        cmd += ["--min_transitions", str(params.get("min_transitions", 5))]
        cmd += ["--max_transitions", str(params.get("max_transitions", 25))]
        cmd += [
            "--fragmentation_model",
            str(params.get("fragmentation_model", "AlphaPeptDeep")),
        ]

        # Allowed fragment types (comma-separated)
        allowed_fragment_types = params.get("allowed_fragment_types", "b,y")
        cmd += ["--allowed_fragment_types", str(allowed_fragment_types)]

        # Output parameters
        if params.get("write_report"):
            cmd += ["--write_report"]
        else:
            cmd += ["--no-write_report"]

        if params.get("parquet_output"):
            cmd += ["--parquet_output"]
        else:
            cmd += ["--no-parquet_output"]

        # RT parameters
        if "rt_scale" in params:
            cmd += ["--rt_scale", str(params.get("rt_scale", 100.0))]

        # Deep Learning parameters
        cmd += ["--instrument", str(params.get("instrument", "QE"))]
        cmd += ["--nce", str(params.get("nce", 20))]
        cmd += ["--batch_size", str(params.get("batch_size", 10))]

        # Fine-tuning parameters
        if params.get("fine_tune"):
            cmd += ["--fine_tune"]
            if "train_data_path" in params and params.get("train_data_path"):
                cmd += ["--train_data_path", str(params.get("train_data_path"))]
        else:
            cmd += ["--no-fine_tune"]

        if params.get("save_model"):
            cmd += ["--save_model"]
        else:
            cmd += ["--no-save_model"]

        # UniMod parameters
        if params.get("unimod_annotation"):
            cmd += ["--unimod_annotation"]
        else:
            cmd += ["--no-unimod_annotation"]

        if "max_delta_unimod" in params:
            cmd += ["--max_delta_unimod", str(params.get("max_delta_unimod", 0.02))]

        if params.get("enable_unannotated"):
            cmd += ["--enable_unannotated"]
        else:
            cmd += ["--no-enable_unannotated"]

        # Optional UniMod XML file
        if "unimodfile" in params and params.get("unimodfile"):
            cmd += ["--unimod_xml_path", str(params.get("unimodfile"))]

        # Threads (may be None; coerce to int with safe fallback to 0)
        threads_raw = params.get("threads", 0)
        try:
            threads = int(threads_raw) if threads_raw is not None else 0
        except (TypeError, ValueError):
            threads = 0
        if threads > 0:
            cmd += ["--threads", str(threads)]

        # Run command via executor (this runs inside the workflow process and logs to workflow logger)
        self.logger.log(
            f"Launching EasyPQP insilico-library with command: {' '.join(cmd)}"
        )
        success = self.executor.run_command(cmd)
        if not success:
            self.logger.log("EasyPQP execution failed.")
            return False

        self.logger.log("EasyPQP execution finished successfully.")
        return True

    def results(self) -> None:
        # Display output file location and simple download if present
        output_candidates = list(
            Path(self.workflow_dir, "results", "insilico").glob("*.tsv")
        )
        if output_candidates:
            out = output_candidates[0]
            st.markdown(f"**Output:** {out}")
            with open(out, "rb") as f:
                st.download_button("⬇️ Download library", data=f, file_name=out.name)
        else:
            st.info(
                "No library output found yet. Run the workflow to generate outputs."
            )
