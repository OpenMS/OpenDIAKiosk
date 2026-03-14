import shutil
import sys
from pathlib import Path

import streamlit as st

from .WorkflowManager import WorkflowManager


class OpenSwathDecoyGeneratorWorkflow(WorkflowManager):
    """Workflow wrapper for running OpenSwathDecoyGenerator."""

    def __init__(self) -> None:
        super().__init__("OpenSwath Decoy Generator", st.session_state["workspace"])

    def execution(self) -> bool:
        self.logger.log("=" * 80)
        self.logger.log("OPENSWATH DECOY GENERATOR EXECUTION STARTED")
        self.logger.log("=" * 80)

        decoy_cmd = shutil.which("OpenSwathDecoyGenerator")
        if not decoy_cmd:
            self.logger.log("ERROR: 'OpenSwathDecoyGenerator' command not found in PATH")
            self.logger.log(f"Searched in PATH. Current sys.prefix: {sys.prefix}")
            return False

        self.logger.log(f"Found OpenSwathDecoyGenerator at: {decoy_cmd}")

        params = self.parameter_manager.get_parameters_from_json()
        self.logger.log(f"Parameters loaded. Keys: {list(params.keys())}")

        results_dir = Path(self.workflow_dir, "results", "decoy")
        results_dir.mkdir(parents=True, exist_ok=True)

        in_param = params.get("input_library")
        if not in_param:
            self.logger.log("ERROR: No input library selected.")
            return False

        in_files = self.file_manager.get_files(in_param)
        if not in_files:
            self.logger.log(f"ERROR: Could not find input library file: {in_param}")
            return False

        input_file = in_files[0]
        output_file = params.get("output_file", str(Path(results_dir, "openswath_decoys.tsv")))

        cmd = [
            decoy_cmd,
            "-in",
            str(input_file),
            "-out",
            str(output_file),
        ]

        # Only pass type options when explicitly chosen.
        input_type = params.get("input_type", "auto")
        if input_type != "auto":
            cmd += ["-in_type", str(input_type)]

        output_type = params.get("output_type", "auto")
        if output_type != "auto":
            cmd += ["-out_type", str(output_type)]

        if "method" in params:
            cmd += ["-method", str(params.get("method", "shuffle"))]

        if "decoy_tag" in params:
            cmd += ["-decoy_tag", str(params.get("decoy_tag", "DECOY_"))]

        if "switchKR" in params:
            cmd += ["-switchKR", "true" if bool(params.get("switchKR")) else "false"]

        if "min_decoy_fraction" in params:
            cmd += ["-min_decoy_fraction", str(params.get("min_decoy_fraction"))]

        if "aim_decoy_fraction" in params:
            cmd += ["-aim_decoy_fraction", str(params.get("aim_decoy_fraction"))]

        if "shuffle_max_attempts" in params:
            cmd += ["-shuffle_max_attempts", str(params.get("shuffle_max_attempts"))]

        if "shuffle_sequence_identity_threshold" in params:
            cmd += [
                "-shuffle_sequence_identity_threshold",
                str(params.get("shuffle_sequence_identity_threshold")),
            ]

        if "shift_precursor_mz_shift" in params:
            cmd += ["-shift_precursor_mz_shift", str(params.get("shift_precursor_mz_shift"))]

        if "shift_product_mz_shift" in params:
            cmd += ["-shift_product_mz_shift", str(params.get("shift_product_mz_shift"))]

        if "product_mz_threshold" in params:
            cmd += ["-product_mz_threshold", str(params.get("product_mz_threshold"))]

        if "allowed_fragment_types" in params:
            cmd += ["-allowed_fragment_types", str(params.get("allowed_fragment_types"))]

        if "allowed_fragment_charges" in params:
            cmd += ["-allowed_fragment_charges", str(params.get("allowed_fragment_charges"))]

        if params.get("enable_detection_specific_losses"):
            cmd += ["-enable_detection_specific_losses"]

        if params.get("enable_detection_unspecific_losses"):
            cmd += ["-enable_detection_unspecific_losses"]

        if params.get("separate"):
            cmd += ["-separate"]

        threads_raw = params.get("threads", 1)
        try:
            threads = int(threads_raw) if threads_raw is not None else 1
        except (TypeError, ValueError):
            threads = 1
        if threads > 0:
            cmd += ["-threads", str(threads)]

        self.logger.log(f"Full command: {' '.join(cmd)}")
        success = self.executor.run_command(cmd)

        if not success:
            self.logger.log("ERROR: OpenSwathDecoyGenerator execution failed.")
            return False

        self.logger.log("OpenSwathDecoyGenerator execution finished successfully.")
        return True
