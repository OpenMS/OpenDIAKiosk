"""
src/workflow/OpenSwathWorkflow.py

WorkflowManager subclass that runs the full OpenSWATH pipeline.

Configuration is read from:
  workspace/params.json          — saved by openswath_configuration.py
  workspace/ini/*.ini            — OpenMS TOPP tool descriptors

Pipeline:
  1. [Optional] EasyPQP in-silico    → easypqp_insilico_library.tsv
  2.            OpenSwathAssayGenerator → openswathassay_targets.tsv
  3.            OpenSwathDecoyGenerator → openswath_targets_and_decoys.pqp
  4.            OpenSwathWorkflow       → openswath_results.osw
  5.            PyProphet               → openswath_results.tsv
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import streamlit as st

from src.common.workspace_files import (
    archive_needs_refresh,
    build_zip_archive,
    total_size_label,
)
from .ParameterManager import ParameterManager
from .WorkflowManager import WorkflowManager


# Fixed file-names for intermediate and final outputs
_EPQP_OUT = "easypqp_insilico_library.tsv"
_OSAG_OUT = "openswathassay_targets.tsv"
_OSDG_OUT = "openswath_targets_and_decoys.pqp"
_OSW_OUT = "openswath_results.osw"
_PY_OUT = "openswath_results.tsv"
_OSW_XIC_OUT = "openswath_results_chromatograms.xic"
_OSW_XIM_OUT = "openswath_results_mobilograms.xim"
_OSW_DEBUG_IM_OUT = "_debug_calibration_im.txt"
_OSW_DEBUG_MZ_OUT = "_debug_calibration_mz.txt"
_OSW_DEBUG_IRT_TRAFO_OUT = "_debug_calibration_irt.trafoXML"
_OSW_DEBUG_IRT_MZML_OUT = "_debug_calibration_irt_chrom.mzML"
_OSW_CACHE_DIR = "openswath-workflow-temp"
_PY_MATRIX_OUTS = {
    "precursor": "openswath_results.precursor.tsv",
    "peptide": "openswath_results.peptide.tsv",
    "protein": "openswath_results.protein.tsv",
}
_PY_INFER_CONTEXTS = ["global", "experiment-wide", "run-specific"]

_RUN_MODE_KEY = "openswath_run_mode"
_RUN_RESUME_STEP_KEY = "openswath_resume_step"
_RUN_MODE_FRESH = "fresh"
_RUN_MODE_RESUME = "resume"

_STEP_LABELS = {
    "easypqp": "EasyPQP",
    "osag": "OpenSwathAssayGenerator",
    "osdg": "OpenSwathDecoyGenerator",
    "openswathworkflow": "OpenSwathWorkflow",
    "pyprophet": "PyProphet",
}


class OpenSwathWorkflow(WorkflowManager):
    """
    Full OpenSWATH pipeline as a WorkflowManager.

    Execution reads all configuration from:
      * ``workspace/params.json``  (written by the configuration page)
      * ``workspace/ini/*.ini``    (TOPP tool INI files)

    Usage (in a content page)::

        wf = OpenSwathWorkflow()
        wf.show_execution_section()
        wf.show_results_section()
    """

    def __init__(self) -> None:
        workspace = st.session_state["workspace"]
        super().__init__("OpenSwath Workflow", workspace)
        # The configuration page saves params relative to workspace_dir,
        # not to our workflow_dir.  Keep a pointer to the shared params file.
        self._workspace_dir = Path(workspace)
        self._workspace_params_file = self._workspace_dir / "params.json"
        # Shared INI dir (same as config page uses)
        self._shared_ini_dir = self._workspace_dir / "ini"
        self._workspace_parameter_manager = ParameterManager(
            self._workspace_dir, workflow_name=self.name
        )
        self.executor.parameter_manager = self._workspace_parameter_manager

    # ------------------------------------------------------------------
    # Helpers

    def _ensure_workspace_context(self) -> None:
        """
        Rebuild shared workspace paths when the object was created outside the
        normal Streamlit page flow (for example queue workers).
        """
        if not hasattr(self, "_workspace_dir"):
            workflow_dir = Path(self.workflow_dir)
            self._workspace_dir = workflow_dir.parent
            self._workspace_params_file = self._workspace_dir / "params.json"
            self._shared_ini_dir = self._workspace_dir / "ini"

        if not hasattr(self, "_workspace_parameter_manager"):
            self._workspace_parameter_manager = ParameterManager(
                self._workspace_dir, workflow_name=self.name
            )

        self.executor.parameter_manager = self._workspace_parameter_manager

    def _load_workspace_params(self) -> dict:
        """Load params.json saved by the configuration page."""
        self._ensure_workspace_context()
        if self._workspace_params_file.exists():
            try:
                with open(self._workspace_params_file, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_workspace_param(self, name: str, value) -> None:
        self._ensure_workspace_context()
        params = self._load_workspace_params()
        params[name] = value
        self._workspace_params_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self._workspace_params_file, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=4)

    def _ini(self, tool: str) -> str | None:
        """Return the INI path for *tool* if it exists in the shared ini dir."""
        self._ensure_workspace_context()
        p = self._shared_ini_dir / f"{tool}.ini"
        return str(p) if p.exists() else None

    def _workspace_file(self, *parts) -> Path:
        self._ensure_workspace_context()
        return self._workspace_dir.joinpath(*parts)

    def _load_tool_config(self, *parts) -> dict:
        path = self._workspace_file(*parts)
        if not path.exists():
            return {}
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _load_pyprophet_schema(self, schema_name: str) -> dict:
        repo_root = Path(__file__).resolve().parents[2]
        schema_path = (
            repo_root
            / "assets"
            / "common-tool-descriptors"
            / "pyprophet"
            / schema_name
        )
        if not schema_path.exists():
            return {}
        try:
            with open(schema_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _append_pyprophet_config_options(
        self,
        command: list[str],
        config: dict,
        schema_name: str,
        exclude_keys: set[str] | None = None,
    ) -> list[str]:
        exclude_keys = exclude_keys or set()
        schema = self._load_pyprophet_schema(schema_name)
        option_defs: dict[str, dict] = {}
        for section in schema.get("sections", {}).values():
            option_defs.update(section.get("options", {}))

        for opt_name, opt_def in option_defs.items():
            if opt_name in exclude_keys or opt_name not in config:
                continue
            value = config.get(opt_name)
            if value in (None, ""):
                continue

            value_type = opt_def.get("value_type")
            cli_flags = opt_def.get("cli_flags", [])
            if not cli_flags:
                continue

            if value_type == "boolean":
                bool_value = self._coerce_config_bool(value)
                positive_flag = next(
                    (flag for flag in cli_flags if not flag.startswith("--no-")),
                    None,
                )
                negative_flag = next(
                    (flag for flag in cli_flags if flag.startswith("--no-")),
                    None,
                )
                selected_flag = positive_flag if bool_value else negative_flag
                if selected_flag:
                    command.append(selected_flag)
                continue

            if value_type == "boolean_flag":
                if self._coerce_config_bool(value):
                    command.append(cli_flags[0])
                continue

            if isinstance(value_type, str) and value_type.startswith("array["):
                array_values = self._coerce_config_array(value)
                if not array_values:
                    continue
                command.append(cli_flags[0])
                command.extend(array_values)
                continue

            if isinstance(value, list):
                if not value:
                    continue
                command.append(cli_flags[0])
                command.extend(str(item) for item in value)
                continue

            command.extend([cli_flags[0], str(value)])

        return command

    @staticmethod
    def _coerce_config_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @staticmethod
    def _coerce_config_array(value) -> list[str]:
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            return [item for item in re.split(r"[\s,]+", stripped) if item]
        return [str(value)] if value is not None else []

    def _selected_pyprophet_contexts(self, config: dict) -> list[str]:
        contexts = config.get("contexts", config.get("context", ["global"]))
        if isinstance(contexts, str):
            contexts = [contexts]
        if not isinstance(contexts, list):
            contexts = []
        selected = [context for context in _PY_INFER_CONTEXTS if context in contexts]
        return selected or ["global"]

    def _debug_output_matches(self, results_dir: Path) -> list[Path]:
        matches: dict[str, Path] = {}
        for suffix in (
            _OSW_DEBUG_IM_OUT,
            _OSW_DEBUG_MZ_OUT,
            _OSW_DEBUG_IRT_TRAFO_OUT,
            _OSW_DEBUG_IRT_MZML_OUT,
        ):
            for path in results_dir.glob(f"*{suffix}"):
                matches[str(path.resolve())] = path
        return sorted(matches.values(), key=lambda path: path.name)

    def _pipeline_steps(self, use_easypqp: bool) -> list[str]:
        steps: list[str] = []
        if use_easypqp:
            steps.append("easypqp")
        steps.extend(["osag", "osdg", "openswathworkflow", "pyprophet"])
        return steps

    def _step_output_paths(self, results_dir: Path) -> dict[str, Path]:
        return {
            "easypqp": results_dir / "insilico" / _EPQP_OUT,
            "osag": results_dir / "osag" / _OSAG_OUT,
            "osdg": results_dir / "osdg" / _OSDG_OUT,
            "openswathworkflow": results_dir / _OSW_OUT,
            "pyprophet": results_dir / _PY_OUT,
        }

    def _cleanup_targets(self, results_dir: Path) -> dict[str, list[Path]]:
        return {
            "easypqp": [results_dir / "insilico"],
            "osag": [results_dir / "osag"],
            "osdg": [results_dir / "osdg"],
            "openswathworkflow": [
                results_dir / _OSW_OUT,
                results_dir / _OSW_XIC_OUT,
                results_dir / _OSW_XIM_OUT,
                results_dir / _PY_OUT,
                *[results_dir / name for name in _PY_MATRIX_OUTS.values()],
            ],
            "pyprophet": [
                results_dir / _PY_OUT,
                *[results_dir / name for name in _PY_MATRIX_OUTS.values()],
            ],
        }

    def _normalize_run_settings(
        self, cfg: dict, use_easypqp: bool
    ) -> tuple[str, str, list[str]]:
        steps = self._pipeline_steps(use_easypqp)
        mode = cfg.get(_RUN_MODE_KEY, _RUN_MODE_FRESH)
        if mode not in (_RUN_MODE_FRESH, _RUN_MODE_RESUME):
            mode = _RUN_MODE_FRESH

        resume_step = cfg.get(_RUN_RESUME_STEP_KEY, steps[0])
        if resume_step not in steps:
            resume_step = steps[0]

        return mode, resume_step, steps

    def _missing_resume_prerequisites(
        self, start_step: str, use_easypqp: bool, results_dir: Path
    ) -> list[tuple[str, Path]]:
        steps = self._pipeline_steps(use_easypqp)
        outputs = self._step_output_paths(results_dir)
        start_index = steps.index(start_step)
        missing: list[tuple[str, Path]] = []
        for step in steps[:start_index]:
            path = outputs[step]
            if not path.exists():
                missing.append((_STEP_LABELS[step], path))
        return missing

    def _clear_results_from_step(
        self, start_step: str, use_easypqp: bool, results_dir: Path
    ) -> None:
        steps = self._pipeline_steps(use_easypqp)
        cleanup_targets = self._cleanup_targets(results_dir)
        start_index = steps.index(start_step)

        seen: set[Path] = set()
        for step in steps[start_index:]:
            for path in cleanup_targets.get(step, []):
                if path in seen or not path.exists():
                    continue
                seen.add(path)
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink(missing_ok=True)

        if "openswathworkflow" in steps[start_index:]:
            for path in self._debug_output_matches(results_dir):
                path.unlink(missing_ok=True)

    def _persist_execution_preference(self, param_name: str, session_key: str) -> None:
        self._save_workspace_param(param_name, st.session_state.get(session_key))

    def should_reset_results_dir(self) -> bool:
        cfg = self._load_workspace_params()
        use_easypqp = (
            cfg.get("osag_input_mode", "Use existing transition list(s)")
            == "Generate from FASTA (predict transitions)"
        )
        mode, _, _ = self._normalize_run_settings(cfg, use_easypqp)
        return mode == _RUN_MODE_FRESH

    def show_execution_section(self) -> None:
        cfg = self._load_workspace_params()
        use_easypqp = (
            cfg.get("osag_input_mode", "Use existing transition list(s)")
            == "Generate from FASTA (predict transitions)"
        )
        mode, resume_step, steps = self._normalize_run_settings(cfg, use_easypqp)

        mode_key = f"{self.workflow_dir.stem}-execution-mode"
        step_key = f"{self.workflow_dir.stem}-execution-resume-step"
        st.session_state[mode_key] = mode
        st.session_state[step_key] = resume_step

        st.markdown("**Rerun Behavior**")
        st.caption(
            "Start from scratch deletes previous workflow outputs. "
            "Reuse mode keeps successful upstream results and reruns the selected step and everything after it."
        )

        st.radio(
            "Execution mode",
            options=[_RUN_MODE_FRESH, _RUN_MODE_RESUME],
            format_func=lambda value: {
                _RUN_MODE_FRESH: "Start from scratch",
                _RUN_MODE_RESUME: "Reuse previous successful outputs",
            }[value],
            key=mode_key,
            horizontal=True,
            on_change=self._persist_execution_preference,
            args=(_RUN_MODE_KEY, mode_key),
        )

        if st.session_state[mode_key] == _RUN_MODE_RESUME:
            st.selectbox(
                "Resume at step",
                options=steps,
                format_func=lambda value: _STEP_LABELS[value],
                key=step_key,
                on_change=self._persist_execution_preference,
                args=(_RUN_RESUME_STEP_KEY, step_key),
            )

            selected_step = st.session_state[step_key]
            missing = self._missing_resume_prerequisites(
                selected_step, use_easypqp, self.workflow_dir / "results"
            )
            if missing:
                missing_text = ", ".join(
                    f"{label} (`{path.relative_to(self.workflow_dir)}`)"
                    for label, path in missing
                )
                st.warning(
                    f"Resume from {_STEP_LABELS[selected_step]} requires existing outputs from prior steps: {missing_text}."
                )
            else:
                reused_steps = steps[: steps.index(selected_step)]
                if reused_steps:
                    st.caption(
                        "Upstream outputs to reuse: "
                        + ", ".join(_STEP_LABELS[step] for step in reused_steps)
                    )

        super().show_execution_section()

    def _mzml_paths(self) -> list[str]:
        """Resolve full paths for all mzML files in the workspace."""
        self._ensure_workspace_context()
        mzml_dir = self._workspace_dir / "mzML-files"
        paths: list[str] = []
        if mzml_dir.exists():
            for p in mzml_dir.iterdir():
                if p.is_file() and "external_files.txt" not in p.name:
                    paths.append(str(p))
            ext_file = mzml_dir / "external_files.txt"
            if ext_file.exists():
                paths += [
                    l.strip() for l in ext_file.read_text().splitlines() if l.strip()
                ]
        return paths

    # ------------------------------------------------------------------
    # Step implementations

    def _run_easypqp(self, cfg: dict, results_dir: Path) -> bool:
        """Build and execute the easypqp insilico-library command."""
        exe = shutil.which("easypqp")
        if not exe:
            self.logger.log("❌ 'easypqp' not found in PATH.")
            return False

        fasta_name = cfg.get("fasta", "")
        if not fasta_name:
            self.logger.log("❌ No FASTA file configured for EasyPQP.")
            return False

        fasta_path = self._workspace_file("input-files", "fasta", fasta_name)
        if not fasta_path.exists():
            self.logger.log(f"❌ FASTA file not found: {fasta_path}")
            return False

        out_dir = results_dir / "insilico"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / _EPQP_OUT
        # Prefer using a saved config file in the workspace tools-configs
        config_path = self._workspace_file(
            "tools-configs", "easypqp", "easypqp_insilico.json"
        )
        if config_path.exists():
            self.logger.log(f"Using EasyPQP config from workspace: {config_path}")
            cmd = [
                exe,
                "insilico-library",
                "--fasta",
                str(fasta_path),
                "--output_file",
                str(out_file),
                "--config",
                str(config_path),
            ]
            return self.executor.run_command(cmd)

    def _run_osag(self, in_file: str, out_file: str) -> bool:
        """Run OpenSwathAssayGenerator via executor.run_topp()."""
        ini = self._ini("OpenSwathAssayGenerator")
        if ini is None:
            self.logger.log(
                "⚠️ OpenSwathAssayGenerator.ini not found — using tool defaults."
            )

        # Temporarily point the shared ini dir into the parameter manager's ini dir
        # so that run_topp picks it up automatically (it appends -ini <path>).
        # We copy the shared ini into the workflow ini dir if needed.
        self._ensure_ini_in_workflow("OpenSwathAssayGenerator")

        return self.executor.run_topp(
            "OpenSwathAssayGenerator",
            input_output={
                "in": [in_file],
                "out": [out_file],
            },
            include_saved_params=False,
        )

    def _run_osdg(self, in_file: str, out_file: str) -> bool:
        """Run OpenSwathDecoyGenerator via executor.run_topp()."""
        self._ensure_ini_in_workflow("OpenSwathDecoyGenerator")
        return self.executor.run_topp(
            "OpenSwathDecoyGenerator",
            input_output={
                "in": [in_file],
                "out": [out_file],
            },
            include_saved_params=False,
        )

    def _run_openswath(
        self,
        cfg: dict,
        tr_file: str,
        mzml_paths: list[str],
        out_file: str,
        results_dir: Path,
    ) -> bool:
        """Run OpenSwathWorkflow via executor.run_topp()."""
        self._ensure_ini_in_workflow("OpenSwathWorkflow")
        custom_params: dict[str, str] = {}
        results_dir_abs = results_dir.resolve()

        osw_cfg = cfg.get("OpenSwathWorkflow", {})
        if osw_cfg.get("out_chrom"):
            custom_params["out_chrom"] = str(results_dir_abs / _OSW_XIC_OUT)
        if osw_cfg.get("out_mobilogram"):
            custom_params["out_mobilogram"] = str(results_dir_abs / _OSW_XIM_OUT)
        if osw_cfg.get("Calibration:MassIMCorrection:debug_im_file"):
            custom_params["Calibration:MassIMCorrection:debug_im_file"] = (
                _OSW_DEBUG_IM_OUT
            )
        if osw_cfg.get("Calibration:MassIMCorrection:debug_mz_file"):
            custom_params["Calibration:MassIMCorrection:debug_mz_file"] = (
                _OSW_DEBUG_MZ_OUT
            )
        if osw_cfg.get("Debugging:irt_trafo"):
            custom_params["Debugging:irt_trafo"] = _OSW_DEBUG_IRT_TRAFO_OUT
        if osw_cfg.get("Debugging:irt_mzml"):
            custom_params["Debugging:irt_mzml"] = _OSW_DEBUG_IRT_MZML_OUT

        read_option = osw_cfg.get("readOptions")
        if read_option:
            custom_params["readOptions"] = read_option
        if read_option in {"cache", "cacheWorkingInMemory"}:
            cache_dir = self._workspace_file(_OSW_CACHE_DIR)
            cache_dir.mkdir(parents=True, exist_ok=True)
            custom_params["tempDirectory"] = str(cache_dir.resolve())
            self.logger.log(f"Using workspace cache directory: {cache_dir.resolve()}")

        command_inputs = {
            "in": [[str(Path(path).resolve()) for path in mzml_paths]],
            "tr": [str(Path(tr_file).resolve())],
            "out_features": [str(Path(out_file).resolve())],
        }

        return self.executor.run_topp(
            "OpenSwathWorkflow",
            input_output=command_inputs,
            custom_params=custom_params,
            cwd=str(results_dir_abs),
            include_saved_params=False,
        )

    def _run_pyprophet(self, osw_in: str, tsv_out: str, results_dir: Path) -> bool:
        """Run pyprophet score → infer peptide → infer protein → exports."""
        exe = shutil.which("pyprophet")
        if not exe:
            self.logger.log("❌ 'pyprophet' not found in PATH.")
            return False

        tsv_cfg = self._load_tool_config(
            "tools-configs", "pyprophet", "pyprophet_export_tsv_config.json"
        )
        tsv_cmd = [exe, "export", "tsv", "--in", osw_in, "--out", tsv_out]
        if tsv_cfg.get("format"):
            tsv_cmd += ["--format", str(tsv_cfg["format"])]
        if tsv_cfg.get("csv"):
            tsv_cmd.append("--csv")
        tsv_cmd.append(
            "--transition_quantification"
            if tsv_cfg.get("transition_quantification", True)
            else "--no-transition_quantification"
        )
        if tsv_cfg.get("max_transition_pep") is not None:
            tsv_cmd += ["--max_transition_pep", str(tsv_cfg["max_transition_pep"])]
        if tsv_cfg.get("ipf"):
            tsv_cmd += ["--ipf", str(tsv_cfg["ipf"])]
        if tsv_cfg.get("ipf_max_peptidoform_pep") is not None:
            tsv_cmd += [
                "--ipf_max_peptidoform_pep",
                str(tsv_cfg["ipf_max_peptidoform_pep"]),
            ]
        if tsv_cfg.get("max_rs_peakgroup_qvalue") is not None:
            tsv_cmd += [
                "--max_rs_peakgroup_qvalue",
                str(tsv_cfg["max_rs_peakgroup_qvalue"]),
            ]
        if tsv_cfg.get("max_global_peptide_qvalue") is not None:
            tsv_cmd += [
                "--max_global_peptide_qvalue",
                str(tsv_cfg["max_global_peptide_qvalue"]),
            ]
        if tsv_cfg.get("max_global_protein_qvalue") is not None:
            tsv_cmd += [
                "--max_global_protein_qvalue",
                str(tsv_cfg["max_global_protein_qvalue"]),
            ]
        tsv_cmd.append(
            "--use_alignment"
            if tsv_cfg.get("use_alignment", True)
            else "--no-use_alignment"
        )
        if tsv_cfg.get("max_alignment_pep") is not None:
            tsv_cmd += ["--max_alignment_pep", str(tsv_cfg["max_alignment_pep"])]

        score_cfg = self._load_tool_config(
            "tools-configs", "pyprophet", "pyprophet_score_config.json"
        )
        score_cmd = self._append_pyprophet_config_options(
            [exe, "score", "--in", osw_in],
            score_cfg,
            "pyprophet_score_config.json",
            exclude_keys={"in", "out", "help"},
        )
        subcommands: list[tuple[str, list[str]]] = [("score", score_cmd)]

        infer_peptide_cfg = self._load_tool_config(
            "tools-configs", "pyprophet", "pyprophet_infer_peptide_config.json"
        )
        for context in self._selected_pyprophet_contexts(infer_peptide_cfg):
            cmd = [exe, "infer", "peptide", "--in", osw_in, "--context", context]
            cmd = self._append_pyprophet_config_options(
                cmd,
                infer_peptide_cfg,
                "pyprophet_infer_peptide_config.json",
                exclude_keys={"in", "out", "help", "context", "contexts"},
            )
            subcommands.append((f"infer peptide ({context})", cmd))

        infer_protein_cfg = self._load_tool_config(
            "tools-configs", "pyprophet", "pyprophet_infer_protein_config.json"
        )
        for context in self._selected_pyprophet_contexts(infer_protein_cfg):
            cmd = [exe, "infer", "protein", "--in", osw_in, "--context", context]
            cmd = self._append_pyprophet_config_options(
                cmd,
                infer_protein_cfg,
                "pyprophet_infer_protein_config.json",
                exclude_keys={"in", "out", "help", "context", "contexts"},
            )
            subcommands.append((f"infer protein ({context})", cmd))

        subcommands.append(("export tsv", tsv_cmd))

        matrix_cfg = self._load_tool_config(
            "tools-configs", "pyprophet", "pyprophet_export_matrix_config.json"
        )
        matrix_levels = [
            level
            for level in matrix_cfg.get("levels", [])
            if level in _PY_MATRIX_OUTS
        ]
        for level in matrix_levels:
            cmd = [
                exe,
                "export",
                "matrix",
                "--in",
                osw_in,
                "--out",
                str(results_dir / _PY_MATRIX_OUTS[level]),
                "--level",
                level,
            ]
            if matrix_cfg.get("csv"):
                cmd.append("--csv")
            cmd.append(
                "--transition_quantification"
                if matrix_cfg.get("transition_quantification", True)
                else "--no-transition_quantification"
            )
            if matrix_cfg.get("max_transition_pep") is not None:
                cmd += ["--max_transition_pep", str(matrix_cfg["max_transition_pep"])]
            if matrix_cfg.get("ipf"):
                cmd += ["--ipf", str(matrix_cfg["ipf"])]
            if matrix_cfg.get("ipf_max_peptidoform_pep") is not None:
                cmd += [
                    "--ipf_max_peptidoform_pep",
                    str(matrix_cfg["ipf_max_peptidoform_pep"]),
                ]
            if matrix_cfg.get("max_rs_peakgroup_qvalue") is not None:
                cmd += [
                    "--max_rs_peakgroup_qvalue",
                    str(matrix_cfg["max_rs_peakgroup_qvalue"]),
                ]
            if matrix_cfg.get("max_global_peptide_qvalue") is not None:
                cmd += [
                    "--max_global_peptide_qvalue",
                    str(matrix_cfg["max_global_peptide_qvalue"]),
                ]
            if matrix_cfg.get("max_global_protein_qvalue") is not None:
                cmd += [
                    "--max_global_protein_qvalue",
                    str(matrix_cfg["max_global_protein_qvalue"]),
                ]
            cmd.append(
                "--use_alignment"
                if matrix_cfg.get("use_alignment", True)
                else "--no-use_alignment"
            )
            if matrix_cfg.get("max_alignment_pep") is not None:
                cmd += ["--max_alignment_pep", str(matrix_cfg["max_alignment_pep"])]
            if matrix_cfg.get("top_n") is not None:
                cmd += ["--top_n", str(matrix_cfg["top_n"])]
            cmd.append(
                "--consistent_top"
                if matrix_cfg.get("consistent_top", True)
                else "--no-consistent_top"
            )
            if matrix_cfg.get("normalization"):
                cmd += ["--normalization", str(matrix_cfg["normalization"])]
            subcommands.append((f"export matrix ({level})", cmd))

        for label, cmd in subcommands:
            self.logger.log(f"Running pyprophet {label}…")
            if not self.executor.run_command(cmd):
                self.logger.log(f"❌ pyprophet {label} failed.")
                return False
            self.logger.log(f"✅ pyprophet {label} completed.")

        return True

    def _ensure_ini_in_workflow(self, tool: str) -> None:
        """
        Copy the shared INI (from workspace/ini/) into the workflow INI dir
        so that executor.run_topp() can find it (it appends -ini <path>).
        """
        self._ensure_workspace_context()
        workspace_params = self._load_workspace_params()
        saved_tool_params = workspace_params.get(tool, {})
        if not isinstance(saved_tool_params, dict):
            saved_tool_params = {}

        refreshed = self._workspace_parameter_manager.refresh_ini_from_binary(
            tool,
            saved_tool_params,
        )
        if refreshed:
            self.logger.log(
                f"Refreshed workspace INI for {tool} from installed binary: "
                f"{self._shared_ini_dir / f'{tool}.ini'}"
            )

        shared = self._shared_ini_dir / f"{tool}.ini"
        dest = self.parameter_manager.ini_dir / f"{tool}.ini"
        if not shared.exists():
            return

        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(shared, dest)
            self.logger.log(f"Using synced INI for {tool}: {dest}")
        except Exception as e:
            self.logger.log(f"⚠️ Could not copy {tool}.ini: {e}")

    # ------------------------------------------------------------------
    # Main execution entry-point-

    def execution(self) -> bool:
        """
        Execute the full OpenSWATH pipeline.

        Reads all configuration from workspace/params.json (written by the
        configuration page).  Intermediate files are written under
        ``workflow_dir/results/``.
        """
        self.logger.log("=" * 70)
        self.logger.log("OPENSWATH PIPELINE — STARTING")
        self.logger.log("=" * 70)

        # -- Load configuration --------------------------------------------
        cfg = self._load_workspace_params()
        ep_cfg = cfg.get("easypqp", {})
        osag_mode = cfg.get("osag_input_mode", "Use existing transition list(s)")
        use_easypqp = osag_mode == "Generate from FASTA (predict transitions)"
        run_mode, start_step, steps = self._normalize_run_settings(cfg, use_easypqp)

        results_dir = self.workflow_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        self.logger.log(f"Assay source mode: {osag_mode}")
        self.logger.log(f"EasyPQP enabled: {use_easypqp}")
        if run_mode == _RUN_MODE_FRESH:
            self.logger.log("Execution mode: start from scratch")
            start_step = steps[0]
        else:
            self.logger.log(
                f"Execution mode: reuse outputs from {_STEP_LABELS[start_step]}"
            )
            missing = self._missing_resume_prerequisites(
                start_step, use_easypqp, results_dir
            )
            if missing:
                self.logger.log(
                    "❌ Resume requested but required upstream outputs are missing:"
                )
                for label, path in missing:
                    self.logger.log(f"  - {label}: {path}")
                return False
            self._clear_results_from_step(start_step, use_easypqp, results_dir)
            self.logger.log(
                f"Cleared outputs for {_STEP_LABELS[start_step]} and downstream steps."
            )
        start_index = steps.index(start_step)

        # -- Resolve mzML file paths ---------------------------------------
        mzml_paths = self._mzml_paths()
        if not mzml_paths:
            self.logger.log("❌ No mzML files found in workspace.")
            return False
        self.logger.log(
            f"mzML files ({len(mzml_paths)}): {[Path(p).name for p in mzml_paths]}"
        )

        step_outputs = self._step_output_paths(results_dir)

        # -- Step 1: EasyPQP (optional) ------------------------------------
        if use_easypqp:
            osag_in = str(step_outputs["easypqp"])
            if steps.index("easypqp") >= start_index:
                self.logger.log("-" * 50)
                self.logger.log("STEP 1: EasyPQP in-silico library generation")
                if not self._run_easypqp(ep_cfg, results_dir):
                    return False
            else:
                self.logger.log(f"Reusing EasyPQP output: {osag_in}")
        else:
            # Use existing library from workspace
            lib_dir = self._workspace_file("input-files", "libraries")
            lib_file = cfg.get("osag_library_input", "")
            if lib_file:
                osag_in = str(lib_dir / lib_file)
            else:
                libs = sorted(lib_dir.iterdir()) if lib_dir.exists() else []
                if not libs:
                    self.logger.log(
                        "❌ No transition library found and EasyPQP is disabled."
                    )
                    return False
                osag_in = str(libs[0])
                self.logger.log(f"Auto-selected library: {libs[0].name}")

        self.logger.log(f"OSAG input: {osag_in}")

        # -- Step 2: OpenSwathAssayGenerator ------------------------------
        osag_dir = results_dir / "osag"
        osag_out = str(step_outputs["osag"])
        if steps.index("osag") >= start_index:
            self.logger.log("-" * 50)
            self.logger.log("STEP 2: OpenSwathAssayGenerator")
            osag_dir.mkdir(parents=True, exist_ok=True)
            if not self._run_osag(osag_in, osag_out):
                return False
        else:
            self.logger.log(f"Reusing OpenSwathAssayGenerator output: {osag_out}")

        # -- Step 3: OpenSwathDecoyGenerator ------------------------------
        osdg_dir = results_dir / "osdg"
        osdg_out = str(step_outputs["osdg"])
        if steps.index("osdg") >= start_index:
            self.logger.log("-" * 50)
            self.logger.log("STEP 3: OpenSwathDecoyGenerator")
            osdg_dir.mkdir(parents=True, exist_ok=True)
            if not self._run_osdg(osag_out, osdg_out):
                return False
        else:
            self.logger.log(f"Reusing OpenSwathDecoyGenerator output: {osdg_out}")

        # -- Step 4: OpenSwathWorkflow -------------------------------------
        osw_out = str(step_outputs["openswathworkflow"])
        if steps.index("openswathworkflow") >= start_index:
            self.logger.log("-" * 50)
            self.logger.log("STEP 4: OpenSwathWorkflow")
            if not self._run_openswath(cfg, osdg_out, mzml_paths, osw_out, results_dir):
                return False
        else:
            self.logger.log(f"Reusing OpenSwathWorkflow output: {osw_out}")

        # -- Step 5: PyProphet ---------------------------------------------
        py_out = str(step_outputs["pyprophet"])
        if steps.index("pyprophet") >= start_index:
            self.logger.log("-" * 50)
            self.logger.log("STEP 5: PyProphet score / infer / export")
            if not self._run_pyprophet(osw_out, py_out, results_dir):
                return False
        else:
            self.logger.log(f"Reusing PyProphet output: {py_out}")

        self.logger.log("=" * 70)
        self.logger.log("OPENSWATH PIPELINE — COMPLETED SUCCESSFULLY")
        self.logger.log("=" * 70)
        return True

    # ------------------------------------------------------------------
    # Results display

    def _result_bundle_specs(self, results_dir: Path) -> list[dict[str, object]]:
        workspace_slug = self._workspace_dir.name.replace(" ", "_")
        debug_files = self._debug_output_matches(results_dir)

        bundles = [
            {
                "id": "quant-results",
                "title": "Quantification tables",
                "description": (
                    "Primary end-user outputs: long identification table plus "
                    "precursor, peptide, and protein matrices."
                ),
                "files": [
                    results_dir / _PY_OUT,
                    results_dir / _PY_MATRIX_OUTS["precursor"],
                    results_dir / _PY_MATRIX_OUTS["peptide"],
                    results_dir / _PY_MATRIX_OUTS["protein"],
                ],
                "archive_name": f"{workspace_slug}_openswath_quant_tables.zip",
                "advanced": False,
            },
            {
                "id": "transition-libraries",
                "title": "Transition libraries",
                "description": (
                    "Upstream spectral-library outputs that can be reused as peptide "
                    "query / transition inputs."
                ),
                "files": [
                    results_dir / "insilico" / _EPQP_OUT,
                    results_dir / "osag" / _OSAG_OUT,
                    results_dir / "osdg" / _OSDG_OUT,
                ],
                "archive_name": (
                    f"{workspace_slug}_openswath_transition_libraries.zip"
                ),
                "advanced": False,
            },
            {
                "id": "feature-database",
                "title": "Feature database (.osw)",
                "description": (
                    "OpenSwath feature store for debugging, inspection, and "
                    "developer workflows."
                ),
                "files": [results_dir / _OSW_OUT],
                "archive_name": f"{workspace_slug}_openswath_feature_database.zip",
                "advanced": True,
            },
            {
                "id": "chromatograms-debug",
                "title": "Chromatograms and debug outputs",
                "description": (
                    "XIC / mobilogram exports plus calibration and diagnostic side-products."
                ),
                "files": [
                    results_dir / _OSW_XIC_OUT,
                    results_dir / _OSW_XIM_OUT,
                    *debug_files,
                ],
                "archive_name": f"{workspace_slug}_openswath_debug_outputs.zip",
                "advanced": True,
            },
        ]

        available_bundles: list[dict[str, object]] = []
        for bundle in bundles:
            files = [Path(path) for path in bundle["files"] if Path(path).exists()]
            if not files:
                continue
            bundle_copy = dict(bundle)
            bundle_copy["files"] = files
            available_bundles.append(bundle_copy)

        return available_bundles

    def _render_result_bundle(
        self, results_dir: Path, bundle: dict[str, object]
    ) -> None:
        files: list[Path] = bundle["files"]
        bundle_id = str(bundle["id"])
        archive_dir = results_dir / "download-archives"
        archive_path = archive_dir / f"{bundle_id}.zip"

        ready_key = f"{self.workflow_dir.resolve()}::bundle::{bundle_id}::ready"
        toggle_key = f"{ready_key}::prepare"

        if archive_needs_refresh(files, archive_path):
            st.session_state[ready_key] = False

        col_a, col_b = st.columns([4.2, 1.2])
        col_a.markdown(f"**{bundle['title']}**")
        col_a.caption(
            f"{bundle['description']} {len(files)} file(s) • {total_size_label(files)}"
        )

        button_placeholder = col_b.empty()
        if st.session_state.get(ready_key, False) and archive_path.exists():
            with open(archive_path, "rb") as fh:
                button_placeholder.download_button(
                    "Download ⬇️",
                    data=fh,
                    file_name=str(bundle["archive_name"]),
                    use_container_width=True,
                    key=f"download_bundle::{bundle_id}",
                )
        else:
            button_label = "Refresh Download" if archive_path.exists() else "Prepare Download"
            if button_placeholder.button(
                button_label,
                use_container_width=True,
                key=toggle_key,
            ):
                with st.spinner("Building archive…"):
                    build_zip_archive(files, archive_path, relative_to=results_dir)
                st.session_state[ready_key] = True
                st.rerun()

        with st.expander(f"Files in {bundle['title']}", expanded=False):
            for path in files:
                try:
                    rel_path = path.relative_to(results_dir)
                except ValueError:
                    rel_path = path.name
                st.caption(f"• `{rel_path}`")

    def results(self) -> None:
        """Display grouped output bundles with on-demand archive creation."""

        results_dir = self.workflow_dir / "results"
        bundles = self._result_bundle_specs(results_dir)

        if not bundles:
            st.info("No output files yet — run the workflow above to generate results.")
            return

        st.caption(
            "Archives are prepared on demand from disk so large result sets are not "
            "loaded into memory during page render."
        )

        primary_bundles = [bundle for bundle in bundles if not bundle["advanced"]]
        advanced_bundles = [bundle for bundle in bundles if bundle["advanced"]]

        if primary_bundles:
            st.markdown("**Primary Downloads**")
            for bundle in primary_bundles:
                self._render_result_bundle(results_dir, bundle)

        if advanced_bundles:
            state_key = f"{self.workflow_dir.resolve()}::show_advanced_downloads"
            show_advanced = bool(st.session_state.get(state_key, False))
            toggle_label = (
                "Hide Advanced Downloads" if show_advanced else "Show Advanced Downloads"
            )
            if st.button(toggle_label, key=f"{state_key}::toggle"):
                st.session_state[state_key] = not show_advanced
                st.rerun()

            if show_advanced:
                st.markdown("**Advanced Downloads**")
                st.caption(
                    "Developer-oriented artifacts: the `.osw` feature database, "
                    "chromatograms, mobilograms, and calibration/debug outputs."
                )
                for bundle in advanced_bundles:
                    self._render_result_bundle(results_dir, bundle)
