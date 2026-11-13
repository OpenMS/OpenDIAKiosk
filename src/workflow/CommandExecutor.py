import time
import os
import subprocess
import threading
from pathlib import Path
from .Logger import Logger
from .ParameterManager import ParameterManager, bool_param_paths_from_param_xml_ini
from .settings_io import load_settings, detect_cpu_quota, resolve_thread_setting, local_thread_default
from .vendor_files import vendor_format
from .progress import RunProgress, read_progress, failure_context, OUTPUT_TAIL
from .processing_report import command_io, report_params
from collections import deque
import sys
import importlib.util
import json
import tempfile
import streamlit as st
import xml.etree.ElementTree as ET

from .process_state import pid_markers, prune_exited_marker, register_process_marker, signal_owned_process

class CommandExecutor:
    """
    Manages the execution of external shell commands such as OpenMS TOPP tools within a Streamlit application.

    This class provides a structured approach to executing shell commands, capturing
    their output, and handling errors. It is designed to facilitate running both single
    commands and batches of commands in parallel, leveraging Python's subprocess module
    for execution.
    """

    # Methods for running commands and logging
    def __init__(
        self, workflow_dir: Path, logger: Logger, parameter_manager: ParameterManager
    ):
        self.pid_dir = Path(workflow_dir, "pids")
        self.logger = logger
        self.parameter_manager = parameter_manager
        # Memoised thread budget, resolved lazily by _get_max_threads().
        self._max_threads = None
        # Per-file progress of run_topp() calls, read by the Run page
        self.progress = RunProgress(Path(workflow_dir))

    def _get_max_threads(self) -> int:
        """
        Get max threads for current deployment mode.

        Settings are read from settings.json on disk, not from
        st.session_state: inside the RQ work horse there is no
        ScriptRunContext, so st.session_state is an empty mock, the online
        branch was unreachable and the hardcoded local default silently won
        instead of max_threads.online.

        In local mode the workspace params.json may override the configured
        value - that is what the "Threads" widget writes, and it is only
        rendered in local mode. Online the worker is shared and sized by its
        pod spec, so user supplied params.json is ignored.

        Online the container's own cgroup CPU quota wins over
        max_threads.online where one exists. The rq-worker runs at
        requests.cpu == limits.cpu, so its pod spec IS its allocation, and the
        memory-tier components size that per deployment - a 4 cpu worker and a
        20 cpu worker cannot both be described by one number in settings.json,
        and the number that was there described neither. max_threads.online
        remains the fallback for every deployment where no quota is readable:
        bare metal, docker-compose without a cpu limit, a dev box.

        ``"auto"`` (the template default for both keys; ``null`` and ``-1``
        mean the same) is every usable CPU but one, see
        settings_io.default_thread_count(). A number pins the budget.

        The trade-off, deliberately taken: an operator can no longer
        under-subscribe a container that declares more CPU than it wants used.
        For a Guaranteed pod that is the right call, because the declaration
        and the allocation are the same thing.

        The result is memoised for the lifetime of this executor: run_topp()
        consults it twice, once for its own thread split and once more inside
        run_multiple_commands(), and the two must agree.

        In either mode, a missing setting or an explicit ``-1`` auto-scales
        to all CPU cores available to the running process so the workflow
        uses every core on the host (or worker pod).

        Returns:
            int: Maximum number of threads to use for parallel processing (minimum 1).
        """
        if self._max_threads is not None:
            return self._max_threads

        settings = load_settings()
        max_threads_config = settings.get("max_threads", {})

        if settings.get("online_deployment", False):
            quota = detect_cpu_quota()
            if quota is not None:
                value = quota
            else:
                value = resolve_thread_setting(max_threads_config.get("online", "auto"))
        else:
            default = local_thread_default(settings)
            params = self.parameter_manager.get_parameters_from_json()
            value = params.get("max_threads", default)

        value = resolve_thread_setting(value)

        try:
            value = int(value)
        except (TypeError, ValueError):
            # params.json is user supplied - the parameter import uploader
            # writes it verbatim - so a non numeric thread budget degrades to
            # serial execution instead of taking the whole run down.
            self.logger.log(f"Invalid max_threads value {value!r}, falling back to 1.", 1)
            value = 1

        self._max_threads = max(1, value)
        return self._max_threads

    def run_multiple_commands(
        self, commands: list[str], step: int = None, cwd: str | None = None
    ) -> bool:
        """
        Executes multiple shell commands concurrently in separate threads.

        This method leverages threading to run each command in parallel, improving
        efficiency for batch command execution. The number of concurrent commands
        is limited by the max_threads setting, which is distributed between
        parallel command execution and per-tool thread allocation.

        Args:
            commands (list[str]): A list where each element is a list representing
                                        a command and its arguments.
            step (int, optional): The progress step these commands belong to
                (from RunProgress.start_step()); command i is its file i.

        Returns:
            bool: True if all commands succeeded, False if any failed.
        """
        # Get thread settings and calculate distribution
        max_threads = self._get_max_threads()
        num_commands = len(commands)
        parallel_commands = min(num_commands, max_threads)

        # Log the start of command execution
        self.logger.log(
            f"Running {num_commands} commands (max {parallel_commands} parallel, {max_threads} total threads)...",
            1,
        )
        start_time = time.time()

        results = [False] * num_commands
        semaphore = threading.Semaphore(parallel_commands)

        def run_and_track(i, cmd):
            with semaphore:
                try:
                    if step is None:
                        success = self.run_command(cmd, cwd=cwd)
                    else:
                        success = self.run_command(cmd, task=(step, i), cwd=cwd)
                except Exception as e:
                    # e.g. the tool is not installed. A thread that dies here
                    # would add no result, and the step would count as passed.
                    self.logger.log(f"ERROR: {cmd[0]}: {e}", 0)
                    success = False
                results[i] = success

        # Initialize a list to keep track of threads
        threads = []

        # Start a new thread for each command
        for i, cmd in enumerate(commands):
            thread = threading.Thread(target=run_and_track, args=(i, cmd))
            thread.start()
            threads.append(thread)

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Calculate and log the total execution time
        end_time = time.time()
        self.logger.log(
            f"Total time to run {num_commands} commands: {end_time - start_time:.2f} seconds",
            1,
        )

        self._last_results = results
        return all(results)

    def run_command(self, command: list[str], task: tuple = None, cwd=None) -> bool:
        """
        Executes a specified shell command and logs its execution details.

        Args:
            command (list[str]): The shell command to execute, provided as a list of strings.
            task (tuple, optional): (step, index) of this command in the run
                progress (see run_topp()). Its start, end and, on failure, exit
                code and output are recorded there, and a failure is logged
                with its context ("While running X on file 2 of 3 (a.mzML)").
            cwd (str | Path, optional): Working directory of the command, for
                programs that read files next to their executable. Defaults to
                the app's working directory.

        Raises:
            Exception: If the command execution results in any errors.
        """
        # Ensure all command parts are strings
        command = [str(c) for c in command]

        cmd_str = " ".join(command)
        print(f"[DEBUG] run_command() called with: {cmd_str}")

        # Log the execution start
        if cwd is not None:
            self.logger.log(f"Working directory: {cwd}", 1)
        self.logger.log(
            f"Running command:\n{cmd_str}\nWaiting for command to finish...", 1
        )
        start_time = time.time()
        if task is not None:
            self.progress.file_started(*task)

        # Create tracking before spawning: failure must not leave an untracked
        # child. Cancellation can still remove this directory after the spawn.
        # Execute the command with real-time output capture
        try:
            self.pid_dir.mkdir(parents=True, exist_ok=True)
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,  # Line buffered
                universal_newlines=True,
                cwd=cwd,
            )
        except OSError as e:
            # e.g. the tool is not installed: record it, then fail as before
            self.logger.log(f"❌ Failed to spawn subprocess: {e}")
            if task is not None:
                self.progress.file_finished(*task, ok=False, returncode=-1, output=[str(e)])
                self._log_failure_context(task, -1)
            raise
        child_pid = process.pid
        self.logger.log(f"Process spawned with PID: {child_pid}")

        # Record the PID to keep track of running processes associated with this workspace/workflow
        # User can close the Streamlit app and return to a running workflow later
        pid_file_path = self.pid_dir / str(child_pid)
        # Buffer for stderr - will only be written to minimal log if process fails
        stderr_buffer: list[str] = []
        # Last lines of stdout, kept for the run progress of a failed file
        stdout_tail: deque = deque(maxlen=OUTPUT_TAIL)

        try:
            register_process_marker(pid_file_path, child_pid)
            # Real-time output capture, followed by reaping the child.
            self._stream_output(process, stderr_buffer, stdout_tail)
            process.wait()
        except Exception as error:
            # An I/O/registration failure after Popen must not orphan a native
            # tool or lose the original failure behind a secondary cleanup.
            self._terminate_failed_command(process)
            try:
                pid_file_path.unlink(missing_ok=True)
            except OSError as cleanup_error:
                self.logger.log(f"Could not remove process marker: {cleanup_error}", 1)
            if task is not None:
                self.progress.file_finished(*task, ok=False, returncode=-1, output=[str(error)])
                self._log_failure_context(task, -1)
            raise

        # Status polling (pid_files.reap) removes the marker of a zombie, and
        # a tool is a zombie between its exit and wait(): it may be gone.
        pid_file_path.unlink(missing_ok=True)

        end_time = time.time()
        execution_time = end_time - start_time

        # Log completion
        self.logger.log(
            f"Process finished:\n{cmd_str}\nTotal time to run command: {execution_time:.2f} seconds\nExit code: {process.returncode}",
            1,
        )

        ok = process.returncode == 0
        if task is not None:
            self.progress.file_finished(
                *task,
                ok=ok,
                returncode=process.returncode,
                output=list(stdout_tail) + stderr_buffer,
            )

        # Check for errors
        if not ok:
            # Write buffered stderr to minimal log only on failure
            for line in stderr_buffer:
                self.logger.log(f"STDERR: {line}", 0)
            self.logger.log(f"ERROR: Command failed with exit code {process.returncode}: {command[0]}", 0)
            if task is not None:
                self._log_failure_context(task, process.returncode)
            return False

        print(f"[DEBUG] Process completed successfully")
        return True

    def _terminate_failed_command(self, process) -> None:
        """Terminate and reap a child when its execution cannot be monitored."""
        try:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        except OSError as cleanup_error:
            self.logger.log(f"Could not stop untracked command: {cleanup_error}", 0)
        finally:
            for pipe in (process.stdout, process.stderr):
                if pipe is not None:
                    pipe.close()

    def _log_failure_context(self, task: tuple, returncode: int) -> None:
        """Log which tool failed on which file, e.g. 'While running X on file 2 of 3 (a.mzML)'."""
        if task[0] < 0:  # the step could not be recorded
            return
        state = read_progress(self.progress.workflow_dir)
        try:
            step = state["steps"][task[0]]
        except (TypeError, KeyError, IndexError):
            return
        self.logger.log(f"ERROR: {failure_context(step, task[1])} (exit code {returncode})", 0)

    @staticmethod
    def _command_labels(input_output: dict, n_processes: int) -> list[str]:
        """A name per command for the run progress: its input file, as a rule.

        Taken from the first input-like key ("in", "in_...") whose list holds
        one entry per command, else the first such list; a collected list of
        files becomes "<n> files".
        """
        keys = list(input_output.keys())
        candidates = [k for k in keys if k.lower().startswith("in")] + keys
        key = next(
            (k for k in candidates if len(input_output[k]) == n_processes),
            candidates[0] if candidates else None,
        )
        labels = []
        for i in range(n_processes):
            if key is None:
                labels.append("")
                continue
            values = input_output[key]
            value = values[i] if len(values) > i else values[0]
            if isinstance(value, list):
                labels.append(Path(value[0]).name if len(value) == 1 else f"{len(value)} files")
            else:
                labels.append(Path(str(value)).name)
        return labels

    def _stream_output(
        self,
        process: subprocess.Popen,
        stderr_buffer: list[str],
        stdout_tail: deque = None,
    ) -> tuple:
        """
        Streams stdout and stderr from a running process in real-time to the logger.
        This method runs in the workflow process, not the GUI thread, so it's safe to block.

        Stderr is buffered and only logged to the detailed log (level 2) during execution.
        The caller is responsible for writing buffered stderr to minimal log if the process fails.

        Args:
            process: The subprocess.Popen object to stream from
            stderr_buffer: A list to accumulate stderr lines for conditional logging
        """

        def read_stdout():
            """Read stdout in real-time"""
            try:
                for line in iter(process.stdout.readline, ""):
                    if line:
                        # Log subprocess output at level 1 (visible) instead of level 2 (detailed)
                        self.logger.log(line.rstrip(), 1)
                        if stdout_tail is not None:
                            stdout_tail.append(line.rstrip())
            except Exception as e:
                self.logger.log(f"Error reading stdout: {e}", 1)
            finally:
                process.stdout.close()

        def read_stderr():
            """Read stderr in real-time, buffering for conditional minimal log output"""
            try:
                for line in iter(process.stderr.readline, ""):
                    if line:
                        stderr_line = line.rstrip()
                        stderr_buffer.append(stderr_line)
                        # Log to process output log (level 1) so it's visible
                        self.logger.log(f"{stderr_line}", 1)
            except Exception as e:
                self.logger.log(f"Error reading stderr: {e}", 1)
            finally:
                process.stderr.close()

        # Start threads to read stdout and stderr simultaneously
        stdout_thread = threading.Thread(target=read_stdout, daemon=True)
        stderr_thread = threading.Thread(target=read_stderr, daemon=True)

        stdout_thread.start()
        stderr_thread.start()

        # Wait for both threads to complete
        stdout_thread.join()
        stderr_thread.join()

    @staticmethod
    def _coerce_cli_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @staticmethod
    def _load_topp_param_types(ini_path: Path, tool: str) -> dict[str, str]:
        """
        Parse a TOPP INI XML file and return a mapping of short parameter key -> type.

        Example:
            OpenSwathWorkflow:1:Calibration:auto_irt:prefilter:enabled -> bool
            returns {"Calibration:auto_irt:prefilter:enabled": "bool"}
        """
        if not ini_path.exists():
            return {}

        try:
            root = ET.parse(ini_path).getroot()
        except Exception:
            return {}

        key_types: dict[str, str] = {}

        def walk(node, parts: list[str]) -> None:
            for child in list(node):
                tag = child.tag.split("}", 1)[-1]
                name = child.attrib.get("name")
                if not name:
                    continue

                if tag == "NODE":
                    walk(child, parts + [name])
                elif tag in {"ITEM", "ITEMLIST"}:
                    key_types[":".join(parts + [name])] = child.attrib.get("type", "")

        walk(root, [])

        prefix = f"{tool}:1:"
        return {
            full_key.split(prefix, 1)[1]: value_type
            for full_key, value_type in key_types.items()
            if full_key.startswith(prefix)
        }

    @staticmethod
    def _load_asset_topp_param_types(tool: str) -> dict[str, str]:
        """
        Load parameter types from bundled descriptor assets for a TOPP tool.

        These descriptors can be newer or more accurate than the runtime
        `-write_ini` output for some OpenMS builds. They are used as a secondary
        type source during CLI argument construction.
        """
        repo_root = Path(__file__).resolve().parents[2]
        asset_dir = repo_root / "assets" / "common-tool-descriptors" / tool.lower()
        if not asset_dir.exists():
            return {}

        merged: dict[str, str] = {}
        for ini_path in sorted(asset_dir.glob("*.ini")):
            current = CommandExecutor._load_topp_param_types(ini_path, tool)
            for key, value_type in current.items():
                # Prefer bool whenever any bundled descriptor marks the key as bool.
                if merged.get(key) == "bool":
                    continue
                if value_type == "bool" or key not in merged:
                    merged[key] = value_type
        return merged

    def _append_topp_param_arg(
        self,
        command: list[str],
        key: str,
        value,
        param_types: dict[str, str],
    ) -> None:
        """
        Append a TOPP CLI argument while respecting INI-declared parameter types.

        OpenMS `type="bool"` parameters are flags. They must be emitted as
        `-flag` when true, and omitted when false.
        """
        # Skip unset optional TOPP values entirely.
        # Note: 0 and 0.0 are valid values, so use explicit checks.
        if value == "" or value is None:
            return
        if isinstance(value, (list, tuple)) and len(value) == 0:
            return
        if isinstance(value, str) and value.strip() == "[]":
            return

        param_type = param_types.get(key)
        cli_flag = f"-{key}"

        if param_type == "bool":
            if self._coerce_cli_bool(value):
                command.append(cli_flag)
            return

        command.append(cli_flag)
        if isinstance(value, (list, tuple)):
            command += [str(x) for x in value]
        elif isinstance(value, str) and "\n" in value:
            command += [entry for entry in value.split("\n") if entry != ""]
        elif isinstance(value, bool):
            # OpenMS reads a boolean as "true"/"false", not "True"
            command += [str(value).lower()]
        else:
            command += [str(value)]

    def run_topp(
        self,
        tool: str,
        input_output: dict,
        custom_params: dict = {},
        cwd: str | None = None,
        include_saved_params: bool = True,
        tool_instance_name: str = None,
        allow_failed: bool = False,
    ) -> bool:
        """
        Constructs and executes commands for the specified tool OpenMS TOPP tool based on the given
        input and output configurations. Ensures that all input/output file lists
        are of the same length, or single strings, to maintain consistency in command
        execution.
        In many tools, a single input file is processed to produce a single output file.
        When dealing with lists of input or output files, the convention is that
        files are paired based on their order. For instance, the n-th input file is
        assumed to correspond to the n-th output file, maintaining a structured
        relationship between input and output data.
        Supports executing commands either as single or multiple processes
        based on the input size.

        Args:
            tool (str): The executable name or path of the tool.
            input_output (dict): A dictionary specifying the input/output parameter names (as key) and their corresponding file paths (as value).
            custom_params (dict): A dictionary of custom parameters to pass to the tool.
            include_saved_params (bool): Whether to append saved non-default
                params.json values to the CLI. Set this to False when the
                synced INI should be the source of truth for TOPP parameters.
            tool_instance_name (str, optional): A unique instance name for this tool
                invocation, used for parameter lookup when multiple instances of the
                same tool exist. If not provided, defaults to the tool name.
            allow_failed (bool, optional): Go on without the files the tool
                fails on, instead of failing the run: the call returns True
                while at least one command succeeded, ``self.failed`` lists
                the indices of the failed commands, and ``keep_succeeded()``
                drops their entries from the lists the next step takes. The
                Run page names the files left out and why.

        Returns:
            bool: True if all commands succeeded (with ``allow_failed``: any),
            False otherwise.

        Raises:
            ValueError: If the lengths of input/output file lists are inconsistent,
                        except for single string inputs.
        """
        # Use tool_instance_name for parameter lookup, fall back to tool name
        params_key = tool_instance_name if tool_instance_name else tool
        # check input: any input lists must be same length, other items can be a single string
        # e.g. input_mzML : [list of n mzML files], output_featureXML : [list of n featureXML files], input_database : database.tsv
        io_lengths = [len(v) for v in input_output.values() if len(v) > 1]

        if len(set(io_lengths)) > 1:
            raise ValueError(
                f"ERROR in {tool} input/output.\nFile list lengths must be 1 and/or the same. They are {io_lengths}."
            )

        if len(io_lengths) == 0:  # all inputs/outputs are length == 1
            n_processes = 1
        else:
            n_processes = max(io_lengths)

        # Calculate threads per command based on max_threads setting
        max_threads = self._get_max_threads()
        parallel_commands = min(n_processes, max_threads)
        threads_per_command = max(1, max_threads // parallel_commands)

        commands = []

        # Load parameters for non-defaults
        params = self.parameter_manager.get_parameters_from_json()
        ini_path = Path(self.parameter_manager.ini_dir, tool + ".ini").resolve()
        param_types = self._load_topp_param_types(ini_path, tool)
        asset_param_types = self._load_asset_topp_param_types(tool)
        for key, value_type in asset_param_types.items():
            if value_type == "bool" or key not in param_types:
                param_types[key] = value_type

        # A stored parameter the installed tool does not have (saved before an
        # upgrade that removed or renamed it) makes the tool refuse the whole
        # run: "Parameters passed to '<tool>' are invalid", exit code 6. Leave
        # it out and say so. Only when the tool's .ini is there to tell.
        known_params = self.parameter_manager.known_topp_params(tool)
        saved_params = params.get(params_key)
        if isinstance(known_params, set) and isinstance(saved_params, dict):
            unknown = sorted(k for k in saved_params if k not in known_params)
            for k in unknown:
                del saved_params[k]
            if unknown:
                self.logger.log(
                    f"WARNING: {tool} has no parameter {', '.join(unknown)}; "
                    "the stored value is not passed on. Check the parameters "
                    "of this tool on the configuration page.",
                    0,
                )

        # Flags declared explicitly via input_TOPP(flag_parameters=[...]) are
        # valueless CLI switches too: params.json takes priority (survives a
        # session restart), session_state is the live fallback.
        flag_list = params.get("_flag_params", {}).get(params_key)
        if flag_list is None:
            try:
                flag_list = st.session_state.get("_topp_flag_params", {}).get(params_key, [])
            except Exception:
                flag_list = []
        for key in flag_list:
            param_types[key] = "bool"
        # Construct commands for each process
        for i in range(n_processes):
            command = [tool]
            # Add input/output files
            for k in input_output.keys():
                # add key as parameter name
                command += [f"-{k}"]
                # get value from input_output dictionary
                value = input_output[k]
                # when multiple input/output files exist (e.g., multiple mzMLs and featureXMLs), but only one additional input file (e.g., one input database file)
                if len(value) == 1:
                    i = 0
                # when the entry is a list of collected files to be passed as one [["sample1", "sample2"]]
                if isinstance(value[i], list):
                    command += value[i]
                # standard case, files was a list of strings, take the file name at index
                else:
                    command += [value[i]]
            managed_params = set(input_output.keys()) | set(custom_params.keys()) | {
                "threads"
            }

            # Add non-default TOPP tool parameters
            if include_saved_params and params_key in params.keys():
                for k, v in params[params_key].items():
                    if k in managed_params:
                        continue
                    self._append_topp_param_arg(command, k, v, param_types)
            # Add custom parameters
            for k, v in custom_params.items():
                self._append_topp_param_arg(command, k, v, param_types)
            # Add threads parameter for TOPP tools
            command += ["-threads", str(threads_per_command)]
            commands.append(command)

            # check if a ini file has been written, if yes use it (contains custom defaults)
            if ini_path.exists():
                command += ["-ini", str(ini_path)]

        if not commands:
            raise Exception("No commands to execute.")

        # The parameters this call passes besides the -ini defaults, for the
        # processing report.
        saved_params_for_report = dict(params.get(params_key) or {}) if include_saved_params else {}

        # Record the call for the Run page: one entry per command (input file),
        # with what the processing report says about it
        step = self.progress.start_step(
            tool if params_key == tool else f"{tool} ({params_key})",
            self._command_labels(input_output, len(commands)),
            threads=threads_per_command,
            parallel=parallel_commands,
            provenance={
                "kind": "topp",
                "program": tool,
                "version": self.parameter_manager.tool_version(tool),
                "params": report_params({**saved_params_for_report, **custom_params}),
                "io": command_io(input_output, len(commands)),
            },
            allow_failed=allow_failed,
        )
        self.failed = []
        try:
            if len(commands) == 1:
                results = [self.run_command(commands[0], task=(step, 0), cwd=cwd)]
            else:
                self._last_results = None
                ok = self.run_multiple_commands(commands, step=step, cwd=cwd)
                results = self._last_results
                if not isinstance(results, list) or len(results) != len(commands):
                    results = [bool(ok)] * len(commands)
        finally:
            self.progress.end_step(step)
        self.failed = [i for i, ok in enumerate(results) if not ok]
        if not allow_failed or not self.failed:
            return not self.failed
        labels = self._command_labels(input_output, len(commands))
        if len(self.failed) == len(results):
            self.logger.log(f"ERROR: {tool} failed for every file.", 0)
            return False
        self.logger.log(
            f"WARNING: {tool} failed for {len(self.failed)} of {len(results)} "
            f"files; the run goes on without them: "
            + ", ".join(labels[i] for i in self.failed),
            0,
        )
        return True

    def keep_succeeded(self, values: list) -> list:
        """``values`` without the entries of the files the last
        ``run_topp(..., allow_failed=True)`` failed on (``self.failed``).

        Pass the lists the next step takes, in the order of that call's
        inputs, e.g. ``mzmls = executor.keep_succeeded(mzmls)``.
        """
        failed = set(getattr(self, "failed", []) or [])
        return [v for i, v in enumerate(values) if i not in failed]

    def stop(self) -> None:
        """
        Terminates all processes initiated by this executor by killing them based on stored PIDs.
        """
        self.logger.log("Stopping all running processes...")
        incomplete = False
        for marker in pid_markers(self.pid_dir):
            try:
                signal_owned_process(marker, 9)
            except ProcessLookupError:
                pass
            except (OSError, OverflowError) as error:
                # In particular, retain tracking when a live process cannot be
                # signalled. Do not claim it stopped or start another over it.
                self.logger.log(f"Failed to kill process {marker.name}: {error}")
                incomplete = True
                continue
            if prune_exited_marker(marker):
                incomplete = True
        try:
            self.pid_dir.rmdir()
        except OSError:
            # Other processes or housekeeping files may still own this folder.
            pass
        self.logger.log("Some processes are still stopping or could not be stopped." if incomplete else "Workflow stopped.")

    def run_python(self, script_file: str, input_output: dict = {}) -> bool:
        """
        Executes a specified Python script with dynamic input and output parameters,
        optionally logging the execution process. The method identifies and loads
        parameter defaults from the script, updates them with any user-specified
        parameters and file paths, and then executes the script via a subprocess
        call.

        This method facilitates the integration of standalone Python scripts into
        a larger application or workflow, allowing for the execution of these scripts
        with varying inputs and outputs without modifying the scripts themselves.

        Args:
            script_file (str):  The name or path of the Python script to be executed.
                                If the path is omitted, the method looks for the script in 'src/python-tools/'.
                                The '.py' extension is appended if not present.
            input_output (dict, optional): A dictionary specifying the input/output parameter names (as key) and their corresponding file paths (as value). Defaults to {}.

        Returns:
            bool: True if the script ran and exited 0, False otherwise. Same
                contract as run_topp(), so execution() can gate on it. This used
                to discard run_command()'s result and return None, which made
                every `if not self.executor.run_python(...)` guard dead code and
                left a failed Python tool indistinguishable from a successful one.
        """
        # Check if script file exists (can be specified without path and extension)
        # default location: src/python-tools/script_file
        if not script_file.endswith(".py"):
            script_file += ".py"
        path = Path(script_file)
        if not path.exists():
            path = Path("src", "python-tools", script_file)
            if not path.exists():
                # Bail out here: falling through to spec_from_file_location on a
                # path that does not exist raises out of the workflow instead of
                # reporting a failed step.
                self.logger.log(f"ERROR: Script file not found: {script_file}", 0)
                return False

        # load DEFAULTS
        if path.parent not in sys.path:
            sys.path.append(str(path.parent))
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        defaults = getattr(module, "DEFAULTS", None)
        if defaults is None:
            self.logger.log(f"WARNING: No DEFAULTS found in {path.name}")
            # run command without params
            return self.run_command([sys.executable, str(path)])
        elif isinstance(defaults, list):
            defaults = {entry["key"]: entry["value"] for entry in defaults}
            script_defaults = dict(defaults)
            # load paramters from JSON file
            params = {
                k: v
                for k, v in self.parameter_manager.get_parameters_from_json().items()
                if path.name in k
            }
            # update defaults
            for k, v in params.items():
                defaults[k.replace(f"{path.name}:", "")] = v
            for k, v in input_output.items():
                defaults[k] = v
            # Each invocation owns its parameters, including concurrent calls
            # to the same script. Close before dispatch for Windows readers.
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", prefix=f"{path.stem}-", suffix=".json",
                dir=self.pid_dir.parent, delete=False,
            ) as temporary:
                tmp_params_file = Path(temporary.name)
            # Record the call like run_topp(), for the Run page and the report
            io = {k: v if isinstance(v, list) else [v] for k, v in input_output.items()}
            io = {k: v for k, v in io.items() if v}
            step = None
            try:
                with tmp_params_file.open("w", encoding="utf-8") as handle:
                    json.dump(defaults, handle, indent=4)
                step = self.progress.start_step(
                    path.name,
                    self._command_labels(io, 1) if io else [""],
                    provenance={
                        "kind": "python",
                        "program": path.name,
                        "version": None,
                        "params": report_params({k: v for k, v in defaults.items()
                                                if k not in input_output and v != script_defaults.get(k)}),
                        "io": command_io({k: [v] for k, v in io.items()}, 1),
                    },
                )
                return self.run_command(
                    [sys.executable, str(path), str(tmp_params_file)], task=(step, 0)
                )
            finally:
                if step is not None:
                    self.progress.end_step(step)
                tmp_params_file.unlink(missing_ok=True)

        # DEFAULTS exists but is not the documented list of parameter dicts, so
        # nothing ran. Reporting that as success would put the workflow's
        # WORKFLOW FINISHED marker on a step which never executed.
        self.logger.log(
            f"ERROR: DEFAULTS in {path.name} is a {type(defaults).__name__}, "
            "expected a list of parameter dicts. Script not run.",
            0,
        )
        return False

    def convert_to_mzml(self, files: list[str], out_dir: str = None) -> list[str]:
        """
        Returns an mzML path for every input file, converting vendor files first.

        mzML files are returned unchanged. Thermo ``.raw`` files are converted
        with the TOPP tool FileConverter: on Windows with ThermoRawFileParser,
        elsewhere with OpenMS's built-in reader, which needs a .NET 8 or newer
        runtime (the desktop app ships one), or, in an OpenMS built without it
        (the Docker image), with ThermoRawFileParser on mono. Bruker timsTOF ``.d`` folders are converted with
        ``src/python-tools/bruker_to_mzml.py``. A conversion is done once: the
        mzML is kept in ``out_dir`` (default ``<workflow_dir>/converted-mzML``)
        and reused while it is newer than its input.

        Args:
            files (list[str]): Input files, e.g. ``self.params["mzML-files"]``.
            out_dir (str, optional): Where converted mzML files are kept.

        Returns:
            list[str]: mzML paths in input order, or None if a conversion failed
                (the reason is in the workflow log).
        """
        out_dir = Path(out_dir) if out_dir else Path(self.pid_dir.parent, "converted-mzML")
        mzmls = []
        for f in files:
            kind = vendor_format(f)
            if kind is None:
                mzmls.append(str(f))
                continue
            out = Path(out_dir, f"{Path(f).stem}.mzML")
            if out.exists() and out.stat().st_mtime >= Path(f).stat().st_mtime:
                self.logger.log(f"Using converted {out.name} for {Path(f).name}.")
                self.progress.record_alias(str(out), str(f), "converted to mzML in an earlier run")
                mzmls.append(str(out))
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            # A private temporary directory prevents simultaneous conversions
            # from removing each other's partial output. Only complete files
            # replace the cache, and exceptions also clean up partial data.
            with tempfile.TemporaryDirectory(prefix=f".{out.stem}-", dir=out_dir) as temporary:
                tmp = Path(temporary, f"{out.stem}.part.mzML")
                self.logger.log(f"Converting {Path(f).name} to mzML...")
                if kind == "raw":
                    ok = self.run_topp("FileConverter", {"in": [str(f)], "out": [str(tmp)]})
                else:
                    ok = self.run_python("bruker_to_mzml", {"in": str(f), "out": str(tmp)})
                if not ok or not tmp.exists():
                    self.logger.log(f"ERROR: Could not convert {Path(f).name} to mzML.", 0)
                    return None
                os.replace(tmp, out)
                # The report follows the mzML back to the file it came from
                self.progress.record_alias(str(out), str(tmp))
            mzmls.append(str(out))
        return mzmls
