import pyopenms as poms
import json
import shutil
import subprocess
import streamlit as st
from pathlib import Path


class ParameterManager:
    """
    Manages the parameters for a workflow, including saving parameters to a JSON file,
    loading parameters from the file, and resetting parameters to defaults. This class
    specifically handles parameters related to TOPP tools in a pyOpenMS context and
    general parameters stored in Streamlit's session state.

    Attributes:
        ini_dir (Path): Directory path where .ini files for TOPP tools are stored.
        params_file (Path): Path to the JSON file where parameters are saved.
        param_prefix (str): Prefix for general parameter keys in Streamlit's session state.
        topp_param_prefix (str): Prefix for TOPP tool parameter keys in Streamlit's session state.
        workflow_name (str): Name of the workflow, used for loading presets.
    """

    # Methods related to parameter handling
    def __init__(self, workflow_dir: Path, workflow_name: str = None):
        self.ini_dir = Path(workflow_dir, "ini")
        self.ini_dir.mkdir(parents=True, exist_ok=True)
        self.defaults_ini_dir = self.ini_dir / ".defaults"
        self.defaults_ini_dir.mkdir(parents=True, exist_ok=True)
        self.params_file = Path(workflow_dir, "params.json")
        self.param_prefix = f"{workflow_dir.stem}-param-"
        self.topp_param_prefix = f"{workflow_dir.stem}-TOPP-"
        # Store workflow name for preset loading; default to directory stem if not provided
        self.workflow_name = workflow_name or workflow_dir.stem

    def create_ini(self, tool: str) -> bool:
        """
        Create an ini file for a TOPP tool if it doesn't exist.

        Args:
            tool: Name of the TOPP tool (e.g., "CometAdapter")

        Returns:
            True if ini file exists (created or already existed), False if creation failed
        """
        ini_path = Path(self.ini_dir, tool + ".ini")
        if ini_path.exists():
            return True
        return self._write_ini(tool, ini_path)

    def _write_ini(self, tool: str, ini_path: Path) -> bool:
        """Write a TOPP tool INI file via `<tool> -write_ini <path>`."""
        try:
            ini_path.parent.mkdir(parents=True, exist_ok=True)
            completed = subprocess.run(
                [tool, "-write_ini", str(ini_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except FileNotFoundError:
            return False
        return completed.returncode == 0 and ini_path.exists()

    def refresh_ini_from_binary(
        self, tool: str, saved_tool_params: dict | None = None
    ) -> bool:
        """
        Refresh a workspace INI from the installed TOPP binary.

        If saved non-default parameters are supplied, they are overlaid onto the
        generated defaults before the workspace INI is replaced.
        """
        target_path = Path(self.ini_dir, tool + ".ini")
        tmp_path = Path(self.ini_dir, f".{tool}.generated.ini")
        tmp_path.unlink(missing_ok=True)

        if not self._write_ini(tool, tmp_path):
            tmp_path.unlink(missing_ok=True)
            return False

        try:
            default_path = self.defaults_ini_dir / f"{tool}.ini"
            default_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tmp_path, default_path)

            if saved_tool_params:
                param = poms.Param()
                poms.ParamXMLFile().load(str(tmp_path), param)
                ini_keys = {
                    k.decode() if isinstance(k, (bytes, bytearray)) else str(k)
                    for k in param.keys()
                }
                for short_key, value in saved_tool_params.items():
                    ini_key = f"{tool}:1:{short_key}"
                    if ini_key not in ini_keys:
                        continue
                    ini_value = param.getValue(ini_key)
                    param.setValue(ini_key, self._coerce_topp_value(ini_value, value))
                poms.ParamXMLFile().store(str(tmp_path), param)

            tmp_path.replace(target_path)
            return True
        finally:
            tmp_path.unlink(missing_ok=True)

    def ensure_default_ini(self, tool: str) -> Path | None:
        """
        Ensure a stable default descriptor exists for a TOPP tool.

        This should represent the tool defaults, not the current workspace state.
        """
        default_path = self.defaults_ini_dir / f"{tool}.ini"
        if default_path.exists():
            return default_path

        if self._write_ini(tool, default_path):
            return default_path

        return None

    def _coerce_topp_value(self, ini_value, value):
        """
        Coerce Streamlit widget values to the type expected by pyOpenMS Param.
        """
        if value is None:
            if isinstance(ini_value, list):
                return []
            if isinstance(ini_value, bytes):
                return b""
            if isinstance(ini_value, str):
                return ""
            return ini_value

        if isinstance(ini_value, list):
            if value in (None, ""):
                return []
            if isinstance(value, str):
                stripped = value.strip()
                if stripped in ("", "[]"):
                    return []
                try:
                    parsed = json.loads(stripped)
                except Exception:
                    parsed = None
                if isinstance(parsed, list):
                    return parsed
                return [entry for entry in value.split("\n") if entry.strip()]
            if isinstance(value, tuple):
                return list(value)
            return value if isinstance(value, list) else [value]

        if isinstance(ini_value, bool):
            if isinstance(value, str):
                return value.strip().lower() in ("true", "1", "yes", "on")
            return bool(value)

        if isinstance(ini_value, int) and not isinstance(ini_value, bool):
            if value in (None, ""):
                return ini_value
            return int(value)

        if isinstance(ini_value, float):
            if value in (None, ""):
                return ini_value
            return float(value)

        return value

    def save_parameters(self) -> None:
        """
        Saves the current parameters from Streamlit's session state to a JSON file.
        It handles both general parameters and parameters specific to TOPP tools,
        ensuring that only non-default values are stored.
        """
        # Everything in session state which begins with self.param_prefix is saved to a json file
        general_params = {
            k.replace(self.param_prefix, ""): v
            for k, v in st.session_state.items()
            if k.startswith(self.param_prefix)
        }

        existing_params = self.get_parameters_from_json()

        # Merge with parameters from json
        # Advanced parameters are only in session state if the view is active
        json_params = existing_params | general_params

        topp_state_items: dict[str, list[tuple[str, object]]] = {}
        for key, value in st.session_state.items():
            if not key.startswith(self.topp_param_prefix):
                continue
            # Skip display-only helper widgets used by the UI
            if key.endswith("_display"):
                continue

            tool_and_key = key.replace(self.topp_param_prefix, "", 1)
            tool = tool_and_key.split(":1:", 1)[0]
            topp_state_items.setdefault(tool, []).append((key, value))

        # get a list of TOPP tools which are in session state
        current_topp_tools = set(topp_state_items.keys())
        current_topp_tools.update(
            tool
            for tool, value in existing_params.items()
            if isinstance(value, dict) and (self.ini_dir / f"{tool}.ini").exists()
        )

        # for each TOPP tool, open the ini file
        for tool in current_topp_tools:
            existing_tool_params = existing_params.get(tool, {})
            json_params[tool] = (
                existing_tool_params.copy()
                if isinstance(existing_tool_params, dict)
                else {}
            )

            tool_state_items = topp_state_items.get(tool, [])
            if not tool_state_items:
                continue

            if not self.create_ini(tool):
                # Could not create ini file - skip this tool
                continue
            ini_path = Path(self.ini_dir, f"{tool}.ini")
            # load the param object
            param = poms.Param()
            poms.ParamXMLFile().load(str(ini_path), param)
            ini_keys = {str(k) for k in param.keys()}
            tool_changed = False

            default_param = poms.Param()
            default_ini_path = self.ensure_default_ini(tool)
            if default_ini_path and default_ini_path.exists():
                poms.ParamXMLFile().load(str(default_ini_path), default_param)
                default_ini_keys = {str(k) for k in default_param.keys()}
            else:
                default_param = param
                default_ini_keys = ini_keys

            # get all session state param keys and values for this tool
            for key, value in tool_state_items:
                # get ini_key
                ini_key = key.replace(self.topp_param_prefix, "")
                # Skip keys that don't correspond to actual ini entries
                # (e.g., widget-specific keys like uploaders or display helpers)
                if ini_key not in ini_keys:
                    continue
                # get ini (default) value by ini_key
                ini_value = param.getValue(ini_key)
                if ini_key in default_ini_keys:
                    default_value = default_param.getValue(ini_key)
                else:
                    default_value = ini_value
                coerced_value = self._coerce_topp_value(ini_value, value)
                short_key = key.split(":1:")[1]

                # Keep the workspace INI aligned with the current UI state,
                # but avoid rewriting unchanged parameters.
                if ini_value != coerced_value:
                    param.setValue(ini_key, coerced_value)
                    tool_changed = True

                # keep non-default value in params.json, otherwise remove stale entry
                if default_value != coerced_value:
                    json_params[tool][short_key] = coerced_value
                else:
                    json_params[tool].pop(short_key, None)

            # Persist the current TOPP state in the workspace INI file only when needed.
            if tool_changed:
                poms.ParamXMLFile().store(str(ini_path), param)

        # Save to json file only if parameters changed
        if json_params != existing_params:
            with open(self.params_file, "w", encoding="utf-8") as f:
                json.dump(json_params, f, indent=4)

    def get_parameters_from_json(self) -> dict:
        """
        Loads parameters from the JSON file if it exists and returns them as a dictionary.
        If the file does not exist, it returns an empty dictionary.

        Returns:
            dict: A dictionary containing the loaded parameters. Keys are parameter names,
                and values are parameter values.
        """
        # Check if parameter file exists
        if not Path(self.params_file).exists():
            return {}
        else:
            # Load parameters from json file
            try:
                with open(self.params_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                st.error(
                    "**ERROR**: Attempting to load an invalid JSON parameter file. Reset to defaults."
                )
                return {}

    def get_topp_parameters(self, tool: str) -> dict:
        """
        Get all parameters for a TOPP tool, merging defaults with user values.

        Args:
            tool: Name of the TOPP tool (e.g., "CometAdapter")

        Returns:
            Dict with parameter names as keys (without tool prefix) and their values.
            Returns empty dict if ini file doesn't exist.
        """
        ini_path = Path(self.ini_dir, f"{tool}.ini")
        if not ini_path.exists():
            return {}

        # Load defaults from ini file
        param = poms.Param()
        poms.ParamXMLFile().load(str(ini_path), param)

        # Build dict from ini (extract short key names)
        prefix = f"{tool}:1:"
        full_params = {}
        for key in param.keys():
            key_str = key.decode() if isinstance(key, bytes) else str(key)
            if prefix in key_str:
                short_key = key_str.split(prefix, 1)[1]
                full_params[short_key] = param.getValue(key)

        # Override with user-modified values from JSON
        user_params = self.get_parameters_from_json().get(tool, {})
        full_params.update(user_params)

        return full_params

    def reset_to_default_parameters(self) -> None:
        """
        Resets the parameters to their default values by deleting the custom parameters
        JSON file.
        """
        # Delete custom params json file
        self.params_file.unlink(missing_ok=True)

    def load_presets(self) -> dict:
        """
        Load preset definitions from presets.json file.

        Returns:
            dict: Dictionary of presets for the current workflow, or empty dict if
                  presets.json doesn't exist or has no presets for this workflow.
        """
        presets_file = Path("presets.json")
        if not presets_file.exists():
            return {}

        try:
            with open(presets_file, "r", encoding="utf-8") as f:
                all_presets = json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}

        # Normalize workflow name to match preset keys (lowercase with hyphens)
        workflow_key = self.workflow_name.replace(" ", "-").lower()
        return all_presets.get(workflow_key, {})

    def get_preset_names(self) -> list:
        """
        Get list of available preset names for the current workflow.

        Returns:
            list: List of preset names (strings), excluding special keys like _description.
        """
        presets = self.load_presets()
        return [name for name in presets.keys() if not name.startswith("_")]

    def get_preset_description(self, preset_name: str) -> str:
        """
        Get the description for a specific preset.

        Args:
            preset_name: Name of the preset

        Returns:
            str: Description text for the preset, or empty string if not found.
        """
        presets = self.load_presets()
        preset = presets.get(preset_name, {})
        return preset.get("_description", "")

    def apply_preset(self, preset_name: str) -> bool:
        """
        Apply a preset by updating params.json and clearing relevant session_state keys.

        Uses the "delete-then-rerun" pattern: instead of overwriting session_state
        values (which widgets may not reflect immediately due to fragment caching),
        we delete the keys so widgets re-initialize fresh from params.json on rerun.

        Args:
            preset_name: Name of the preset to apply

        Returns:
            bool: True if preset was applied successfully, False otherwise.
        """
        presets = self.load_presets()
        preset = presets.get(preset_name)
        if not preset:
            return False

        # Load existing parameters
        current_params = self.get_parameters_from_json()

        # Collect keys to delete from session_state
        keys_to_delete = []

        for key, value in preset.items():
            # Skip description key
            if key == "_description":
                continue

            if key == "_general":
                # Handle general workflow parameters
                for param_name, param_value in value.items():
                    session_key = f"{self.param_prefix}{param_name}"
                    keys_to_delete.append(session_key)
                    current_params[param_name] = param_value
            elif isinstance(value, dict) and not key.startswith("_"):
                # Handle TOPP tool parameters
                tool_name = key
                if tool_name not in current_params:
                    current_params[tool_name] = {}
                for param_name, param_value in value.items():
                    session_key = f"{self.topp_param_prefix}{tool_name}:1:{param_name}"
                    keys_to_delete.append(session_key)
                    current_params[tool_name][param_name] = param_value

        # Delete affected keys from session_state so widgets re-initialize fresh
        for session_key in keys_to_delete:
            if session_key in st.session_state:
                del st.session_state[session_key]

        # Save updated parameters to file
        with open(self.params_file, "w", encoding="utf-8") as f:
            json.dump(current_params, f, indent=4)

        return True

    def clear_parameter_session_state(self) -> None:
        """
        Clear all parameter-related keys from session_state.

        This forces widgets to re-initialize from params.json or defaults
        on the next rerun, rather than using potentially stale session_state values.
        """
        keys_to_delete = [
            key
            for key in list(st.session_state.keys())
            if key.startswith(self.param_prefix)
            or key.startswith(self.topp_param_prefix)
        ]
        for key in keys_to_delete:
            del st.session_state[key]
