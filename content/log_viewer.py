import streamlit as st
from pathlib import Path
from typing import List

from src.common.common import page_setup

page_setup()

st.title("Log Viewer")

st.markdown(
    """View and download logs for a workspace tool. Select a workspaces root, then a tool/folder
that contains a `logs/` subfolder. Choose a log file to view and download the contents.
"""
)


def find_tool_dirs(base: Path) -> List[Path]:
    if not base.exists() or not base.is_dir():
        return []
    dirs = [d for d in base.iterdir() if d.is_dir() and (d / "logs").exists()]
    return sorted(dirs, key=lambda p: p.name)


# Default guess for workspaces root
def discover_roots() -> List[Path]:
    roots: List[Path] = []
    # Derive workspaces base dir the same way `page_setup()` does
    try:
        settings = st.session_state.settings
        location = st.session_state.get("location", "local")
        if settings.get("workspaces_dir") and location == "local":
            default_ws = Path(
                settings["workspaces_dir"],
                "workspaces-" + settings.get("repository-name", ""),
            )
        else:
            default_ws = Path("..")
    except Exception:
        default_ws = Path("..")

    # add likely candidates if they exist
    candidates = [
        default_ws,
        default_ws / "default",
        Path("example-data") / "workspaces",
    ]

    for c in candidates:
        if c.exists() and c.is_dir():
            roots.append(c)

    # also include any sibling directories starting with 'workspaces' next to the configured path
    try:
        parent = default_ws.parent
        for p in parent.iterdir():
            if p.is_dir() and p.name.startswith("workspaces"):
                roots.append(p)
    except Exception:
        pass

    # preserve order and uniqueness
    seen = set()
    uniq: List[Path] = []
    for p in roots:
        rp = p.resolve()
        if str(rp) not in seen:
            seen.add(str(rp))
            uniq.append(p)
    return uniq


root_choices = discover_roots()
if root_choices:
    root_strs = [str(p) for p in root_choices]

    # Try to detect current workspace for smart default
    try:
        current_workspace = st.session_state.get("workspace", None)
        default_root_idx = 0

        # If current workspace exists, try to find the root that contains it
        if current_workspace:
            current_ws_path = Path(current_workspace).resolve()
            for i, root_choice in enumerate(root_choices):
                root_path = Path(root_choice).resolve()
                # Check if this root is a parent of the current workspace
                try:
                    current_ws_path.relative_to(root_path)
                    default_root_idx = i
                    break
                except ValueError:
                    # current_ws_path is not relative to root_path
                    pass
    except Exception:
        default_root_idx = 0

    selected_root = st.selectbox(
        "Workspaces root directory", root_strs, index=default_root_idx
    )
    base_dir = Path(selected_root)
else:
    default_root = Path(st.session_state.settings.get("workspaces_dir", ".."))
    base_dir_input = st.text_input("Workspaces root directory", value=str(default_root))
    base_dir = Path(base_dir_input)

tool_dirs = find_tool_dirs(base_dir)

if not tool_dirs:
    st.warning(
        "No tool/workspace folders with a `logs/` subfolder were found under the given root."
    )
    st.info(
        "Examples: ../workspaces-OpenDIAKiosk/default/easypqp-insilico (must contain logs/)"
    )
    st.stop()

# Determine default tool/workspace folder based on current workspace
tool_names = [p.name for p in tool_dirs]
default_tool_idx = 0

try:
    current_workspace = st.session_state.get("workspace", None)
    if current_workspace:
        current_ws_path = Path(current_workspace).resolve()
        # Extract the workspace name (e.g., "default" from the path)
        for i, tool_dir in enumerate(tool_dirs):
            tool_path = tool_dir.resolve()
            try:
                # Check if current workspace is under this tool directory
                current_ws_path.relative_to(tool_path)
                default_tool_idx = i
                break
            except ValueError:
                # Also check if the tool directory name matches the workspace name
                if tool_dir.name in str(current_ws_path):
                    default_tool_idx = i
                    break
except Exception:
    pass

tool_choice = st.selectbox(
    "Select tool / workspace folder", tool_names, index=default_tool_idx
)
selected_tool_dir = next((p for p in tool_dirs if p.name == tool_choice), tool_dirs[0])
logs_dir = selected_tool_dir / "logs"

# list available log files
log_files = sorted([p.name for p in logs_dir.iterdir() if p.is_file()])
log_default = (
    "all.log" if "all.log" in log_files else (log_files[0] if log_files else None)
)

log_choice = st.selectbox(
    "Select log file",
    log_files,
    index=log_files.index(log_default) if log_default in log_files else 0,
)

view = st.empty()


def read_log(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except Exception as e:
        return f"Could not read log: {e}"


selected_log_path = logs_dir / log_choice
content = read_log(selected_log_path)

st.subheader(f"{tool_choice} — {log_choice}")
st.download_button(
    label="Download log", data=content, file_name=f"{tool_choice}-{log_choice}"
)

st.code(content)
