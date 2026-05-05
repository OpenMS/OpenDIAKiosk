import streamlit as st
from pathlib import Path
from typing import List
import os

from src.common.common import page_setup

page_setup()

st.title("Workspace Viewer")

st.markdown(
    """View and manage files in your workspace. Select a workspace to browse its contents,
view file sizes, and download files.
"""
)

# Files and directories to exclude from workspace viewer (security/clutter)
EXCLUDED_ITEMS = {
    # Hidden files/dirs
    ".git",
    ".github",
    ".vscode",
    ".env",
    ".env.local",
    ".env.*.local",
    # Python
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".pyc",
    "*.egg-info",
    "dist",
    "build",
    # System
    ".DS_Store",
    "Thumbs.db",
    "node_modules",
    # Secrets/sensitive
    ".ssh",
    ".gpg",
    "secrets",
    "credentials",
    ".credentials",
    # Cache
    ".cache",
    ".tmp",
    "tmp",
}


def is_excluded(name: str) -> bool:
    """Check if a file/directory should be excluded from the viewer"""
    name_lower = name.lower()
    for excluded in EXCLUDED_ITEMS:
        if excluded.startswith("*."):
            # Pattern match (e.g., *.egg-info)
            if name_lower.endswith(excluded[1:]):
                return True
        elif name_lower == excluded or name_lower.startswith(excluded):
            return True
    return False


def is_valid_workspace_path(path: Path, allowed_roots: List[Path]) -> bool:
    """
    Verify that a path is within one of the allowed workspace root directories.
    This prevents directory traversal attacks and restricts access to designated workspaces.
    """
    try:
        path_resolved = path.resolve()
        for root in allowed_roots:
            root_resolved = root.resolve()
            try:
                path_resolved.relative_to(root_resolved)
                return True
            except ValueError:
                pass
        return False
    except Exception:
        return False


def discover_roots() -> List[Path]:
    """Discover available workspace roots - restricted to designated directories only"""
    roots: List[Path] = []
    try:
        settings = st.session_state.settings
        location = st.session_state.get("location", "local")
        workspaces_dir = settings.get("workspaces_dir", "..")
        repo_name = settings.get("repository-name", "")

        # For server deployments, use only configured directories
        if workspaces_dir and location != "local":
            # Server mode: strict configuration-based discovery
            try:
                default_ws = Path(
                    workspaces_dir,
                    "workspaces-" + repo_name if repo_name else "workspaces",
                ).resolve()
                if default_ws.exists() and default_ws.is_dir():
                    roots.append(default_ws)
            except Exception:
                pass
        else:
            # Local mode: more flexible discovery
            try:
                default_ws = Path(
                    workspaces_dir,
                    "workspaces-" + repo_name if repo_name else "workspaces",
                ).resolve()
            except Exception:
                default_ws = Path("../workspaces").resolve()

            candidates = [
                default_ws,
                default_ws / "default",
                Path("example-data").resolve() / "workspaces",
            ]

            for c in candidates:
                try:
                    c_resolved = c.resolve()
                    if c_resolved.exists() and c_resolved.is_dir():
                        roots.append(c_resolved)
                except Exception:
                    pass

            # Also check for workspace directories next to the configured path
            try:
                parent = default_ws.parent
                for p in parent.iterdir():
                    if p.is_dir() and p.name.startswith("workspaces"):
                        roots.append(p.resolve())
            except Exception:
                pass

    except Exception:
        pass

    seen = set()
    uniq: List[Path] = []
    for p in roots:
        rp = str(p)
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def get_workspace_dirs(base: Path) -> List[Path]:
    """Find all workspace directories"""
    if not base.exists() or not base.is_dir():
        return []
    dirs = [d for d in base.iterdir() if d.is_dir()]
    return sorted(dirs, key=lambda p: p.name)


def format_size(size_bytes: int) -> str:
    """Convert bytes to human-readable format"""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def build_ascii_tree(
    path: Path,
    prefix: str = "",
    is_last: bool = True,
    max_depth: int = 10,
    current_depth: int = 0,
) -> str:
    """Build ASCII tree structure like the 'tree' command, excluding sensitive files"""
    if current_depth > max_depth:
        return ""

    tree_str = ""

    try:
        items = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        # Filter out excluded items
        items = [item for item in items if not is_excluded(item.name)]

        for i, item in enumerate(items):
            is_last_item = i == len(items) - 1
            current_prefix = "└── " if is_last_item else "├── "

            # Format: icon + name + size/extra info
            if item.is_dir():
                tree_str += f"{prefix}{current_prefix}📁 {item.name}/\n"

                # Recursively add subdirectory contents
                next_prefix = prefix + ("    " if is_last_item else "│   ")
                tree_str += build_ascii_tree(
                    item, next_prefix, is_last_item, max_depth, current_depth + 1
                )
            else:
                size = format_size(item.stat().st_size)
                tree_str += f"{prefix}{current_prefix}📄 {item.name} ({size})\n"

    except PermissionError:
        tree_str += f"{prefix}[Permission Denied]\n"

    return tree_str


# Select workspace root - restricted access
root_choices = discover_roots()

if not root_choices:
    st.error(
        "❌ No workspace directories are available. "
        "Contact your administrator to configure workspace access."
    )
    st.info(
        "For security reasons, workspace access is restricted to configured directories. "
        "This is expected behavior in a server deployment."
    )
    st.stop()

root_strs = [str(p) for p in root_choices]
selected_root = st.selectbox("Workspaces root directory", root_strs)
base_dir = Path(selected_root)

# Security check: ensure selected root is in the allowed list
if not is_valid_workspace_path(base_dir, root_choices):
    st.error("❌ Invalid workspace path selected. Access denied.")
    st.stop()

workspace_dirs = get_workspace_dirs(base_dir)

if not workspace_dirs:
    st.warning("No workspace directories found in the selected root.")
    st.info("Please create a workspace or contact your administrator.")
    st.stop()

workspace_choice = st.selectbox("Select workspace", [p.name for p in workspace_dirs])
selected_workspace = next(
    (p for p in workspace_dirs if p.name == workspace_choice), workspace_dirs[0]
)

# Security check: ensure selected workspace is under the allowed root
if not is_valid_workspace_path(selected_workspace, root_choices):
    st.error("❌ Invalid workspace path. Access denied.")
    st.stop()

st.divider()

# Create tabs for different views
tab1, tab2 = st.tabs(["Tree View", "Files List"])

with tab1:
    st.markdown("**Directory Structure**")

    if not any(selected_workspace.iterdir()):
        st.info("Workspace appears to be empty")
    else:
        # Build and display ASCII tree
        tree_output = f"{selected_workspace.name}/\n"
        tree_output += build_ascii_tree(selected_workspace)

        # Display in code block with monospace font for proper alignment
        st.code(tree_output, language="plaintext")

with tab2:
    st.markdown("**All Files**")

    # Collect all files with metadata
    all_files = []
    for item in selected_workspace.rglob("*"):
        # Skip excluded files and directories
        if item.is_file():
            # Check if any part of the path is excluded
            is_file_excluded = False
            for part in item.parts:
                if is_excluded(part):
                    is_file_excluded = True
                    break

            if not is_file_excluded:
                try:
                    rel_path = item.relative_to(selected_workspace)
                    size = item.stat().st_size
                    all_files.append((str(rel_path), size, item))
                except (OSError, ValueError):
                    pass

    if not all_files:
        st.info("No files found in workspace")
    else:
        # Create header columns
        col1, col2, col3, col4 = st.columns([3, 1, 1, 0.5])
        col1.markdown("**File Path**")
        col2.markdown("**Size**")
        col3.markdown("**Type**")
        col4.markdown("**Action**")

        st.divider()

        for file_path, size, full_path in sorted(all_files, key=lambda x: x[0]):
            col1, col2, col3, col4 = st.columns([3, 1, 1, 0.5])

            # File path with icon
            file_ext = full_path.suffix.lower()
            if file_ext in [".log", ".txt"]:
                icon = "📄"
            elif file_ext in [".osw", ".parquet", ".tsv", ".csv"]:
                icon = "📊"
            elif file_ext in [".mzml", ".raw", ".d"]:
                icon = "🧬"
            elif file_ext in [".json", ".ini", ".yaml", ".yml"]:
                icon = "⚙️"
            elif file_ext in [".fasta", ".fa"]:
                icon = "🧬"
            else:
                icon = "📦"

            col1.text(f"{icon} {file_path}")
            col2.text(format_size(size))
            col3.text(file_ext or "file")

            # Download button
            try:
                with open(full_path, "rb") as f:
                    file_data = f.read()
                col4.download_button(
                    label="⬇️",
                    data=file_data,
                    file_name=full_path.name,
                    key=f"download_{full_path}",
                    help="Download file",
                )
            except Exception:
                col4.text("❌")

# Summary statistics
with st.expander("Workspace Statistics"):
    total_size = 0
    file_count = 0
    dir_count = 0

    for item in selected_workspace.rglob("*"):
        # Skip excluded items
        is_item_excluded = False
        for part in item.parts:
            if is_excluded(part):
                is_item_excluded = True
                break

        if not is_item_excluded:
            if item.is_file():
                file_count += 1
                total_size += item.stat().st_size
            elif item.is_dir():
                dir_count += 1

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Files", file_count)
    col2.metric("Total Directories", dir_count)
    col3.metric("Total Size", format_size(total_size))
    col4.metric("Workspace Path", selected_workspace.name)
