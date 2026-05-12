from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable
from zipfile import ZIP_DEFLATED, ZipFile


FASTA_EXTENSIONS = {".fasta", ".fa", ".faa", ".fna"}
LIBRARY_EXTENSIONS = {".traml", ".tsv", ".mrm", ".pqp", ".oswpq"}
MZML_EXTENSIONS = {".mzml"}
XIC_EXTENSIONS = {".xic", ".parquet"}

OPENSWATH_WORKFLOW_NAME = "openswath-workflow"


def openswath_workflow_dir(workspace_dir: Path) -> Path:
    """Return the OpenSwath workflow_dir for *workspace_dir*.

    Matches the convention `WorkflowManager.__init__` uses for an
    `OpenSwathWorkflow` instance (`workspace_dir / "openswath-workflow"`).
    """
    path = Path(workspace_dir, OPENSWATH_WORKFLOW_NAME)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _openswath_input_dir(workspace_dir: Path, key: str) -> Path:
    path = openswath_workflow_dir(workspace_dir) / "input-files" / key
    path.mkdir(parents=True, exist_ok=True)
    return path


def workspace_fasta_dir(workspace_dir: Path) -> Path:
    return _openswath_input_dir(workspace_dir, "fasta")


def workspace_library_dir(workspace_dir: Path) -> Path:
    return _openswath_input_dir(workspace_dir, "libraries")


def workspace_mzml_dir(workspace_dir: Path) -> Path:
    return _openswath_input_dir(workspace_dir, "mzML-files")


def workspace_xic_dir(workspace_dir: Path) -> Path:
    return _openswath_input_dir(workspace_dir, "xic-files")


def list_input_dir_files(directory: Path) -> list[str]:
    """Names of files inside an upload_widget input dir.

    Skips the ``external_files.txt`` manifest itself and resolves the
    basenames of any paths it lists (the non-copy mode of upload_widget
    writes absolute host paths there instead of copying the files).
    """
    names: list[str] = []
    if not directory.exists():
        return names
    names = [
        p.name
        for p in directory.iterdir()
        if p.is_file() and p.name != "external_files.txt"
    ]
    ext_file = directory / "external_files.txt"
    if ext_file.exists():
        names += [
            Path(line.strip()).name
            for line in ext_file.read_text().splitlines()
            if line.strip()
        ]
    return names


def resolve_input_dir_paths(directory: Path) -> list[Path]:
    """Resolve every file in an upload_widget input dir to a full Path.

    Returns the absolute paths of regular files in *directory* (skipping
    ``external_files.txt``) followed by the entries listed in that
    manifest whose paths still exist on disk. Dead manifest entries are
    silently dropped.
    """
    paths: list[Path] = []
    if not directory.exists():
        return paths
    for p in directory.iterdir():
        if p.is_file() and p.name != "external_files.txt":
            paths.append(p)
    ext_file = directory / "external_files.txt"
    if ext_file.exists():
        for line in ext_file.read_text().splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            ext_path = Path(stripped)
            if ext_path.exists():
                paths.append(ext_path)
    return paths


def resolve_input_file(directory: Path, name: str) -> Path | None:
    """Resolve a basename to a full path inside an upload_widget input dir.

    Checks the directory first; if no direct match, falls back to
    ``external_files.txt`` entries with a matching basename. Returns
    ``None`` if nothing matches.
    """
    if not name or not directory.exists():
        return None
    direct = directory / name
    if direct.is_file() and direct.name != "external_files.txt":
        return direct
    ext_file = directory / "external_files.txt"
    if ext_file.exists():
        for line in ext_file.read_text().splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            ext_path = Path(stripped)
            if ext_path.name == name and ext_path.exists():
                return ext_path
    return None


def list_workspace_files(
    directory: Path, valid_extensions: set[str] | None = None
) -> list[Path]:
    if not directory.exists() or not directory.is_dir():
        return []
    files = [path for path in directory.iterdir() if path.is_file()]
    if valid_extensions is not None:
        valid = {ext.lower() for ext in valid_extensions}
        files = [path for path in files if path.suffix.lower() in valid]
    return sorted(files, key=lambda item: item.name.lower())


def save_uploaded_file(uploaded_file, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / getattr(uploaded_file, "name", "uploaded.bin")
    with open(target_path, "wb") as fh:
        fh.write(uploaded_file.getbuffer())
    return target_path


def sync_file_into_directory(source_path: Path, target_dir: Path) -> Path | None:
    source_path = Path(source_path)
    if not source_path.exists() or not source_path.is_file():
        return None

    target_dir.mkdir(parents=True, exist_ok=True)
    dest_path = target_dir / source_path.name

    try:
        if dest_path.exists() and dest_path.resolve() == source_path.resolve():
            return dest_path
    except FileNotFoundError:
        pass

    if dest_path.exists():
        source_stat = source_path.stat()
        dest_stat = dest_path.stat()
        if (
            source_stat.st_size == dest_stat.st_size
            and source_stat.st_mtime <= dest_stat.st_mtime
        ):
            return dest_path

    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    shutil.copy2(source_path, tmp_path)
    tmp_path.replace(dest_path)
    return dest_path


def file_size_label(path: Path) -> str:
    size = Path(path).stat().st_size
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    unit = units[0]
    for unit in units:
        if value < 1024 or unit == units[-1]:
            break
        value /= 1024
    if unit == "B":
        return f"{int(value)} {unit}"
    return f"{value:.1f} {unit}"


def total_size_label(paths: Iterable[Path]) -> str:
    total = sum(Path(path).stat().st_size for path in paths if Path(path).exists())
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(total)
    unit = units[0]
    for unit in units:
        if value < 1024 or unit == units[-1]:
            break
        value /= 1024
    if unit == "B":
        return f"{int(value)} {unit}"
    return f"{value:.1f} {unit}"


def _archive_manifest_path(archive_path: Path) -> Path:
    archive = Path(archive_path)
    return archive.with_suffix(archive.suffix + ".manifest.json")


def _archive_manifest_payload(source_paths: Iterable[Path]) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for path in sorted(
        [Path(item) for item in source_paths if Path(item).exists() and Path(item).is_file()],
        key=lambda item: str(item.resolve()).lower(),
    ):
        stat = path.stat()
        payload.append(
            {
                "path": str(path.resolve()),
                "size": stat.st_size,
                "mtime": stat.st_mtime,
            }
        )
    return payload


def archive_needs_refresh(source_paths: Iterable[Path], archive_path: Path) -> bool:
    archive = Path(archive_path)
    if not archive.exists():
        return True

    manifest_path = _archive_manifest_path(archive)
    if not manifest_path.exists():
        return True

    try:
        with open(manifest_path, encoding="utf-8") as fh:
            saved_manifest = json.load(fh)
    except Exception:
        return True

    return saved_manifest != _archive_manifest_payload(source_paths)


def build_zip_archive(
    source_paths: Iterable[Path], archive_path: Path, relative_to: Path | None = None
) -> None:
    archive = Path(archive_path)
    archive.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = archive.with_suffix(archive.suffix + ".part")
    manifest_path = _archive_manifest_path(archive)
    tmp_manifest_path = manifest_path.with_suffix(manifest_path.suffix + ".part")

    root = relative_to.resolve() if relative_to is not None else None
    files = [Path(path) for path in source_paths if Path(path).exists() and Path(path).is_file()]
    manifest_payload = _archive_manifest_payload(files)

    with ZipFile(tmp_path, "w", ZIP_DEFLATED, allowZip64=True) as zip_file:
        for file_path in sorted(files, key=lambda item: item.name.lower()):
            if root is not None:
                try:
                    arcname = file_path.resolve().relative_to(root)
                except ValueError:
                    arcname = file_path.name
            else:
                arcname = file_path.name
            zip_file.write(file_path, arcname=str(arcname))

    tmp_path.replace(archive)
    with open(tmp_manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest_payload, fh, indent=2)
    tmp_manifest_path.replace(manifest_path)
