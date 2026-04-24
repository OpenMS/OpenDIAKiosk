from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable
from zipfile import ZIP_DEFLATED, ZipFile


FASTA_EXTENSIONS = {".fasta", ".fa", ".faa", ".fna"}
LIBRARY_EXTENSIONS = {".traml", ".tsv", ".mrm", ".pqp", ".oswpq"}


def workspace_fasta_dir(workspace_dir: Path) -> Path:
    path = Path(workspace_dir, "input-files", "fasta")
    path.mkdir(parents=True, exist_ok=True)
    return path


def workspace_library_dir(workspace_dir: Path) -> Path:
    path = Path(workspace_dir, "input-files", "libraries")
    path.mkdir(parents=True, exist_ok=True)
    return path


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
