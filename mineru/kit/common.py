"""Common helpers for mineru-kit commands."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Literal, TypeAlias

from ..filetypes import PARSEABLE_EXTENSIONS
from ..parser.base import ParseResult
from ..render.writer import FileBasedDataWriter
from ..types import Tier
from ..utils.image_payload import validate_image_sidecar_path

KitFormat = Literal["markdown", "middle_json", "zip"]
LocalTier: TypeAlias = Tier

PARSEABLE_SUFFIXES = frozenset(f".{ext}" for ext in PARSEABLE_EXTENSIONS)
OUTPUT_FILE_SUFFIXES = {
    "markdown": ".md",
    "middle_json": ".json",
    "zip": ".zip",
}


def expand_input_paths(inputs: list[str]) -> list[Path]:
    paths = [Path(raw).expanduser() for raw in inputs]
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            for child in sorted(path.iterdir()):
                if child.is_file() and child.suffix.lower() in PARSEABLE_SUFFIXES:
                    expanded.append(child)
        else:
            expanded.append(path)
    return expanded


def ensure_supported_inputs(paths: list[Path]) -> None:
    if not paths:
        raise ValueError("No input files found.")
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_dir():
            raise ValueError(f"Directory input must be expanded before validation: {path}")
        if path.suffix.lower() not in PARSEABLE_SUFFIXES:
            raise ValueError(f"Unsupported file type: {path}")


def is_output_path_file_like(path: Path) -> bool:
    if path.exists():
        return path.is_file()
    return path.suffix.lower() in OUTPUT_FILE_SUFFIXES.values()


def resolve_single_output_path(source: Path, output: Path, format: KitFormat) -> Path:
    if output.exists() and output.is_dir():
        return output / f"{source.stem}{OUTPUT_FILE_SUFFIXES[format]}"
    if not output.exists() and output.suffix == "":
        return output / f"{source.stem}{OUTPUT_FILE_SUFFIXES[format]}"
    return output


def resolve_batch_output_paths(paths: list[Path], output: Path, format: KitFormat) -> dict[Path, Path]:
    if any(path.parent == path for path in paths):
        raise ValueError("Invalid input path.")
    multi_input = len(paths) > 1
    has_directory_input = False
    if multi_input or has_directory_input:
        if is_output_path_file_like(output):
            raise ValueError("When input is multiple files or directories, --output must be a directory path.")

    output_dir = output
    if output.exists() and output.is_file():
        raise ValueError("When input is multiple files or directories, --output must be a directory path.")

    destinations: dict[Path, Path] = {}
    seen: dict[Path, Path] = {}
    for source in paths:
        dest = (
            resolve_single_output_path(source, output_dir, format)
            if len(paths) == 1
            else output_dir / f"{source.stem}{OUTPUT_FILE_SUFFIXES[format]}"
        )
        existing = seen.get(dest)
        if existing is not None:
            raise ValueError(f"Output name collision: {existing.name} and {source.name} both map to {dest}")
        seen[dest] = source
        destinations[source] = dest
    return destinations


def effective_local_tier_and_backend(tier: Tier | None, backend: str | None) -> tuple[Tier, str]:
    from ..parser.tier import backend_for_tier, resolve_tier_and_backend

    if tier is None and backend is None:
        return "standard", backend_for_tier("standard")
    resolved_tier, resolved_backend = resolve_tier_and_backend(tier=tier, backend=backend)
    if tier is None and backend is None:
        resolved_backend = backend_for_tier("standard")
    return resolved_tier, resolved_backend


def build_remote_api_url(remote: bool, remote_url: str | None) -> str | None:
    if remote and remote_url:
        raise ValueError("--remote and --remote-url are mutually exclusive.")
    if remote_url:
        return remote_url
    if remote:
        return "https://mineru.net/api"
    return None


def save_parse_result(result: ParseResult, dest: Path, format: KitFormat) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if format == "markdown":
        _write_utf8_text(dest, result.markdown())
        _write_image_sidecars(dest.parent, result.images())
        return
    if format == "middle_json":
        _write_utf8_text(dest, result.to_json())
        _write_image_sidecars(dest.parent, result.images())
        return
    if format == "zip":
        tmp_dir = dest.parent / f".{dest.stem}"
        writer = FileBasedDataWriter(str(tmp_dir))
        result.save(writer)
        with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
            for child in sorted(tmp_dir.rglob("*")):
                if child.is_file():
                    zf.write(child, arcname=child.relative_to(tmp_dir).as_posix())
        for child in sorted(tmp_dir.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink()
            else:
                child.rmdir()
        if tmp_dir.exists():
            tmp_dir.rmdir()
        return
    raise ValueError(f"Unsupported format: {format}")


def _write_utf8_text(path: Path, content: str) -> None:
    path.write_bytes(content.encode("utf-8", errors="replace"))


def _resolve_safe_sidecar_path(output_dir: Path, image_path: str) -> str:
    """校验图片 sidecar 路径必须落在输出目录内，并返回安全的相对路径。"""
    safe_image_path = validate_image_sidecar_path(image_path)
    output_root = output_dir.resolve()
    target_path = (output_root / safe_image_path).resolve()
    try:
        target_path.relative_to(output_root)
    except ValueError as exc:
        raise ValueError(f"Unsafe image sidecar path: {image_path}") from exc
    return target_path.relative_to(output_root).as_posix()


def _write_image_sidecars(output_dir: Path, images: dict[str, bytes]) -> None:
    """将 public middle_json 引用的图片 sidecar 写到输出目录，避免 image_path 悬空。"""
    writer = FileBasedDataWriter(str(output_dir))
    safe_images = [
        (_resolve_safe_sidecar_path(output_dir, image_path), image_bytes) for image_path, image_bytes in images.items()
    ]
    for image_path, image_bytes in safe_images:
        writer.write(image_path, image_bytes)


def parse_result_payload(path: Path, dest: Path, format: KitFormat) -> dict[str, str]:
    return {
        "input": str(path),
        "output": str(dest),
        "format": format,
    }
