"""Run MinerU directly and publish a validated Knowhere artifact manifest."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from mineru.cli.common import do_parse, read_fn
from mineru.cli.output_paths import resolve_parse_dir
from mineru.integrations.knowhere.contract import (
    ARTIFACT_SCHEMA_VERSION,
    HTTP_CLIENT_BACKENDS,
    SUPPORTED_SUFFIXES,
    KnowhereExportError,
    KnowhereExportOptions,
    resolve_artifact_path,
    sha256_file,
)
from mineru.version import __version__


_OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "MODELSCOPE_OFFLINE": "1",
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


@contextmanager
def _offline_environment(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return
    previous = {name: os.environ.get(name) for name in _OFFLINE_ENVIRONMENT}
    os.environ.update(_OFFLINE_ENVIRONMENT)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _validate_options(options: KnowhereExportOptions) -> tuple[Path, Path, str]:
    source = options.input_path.expanduser().resolve()
    if not source.is_file():
        raise KnowhereExportError(f"Input file does not exist: {source.name}")
    suffix = source.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise KnowhereExportError("Input must be one local PDF or DOCX file.")

    backend = options.backend.strip().lower()
    if options.offline and backend in HTTP_CLIENT_BACKENDS:
        raise KnowhereExportError(
            f"Backend {backend!r} is not allowed in offline mode."
        )
    if options.offline and options.server_url and options.server_url.strip():
        raise KnowhereExportError("A server URL is not allowed in offline mode.")

    output_root = options.output_root.expanduser().resolve()
    return source, output_root, suffix


def _load_json(path: Path, *, artifact_name: str, expected_type: type) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise KnowhereExportError(
            f"Invalid required {artifact_name} artifact: {path.name}"
        ) from error
    if not isinstance(value, expected_type):
        raise KnowhereExportError(
            f"Invalid required {artifact_name} artifact structure: {path.name}"
        )
    return value


def _required_artifacts(
    output_root: Path,
    parse_dir: Path,
    stem: str,
) -> tuple[dict[str, dict[str, str]], int]:
    declared_paths = {
        "markdown": parse_dir / f"{stem}.md",
        "middle_json": parse_dir / f"{stem}_middle.json",
        "content_list": parse_dir / f"{stem}_content_list.json",
        "content_list_v2": parse_dir / f"{stem}_content_list_v2.json",
    }
    artifacts: dict[str, dict[str, str]] = {}
    for name, path in declared_paths.items():
        if not path.is_file():
            raise KnowhereExportError(
                f"Missing required {name} artifact: {path.name}"
            )
        relative_path = path.resolve().relative_to(output_root).as_posix()
        resolve_artifact_path(output_root, relative_path)
        artifacts[name] = {"path": relative_path, "sha256": sha256_file(path)}

    _load_json(
        declared_paths["middle_json"],
        artifact_name="middle_json",
        expected_type=dict,
    )
    _load_json(
        declared_paths["content_list"],
        artifact_name="content_list",
        expected_type=list,
    )
    content_list_v2 = _load_json(
        declared_paths["content_list_v2"],
        artifact_name="content_list_v2",
        expected_type=list,
    )
    if any(not isinstance(page, list) for page in content_list_v2):
        raise KnowhereExportError(
            "Invalid required content_list_v2 artifact structure: "
            f"{declared_paths['content_list_v2'].name}"
        )

    images_dir = parse_dir / "images"
    if not images_dir.is_dir():
        raise KnowhereExportError("Missing required images_dir artifact: images")
    images_relative = images_dir.resolve().relative_to(output_root).as_posix()
    resolve_artifact_path(output_root, images_relative)
    artifacts["images_dir"] = {"path": images_relative}
    return artifacts, len(content_list_v2)


def _git_commit() -> str | None:
    project_root = Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def _write_manifest_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _parse_error_message(error: Exception, *, offline: bool) -> str:
    detail = " ".join(str(error).split())[:500]
    prefix = "MinerU parsing failed"
    if offline:
        prefix += "; offline runs require pre-downloaded local models"
    return f"{prefix}: {type(error).__name__}: {detail}"


def run_knowhere_export(options: KnowhereExportOptions) -> Path:
    """Run local MinerU parsing and return the generated manifest path."""
    source, output_root, suffix = _validate_options(options)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "mineru_manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()

    started_at = _utc_now()
    source_bytes = read_fn(source, suffix.lstrip("."))
    try:
        with _offline_environment(options.offline):
            do_parse(
                str(output_root),
                [source.stem],
                [source_bytes],
                [options.language],
                backend=options.backend,
                parse_method=options.method,
                formula_enable=options.formula_enabled,
                table_enable=options.table_enabled,
                server_url=options.server_url,
                f_draw_layout_bbox=False,
                f_draw_span_bbox=False,
                f_dump_md=True,
                f_dump_middle_json=True,
                f_dump_model_output=False,
                f_dump_orig_pdf=False,
                f_dump_content_list=True,
                image_analysis=options.image_analysis_enabled,
            )
    except Exception as error:
        raise KnowhereExportError(
            _parse_error_message(error, offline=options.offline)
        ) from error

    is_office = suffix == ".docx"
    parse_dir = resolve_parse_dir(
        output_root,
        source.stem,
        options.backend,
        options.method,
        is_office=is_office,
    ).resolve()
    try:
        parse_dir.relative_to(output_root)
    except ValueError as error:
        raise KnowhereExportError("Resolved parse directory escapes output root.") from error

    artifacts, logical_page_count = _required_artifacts(
        output_root,
        parse_dir,
        source.stem,
    )
    effective_backend = "office" if is_office else options.backend
    manifest: dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "completed",
        "source": {
            "filename": source.name,
            "suffix": suffix,
            "sha256": sha256_file(source),
            "size_bytes": source.stat().st_size,
        },
        "parser": {
            "name": "MinerU",
            "version": __version__,
            "git_commit": _git_commit(),
            "backend_requested": options.backend,
            "backend_effective": effective_backend,
            "method": options.method,
            "language": options.language,
            "formula_enabled": options.formula_enabled,
            "table_enabled": options.table_enabled,
            "image_analysis_enabled": options.image_analysis_enabled,
        },
        "execution": {
            "mode": "local-direct-python",
            "offline_requested": options.offline,
            "offline_verified": False,
            "started_at": started_at,
            "completed_at": _utc_now(),
        },
        "document": {"logical_page_count": logical_page_count},
        "artifacts": artifacts,
        "warnings": [],
    }
    _write_manifest_atomic(manifest_path, manifest)
    return manifest_path
