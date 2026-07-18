"""Build the source-owned document-extraction-manifest-v1 payload."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Iterator, Mapping

from mineru.integrations.knowhere.contract import (
    CANONICAL_MANIFEST_VERSION,
    CanonicalManifestOptions,
    KnowhereExportError,
    KnowhereExportOptions,
    resolve_artifact_path,
    sha256_file,
)


_IMAGE_REFERENCE_KEYS = {"path", "img_path", "image_path"}
_TABLE_TYPES = {"table", "simple_table", "complex_table"}


def validate_canonical_manifest_options(
    options: CanonicalManifestOptions,
) -> None:
    """Reject incomplete source identity before a parser run starts."""
    for field_name in ("source_id", "source_version_id", "extraction_run_id"):
        value = getattr(options, field_name)
        if not isinstance(value, str) or not value.strip():
            raise KnowhereExportError(
                f"Canonical manifest requires a non-empty {field_name}."
            )
    if not isinstance(options.accelerator_profile, str) or not options.accelerator_profile.strip():
        raise KnowhereExportError(
            "Canonical manifest requires a non-empty accelerator_profile."
        )
    if not isinstance(options.model_identifiers, dict):
        raise KnowhereExportError("Canonical model_identifiers must be an object.")


def _configuration_sha256(options: KnowhereExportOptions, effective_backend: str) -> str:
    configuration = {
        "backend_requested": options.backend,
        "backend_effective": effective_backend,
        "method": options.method,
        "language": options.language,
        "formula_enabled": options.formula_enabled,
        "table_enabled": options.table_enabled,
        "image_analysis_enabled": options.image_analysis_enabled,
        "offline": options.offline,
        "server_url_configured": bool(options.server_url and options.server_url.strip()),
    }
    encoded = json.dumps(
        configuration,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _artifact_id(artifact_type: str, relative_path: str) -> str:
    return f"{artifact_type}:{relative_path}"


def _relative_output_path(path: Path, *, output_root: Path) -> str:
    try:
        return path.resolve().relative_to(output_root).as_posix()
    except ValueError as error:
        raise KnowhereExportError(
            f"Canonical output path escapes the output root: {path}"
        ) from error


def _output_artifacts(
    legacy_manifest: Mapping[str, Any],
    *,
    output_root: Path,
) -> tuple[list[dict[str, str]], dict[str, str], Path]:
    declared_artifacts = legacy_manifest.get("artifacts")
    if not isinstance(declared_artifacts, dict):
        raise KnowhereExportError("Legacy manifest has no artifact inventory.")

    outputs: list[dict[str, str]] = []
    artifact_ids: dict[str, str] = {}
    images_dir: Path | None = None

    for artifact_name, artifact in sorted(declared_artifacts.items()):
        if not isinstance(artifact, dict):
            raise KnowhereExportError(
                f"Legacy artifact entry is not an object: {artifact_name}"
            )
        relative_path = artifact.get("path")
        if not isinstance(relative_path, str) or not relative_path:
            raise KnowhereExportError(
                f"Legacy artifact has no relative path: {artifact_name}"
            )
        resolved_path = resolve_artifact_path(output_root, relative_path)
        if artifact_name == "images_dir":
            if not resolved_path.is_dir():
                raise KnowhereExportError("Canonical images directory is missing.")
            images_dir = resolved_path
            continue
        declared_sha256 = artifact.get("sha256")
        if not isinstance(declared_sha256, str) or not declared_sha256:
            raise KnowhereExportError(
                f"Legacy artifact has no SHA-256 value: {artifact_name}"
            )
        if not resolved_path.is_file():
            raise KnowhereExportError(
                f"Canonical artifact file is missing: {relative_path}"
            )
        actual_sha256 = sha256_file(resolved_path)
        if actual_sha256 != declared_sha256:
            raise KnowhereExportError(
                f"Canonical artifact SHA-256 mismatch: {relative_path}"
            )
        artifact_type = artifact_name
        artifact_id = _artifact_id(artifact_type, Path(relative_path).as_posix())
        artifact_ids[artifact_name] = artifact_id
        outputs.append(
            {
                "artifact_id": artifact_id,
                "artifact_type": artifact_type,
                "relative_path": Path(relative_path).as_posix(),
                "sha256": actual_sha256,
            }
        )

    if images_dir is None:
        raise KnowhereExportError("Canonical manifest requires an images directory.")

    for image_path in sorted(path for path in images_dir.rglob("*") if path.is_file()):
        relative_path = _relative_output_path(image_path, output_root=output_root)
        artifact_id = _artifact_id("image", relative_path)
        outputs.append(
            {
                "artifact_id": artifact_id,
                "artifact_type": "image",
                "relative_path": relative_path,
                "sha256": sha256_file(image_path),
            }
        )

    outputs.sort(key=lambda item: item["relative_path"])
    return outputs, artifact_ids, images_dir


def _iter_string_references(
    value: Any,
    *,
    key: str | None = None,
) -> Iterator[str]:
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            yield from _iter_string_references(child_value, key=str(child_key))
    elif isinstance(value, list):
        for child_value in value:
            yield from _iter_string_references(child_value, key=key)
    elif key in _IMAGE_REFERENCE_KEYS and isinstance(value, str) and value.strip():
        yield value


def _resolve_image_reference(
    reference: str,
    *,
    parse_dir: Path,
    output_root: Path,
) -> Path | None:
    normalized_path = Path(reference.replace("\\", "/"))
    if normalized_path.is_absolute() or ".." in normalized_path.parts:
        raise KnowhereExportError(
            f"Image reference escapes the output root: {reference!r}"
        )
    normalized = normalized_path.as_posix()
    candidate_paths = [parse_dir / normalized]
    if not normalized.startswith("images/"):
        candidate_paths.append(parse_dir / "images" / normalized)
    for candidate in candidate_paths:
        try:
            resolved = candidate.resolve()
            resolved.relative_to(output_root)
        except ValueError as error:
            raise KnowhereExportError(
                f"Image reference escapes the output root: {reference!r}"
            ) from error
        if resolved.is_file():
            return resolved
    return None


def _page_blocks_and_references(
    content_list_v2: list[list[Any]],
    *,
    content_list_v2_artifact_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, set[str]]]:
    page_blocks: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    image_references: dict[str, set[str]] = {}

    for page_index, page in enumerate(content_list_v2):
        for block_index, block in enumerate(page):
            if not isinstance(block, dict):
                raise KnowhereExportError(
                    "Canonical content_list_v2 blocks must be objects."
                )
            block_id = f"page-{page_index + 1:04d}-block-{block_index + 1:04d}"
            locator = {
                "artifact_id": content_list_v2_artifact_id,
                "json_pointer": f"/{page_index}/{block_index}",
            }
            block_type = str(block.get("type", "unknown"))
            page_block = {
                "block_id": block_id,
                "page_number": page_index + 1,
                "page_index": page_index,
                "block_index": block_index,
                "block_type": block_type,
                "native_locator": locator,
                "content": block,
            }
            page_blocks.append(page_block)

            references = set(_iter_string_references(block))
            for reference in references:
                image_references.setdefault(reference, set()).add(block_id)

            if block_type in _TABLE_TYPES:
                tables.append(
                    {
                        "table_id": f"{block_id}:table",
                        "block_id": block_id,
                        "page_number": page_index + 1,
                        "native_locator": locator,
                        "content": block.get("content", block),
                    }
                )

    return page_blocks, tables, image_references


def _image_records(
    image_paths: list[Path],
    image_references: Mapping[str, set[str]],
    *,
    parse_dir: Path,
    output_root: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    for image_path in image_paths:
        relative_path = _relative_output_path(image_path, output_root=output_root)
        matching_blocks: set[str] = set()
        for reference, block_ids in image_references.items():
            resolved_reference = _resolve_image_reference(
                reference,
                parse_dir=parse_dir,
                output_root=output_root,
            )
            if resolved_reference is not None and resolved_reference == image_path.resolve():
                matching_blocks.update(block_ids)
        records.append(
            {
                "image_id": _artifact_id("image", relative_path),
                "relative_path": relative_path,
                "sha256": sha256_file(image_path),
                "block_ids": sorted(matching_blocks),
            }
        )

    for reference in sorted(image_references):
        if _resolve_image_reference(
            reference,
            parse_dir=parse_dir,
            output_root=output_root,
        ) is None:
            warnings.append(f"unresolved_image_reference:{reference}")
    return records, warnings


def _ocr_profile(
    options: KnowhereExportOptions,
    *,
    effective_backend: str,
) -> dict[str, str] | None:
    if options.method.strip().lower() != "ocr":
        return None
    return {
        "mode": "ocr",
        "language": options.language,
        "backend": effective_backend,
    }


def build_document_extraction_manifest(
    *,
    legacy_manifest: Mapping[str, Any],
    content_list_v2: list[list[Any]],
    output_root: Path,
    parse_dir: Path,
    options: KnowhereExportOptions,
    canonical_options: CanonicalManifestOptions,
    repository_sha: str,
    package_version: str,
    effective_backend: str,
) -> dict[str, Any]:
    """Build a canonical manifest without adding RA or sufficiency state."""
    validate_canonical_manifest_options(canonical_options)
    if not isinstance(repository_sha, str) or not repository_sha.strip():
        raise KnowhereExportError(
            "Canonical manifest requires a producer revision (repository SHA)."
        )
    normalized_repository_sha = repository_sha.strip().lower()
    if not (7 <= len(normalized_repository_sha) <= 64) or any(
        character not in "0123456789abcdef" for character in normalized_repository_sha
    ):
        raise KnowhereExportError(
            "Canonical manifest producer revision must be a lowercase hexadecimal SHA."
        )

    source = legacy_manifest.get("source")
    if not isinstance(source, dict):
        raise KnowhereExportError("Legacy manifest has no source identity.")
    input_sha256 = source.get("sha256")
    if not isinstance(input_sha256, str) or len(input_sha256) != 64:
        raise KnowhereExportError("Legacy manifest source SHA-256 is invalid.")

    outputs, artifact_ids, images_dir = _output_artifacts(
        legacy_manifest,
        output_root=output_root,
    )
    content_list_v2_artifact_id = artifact_ids.get("content_list_v2")
    if content_list_v2_artifact_id is None:
        raise KnowhereExportError(
            "Canonical manifest requires a content_list_v2 artifact."
        )
    page_blocks, tables, image_references = _page_blocks_and_references(
        content_list_v2,
        content_list_v2_artifact_id=content_list_v2_artifact_id,
    )
    image_paths = sorted(path for path in images_dir.rglob("*") if path.is_file())
    images, image_warnings = _image_records(
        image_paths,
        image_references,
        parse_dir=parse_dir,
        output_root=output_root,
    )

    warnings = [
        warning
        for warning in legacy_manifest.get("warnings", [])
        if isinstance(warning, str)
    ]
    warnings.extend(image_warnings)
    if not canonical_options.model_identifiers:
        warnings.append("model_identifiers_not_exposed_by_adapter")

    native_page_count = len(content_list_v2)
    legacy_page_count = legacy_manifest.get("document", {}).get("logical_page_count")
    if legacy_page_count != native_page_count:
        raise KnowhereExportError(
            "Canonical page count does not match the legacy manifest."
        )

    return {
        "contract_version": CANONICAL_MANIFEST_VERSION,
        "extraction_run_id": canonical_options.extraction_run_id.strip(),
        "source_id": canonical_options.source_id.strip(),
        "source_version_id": canonical_options.source_version_id.strip(),
        "input_sha256": input_sha256,
        "mineru_identity": {
            "repository_sha": normalized_repository_sha,
            "package_version": package_version,
            "backend": effective_backend,
            "model_identifiers": dict(canonical_options.model_identifiers),
            "configuration_sha256": _configuration_sha256(options, effective_backend),
        },
        "runtime_identity": {
            "os": platform.system() or "unknown",
            "architecture": platform.machine() or "unknown",
            "accelerator_profile": canonical_options.accelerator_profile.strip(),
        },
        "status": "completed",
        "native_page_count": native_page_count,
        "outputs": outputs,
        "page_blocks": page_blocks,
        "tables": tables,
        "images": images,
        "ocr_profile": _ocr_profile(options, effective_backend=effective_backend),
        "warnings": warnings,
        "errors": [],
        "fallback_used": False,
        "derivative_not_native_source_evidence": True,
        "does_not_establish_source_sufficiency": True,
    }
