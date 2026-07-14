"""Contract primitives shared by the Knowhere export adapter."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


ARTIFACT_SCHEMA_VERSION = "knowhere-mineru-artifacts/1.0"
SUPPORTED_SUFFIXES = {".pdf", ".docx"}
HTTP_CLIENT_BACKENDS = {"vlm-http-client", "hybrid-http-client"}


class KnowhereExportError(RuntimeError):
    """Raised when the local export cannot produce a valid artifact bundle."""


@dataclass(frozen=True)
class KnowhereExportOptions:
    """Options for one local MinerU parse and artifact export."""

    input_path: Path
    output_root: Path
    backend: str
    method: str
    language: str
    formula_enabled: bool
    table_enabled: bool
    image_analysis_enabled: bool
    offline: bool
    server_url: str | None = None


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a local file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_artifact_path(output_root: Path, relative_path: str) -> Path:
    """Resolve a portable artifact path while confining it to *output_root*."""
    candidate_path = Path(relative_path)
    if candidate_path.is_absolute() or ".." in candidate_path.parts:
        raise KnowhereExportError(
            f"Artifact path must not escape the output root: {relative_path!r}"
        )

    root = output_root.resolve()
    candidate = (root / candidate_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise KnowhereExportError(
            f"Artifact path must not escape the output root: {relative_path!r}"
        ) from error
    return candidate

