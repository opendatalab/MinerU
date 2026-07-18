# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any, NoReturn

import pytest

from mineru.integrations.knowhere.contract import (
    CanonicalManifestOptions,
    KnowhereExportError,
    KnowhereExportOptions,
    resolve_artifact_path,
)
from mineru.integrations.knowhere.runner import run_knowhere_export


def test_pipeline_extra_declares_legacy_ocr_runtime_dependency() -> None:
    project_root = Path(__file__).resolve().parents[2]
    project = tomllib.loads(
        (project_root / "pyproject.toml").read_text(encoding="utf-8")
    )
    pipeline_dependencies = project["project"]["optional-dependencies"]["pipeline"]

    assert any(
        dependency.split(";", 1)[0].strip().startswith("six")
        for dependency in pipeline_dependencies
    )


def _write_parser_outputs(
    output_root: Path,
    *,
    stem: str,
    parse_dir_name: str,
    missing: str | None = None,
) -> None:
    parse_dir = output_root / stem / parse_dir_name
    images_dir = parse_dir / "images"
    images_dir.mkdir(parents=True)
    outputs = {
        "markdown": (parse_dir / f"{stem}.md", "# Synthetic\n"),
        "middle_json": (
            parse_dir / f"{stem}_middle.json",
            json.dumps({"pdf_info": [{"page_idx": 0}]}),
        ),
        "content_list": (
            parse_dir / f"{stem}_content_list.json",
            json.dumps([{"type": "text", "text": "Synthetic"}]),
        ),
        "content_list_v2": (
            parse_dir / f"{stem}_content_list_v2.json",
            json.dumps([[{"type": "paragraph", "content": []}]]),
        ),
    }
    for name, (path, content) in outputs.items():
        if name != missing:
            path.write_text(content, encoding="utf-8")


def _options(
    source: Path,
    output_root: Path,
    **overrides: Any,
) -> KnowhereExportOptions:
    values = {
        "input_path": source,
        "output_root": output_root,
        "backend": "pipeline",
        "method": "auto",
        "language": "en",
        "formula_enabled": True,
        "table_enabled": True,
        "image_analysis_enabled": False,
        "offline": True,
    }
    values.update(overrides)
    return KnowhereExportOptions(**values)


def _install_fake_parser(
    monkeypatch: pytest.MonkeyPatch,
    *,
    parse_dir_name: str,
    missing: str | None = None,
) -> None:
    from mineru.integrations.knowhere import runner

    def fake_do_parse(
        output_dir: str,
        pdf_file_names: list[str],
        _bytes: list[bytes],
        _languages: list[str],
        **_kwargs: Any,
    ) -> None:
        _write_parser_outputs(
            Path(output_dir),
            stem=pdf_file_names[0],
            parse_dir_name=parse_dir_name,
            missing=missing,
        )

    monkeypatch.setattr(runner, "do_parse", fake_do_parse)


def test_pdf_export_writes_completed_manifest_with_required_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "report.pdf"
    source.write_bytes(b"%PDF-1.7 synthetic")
    output_root = tmp_path / "output"
    _install_fake_parser(monkeypatch, parse_dir_name="auto")

    manifest_path = run_knowhere_export(_options(source, output_root))

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "knowhere-mineru-artifacts/1.0"
    assert manifest["status"] == "completed"
    assert manifest["parser"]["backend_effective"] == "pipeline"
    assert manifest["execution"]["mode"] == "local-direct-python"
    assert manifest["execution"]["offline_verified"] is False
    assert manifest["document"]["logical_page_count"] == 1
    assert set(manifest["artifacts"]) == {
        "markdown",
        "middle_json",
        "content_list",
        "content_list_v2",
        "images_dir",
    }
    assert not (output_root / "document-extraction-manifest-v1.json").exists()


def test_docx_export_resolves_office_parse_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "report.docx"
    source.write_bytes(b"PK synthetic docx")
    output_root = tmp_path / "output"
    _install_fake_parser(monkeypatch, parse_dir_name="office")

    manifest_path = run_knowhere_export(_options(source, output_root))

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["parser"]["backend_effective"] == "office"
    assert manifest["artifacts"]["middle_json"]["path"] == (
        "report/office/report_middle.json"
    )


@pytest.mark.parametrize("missing", ["middle_json", "content_list_v2"])
def test_export_fails_when_required_json_artifact_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
) -> None:
    source = tmp_path / "report.pdf"
    source.write_bytes(b"%PDF-1.7 synthetic")
    output_root = tmp_path / "output"
    _install_fake_parser(monkeypatch, parse_dir_name="auto", missing=missing)

    with pytest.raises(KnowhereExportError, match=missing):
        run_knowhere_export(_options(source, output_root))

    assert not (output_root / "mineru_manifest.json").exists()


def test_invalid_extension_is_rejected_before_parsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "report.txt"
    source.write_text("not supported", encoding="utf-8")
    from mineru.integrations.knowhere import runner

    def unexpected_parse(*_args: Any, **_kwargs: Any) -> NoReturn:
        raise AssertionError("parser must not be called")

    monkeypatch.setattr(runner, "do_parse", unexpected_parse)

    with pytest.raises(KnowhereExportError, match="PDF or DOCX"):
        run_knowhere_export(_options(source, tmp_path / "output"))


@pytest.mark.parametrize("backend", ["vlm-http-client", "hybrid-http-client"])
def test_offline_mode_rejects_http_client_backends(
    tmp_path: Path,
    backend: str,
) -> None:
    source = tmp_path / "report.pdf"
    source.write_bytes(b"%PDF-1.7 synthetic")

    with pytest.raises(KnowhereExportError, match="offline mode"):
        run_knowhere_export(
            _options(source, tmp_path / "output", backend=backend)
        )


def test_offline_mode_rejects_remote_server_url(tmp_path: Path) -> None:
    source = tmp_path / "report.pdf"
    source.write_bytes(b"%PDF-1.7 synthetic")

    with pytest.raises(KnowhereExportError, match="server URL"):
        run_knowhere_export(
            _options(
                source,
                tmp_path / "output",
                server_url="https://parser.example.test",
            )
        )


def test_manifest_paths_are_relative_and_hashes_are_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "report.pdf"
    source.write_bytes(b"%PDF-1.7 synthetic")
    _install_fake_parser(monkeypatch, parse_dir_name="auto")

    first = json.loads(
        run_knowhere_export(_options(source, tmp_path / "first")).read_text(
            encoding="utf-8"
        )
    )
    second = json.loads(
        run_knowhere_export(_options(source, tmp_path / "second")).read_text(
            encoding="utf-8"
        )
    )

    for name, artifact in first["artifacts"].items():
        assert not Path(artifact["path"]).is_absolute(), name
        assert ".." not in Path(artifact["path"]).parts, name
    assert first["source"]["sha256"] == second["source"]["sha256"]
    for name in ("markdown", "middle_json", "content_list", "content_list_v2"):
        assert first["artifacts"][name]["sha256"] == second["artifacts"][name][
            "sha256"
        ]


def test_artifact_path_cannot_escape_output_root(tmp_path: Path) -> None:
    output_root = tmp_path / "output"

    with pytest.raises(KnowhereExportError, match="escape"):
        resolve_artifact_path(output_root, "../outside.json")


def test_parse_failure_does_not_write_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "report.pdf"
    source.write_bytes(b"%PDF-1.7 synthetic")
    output_root = tmp_path / "output"
    from mineru.integrations.knowhere import runner

    def failing_parse(*_args: Any, **_kwargs: Any) -> NoReturn:
        raise RuntimeError("synthetic parse failure")

    monkeypatch.setattr(runner, "do_parse", failing_parse)

    with pytest.raises(KnowhereExportError, match="MinerU parsing failed"):
        run_knowhere_export(_options(source, output_root))

    assert not (output_root / "mineru_manifest.json").exists()


def test_export_requires_no_api_key_and_sets_offline_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "report.pdf"
    source.write_bytes(b"%PDF-1.7 synthetic")
    output_root = tmp_path / "output"
    monkeypatch.delenv("MINERU_API_KEYS", raising=False)
    from mineru.integrations.knowhere import runner

    seen_environment: dict[str, str | None] = {}

    def fake_do_parse(
        output_dir: str,
        pdf_file_names: list[str],
        _bytes: list[bytes],
        _languages: list[str],
        **_kwargs: Any,
    ) -> None:
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "MODELSCOPE_OFFLINE"):
            seen_environment[name] = runner.os.environ.get(name)
        _write_parser_outputs(
            Path(output_dir), stem=pdf_file_names[0], parse_dir_name="auto"
        )

    monkeypatch.setattr(runner, "do_parse", fake_do_parse)

    run_knowhere_export(_options(source, output_root))

    assert seen_environment == {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "MODELSCOPE_OFFLINE": "1",
    }


def _install_rich_fake_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mineru.integrations.knowhere import runner

    def fake_do_parse(
        output_dir: str,
        pdf_file_names: list[str],
        _bytes: list[bytes],
        _languages: list[str],
        **_kwargs: Any,
    ) -> None:
        stem = pdf_file_names[0]
        parse_dir = Path(output_dir) / stem / "auto"
        images_dir = parse_dir / "images"
        images_dir.mkdir(parents=True)
        (images_dir / "figure.png").write_bytes(b"synthetic image")
        (parse_dir / f"{stem}.md").write_text(
            "# Synthetic\n", encoding="utf-8"
        )
        (parse_dir / f"{stem}_middle.json").write_text(
            json.dumps(
                {
                    "pdf_info": [
                        {"page_idx": 0},
                        {"page_idx": 1},
                    ]
                }
            ),
            encoding="utf-8",
        )
        (parse_dir / f"{stem}_content_list.json").write_text(
            json.dumps(
                [
                    {"type": "text", "text": "Synthetic"},
                    {"type": "table", "table_body": "<table></table>"},
                    {"type": "image", "img_path": "images/figure.png"},
                ]
            ),
            encoding="utf-8",
        )
        (parse_dir / f"{stem}_content_list_v2.json").write_text(
            json.dumps(
                [
                    [
                        {
                            "type": "paragraph",
                            "content": {
                                "paragraph_content": [
                                    {"type": "text", "content": "Synthetic"}
                                ]
                            },
                        },
                        {
                            "type": "table",
                            "content": {
                                "html": "<table></table>",
                                "image_source": {"path": "images/figure.png"},
                            },
                        },
                    ],
                    [
                        {
                            "type": "image",
                            "content": {
                                "image_source": {"path": "images/figure.png"}
                            },
                        }
                    ],
                ]
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(runner, "do_parse", fake_do_parse)


def test_canonical_manifest_is_opt_in_and_maps_page_table_and_image_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "report.pdf"
    source.write_bytes(b"%PDF-1.7 synthetic")
    output_root = tmp_path / "output"
    _install_rich_fake_parser(monkeypatch)

    canonical_path = run_knowhere_export(
        _options(
            source,
            output_root,
            canonical_manifest=CanonicalManifestOptions(
                source_id="SRC-SYNTHETIC-001",
                source_version_id="SRC-SYNTHETIC-001-V001",
                extraction_run_id="EXT-SYNTHETIC-001",
                accelerator_profile="cpu",
                model_identifiers={"layout": "synthetic-fixture"},
            ),
        )
    )

    assert canonical_path.name == "document-extraction-manifest-v1.json"
    assert (output_root / "mineru_manifest.json").exists()
    manifest = json.loads(canonical_path.read_text(encoding="utf-8"))
    assert manifest["contract_version"] == "document-extraction-manifest-v1"
    assert manifest["source_id"] == "SRC-SYNTHETIC-001"
    assert manifest["source_version_id"] == "SRC-SYNTHETIC-001-V001"
    assert manifest["extraction_run_id"] == "EXT-SYNTHETIC-001"
    assert manifest["native_page_count"] == 2
    assert len(manifest["page_blocks"]) == 3
    assert manifest["page_blocks"][0]["page_number"] == 1
    assert manifest["page_blocks"][2]["page_number"] == 2
    assert len(manifest["tables"]) == 1
    assert manifest["tables"][0]["block_id"] == manifest["page_blocks"][1]["block_id"]
    assert len(manifest["images"]) == 1
    assert manifest["images"][0]["relative_path"].endswith(
        "report/auto/images/figure.png"
    )
    assert manifest["ocr_profile"] is None
    assert manifest["derivative_not_native_source_evidence"] is True
    assert manifest["does_not_establish_source_sufficiency"] is True
    assert all(len(output["sha256"]) == 64 for output in manifest["outputs"])
    schema = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "schemas"
            / "document-extraction-manifest-v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    assert set(schema["required"]) <= set(manifest)
    assert set(manifest) <= set(schema["properties"])
    assert not {
        "evidence_status",
        "readiness_status",
        "regulatory_conclusion",
    } & set(manifest)


def test_canonical_manifest_requires_producer_revision_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "report.pdf"
    source.write_bytes(b"%PDF-1.7 synthetic")
    output_root = tmp_path / "output"
    _install_fake_parser(monkeypatch, parse_dir_name="auto")
    from mineru.integrations.knowhere import runner

    monkeypatch.setattr(runner, "_git_commit", lambda: None)

    with pytest.raises(KnowhereExportError, match="producer revision"):
        run_knowhere_export(
            _options(
                source,
                output_root,
                canonical_manifest=CanonicalManifestOptions(
                    source_id="SRC-SYNTHETIC-001",
                    source_version_id="SRC-SYNTHETIC-001-V001",
                    extraction_run_id="EXT-SYNTHETIC-001",
                ),
            )
        )

    assert not (output_root / "document-extraction-manifest-v1.json").exists()
    assert not (output_root / "mineru_manifest.json").exists()


def test_canonical_manifest_rejects_missing_source_identity_before_parsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "report.pdf"
    source.write_bytes(b"%PDF-1.7 synthetic")
    from mineru.integrations.knowhere import runner

    def unexpected_parse(*_args: Any, **_kwargs: Any) -> NoReturn:
        raise AssertionError("parser must not be called")

    monkeypatch.setattr(runner, "do_parse", unexpected_parse)

    with pytest.raises(KnowhereExportError, match="source_id"):
        run_knowhere_export(
            _options(
                source,
                tmp_path / "output",
                canonical_manifest=CanonicalManifestOptions(
                    source_id="",
                    source_version_id="SRC-SYNTHETIC-001-V001",
                    extraction_run_id="EXT-SYNTHETIC-001",
                ),
            )
        )
