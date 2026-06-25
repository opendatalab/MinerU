from pathlib import Path
import inspect

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient
from pydantic import ValidationError

import mineru.parser.api_client as api_client
import mineru.parser.api_server as api_server
from mineru.parser.api_client import MinerUApiParser, _pages_from_middle_json, _parse_result_from_job
from mineru.parser import parse, parse_async
from mineru.parser.api_server import (
    _API_SERVER_BACKENDS,
    CreateJobRequest,
    FileStore,
    FileParseInfo,
    HealthResponse,
    ImageOutputRef,
    JobListItem,
    OutputFileRef,
    OutputFiles,
    create_app,
    main,
)

runner = CliRunner()


def test_api_client_builds_file_page_range_without_options(tmp_path: Path) -> None:
    pdf = tmp_path / "demo.pdf"
    pdf.write_bytes(b"%PDF-1.7\n")

    parser = MinerUApiParser(api_url="http://localhost:8000", tier="standard")
    payload = parser._build_payload(pdf, "1,3~5")

    assert payload["output_formats"] == ["middle_json", "images"]
    assert payload["files"] == [
        {"source": {"type": "local", "path": str(pdf)}, "page_range": "1,3~5"}
    ]
    assert "options" not in payload["files"][0]


def test_api_client_uses_tier_without_backend_semantics(tmp_path: Path) -> None:
    pdf = tmp_path / "demo.pdf"
    pdf.write_bytes(b"%PDF-1.7\n")

    parser = MinerUApiParser(api_url="http://localhost:8000", tier="pro")
    payload = parser._build_payload(pdf, "")

    assert payload["tier"] == "pro"
    assert not hasattr(parser, "backend")


def test_api_client_uses_json_format_for_staging_compat() -> None:
    parser = MinerUApiParser(api_url="https://staging.mineru.org.cn/api", tier="pro")

    assert parser._output_formats() == ["json"]


def test_api_client_downloads_image_sidecars_and_preserves_pdf_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = MinerUApiParser(api_url="http://localhost:8000", tier="standard")
    middle_json = {
        "schema_version": "1.0.0",
        "_pdf_retained_page_indices": [0, 2],
        "_pdf_broken_page_indices": [1],
        "pages": [{"page_idx": 0, "page_size": [100, 200]}],
    }
    image_ref = {"path": "images/chart.png", "file_id": "file-image", "bytes": 11}

    monkeypatch.setattr(api_client, "_download_json", lambda _parser, _outputs: middle_json)
    monkeypatch.setattr(api_client, "_download_bytes", lambda _parser, ref: b"chart-bytes" if ref is image_ref else b"")

    result = _parse_result_from_job(
        {
            "job_id": "job_1",
            "status": "completed",
            "files": [{"output_files": {"middle_json": {"file_id": "file-middle", "bytes": 10}, "images": [image_ref]}}],
        },
        "demo.pdf",
        parser,
    )

    assert result._retained_page_indices == [0, 2]
    assert result._broken_page_indices == [1]
    assert result.images() == {"images/chart.png": b"chart-bytes"}


def test_api_client_omits_tier_when_unspecified(tmp_path: Path) -> None:
    pdf = tmp_path / "demo.pdf"
    pdf.write_bytes(b"%PDF-1.7\n")

    parser = MinerUApiParser(api_url="http://localhost:8000")
    payload = parser._build_payload(pdf, "")

    assert "tier" not in payload


def test_api_client_constructor_does_not_expose_ocr_or_image_options() -> None:
    parameters = inspect.signature(MinerUApiParser).parameters

    assert "method" not in parameters
    assert "image_analysis" not in parameters


def test_parser_entrypoints_expose_api_server_style_options() -> None:
    for entrypoint in (parse, parse_async):
        parameters = inspect.signature(entrypoint).parameters

        assert "tier" in parameters
        assert "backend" in parameters
        assert "language" in parameters
        assert "ocr_mode" in parameters
        assert "disable_table" in parameters
        assert "disable_formula" in parameters
        assert "disable_image_analysis" in parameters


def test_api_client_omits_page_range_when_unspecified(tmp_path: Path) -> None:
    pdf = tmp_path / "demo.pdf"
    pdf.write_bytes(b"%PDF-1.7\n")

    parser = MinerUApiParser(api_url="http://localhost:8000", tier="standard")
    payload = parser._build_payload(pdf, "")

    assert payload["files"] == [{"source": {"type": "local", "path": str(pdf)}}]


def test_api_client_surfaces_failed_job_error() -> None:
    parser = MinerUApiParser(api_url="http://localhost:8000", tier="standard")

    with pytest.raises(Exception, match="upstream failed"):
        _parse_result_from_job(
            {
                "job_id": "job_1",
                "status": "failed",
                "error": {"code": "remote_failed", "message": "upstream failed"},
            },
            "demo.pdf",
            parser,
        )


def test_api_client_rejects_missing_middle_json_output() -> None:
    parser = MinerUApiParser(api_url="http://localhost:8000", tier="standard")

    with pytest.raises(Exception, match="did not return middle_json output"):
        _parse_result_from_job(
            {
                "job_id": "job_1",
                "status": "completed",
                "files": [{"output_files": {}}],
            },
            "demo.pdf",
            parser,
        )


def test_api_client_accepts_pages_middle_json_only() -> None:
    pages = _pages_from_middle_json({"pages": [{"page_idx": 2, "page_size": [100, 200]}]})

    assert len(pages) == 1
    assert pages[0].page_idx == 2
    assert pages[0].page_size == (100, 200)


@pytest.mark.parametrize("payload", [[{"page_idx": 0}], {"pdf_info": [{"page_idx": 0}]}])
def test_api_client_rejects_legacy_middle_json_shapes(payload: object) -> None:
    with pytest.raises(Exception, match="pages"):
        _pages_from_middle_json(payload)  # type: ignore[arg-type]


def test_create_job_request_accepts_new_format_names_and_rejects_options() -> None:
    req = CreateJobRequest.model_validate(
        {
            "files": [
                {
                    "source": {"type": "local", "path": "/tmp/demo.pdf"},
                    "page_range": None,
                }
            ],
            "output_formats": ["middle_json", "structured_content"],
        }
    )
    assert req.files[0].page_range is None
    assert req.output_formats == ["middle_json", "structured_content"]

    with pytest.raises(ValidationError):
        CreateJobRequest.model_validate(
            {
                "files": [
                    {
                        "source": {"type": "local", "path": "/tmp/demo.pdf"},
                        "options": {"page_range": "1"},
                    }
                ],
            }
        )


def test_local_parse_server_rejects_callback(tmp_path: Path) -> None:
    app = create_app(upload_dir=str(tmp_path))
    client = TestClient(app)

    response = client.post(
        "/v1/parse/jobs",
        json={
            "files": [{"source": {"type": "local", "path": str(tmp_path / "demo.pdf")}}],
            "callback": {"url": "https://example.com/mineru-webhook", "secret": "secret"},
        },
    )

    assert response.status_code == 400
    body = response.json()["detail"]
    assert body["error"]["code"] == "invalid_request"
    assert "Webhook callback is not supported" in body["error"]["message"]
    assert app.state.job_store._jobs == {}


def test_create_app_does_not_read_runtime_settings_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINERU_TIER", "pro")
    monkeypatch.setenv("MINERU_BACKEND", "hybrid-auto-engine")
    monkeypatch.setenv("MINERU_CONCURRENCY", "9")
    monkeypatch.setenv("MINERU_URL_TIMEOUT", "99")
    monkeypatch.setenv("MINERU_MAX_WAIT", "999")
    monkeypatch.setenv("MINERU_LANGUAGE", "en")
    monkeypatch.setenv("MINERU_OCR_MODE", "ocr")
    monkeypatch.setenv("MINERU_TABLE_ENABLE", "false")
    monkeypatch.setenv("MINERU_FORMULA_ENABLE", "false")
    monkeypatch.setenv("MINERU_IMAGE_ANALYSIS", "false")

    app = create_app(upload_dir=str(tmp_path))

    assert app.state.tier == "standard"
    assert app.state.backend == "pipeline"
    assert app.state.concurrency == 1
    assert app.state.url_timeout == 60
    assert app.state.max_wait == 600
    assert app.state.language == "ch"
    assert app.state.ocr_mode == "auto"
    assert app.state.table_enable is True
    assert app.state.formula_enable is True
    assert app.state.image_analysis is True


def test_api_server_cli_no_longer_exposes_reload() -> None:
    result = runner.invoke(main, ["--reload"])

    assert result.exit_code != 0
    assert "No such option" in result.output
    assert "--reload" in result.output


def test_output_files_expose_new_names_without_inline_content() -> None:
    ref = OutputFileRef(file_id="file-1", bytes=12)
    outputs = OutputFiles(middle_json=ref, structured_content=ref)

    assert outputs.model_dump(by_alias=True, exclude_none=True) == {
        "middle_json": {"file_id": "file-1", "bytes": 12},
        "structured_content": {"file_id": "file-1", "bytes": 12},
    }

    with pytest.raises(ValidationError):
        OutputFileRef.model_validate(
            {"file_id": "file-1", "bytes": 12, "content": "inline"}
        )


def test_store_image_outputs_creates_downloadable_image_refs(tmp_path: Path) -> None:
    file_store = FileStore(tmp_path / "api-files")

    assert hasattr(api_server, "_store_image_outputs")
    refs = api_server._store_image_outputs(file_store, {"images/chart.png": b"chart-bytes"})

    assert refs == [
        ImageOutputRef(path="images/chart.png", file_id=refs[0].file_id, bytes=len(b"chart-bytes"))
    ]
    assert file_store.read_file_data(refs[0].file_id) == b"chart-bytes"


def test_job_list_item_uses_file_count() -> None:
    item = JobListItem(
        job_id="job_1", status="queued", created_at="2026-06-10T00:00:00Z", file_count=2
    )

    assert item.model_dump() == {
        "job_id": "job_1",
        "status": "queued",
        "created_at": "2026-06-10T00:00:00Z",
        "file_count": 2,
    }


def test_api_contract_uses_parser_version_not_backend_version() -> None:
    parse = FileParseInfo(model_used="model-a", duration_ms=12, parser_version="3.2.1")
    assert parse.model_dump(exclude_none=True) == {
        "model_used": "model-a",
        "duration_ms": 12,
        "parser_version": "3.2.1",
    }

    health = HealthResponse(status="ok", version="3.2.1", parser_version="3.2.1")
    assert health.model_dump(exclude_none=True) == {
        "status": "ok",
        "version": "3.2.1",
        "parser_version": "3.2.1",
        "features": {"sse": False, "webhook": False},
    }

    with pytest.raises(ValidationError):
        FileParseInfo.model_validate({"backend_version": "3.2.1"})

    with pytest.raises(ValidationError):
        HealthResponse.model_validate({"status": "ok", "version": "3.2.1", "backend_version": "3.2.1"})


def test_api_server_tier_selects_compatible_backend(tmp_path: Path) -> None:
    pro_app = create_app(upload_dir=str(tmp_path / "pro"), tier="pro")
    standard_app = create_app(upload_dir=str(tmp_path / "standard"), tier="standard")

    assert pro_app.state.tier == "pro"
    assert pro_app.state.backend == "hybrid-auto-engine"
    assert [tier["id"] for tier in pro_app.state.tiers] == ["pro"]
    assert standard_app.state.tier == "standard"
    assert standard_app.state.backend == "pipeline"
    assert [tier["id"] for tier in standard_app.state.tiers] == ["standard"]


def test_api_server_defaults_to_standard_tier(tmp_path: Path) -> None:
    app = create_app(upload_dir=str(tmp_path))

    assert app.state.tier == "standard"
    assert app.state.backend == "pipeline"
    assert [tier["id"] for tier in app.state.tiers] == ["standard"]


def test_api_server_rejects_incompatible_tier_and_backend(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="incompatible"):
        create_app(upload_dir=str(tmp_path), tier="pro", backend="pipeline")


def test_api_server_allows_compatible_tier_and_explicit_backend(tmp_path: Path) -> None:
    app = create_app(upload_dir=str(tmp_path), tier="pro", backend="vlm-auto-engine")

    assert app.state.tier == "pro"
    assert app.state.backend == "vlm-auto-engine"
    assert [tier["id"] for tier in app.state.tiers] == ["pro"]


def test_api_server_explicit_backend_infers_tier(tmp_path: Path) -> None:
    app = create_app(upload_dir=str(tmp_path), backend="hybrid-auto-engine")

    assert app.state.tier == "pro"
    assert app.state.backend == "hybrid-auto-engine"
    assert [tier["id"] for tier in app.state.tiers] == ["pro"]


def test_api_server_rejects_bare_vlm_backend(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported backend"):
        create_app(upload_dir=str(tmp_path), backend="vlm")


def test_api_server_accepts_parser_backend_values(tmp_path: Path) -> None:
    assert "vlm" not in _API_SERVER_BACKENDS
    assert "hybrid" not in _API_SERVER_BACKENDS

    for backend in _API_SERVER_BACKENDS:
        app = create_app(upload_dir=str(tmp_path / backend), backend=backend)
        expected_tier = "standard" if backend == "pipeline" else "pro"

        assert app.state.backend == backend
        assert app.state.tier == expected_tier


def test_api_server_stores_parser_runtime_options(tmp_path: Path) -> None:
    app = create_app(
        upload_dir=str(tmp_path),
        language="en",
        ocr_mode="ocr",
        table_enable=False,
        formula_enable=False,
        image_analysis=False,
    )

    assert app.state.language == "en"
    assert app.state.ocr_mode == "ocr"
    assert app.state.table_enable is False
    assert app.state.formula_enable is False
    assert app.state.image_analysis is False


def test_api_server_cli_exposes_parser_runtime_options() -> None:
    option_names = {name for param in main.params for name in param.opts}

    assert "--language" in option_names
    assert "--ocr-mode" in option_names
    assert "--disable-table" in option_names
    assert "--disable-formula" in option_names
    assert "--disable-image-analysis" in option_names
