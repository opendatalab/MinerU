import asyncio
import importlib
import inspect
import io
import json
import logging
import subprocess
import sys
import types
from pathlib import Path

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient
from pydantic import ValidationError

import mineru.parser.api_client as api_client
import mineru.parser.api_server as api_server
import mineru.parser.pdf as parser_pdf
import mineru.parser.tier as parser_tier
from mineru.parser import _build_parser, parse, parse_async
from mineru.parser.api_client import MinerUApiParser, _pages_from_middle_json, _parse_result_from_job, should_trust_env_for_url
from mineru.parser.api_server import (
    _API_SERVER_LANGUAGES,
    CreateJobRequest,
    FileParseInfo,
    FileStore,
    HealthResponse,
    ImageOutputRef,
    JobListItem,
    OutputFileRef,
    OutputFiles,
    _install_managed_parse_server_stdin_watcher,
    create_app,
    main,
)
from mineru.parser.base import ParseResult
from mineru.types import Block, Line, PageInfo, Span
from mineru.utils.image_payload import ImagePayloadCache

runner = CliRunner()


def _stub_api_server_dependency_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib, "import_module", lambda _module_name: object())


def test_hybrid_analyze_import_does_not_require_vlm_utils() -> None:
    """校验 Hybrid medium 所需模块导入阶段不再强依赖 VLM 工具包。"""
    repo_root = Path(__file__).resolve().parents[2]
    code = """
import importlib.abc
import sys


class BlockVlmUtilsFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        # 阻断 mineru_vl_utils 导入，用来验证 Hybrid medium 的 lazy import 边界。
        if fullname == "mineru_vl_utils" or fullname.startswith("mineru_vl_utils."):
            raise ModuleNotFoundError(f"blocked VLM utility import: {fullname}")
        return None


sys.meta_path.insert(0, BlockVlmUtilsFinder())
import mineru.backend.hybrid.hybrid_analyze
print("ok")
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_hybrid_local_runtime_uses_context_names_and_local_lock_env() -> None:
    """校验 Hybrid 本地模型运行时只暴露 context 命名，并优先读取新的本地锁环境变量。"""
    repo_root = Path(__file__).resolve().parents[2]
    code = """
import os

os.environ["MINERU_ENABLE_LOCAL_MODEL_INFERENCE_LOCKS"] = "true"
os.environ["MINERU_ENABLE_PIPELINE_INFERENCE_LOCKS"] = "false"

from mineru.backend import local_model_runtime

assert hasattr(local_model_runtime, "HybridLocalModelContext")
assert hasattr(local_model_runtime, "HybridLocalModelContextSingleton")
for removed_name in [
    "Mineru" + "PipelineModel",
    "Mineru" + "HybridModel",
    "Hybrid" + "ModelSingleton",
]:
    assert not hasattr(local_model_runtime, removed_name)
assert local_model_runtime.LOCAL_MODEL_INFERENCE_LOCKS_ENABLED is True
print("ok")
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_validate_effort_rejects_low_and_maps_legacy_backends() -> None:
    """校验 Hybrid effort 只接受 medium/high/extra_high 三档。"""
    from mineru.utils.backend_options import (
        HYBRID_EFFORT_CHOICES,
        effort_for_tier,
        resolve_backend_and_effort,
        validate_effort,
    )

    assert HYBRID_EFFORT_CHOICES == ("medium", "high", "extra_high")
    assert effort_for_tier("medium") == "medium"
    assert effort_for_tier("high") == "high"
    assert effort_for_tier("extra_high") == "extra_high"
    with pytest.raises(ValueError, match="Unsupported effort 'low'"):
        validate_effort("low")
    with pytest.raises(ValueError, match="Unsupported tier 'standard'"):
        effort_for_tier("standard")
    assert resolve_backend_and_effort("vlm-engine", "medium") == ("hybrid-engine", "extra_high")
    assert resolve_backend_and_effort("pipeline", "extra_high") == ("hybrid-engine", "medium")


def test_tier_runtime_options_map_hybrid_effort() -> None:
    """校验 tier 到 Hybrid runtime 参数的共享映射，避免 API/Gradio 分叉维护。"""
    from mineru.parser.tier import runtime_options_for_tier

    assert runtime_options_for_tier("flash").as_kwargs() == {
        "tier": "flash",
        "backend": "flash",
        "effort": "medium",
    }
    assert runtime_options_for_tier("medium").as_kwargs() == {
        "tier": "medium",
        "backend": "hybrid-engine",
        "effort": "medium",
    }
    assert runtime_options_for_tier("high").as_kwargs() == {
        "tier": "high",
        "backend": "hybrid-engine",
        "effort": "high",
    }
    assert runtime_options_for_tier("extra_high").as_kwargs() == {
        "tier": "extra_high",
        "backend": "hybrid-engine",
        "effort": "extra_high",
    }
    with pytest.raises(ValueError, match="Unsupported tier 'standard'"):
        runtime_options_for_tier("standard")


def test_api_client_builds_file_page_range_without_options(tmp_path: Path) -> None:
    pdf = tmp_path / "demo.pdf"
    pdf.write_bytes(b"%PDF-1.7\n")

    parser = MinerUApiParser(api_url="http://localhost:8000", tier="high")
    payload = parser._build_payload(pdf, "1,3~5")

    assert payload["output_formats"] == ["middle_json", "images"]
    assert payload["files"] == [{"source": {"type": "local", "path": str(pdf)}, "page_range": "1,3~5"}]
    assert "options" not in payload["files"][0]


def test_api_client_uses_tier_without_backend_semantics(tmp_path: Path) -> None:
    pdf = tmp_path / "demo.pdf"
    pdf.write_bytes(b"%PDF-1.7\n")

    parser = MinerUApiParser(api_url="http://localhost:8000", tier="extra_high")
    payload = parser._build_payload(pdf, "")

    assert payload["tier"] == "extra_high"
    assert not hasattr(parser, "backend")


@pytest.mark.parametrize("tier", ["standard", "pro"])
def test_api_client_rejects_old_tier_names(tier: str) -> None:
    with pytest.raises(ValueError, match=f"Unsupported tier '{tier}'"):
        MinerUApiParser(api_url="http://localhost:8000", tier=tier)  # type: ignore[arg-type]


def test_api_client_uses_json_format_for_staging_compat() -> None:
    parser = MinerUApiParser(api_url="https://staging.mineru.org.cn/api", tier="extra_high")

    assert parser._output_formats() == ["json"]


def test_api_client_downloads_image_sidecars_and_preserves_pdf_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = MinerUApiParser(api_url="http://localhost:8000", tier="high")
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


def test_api_client_accepts_remote_pdf_info_json(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = MinerUApiParser(api_url="https://staging.mineru.org.cn/api", tier="high")
    middle_json = {
        "_backend": "hybrid",
        "_version_name": "remote",
        "pdf_info": [{"page_idx": 0, "page_size": [100, 200]}],
    }

    monkeypatch.setattr(api_client, "_download_json", lambda _parser, _outputs: middle_json)
    monkeypatch.setattr(api_client, "_download_image_sidecars", lambda _parser, _outputs: {})

    result = _parse_result_from_job(
        {
            "job_id": "job_1",
            "status": "completed",
            "files": [{"output_files": {"json": {"file_id": "file-json", "bytes": 10}}}],
        },
        "demo.pdf",
        parser,
    )

    assert len(result.pages) == 1
    assert result.pages[0].page_idx == 0
    assert result.pages[0].page_size == (100, 200)
    assert result.pages[0]._backend == "hybrid"


def test_async_api_client_accepts_remote_pdf_info_json(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = MinerUApiParser(api_url="https://staging.mineru.org.cn/api", tier="high")
    middle_json = {
        "_backend": "hybrid",
        "pdf_info": [{"page_idx": 0, "page_size": [100, 200]}],
    }

    async def _download_json(*_args: object, **_kwargs: object) -> dict[str, object]:
        return middle_json

    async def _download_image_sidecars(*_args: object, **_kwargs: object) -> dict[str, bytes]:
        return {}

    monkeypatch.setattr(api_client, "_async_download_json", _download_json)
    monkeypatch.setattr(api_client, "_async_download_image_sidecars", _download_image_sidecars)

    result = asyncio.run(
        api_client._async_parse_result_from_job(
            {
                "job_id": "job_1",
                "status": "completed",
                "files": [{"output_files": {"json": {"file_id": "file-json", "bytes": 10}}}],
            },
            "demo.pdf",
            parser,
        )
    )

    assert len(result.pages) == 1
    assert result.pages[0].page_idx == 0
    assert result.pages[0].page_size == (100, 200)
    assert result.pages[0]._backend == "hybrid"


@pytest.mark.parametrize("image_path", ["../escape.png", "/tmp/escape.png", "\\escape.png", "C:\\escape.png"])
def test_api_client_rejects_unsafe_image_sidecar_paths(
    monkeypatch: pytest.MonkeyPatch,
    image_path: str,
) -> None:
    parser = MinerUApiParser(api_url="http://localhost:8000", tier="high")
    middle_json = {"schema_version": "1.0.0", "pages": [{"page_idx": 0}]}
    image_ref = {"path": image_path, "file_id": "file-image", "bytes": 11}

    monkeypatch.setattr(api_client, "_download_json", lambda _parser, _outputs: middle_json)

    def _fail_download(*_args: object, **_kwargs: object) -> bytes:
        """危险 sidecar 路径必须在下载图片字节前被拒绝。"""
        raise AssertionError("unsafe sidecar path should be rejected before download")

    monkeypatch.setattr(api_client, "_download_bytes", _fail_download)

    with pytest.raises(ValueError, match="Unsafe image sidecar path"):
        _parse_result_from_job(
            {
                "job_id": "job_1",
                "status": "completed",
                "files": [{"output_files": {"middle_json": {"file_id": "file-middle", "bytes": 10}, "images": [image_ref]}}],
            },
            "demo.pdf",
            parser,
        )


def test_async_api_client_rejects_unsafe_image_sidecar_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = MinerUApiParser(api_url="http://localhost:8000", tier="high")
    middle_json = {"schema_version": "1.0.0", "pages": [{"page_idx": 0}]}
    image_ref = {"path": "../escape.png", "file_id": "file-image", "bytes": 11}

    async def _download_json(*_args: object, **_kwargs: object) -> dict[str, object]:
        """返回最小 middle_json，聚焦验证 sidecar 路径校验。"""
        return middle_json

    async def _fail_download(*_args: object, **_kwargs: object) -> bytes:
        """异步路径同样必须在下载图片字节前拒绝危险 sidecar。"""
        raise AssertionError("unsafe sidecar path should be rejected before download")

    monkeypatch.setattr(api_client, "_async_download_json", _download_json)
    monkeypatch.setattr(api_client, "_async_download_bytes", _fail_download)

    with pytest.raises(ValueError, match="Unsafe image sidecar path"):
        asyncio.run(
            api_client._async_parse_result_from_job(
                {
                    "job_id": "job_1",
                    "status": "completed",
                    "files": [
                        {"output_files": {"middle_json": {"file_id": "file-middle", "bytes": 10}, "images": [image_ref]}}
                    ],
                },
                "demo.pdf",
                parser,
            )
        )


def test_api_client_disables_env_proxy_for_local_network_urls() -> None:
    assert should_trust_env_for_url("http://localhost:8000/api") is False
    assert should_trust_env_for_url("http://127.0.0.1:8000/api") is False
    assert should_trust_env_for_url("http://192.168.1.20:8000/api") is False
    assert should_trust_env_for_url("https://staging.mineru.org.cn/api") is True


def test_api_client_passes_trust_env_from_api_url(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bool] = []

    class _Response:
        status_code = 200
        text = ""

        def json(self) -> dict[str, str]:
            return {"job_id": "job_1", "status": "completed"}

    class _Client:
        def __init__(self, *, timeout: object, trust_env: bool, **_: object) -> None:
            calls.append(trust_env)

        def __enter__(self) -> "_Client":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def post(self, *args: object, **kwargs: object) -> _Response:
            return _Response()

    monkeypatch.setattr("mineru.parser.api_client.httpx.Client", _Client)

    MinerUApiParser(api_url="http://127.0.0.1:8000/api")._do_parse({"files": []})
    MinerUApiParser(api_url="https://mineru.net/api")._do_parse({"files": []})

    assert calls == [False, True]


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
        assert "effort" in parameters
        assert "disable_table" in parameters
        assert "disable_formula" in parameters
        assert "disable_image_analysis" in parameters


def test_api_client_omits_page_range_when_unspecified(tmp_path: Path) -> None:
    pdf = tmp_path / "demo.pdf"
    pdf.write_bytes(b"%PDF-1.7\n")

    parser = MinerUApiParser(api_url="http://localhost:8000", tier="high")
    payload = parser._build_payload(pdf, "")

    assert payload["files"] == [{"source": {"type": "local", "path": str(pdf)}}]


def test_api_client_surfaces_failed_job_error() -> None:
    parser = MinerUApiParser(api_url="http://localhost:8000", tier="high")

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


def test_api_client_surfaces_failed_file_error_when_job_has_no_top_level_error() -> None:
    parser = MinerUApiParser(api_url="http://localhost:8000", tier="high")

    with pytest.raises(api_client._V1APIError) as exc_info:
        _parse_result_from_job(
            {
                "job_id": "job_1",
                "status": "failed",
                "files": [
                    {
                        "name": "demo.pdf",
                        "status": "failed",
                        "error": {
                            "type": "engine_error",
                            "code": "parse_failed",
                            "message": "No module named 'torch'",
                        },
                    }
                ],
            },
            "demo.pdf",
            parser,
        )

    assert exc_info.value.code == "parse_failed"
    assert exc_info.value.message == "No module named 'torch'"


def test_api_client_rejects_missing_middle_json_output() -> None:
    parser = MinerUApiParser(api_url="http://localhost:8000", tier="high")

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


def test_api_client_accepts_legacy_pdf_info_middle_json() -> None:
    pages = _pages_from_middle_json({"pdf_info": [{"page_idx": 2, "page_size": [100, 200]}]})

    assert len(pages) == 1
    assert pages[0].page_idx == 2
    assert pages[0].page_size == (100, 200)


@pytest.mark.parametrize("payload", [[{"page_idx": 0}], {"pdf_info": {"preproc_blocks": []}}])
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


def test_local_parse_server_rejects_callback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_api_server_dependency_preflight(monkeypatch)
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
    _stub_api_server_dependency_preflight(monkeypatch)
    monkeypatch.setenv("MINERU_TIER", "extra_high")
    monkeypatch.setenv("MINERU_BACKEND", "hybrid-auto-engine")
    monkeypatch.setenv("MINERU_CONCURRENCY", "9")
    monkeypatch.setenv("MINERU_URL_TIMEOUT", "99")
    monkeypatch.setenv("MINERU_MAX_WAIT", "999")
    monkeypatch.setenv("MINERU_LANGUAGE", "en")
    monkeypatch.setenv("MINERU_OCR_MODE", "ocr")
    monkeypatch.setenv("MINERU_EFFORT", "high")
    monkeypatch.setenv("MINERU_TABLE_ENABLE", "false")
    monkeypatch.setenv("MINERU_FORMULA_ENABLE", "false")
    monkeypatch.setenv("MINERU_IMAGE_ANALYSIS", "false")

    app = create_app(upload_dir=str(tmp_path))

    assert app.state.tier == "high"
    assert app.state.backend == "hybrid-engine"
    assert app.state.concurrency == 1
    assert app.state.url_timeout == 60
    assert app.state.max_wait == 600
    assert app.state.language == "ch"
    assert app.state.ocr_mode == "auto"
    assert app.state.effort == "high"
    assert app.state.table_enable is True
    assert app.state.formula_enable is True
    assert app.state.image_analysis is True


def test_api_server_cli_no_longer_exposes_reload() -> None:
    result = runner.invoke(main, ["--reload"])

    assert result.exit_code != 0
    assert "No such option" in result.output
    assert "--reload" in result.output


def test_managed_parse_server_env_enables_stdin_eof_shutdown_watcher(monkeypatch: pytest.MonkeyPatch) -> None:
    server = type("Server", (), {"should_exit": False})()
    monkeypatch.setenv("MINERU_MANAGED_PARSE_SERVER", "1")
    monkeypatch.setattr(api_server.sys, "stdin", type("Stdin", (), {"buffer": io.BytesIO(b"")})())

    watcher = _install_managed_parse_server_stdin_watcher(server)

    assert watcher is not None
    watcher.join(timeout=1)
    assert server.should_exit is True


def test_managed_parse_server_stdin_watcher_is_disabled_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    server = type("Server", (), {"should_exit": False})()
    monkeypatch.delenv("MINERU_MANAGED_PARSE_SERVER", raising=False)

    watcher = _install_managed_parse_server_stdin_watcher(server)

    assert watcher is None
    assert server.should_exit is False


def test_output_files_expose_new_names_without_inline_content() -> None:
    ref = OutputFileRef(file_id="file-1", bytes=12)
    outputs = OutputFiles(middle_json=ref, structured_content=ref)

    assert outputs.model_dump(by_alias=True, exclude_none=True) == {
        "middle_json": {"file_id": "file-1", "bytes": 12},
        "structured_content": {"file_id": "file-1", "bytes": 12},
    }

    with pytest.raises(ValidationError):
        OutputFileRef.model_validate({"file_id": "file-1", "bytes": 12, "content": "inline"})


def test_store_image_outputs_creates_downloadable_image_refs(tmp_path: Path) -> None:
    file_store = FileStore(tmp_path / "api-files")

    assert hasattr(api_server, "_store_image_outputs")
    refs = api_server._store_image_outputs(file_store, {"images/chart.png": b"chart-bytes"})

    assert refs == [ImageOutputRef(path="images/chart.png", file_id=refs[0].file_id, bytes=len(b"chart-bytes"))]
    assert file_store.read_file_data(refs[0].file_id) == b"chart-bytes"


@pytest.mark.parametrize(
    ("output_format", "output_attr"),
    [
        ("markdown", "markdown"),
        ("content_list", "content_list"),
        ("structured_content", "structured_content"),
    ],
)
def test_api_server_rendered_outputs_store_image_sidecars(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    output_format: str,
    output_attr: str,
) -> None:
    img_bytes = b"rendered-image-bytes"
    image_cache = ImagePayloadCache()
    image_path = image_cache.register_bytes(img_bytes, "png", image_path="rendered.png")
    span = Span(type="image", bbox=(0, 0, 10, 10), image_path=image_path)
    line = Line(bbox=(0, 0, 10, 10), spans=[span])
    block = Block(index=0, type="image", bbox=(0, 0, 10, 10), lines=[line])
    parse_result = ParseResult(
        pages=[
            PageInfo(
                page_idx=0,
                page_size=(100, 100),
                para_blocks=[block],
                _backend="hybrid",
            )
        ],
        _image_cache=image_cache,
    )

    async def fake_parse_async(*args, **kwargs) -> ParseResult:
        return parse_result

    monkeypatch.setattr("mineru.parser.api_server.parse_async", fake_parse_async)
    file_store = FileStore(tmp_path / "api-files")
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF-1.7\n")
    request = CreateJobRequest.model_validate(
        {
            "files": [{"source": {"type": "local", "path": str(source)}}],
            "tier": "high",
            "output_formats": [output_format],
        }
    )
    job_store = api_server.JobStore()
    rec = job_store.create(request, file_store)

    asyncio.run(
        api_server._run_job(
            rec,
            request,
            file_store,
            server_backend="hybrid-engine",
            language="ch",
            ocr_mode="auto",
            table_enable=True,
            formula_enable=True,
            image_analysis=True,
            effort="medium",
        )
    )

    output_files = rec.files[0].output_files
    assert getattr(output_files, output_attr) is not None
    assert output_files.images == [
        ImageOutputRef(
            path=output_files.images[0].path,
            file_id=output_files.images[0].file_id,
            bytes=len(img_bytes),
        )
    ]
    assert file_store.read_file_data(output_files.images[0].file_id) == img_bytes


def test_api_server_middle_json_preserves_backend_for_client_rendering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    block = Block(
        index=0,
        type="text",
        bbox=(0, 0, 10, 10),
        lines=[Line(bbox=(0, 0, 10, 10), spans=[Span(type="text", bbox=(0, 0, 10, 10), content="hello")])],
    )
    parse_result = ParseResult(
        pages=[
            PageInfo(
                page_idx=0,
                page_size=(100, 100),
                para_blocks=[block],
                _backend="hybrid",
            )
        ]
    )

    async def fake_parse_async(*args, **kwargs) -> ParseResult:
        return parse_result

    monkeypatch.setattr("mineru.parser.api_server.parse_async", fake_parse_async)
    file_store = FileStore(tmp_path / "api-files")
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF-1.7\n")
    request = CreateJobRequest.model_validate(
        {
            "files": [{"source": {"type": "local", "path": str(source)}}],
            "tier": "high",
            "output_formats": ["middle_json"],
        }
    )
    job_store = api_server.JobStore()
    rec = job_store.create(request, file_store)

    asyncio.run(
        api_server._run_job(
            rec,
            request,
            file_store,
            server_backend="hybrid-engine",
            language="ch",
            ocr_mode="auto",
            table_enable=True,
            formula_enable=True,
            image_analysis=True,
            effort="medium",
        )
    )

    output_ref = rec.files[0].output_files.middle_json
    payload = json.loads(file_store.read_file_data(output_ref.file_id).decode("utf-8"))
    roundtrip = ParseResult.from_dict(payload)

    assert payload["_backend"] == "hybrid"
    assert roundtrip.content_list()
    assert roundtrip.structured_content()


def test_api_server_sanitizes_surrogates_in_text_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    block = Block(
        index=0,
        type="text",
        bbox=(0, 0, 10, 10),
        lines=[
            Line(
                bbox=(0, 0, 10, 10),
                spans=[Span(type="text", bbox=(0, 0, 10, 10), content="bad \ud800 text")],
            )
        ],
    )
    parse_result = ParseResult(
        pages=[
            PageInfo(
                page_idx=0,
                page_size=(100, 100),
                para_blocks=[block],
                _backend="hybrid",
            )
        ]
    )

    async def fake_parse_async(*args: object, **kwargs: object) -> ParseResult:
        return parse_result

    monkeypatch.setattr("mineru.parser.api_server.parse_async", fake_parse_async)
    file_store = FileStore(tmp_path / "api-files")
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF-1.7\n")
    request = CreateJobRequest.model_validate(
        {
            "files": [{"source": {"type": "local", "path": str(source)}}],
            "tier": "high",
            "output_formats": ["markdown", "middle_json", "content_list", "structured_content"],
        }
    )
    job_store = api_server.JobStore()
    rec = job_store.create(request, file_store)

    asyncio.run(
        api_server._run_job(
            rec,
            request,
            file_store,
            server_backend="hybrid-engine",
            language="ch",
            ocr_mode="auto",
            table_enable=True,
            formula_enable=True,
            image_analysis=True,
            effort="medium",
        )
    )

    output_files = rec.files[0].output_files
    refs = [
        output_files.markdown,
        output_files.middle_json,
        output_files.content_list,
        output_files.structured_content,
    ]

    assert rec.status == "completed"
    assert all(ref is not None for ref in refs)
    for ref in refs:
        assert ref is not None
        decoded = file_store.read_file_data(ref.file_id).decode("utf-8")
        assert "\ud800" not in decoded
        assert "\ufffd" in decoded


def test_api_server_logs_traceback_when_job_file_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def fake_parse_async(*args: object, **kwargs: object) -> ParseResult:
        raise RuntimeError("boom")

    monkeypatch.setattr("mineru.parser.api_server.parse_async", fake_parse_async)
    file_store = FileStore(tmp_path / "api-files")
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF-1.7\n")
    request = CreateJobRequest.model_validate(
        {
            "files": [{"source": {"type": "local", "path": str(source)}}],
            "tier": "high",
            "output_formats": ["middle_json"],
        }
    )
    job_store = api_server.JobStore()
    rec = job_store.create(request, file_store)

    with caplog.at_level(logging.ERROR, logger="mineru.parser.api_server"):
        asyncio.run(
            api_server._run_job(
                rec,
                request,
                file_store,
                server_backend="hybrid-engine",
                language="ch",
                ocr_mode="auto",
                table_enable=True,
                formula_enable=True,
                image_analysis=True,
                effort="medium",
            )
        )

    assert rec.status == "failed"
    assert rec.files[0].error is not None
    assert rec.files[0].error.message == "boom"
    assert "Parse-server job file failed" in caplog.text
    assert f"job_id={rec.id}" in caplog.text
    assert "RuntimeError: boom" in caplog.text


def test_job_list_item_uses_file_count() -> None:
    item = JobListItem(job_id="job_1", status="queued", created_at="2026-06-10T00:00:00Z", file_count=2)

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


def test_api_server_tier_selects_compatible_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_api_server_dependency_preflight(monkeypatch)
    flash_app = create_app(upload_dir=str(tmp_path / "flash"), tier="flash")
    medium_app = create_app(upload_dir=str(tmp_path / "medium"), tier="medium")
    high_app = create_app(upload_dir=str(tmp_path / "high"), tier="high")
    extra_high_app = create_app(upload_dir=str(tmp_path / "extra_high"), tier="extra_high")

    assert flash_app.state.tier == "flash"
    assert flash_app.state.backend == "flash"
    assert [tier["id"] for tier in flash_app.state.tiers] == ["flash"]
    assert medium_app.state.tier == "medium"
    assert medium_app.state.backend == "hybrid-engine"
    assert medium_app.state.effort == "medium"
    assert [tier["id"] for tier in medium_app.state.tiers] == ["medium"]
    assert high_app.state.tier == "high"
    assert high_app.state.backend == "hybrid-engine"
    assert high_app.state.effort == "high"
    assert [tier["id"] for tier in high_app.state.tiers] == ["high"]
    assert extra_high_app.state.tier == "extra_high"
    assert extra_high_app.state.backend == "hybrid-engine"
    assert extra_high_app.state.effort == "extra_high"
    assert [tier["id"] for tier in extra_high_app.state.tiers] == ["extra_high"]


def test_api_server_defaults_to_all_quality_tiers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_api_server_dependency_preflight(monkeypatch)
    app = create_app(upload_dir=str(tmp_path))

    assert app.state.tier == "high"
    assert app.state.default_tier == "high"
    assert app.state.backend == "hybrid-engine"
    assert app.state.effort == "high"
    assert [tier["id"] for tier in app.state.tiers] == ["flash", "medium", "high", "extra_high"]
    assert app.state.tier_runtime_options["flash"].as_kwargs() == {
        "tier": "flash",
        "backend": "flash",
        "effort": "medium",
    }
    assert app.state.tier_runtime_options["medium"].as_kwargs() == {
        "tier": "medium",
        "backend": "hybrid-engine",
        "effort": "medium",
    }
    assert app.state.tier_runtime_options["high"].as_kwargs() == {
        "tier": "high",
        "backend": "hybrid-engine",
        "effort": "high",
    }
    assert app.state.tier_runtime_options["extra_high"].as_kwargs() == {
        "tier": "extra_high",
        "backend": "hybrid-engine",
        "effort": "extra_high",
    }


def test_api_server_multi_tier_state_and_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """校验同一个 API server 可发布多个 tier，并在包含 high 时以 high 作为默认 tier。"""
    _stub_api_server_dependency_preflight(monkeypatch)
    app = create_app(upload_dir=str(tmp_path), tier=["medium", "high", "extra_high"])

    assert app.state.tier == "high"
    assert app.state.default_tier == "high"
    assert app.state.backend == "hybrid-engine"
    assert app.state.effort == "high"
    assert [tier["id"] for tier in app.state.tiers] == ["medium", "high", "extra_high"]
    assert app.state.model_ids == ["Hybrid-Medium", "MinerU-HTML", "MinerU2.5-Pro-2604-1.2B"]
    assert app.state.tier_runtime_options["medium"].as_kwargs() == {
        "tier": "medium",
        "backend": "hybrid-engine",
        "effort": "medium",
    }
    assert app.state.tier_runtime_options["high"].as_kwargs() == {
        "tier": "high",
        "backend": "hybrid-engine",
        "effort": "high",
    }
    assert app.state.tier_runtime_options["extra_high"].as_kwargs() == {
        "tier": "extra_high",
        "backend": "hybrid-engine",
        "effort": "extra_high",
    }


def test_api_server_multi_tier_http_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """校验 /v1/tiers 与 /v1/models 返回启动时声明的全部 tier 能力。"""
    _stub_api_server_dependency_preflight(monkeypatch)
    app = create_app(upload_dir=str(tmp_path), tier=["flash", "medium", "high", "extra_high"])

    with TestClient(app) as client:
        tiers_response = client.get("/v1/tiers")
        models_response = client.get("/v1/models")

    assert tiers_response.status_code == 200
    assert [tier["id"] for tier in tiers_response.json()["data"]] == ["flash", "medium", "high", "extra_high"]
    assert models_response.status_code == 200
    assert [model["id"] for model in models_response.json()["data"]] == [
        "MinerU-Flash",
        "Hybrid-Medium",
        "MinerU-HTML",
        "MinerU2.5-Pro-2604-1.2B",
    ]


def test_api_server_multi_tier_jobs_use_requested_tier_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """校验多 tier server 按每个 job 的 tier 选择 backend/effort，而不是复用全局默认值。"""
    _stub_api_server_dependency_preflight(monkeypatch)
    calls: list[dict[str, object]] = []

    async def fake_parse_async(*args: object, **kwargs: object) -> ParseResult:
        """记录 API server 传给 parser 的 runtime 参数，并返回最小解析结果。"""
        calls.append(dict(kwargs))
        return ParseResult(pages=[PageInfo(page_idx=0, _backend="hybrid")])

    monkeypatch.setattr("mineru.parser.api_server.parse_async", fake_parse_async)
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF-1.7\n")
    app = create_app(upload_dir=str(tmp_path / "api"), tier=["medium", "high"])

    with TestClient(app) as client:
        default_response = client.post(
            "/v1/parse/jobs",
            json={
                "files": [{"source": {"type": "local", "path": str(source)}}],
                "output_formats": ["middle_json"],
                "wait": 5,
            },
        )
        medium_response = client.post(
            "/v1/parse/jobs",
            json={
                "files": [{"source": {"type": "local", "path": str(source)}}],
                "tier": "medium",
                "output_formats": ["middle_json"],
                "wait": 5,
            },
        )

    assert default_response.status_code == 200
    assert default_response.json()["tier"] == "high"
    assert medium_response.status_code == 200
    assert medium_response.json()["tier"] == "medium"
    assert [(call["tier"], call["backend"], call["effort"]) for call in calls] == [
        ("high", "hybrid-engine", "high"),
        ("medium", "hybrid-engine", "medium"),
    ]


def test_api_server_single_tier_rejects_unavailable_tier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """校验单 tier server 不接受未启动的 tier，避免请求绕过启动能力边界。"""
    _stub_api_server_dependency_preflight(monkeypatch)
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF-1.7\n")
    app = create_app(upload_dir=str(tmp_path / "api"), tier="high")

    with TestClient(app) as client:
        response = client.post(
            "/v1/parse/jobs",
            json={
                "files": [{"source": {"type": "local", "path": str(source)}}],
                "tier": "extra_high",
                "wait": 0,
            },
        )

    assert response.status_code == 400
    assert "Tier 'extra_high' not available in this server" in response.text


def test_api_server_preflights_medium_tier_dependencies(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    imported_modules: list[str] = []

    def fake_import_module(module_name: str):
        imported_modules.append(module_name)
        return object()

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    create_app(upload_dir=str(tmp_path), tier="medium")

    assert imported_modules == [
        "ftfy",
        "shapely",
        "pyclipper",
        "torch",
        "torchvision",
        "transformers",
    ]


def test_api_server_preflights_extra_high_tier_dependencies_for_platform(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    imported_modules: list[str] = []

    def fake_import_module(module_name: str):
        imported_modules.append(module_name)
        return object()

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    monkeypatch.setattr(parser_tier.sys, "platform", "darwin")

    create_app(upload_dir=str(tmp_path), tier="extra_high")

    assert imported_modules == [
        "ftfy",
        "shapely",
        "pyclipper",
        "torch",
        "torchvision",
        "transformers",
        "accelerate",
        "mlx",
        "mlx_vlm",
    ]


def test_api_server_preflight_rejects_missing_tier_dependency(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_import_module(module_name: str):
        if module_name == "torch":
            raise ModuleNotFoundError("No module named 'torch'")
        return object()

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    monkeypatch.setattr(parser_tier.importlib_metadata, "packages_distributions", lambda: {"mineru": ["mineru"]})

    with pytest.raises(api_server.ParseServerStartupError, match="tier 'medium'.*torch.*mineru\\[medium\\]"):
        create_app(upload_dir=str(tmp_path), tier="medium")


def test_api_server_cli_reports_dependency_preflight_without_traceback(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_import_module(module_name: str):
        if module_name == "mlx":
            raise ModuleNotFoundError("No module named 'mlx'")
        return object()

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    monkeypatch.setattr(parser_tier.importlib_metadata, "packages_distributions", lambda: {"mineru": ["mineru-next-dev"]})
    monkeypatch.setattr(parser_tier.sys, "platform", "darwin")

    result = runner.invoke(main, ["--tier", "extra_high"])

    assert result.exit_code == 1
    assert result.output == (
        "Error: Parse server cannot start for tier 'extra_high'; missing runtime dependencies: mlx. "
        "Install optional dependencies for this tier in the same Python environment as MinerU, "
        "for example: pip install 'mineru-next-dev[extra_high]'.\n"
    )
    assert "Traceback" not in result.output


def test_api_server_create_app_rejects_backend_and_effort_parameters(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="backend"):
        create_app(upload_dir=str(tmp_path / "backend"), backend="pipeline")  # type: ignore[call-arg]

    with pytest.raises(TypeError, match="effort"):
        create_app(upload_dir=str(tmp_path / "effort"), effort="high")  # type: ignore[call-arg]


def test_build_parser_forwards_effort_to_hybrid_parser(tmp_path: Path) -> None:
    pdf = tmp_path / "demo.pdf"
    pdf.write_bytes(b"%PDF-1.7\n")

    parser = _build_parser(pdf, backend="hybrid-engine", effort="high")

    assert parser.__class__.__name__ == "PdfHybridParser"
    assert parser.effort == "high"


def test_build_parser_maps_legacy_vlm_backend_to_hybrid_extra_high(tmp_path: Path) -> None:
    pdf = tmp_path / "demo.pdf"
    pdf.write_bytes(b"%PDF-1.7\n")

    parser = _build_parser(pdf, backend="vlm-engine", effort="medium")

    assert parser.__class__.__name__ == "PdfHybridParser"
    assert parser.backend == "hybrid-engine"
    assert parser.effort == "extra_high"


def test_build_parser_maps_legacy_pipeline_backend_to_hybrid_medium(tmp_path: Path) -> None:
    pdf = tmp_path / "demo.pdf"
    pdf.write_bytes(b"%PDF-1.7\n")

    parser = _build_parser(pdf, backend="pipeline", effort="high")

    assert parser.__class__.__name__ == "PdfHybridParser"
    assert parser.backend == "hybrid-engine"
    assert parser.effort == "medium"


def test_pdf_pipeline_parser_compat_delegates_to_hybrid_medium() -> None:
    parser = parser_pdf.PdfPipelineParser(method="ocr", lang="en", effort="high")

    assert isinstance(parser, parser_pdf.PdfHybridParser)
    assert parser.backend == "hybrid-engine"
    assert parser.method == "ocr"
    assert parser.lang == "en"
    assert parser.effort == "medium"


def test_pdf_hybrid_medium_parser_skips_vlm_backend_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """校验 Hybrid medium 直接进入本地分支，不解析 VLM engine。"""
    seen: dict[str, object] = {}
    fake_module = types.ModuleType("mineru.backend.hybrid.hybrid_analyze")

    def fake_doc_analyze(_pdf_bytes: bytes, **kwargs: object) -> tuple[list[PageInfo], list[object], bool]:
        """记录 medium 分支收到的参数，并返回 Hybrid middle-json 形态。"""
        seen.update(kwargs)
        return [PageInfo(page_idx=0, _backend="hybrid")], [], False

    fake_module.doc_analyze = fake_doc_analyze
    monkeypatch.setitem(sys.modules, "mineru.backend.hybrid.hybrid_analyze", fake_module)

    def fail_resolve_backend(*_args: object, **_kwargs: object) -> str:
        """medium 不应触发 VLM backend resolver。"""
        raise AssertionError("medium effort should not resolve VLM backend")

    monkeypatch.setattr(parser_pdf, "_resolve_hybrid_backend", fail_resolve_backend)

    parser = parser_pdf.PdfHybridParser(backend="hybrid-engine", effort="medium", lang="en")
    pages = parser._run_analysis(b"%PDF-1.7\n")

    assert pages[0]._backend == "hybrid"
    assert seen["backend"] == "hybrid-engine"
    assert seen["effort"] == "medium"
    assert seen["language"] == "en"


@pytest.mark.parametrize(
    ("resolver", "backend"),
    [
        (parser_pdf._resolve_hybrid_backend, "hybrid-engine"),
    ],
)
def test_pdf_engine_resolvers_use_sync_or_async_mode(
    monkeypatch: pytest.MonkeyPatch,
    resolver,
    backend: str,
) -> None:
    calls: list[tuple[str, bool]] = []

    def fake_get_vlm_engine(inference_engine: str, is_async: bool = False) -> str:
        """记录自动 engine 选择是否收到当前解析调用形态。"""
        calls.append((inference_engine, is_async))
        return "vllm-async-engine" if is_async else "vllm-engine"

    monkeypatch.setattr("mineru.utils.engine_utils.get_vlm_engine", fake_get_vlm_engine)

    assert resolver(backend, is_async=False) == "vllm-engine"
    assert resolver(backend, is_async=True) == "vllm-async-engine"
    assert calls == [("auto", False), ("auto", True)]


def test_pdf_engine_resolvers_keep_http_client_backend() -> None:
    assert parser_pdf._resolve_hybrid_backend("hybrid-http-client", is_async=False) == "http-client"
    assert parser_pdf._resolve_hybrid_backend("hybrid-http-client", is_async=True) == "http-client"


def test_pdf_vlm_parser_compat_delegates_to_hybrid_extra_high(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, bool]] = []
    backends: list[str] = []
    efforts: list[str] = []
    fake_module = types.ModuleType("mineru.backend.hybrid.hybrid_analyze")

    def fake_resolve_backend(backend: str, is_async: bool = False) -> str:
        """记录兼容 VLM parser 委托 Hybrid resolver 的调用形态。"""
        calls.append((backend, is_async))
        return "async-backend" if is_async else "sync-backend"

    def fake_doc_analyze(_pdf_bytes: bytes, **kwargs: object) -> tuple[list[PageInfo], list[object], bool]:
        """同步 Hybrid 分析桩只记录最终 backend/effort，不加载真实模型。"""
        backends.append(str(kwargs["backend"]))
        efforts.append(str(kwargs["effort"]))
        return [], [], False

    async def fake_aio_doc_analyze(_pdf_bytes: bytes, **kwargs: object) -> tuple[list[PageInfo], list[object], bool]:
        """异步 Hybrid 分析桩只记录最终 backend/effort，不加载真实模型。"""
        backends.append(str(kwargs["backend"]))
        efforts.append(str(kwargs["effort"]))
        return [], [], False

    fake_module.doc_analyze = fake_doc_analyze
    fake_module.aio_doc_analyze = fake_aio_doc_analyze
    monkeypatch.setitem(sys.modules, "mineru.backend.hybrid.hybrid_analyze", fake_module)
    monkeypatch.setattr(parser_pdf, "_resolve_hybrid_backend", fake_resolve_backend)

    parser = parser_pdf.PdfVlmParser(backend="vlm-engine", effort="medium")

    assert parser._run_analysis(b"%PDF") == []
    assert asyncio.run(parser._arun_analysis(b"%PDF")) == []
    assert calls == [("hybrid-engine", False), ("hybrid-engine", True)]
    assert backends == ["sync-backend", "async-backend"]
    assert efforts == ["extra_high", "extra_high"]


def test_pdf_hybrid_parser_passes_call_mode_to_backend_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, bool]] = []
    backends: list[str] = []
    fake_module = types.ModuleType("mineru.backend.hybrid.hybrid_analyze")

    def fake_resolve_backend(backend: str, is_async: bool = False) -> str:
        """记录 Hybrid parser 同步/异步路径传入 resolver 的调用形态。"""
        calls.append((backend, is_async))
        return "async-backend" if is_async else "sync-backend"

    def fake_doc_analyze(_pdf_bytes: bytes, **kwargs: object) -> tuple[list[PageInfo], list[object], bool]:
        """同步 Hybrid 分析桩只记录最终 backend，不加载真实模型。"""
        backends.append(str(kwargs["backend"]))
        return [], [], False

    async def fake_aio_doc_analyze(_pdf_bytes: bytes, **kwargs: object) -> tuple[list[PageInfo], list[object], bool]:
        """异步 Hybrid 分析桩只记录最终 backend，不加载真实模型。"""
        backends.append(str(kwargs["backend"]))
        return [], [], False

    fake_module.doc_analyze = fake_doc_analyze
    fake_module.aio_doc_analyze = fake_aio_doc_analyze
    monkeypatch.setitem(sys.modules, "mineru.backend.hybrid.hybrid_analyze", fake_module)
    monkeypatch.setattr(parser_pdf, "_resolve_hybrid_backend", fake_resolve_backend)

    parser = parser_pdf.PdfHybridParser(backend="hybrid-engine")

    assert parser._run_analysis(b"%PDF") == []
    assert asyncio.run(parser._arun_analysis(b"%PDF")) == []
    assert calls == [("hybrid-engine", False), ("hybrid-engine", True)]
    assert backends == ["sync-backend", "async-backend"]


def test_api_server_stores_parser_runtime_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_api_server_dependency_preflight(monkeypatch)
    app = create_app(
        upload_dir=str(tmp_path),
        tier="medium",
        language="en",
        ocr_mode="ocr",
        table_enable=False,
        formula_enable=False,
        image_analysis=False,
    )

    assert app.state.language == "ch"
    assert app.state.ocr_mode == "ocr"
    assert app.state.effort == "medium"
    assert app.state.table_enable is False
    assert app.state.formula_enable is False
    assert app.state.image_analysis is False


def test_api_server_rejects_removed_ch_lite_language(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Language ch_lite not supported"):
        create_app(upload_dir=str(tmp_path), language="ch_lite")


def test_api_server_cli_exposes_parser_runtime_options() -> None:
    option_names = {name for param in main.params for name in param.opts}

    assert "--tier" in option_names
    assert "--backend" not in option_names
    assert "--language" in option_names
    assert "--ocr-mode" in option_names
    assert "--effort" not in option_names
    assert "--disable-table" in option_names
    assert "--disable-formula" in option_names
    assert "--disable-image-analysis" in option_names
    assert _API_SERVER_LANGUAGES == (
        "ch",
        "ch_server",
        "korean",
        "ta",
        "te",
        "ka",
        "th",
        "el",
        "arabic",
        "east_slavic",
        "cyrillic",
        "devanagari",
    )


def test_api_server_cli_rejects_backend_and_effort_options() -> None:
    backend_result = runner.invoke(main, ["--backend", "hybrid-engine"])
    effort_result = runner.invoke(main, ["--effort", "high"])

    assert backend_result.exit_code != 0
    assert "No such option" in backend_result.output
    assert "--backend" in backend_result.output
    assert effort_result.exit_code != 0
    assert "No such option" in effort_result.output
    assert "--effort" in effort_result.output


def test_api_server_cli_defaults_to_all_quality_tiers(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def _fake_run(server) -> None:
        """记录无 --tier 启动后的默认 API server 能力，不启动真实服务。"""
        seen["tiers"] = [tier["id"] for tier in server.config.app.state.tiers]
        seen["default_tier"] = server.config.app.state.default_tier
        seen["effort_by_tier"] = {
            tier: runtime.effort for tier, runtime in server.config.app.state.tier_runtime_options.items()
        }

    monkeypatch.setattr("uvicorn.Server.run", _fake_run)

    result = runner.invoke(main, ["--host", "0.0.0.0", "--port", "15984"])

    assert result.exit_code == 0
    assert seen == {
        "tiers": ["flash", "medium", "high", "extra_high"],
        "default_tier": "high",
        "effort_by_tier": {"flash": "medium", "medium": "medium", "high": "high", "extra_high": "extra_high"},
    }


def test_api_server_cli_accepts_repeated_tier_list(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def _fake_run(server) -> None:
        """记录重复 --tier 启动后的 API server 能力列表，避免测试启动真实服务。"""
        seen["tiers"] = [tier["id"] for tier in server.config.app.state.tiers]
        seen["default_tier"] = server.config.app.state.default_tier
        seen["effort_by_tier"] = {
            tier: runtime.effort for tier, runtime in server.config.app.state.tier_runtime_options.items()
        }

    monkeypatch.setattr("uvicorn.Server.run", _fake_run)

    result = runner.invoke(main, ["--tier", "medium", "--tier", "extra_high", "--host", "0.0.0.0", "--port", "15984"])

    assert result.exit_code == 0
    assert seen == {
        "tiers": ["medium", "extra_high"],
        "default_tier": "medium",
        "effort_by_tier": {"medium": "medium", "extra_high": "extra_high"},
    }


def test_api_server_cli_rejects_old_tier_names() -> None:
    result = runner.invoke(main, ["--tier", "standard"])

    assert result.exit_code != 0
    assert "Invalid value for '--tier'" in result.output


def test_api_server_cli_normalizes_hidden_language_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, str] = {}

    def _fake_run(server) -> None:
        """记录 Click CLI 创建出的应用语言配置，避免测试启动真实服务。"""
        seen["language"] = server.config.app.state.language

    monkeypatch.setattr("uvicorn.Server.run", _fake_run)

    result = runner.invoke(main, ["--language", "latin", "--host", "0.0.0.0", "--port", "15982"])

    assert result.exit_code == 0
    assert seen == {"language": "ch"}


def test_api_server_cli_rejects_removed_ch_lite_language() -> None:
    result = runner.invoke(main, ["--language", "ch_lite"])

    assert result.exit_code != 0
    assert "Language ch_lite not supported" in result.output


def test_api_server_cli_help_exposes_tier_only_runtime_selection() -> None:
    result = runner.invoke(main, ["--help"])
    normalized_output = result.output.replace("-\n                                  ", "-")

    assert result.exit_code == 0
    assert "--tier" in normalized_output
    assert "flash" in normalized_output
    assert "medium" in normalized_output
    assert "high" in normalized_output
    assert "extra_high" in normalized_output
    assert "--backend" not in normalized_output
    assert "--effort" not in normalized_output
    assert "hybrid-engine" not in normalized_output
    assert "pipeline" not in normalized_output
    assert "vlm-engine" not in normalized_output
    assert "vlm-http-client" not in normalized_output
    assert "hybrid-auto-engine" not in normalized_output
    assert "vlm-auto-engine" not in normalized_output


def test_api_server_public_exports_are_runtime_entrypoints_only() -> None:
    assert set(api_server.__all__) == {"create_app", "main"}
