import asyncio
import base64
import importlib
import inspect
import io
import json
import logging
import os
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import httpx
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
    JobLinks,
    JobListItem,
    OutputFileRef,
    OutputFiles,
    _install_managed_parse_server_stdin_watcher,
    create_app,
    main,
)
from mineru.parser.base import ParseResult
from mineru.types import Block, Line, PageInfo, Span, TIERS, validate_tier
from mineru.utils.image_payload import ImagePayloadCache

runner = CliRunner()

_REMOVED_DISABLE_TABLE_PARAM = "disable" + "_table"
_REMOVED_DISABLE_FORMULA_PARAM = "disable" + "_formula"
_REMOVED_TABLE_ENABLE_PARAM = "table" + "_enable"
_REMOVED_FORMULA_ENABLE_PARAM = "formula" + "_enable"
_REMOVED_INLINE_FORMULA_PARAM = "inline_" + _REMOVED_FORMULA_ENABLE_PARAM
_REMOVED_DISABLE_TABLE_OPTION = "--disable-" + "table"
_REMOVED_DISABLE_FORMULA_OPTION = "--disable-" + "formula"
_REMOVED_TABLE_ENABLE_ENV = "MINERU_" + "TABLE" + "_ENABLE"
_REMOVED_FORMULA_ENABLE_ENV = "MINERU_" + "FORMULA" + "_ENABLE"


def _stub_api_server_dependency_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib, "import_module", lambda _module_name: object())


def test_hybrid_analyze_import_does_not_require_vlm_utils() -> None:
    """校验 Hybrid basic 所需模块导入阶段不再强依赖 VLM 工具包。"""
    repo_root = Path(__file__).resolve().parents[2]
    code = """
import importlib.abc
import sys


class BlockVlmUtilsFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        # 阻断 mineru_vl_utils 导入，用来验证 Hybrid basic 的 lazy import 边界。
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
    """校验 Hybrid effort 只接受 medium/high/xhigh 三档。"""
    from mineru.utils.backend_options import (
        HYBRID_EFFORT_CHOICES,
        effort_for_tier,
        resolve_backend_and_effort,
        tier_for_effort,
        validate_effort,
    )

    assert HYBRID_EFFORT_CHOICES == ("medium", "high", "xhigh")
    assert effort_for_tier("basic") == "medium"
    assert effort_for_tier("standard") == "high"
    assert effort_for_tier("advanced") == "xhigh"
    assert tier_for_effort("medium") == "basic"
    assert tier_for_effort("high") == "standard"
    assert tier_for_effort("xhigh") == "advanced"
    with pytest.raises(ValueError, match="Unsupported effort 'low'"):
        validate_effort("low")
    with pytest.raises(ValueError, match="Unsupported tier 'ultra'"):
        effort_for_tier("ultra")
    assert resolve_backend_and_effort("vlm-engine", "medium") == ("hybrid-engine", "xhigh")
    assert resolve_backend_and_effort("pipeline", "xhigh") == ("hybrid-engine", "medium")


def test_tier_runtime_options_map_hybrid_effort() -> None:
    """校验 tier 到 Hybrid runtime 参数的共享映射，避免 API/Gradio 分叉维护。"""
    from mineru.parser.tier import runtime_options_for_tier

    assert runtime_options_for_tier("flash").as_kwargs() == {
        "tier": "flash",
        "backend": "flash",
        "effort": "medium",
    }
    assert runtime_options_for_tier("basic").as_kwargs() == {
        "tier": "basic",
        "backend": "hybrid-engine",
        "effort": "medium",
    }
    assert runtime_options_for_tier("standard").as_kwargs() == {
        "tier": "standard",
        "backend": "hybrid-engine",
        "effort": "high",
    }
    assert runtime_options_for_tier("advanced").as_kwargs() == {
        "tier": "advanced",
        "backend": "hybrid-engine",
        "effort": "xhigh",
    }


def test_advanced_dependency_error_recommends_standard_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(parser_tier, "installed_distribution_name", lambda: "mineru")

    error = parser_tier.TierDependencyError("advanced", ["vllm"])

    assert "pip install 'mineru[standard]'" in str(error)


def test_public_tier_literals_match_product_contract() -> None:
    from mineru.parser.tier import runtime_options_for_tier

    assert TIERS == {"flash", "basic", "standard", "advanced"}
    assert validate_tier("advanced") == "advanced"
    with pytest.raises(ValueError, match="Unsupported tier 'ultra'"):
        runtime_options_for_tier("ultra")


def test_api_client_builds_file_page_range_without_options(tmp_path: Path) -> None:
    pdf = tmp_path / "demo.pdf"
    pdf.write_bytes(b"%PDF-1.7\n")

    parser = MinerUApiParser(api_url="http://localhost:8000", tier="standard")
    payload = parser._build_payload({"type": "local", "path": str(pdf)}, "1,3~5")

    assert payload["output_formats"] == ["middle_json"]
    assert payload["files"] == [{"source": {"type": "local", "path": str(pdf)}, "page_range": "1,3~5"}]
    assert "options" not in payload["files"][0]


def test_api_client_uses_file_id_when_local_server_does_not_advertise_local_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = tmp_path / "demo.pdf"
    pdf.write_bytes(b"%PDF-1.7\n")
    parser = MinerUApiParser(api_url="http://localhost:8000", tier="standard")
    monkeypatch.setattr(parser, "_upload", lambda _path: "file_1")

    class _Response:
        status_code = 200
        text = ""

        def json(self) -> dict[str, object]:
            return {"features": {"sources": ["file_id", "url", "inline"]}}

    class _Client:
        def __init__(self, *, timeout: object, trust_env: bool, **_: object) -> None:
            assert trust_env is False

        def __enter__(self) -> "_Client":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def get(self, url: str, headers: dict[str, str]) -> _Response:
            assert url == "http://localhost:8000/v1/health"
            assert headers == {}
            return _Response()

    monkeypatch.setattr("mineru.parser.api_client.httpx.Client", _Client)

    source = parser._build_source(pdf)

    assert source == {"type": "file_id", "file_id": "file_1"}


def test_api_client_uses_local_source_when_health_advertises_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = tmp_path / "demo.pdf"
    pdf.write_bytes(b"%PDF-1.7\n")
    parser = MinerUApiParser(api_url="http://localhost:8000", tier="standard")
    monkeypatch.setattr(parser, "_upload", lambda _path: pytest.fail("local source should not upload"))

    class _Response:
        status_code = 200
        text = ""

        def json(self) -> dict[str, object]:
            return {"features": {"sources": ["file_id", "url", "inline", "local"]}}

    class _Client:
        def __init__(self, **_: object) -> None:
            pass

        def __enter__(self) -> "_Client":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def get(self, url: str, headers: dict[str, str]) -> _Response:
            return _Response()

    monkeypatch.setattr("mineru.parser.api_client.httpx.Client", _Client)

    source = parser._build_source(pdf)

    assert source == {"type": "local", "path": str(pdf)}


def test_async_api_client_uses_file_id_when_local_server_does_not_advertise_local_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = tmp_path / "demo.pdf"
    pdf.write_bytes(b"%PDF-1.7\n")
    parser = MinerUApiParser(api_url="http://localhost:8000", tier="standard")

    async def _async_upload(_path: Path) -> str:
        return "file_1"

    monkeypatch.setattr(parser, "_async_upload", _async_upload)

    class _Response:
        status_code = 200
        text = ""

        def json(self) -> dict[str, object]:
            return {"features": {"sources": ["file_id", "url", "inline"]}}

    class _AsyncClient:
        def __init__(self, *, timeout: object, trust_env: bool, **_: object) -> None:
            assert trust_env is False

        async def __aenter__(self) -> "_AsyncClient":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def get(self, url: str, headers: dict[str, str]) -> _Response:
            assert url == "http://localhost:8000/v1/health"
            assert headers == {}
            return _Response()

    monkeypatch.setattr("mineru.parser.api_client.httpx.AsyncClient", _AsyncClient)

    source = asyncio.run(parser._async_build_source(pdf))

    assert source == {"type": "file_id", "file_id": "file_1"}


def test_api_client_can_request_zip_for_model_output(tmp_path: Path) -> None:
    pdf = tmp_path / "demo.pdf"
    pdf.write_bytes(b"%PDF-1.7\n")

    parser = MinerUApiParser(api_url="http://localhost:8000", tier="standard", include_model_output=True)
    payload = parser._build_payload({"type": "local", "path": str(pdf)}, "")

    assert payload["output_formats"] == ["zip"]


def test_api_client_can_request_zip_for_image_cache(tmp_path: Path) -> None:
    pdf = tmp_path / "demo.pdf"
    pdf.write_bytes(b"%PDF-1.7\n")

    parser = MinerUApiParser(
        api_url="http://localhost:8000",
        tier="standard",
        include_images=True,
    )
    payload = parser._build_payload({"type": "local", "path": str(pdf)}, "")

    assert payload["output_formats"] == ["zip"]


def test_api_client_uses_zip_for_official_api_when_model_output_requested() -> None:
    parser = MinerUApiParser(
        api_url="https://mineru.net/api",
        tier="advanced",
        include_model_output=True,
    )

    assert parser._output_formats() == ["zip"]


def test_api_client_uses_zip_for_official_api_when_image_cache_requested() -> None:
    parser = MinerUApiParser(
        api_url="https://mineru.net/api",
        tier="advanced",
        include_images=True,
    )

    assert parser._output_formats() == ["zip"]


def test_api_client_uses_tier_without_backend_semantics(tmp_path: Path) -> None:
    pdf = tmp_path / "demo.pdf"
    pdf.write_bytes(b"%PDF-1.7\n")

    parser = MinerUApiParser(api_url="http://localhost:8000", tier="advanced")
    payload = parser._build_payload({"type": "local", "path": str(pdf)}, "")

    assert payload["tier"] == "advanced"
    assert not hasattr(parser, "backend")


@pytest.mark.parametrize("tier", ["ultra", "experimental"])
def test_api_client_rejects_invalid_tier_names(tier: str) -> None:
    with pytest.raises(ValueError, match=f"Unsupported tier '{tier}'"):
        MinerUApiParser(api_url="http://localhost:8000", tier=tier)  # type: ignore[arg-type]


def test_api_client_uses_middle_json_format_for_official_api() -> None:
    parser = MinerUApiParser(api_url="https://mineru.net/api", tier="advanced")

    assert parser._output_formats() == ["middle_json"]


def test_api_client_uses_env_api_key_when_no_api_key_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINERU_API_KEY", "env-key")

    parser = MinerUApiParser(api_url="https://mineru.net/api", tier="standard")

    assert parser._headers()["Authorization"] == "Bearer env-key"


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, object], text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> dict[str, object]:
        return self._payload


def test_api_client_maps_remote_401_to_invalid_api_key() -> None:
    response = _FakeResponse(
        401,
        {
            "traceId": "trace-1",
            "msgCode": "A0202",
            "msg": "user authenticate failed",
            "data": None,
            "success": False,
            "total": 0,
        },
        text='{"msgCode":"A0202","msg":"user authenticate failed"}',
    )

    with pytest.raises(api_client._V1APIError) as exc_info:
        MinerUApiParser._check(response)

    assert exc_info.value.code == "invalid_api_key"
    assert exc_info.value.message == "Remote authentication failed: user authenticate failed"
    assert exc_info.value.param == "parse_server.remote.api_key"


def test_api_client_preserves_structured_error_body_on_http_error() -> None:
    response = _FakeResponse(
        401,
        {
            "error": {
                "type": "authentication_error",
                "code": "invalid_api_key",
                "message": "Invalid or missing API key",
                "param": "api_key",
            }
        },
        text='{"error":{"code":"invalid_api_key"}}',
    )

    with pytest.raises(api_client._V1APIError) as exc_info:
        MinerUApiParser._check(response)

    assert exc_info.value.code == "invalid_api_key"
    assert exc_info.value.message == "Invalid or missing API key"
    assert exc_info.value.param == "api_key"


def test_api_client_ignores_legacy_detail_error_envelope() -> None:
    response = _FakeResponse(
        400,
        {
            "detail": {
                "error": {
                    "type": "invalid_request_error",
                    "code": "invalid_request",
                    "message": "legacy detail envelope",
                }
            }
        },
        text='{"detail":{"error":{"code":"invalid_request"}}}',
    )

    with pytest.raises(api_client._V1APIError) as exc_info:
        MinerUApiParser._check(response)

    assert exc_info.value.code == "http_error"
    assert exc_info.value.message.startswith("HTTP 400:")


def test_api_client_reads_image_cache_from_zip_and_preserves_pdf_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = MinerUApiParser(api_url="http://localhost:8000", tier="standard", include_images=True)
    middle_json = {
        "schema_version": "1.0.0",
        "_pdf_retained_page_indices": [0, 2],
        "_pdf_broken_page_indices": [1],
        "pages": [
            {
                "page_idx": 0,
                "page_size": [100, 200],
                "para_blocks": [
                    {
                        "index": 0,
                        "type": "image",
                        "bbox": [0, 0, 10, 10],
                        "lines": [
                            {
                                "bbox": [0, 0, 10, 10],
                                "spans": [
                                    {
                                        "type": "image",
                                        "bbox": [0, 0, 10, 10],
                                        "image_path": "images/chart.png",
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        ],
    }
    zip_ref = {"file_id": "file-zip", "bytes": 10}
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("middle_json.json", json.dumps(middle_json, ensure_ascii=False))
        archive.writestr("images/chart.png", b"chart-bytes")

    monkeypatch.setattr(api_client, "_download_bytes", lambda _parser, ref: zip_buffer.getvalue() if ref is zip_ref else b"")

    result = _parse_result_from_job(
        {
            "job_id": "job_1",
            "status": "completed",
            "files": [{"output_files": {"zip": zip_ref}}],
        },
        "demo.pdf",
        parser,
    )

    assert result._retained_page_indices == [0, 2]
    assert result._broken_page_indices == [1]
    assert result.images() == {"images/chart.png": b"chart-bytes"}


def test_api_client_downloads_model_output_from_zip(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = MinerUApiParser(api_url="http://localhost:8000", tier="standard", include_model_output=True)
    middle_json = {
        "schema_version": "1.0.0",
        "pages": [{"page_idx": 0, "page_size": [100, 200]}],
    }
    zip_ref = {"file_id": "file-zip", "bytes": 10}
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("middle_json.json", json.dumps(middle_json, ensure_ascii=False))
        archive.writestr(
            "model_output.json",
            json.dumps([[{"raw": "model"}]], ensure_ascii=False, indent=4),
        )

    monkeypatch.setattr(
        api_client,
        "_download_bytes",
        lambda _parser, ref: zip_buffer.getvalue() if ref is zip_ref else b"",
    )

    result = _parse_result_from_job(
        {
            "job_id": "job_1",
            "status": "completed",
            "files": [{"output_files": {"zip": zip_ref}}],
        },
        "demo.pdf",
        parser,
    )

    assert result._model_output == [[{"raw": "model"}]]


def test_api_client_reads_official_layout_json_and_model_output_from_zip(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = MinerUApiParser(
        api_url="https://mineru.net/api",
        tier="standard",
        include_images=True,
        include_model_output=True,
    )
    zip_ref = {"file_id": "file-zip", "bytes": 10}
    middle_json = {
        "_backend": "hybrid",
        "pdf_info": [
            {
                "page_idx": 0,
                "page_size": [100, 200],
                "para_blocks": [
                    {
                        "index": 0,
                        "type": "image",
                        "bbox": [0, 0, 10, 10],
                        "lines": [
                            {
                                "bbox": [0, 0, 10, 10],
                                "spans": [{"type": "image", "bbox": [0, 0, 10, 10], "image_path": "chart.png"}],
                            }
                        ],
                    }
                ],
            }
        ],
    }
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("layout.json", json.dumps(middle_json, ensure_ascii=False))
        archive.writestr("images/chart.png", b"chart-bytes")
        archive.writestr("ab3a55b0-2017-4e35-8507-ab8e2c012160_model.json", json.dumps([[{"raw": "model"}]]))

    monkeypatch.setattr(api_client, "_download_bytes", lambda _parser, ref: zip_buffer.getvalue() if ref is zip_ref else b"")

    result = _parse_result_from_job(
        {
            "job_id": "job_1",
            "status": "completed",
            "files": [{"output_files": {"zip": zip_ref}}],
        },
        "demo.pdf",
        parser,
    )

    assert len(result.pages) == 1
    assert result.pages[0]._backend == "hybrid"
    assert result.images() == {"chart.png": b"chart-bytes"}
    assert result._model_output == [[{"raw": "model"}]]


def test_api_client_async_downloads_model_output_from_zip(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = MinerUApiParser(api_url="http://localhost:8000", tier="standard", include_model_output=True)
    middle_json = {
        "schema_version": "1.0.0",
        "pages": [{"page_idx": 0, "page_size": [100, 200]}],
    }
    zip_ref = {"file_id": "file-zip", "bytes": 10}
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("middle_json.json", json.dumps(middle_json, ensure_ascii=False))
        archive.writestr(
            "model_output.json",
            json.dumps([[{"raw": "model"}]], ensure_ascii=False, indent=4),
        )

    async def fake_download_bytes(_parser: object, ref: dict[str, object]) -> bytes:
        return zip_buffer.getvalue() if ref is zip_ref else b""

    monkeypatch.setattr(api_client, "_async_download_bytes", fake_download_bytes)

    result = asyncio.run(
        api_client._async_parse_result_from_job(
            {
                "job_id": "job_1",
                "status": "completed",
                "files": [{"output_files": {"zip": zip_ref}}],
            },
            "demo.pdf",
            parser,
        )
    )

    assert result._model_output == [[{"raw": "model"}]]


def test_api_client_include_images_downloads_single_zip(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = MinerUApiParser(
        api_url="http://localhost:8000",
        tier="standard",
        include_images=True,
        include_model_output=True,
    )
    zip_ref = {"file_id": "file-zip", "bytes": 10}
    image_cache = ImagePayloadCache()
    image_path = image_cache.register_bytes(b"chart-bytes", "png", image_path="images/chart.png")
    span = Span(type="image", bbox=(0, 0, 10, 10), image_path=image_path)
    page = PageInfo(
        page_idx=0,
        page_size=(100, 200),
        para_blocks=[Block(index=0, type="image", bbox=(0, 0, 10, 10), lines=[Line(bbox=(0, 0, 10, 10), spans=[span])])],
        _backend="hybrid",
    )
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("middle_json.json", json.dumps(ParseResult(pages=[page]).to_dict(), ensure_ascii=False))
        archive.writestr("images/chart.png", b"chart-bytes")
        archive.writestr("model_output.json", json.dumps([[{"raw": "model"}]], ensure_ascii=False, indent=4))
    download_calls: list[dict[str, object]] = []

    def fake_download_bytes(_parser: object, ref: dict[str, object]) -> bytes:
        download_calls.append(ref)
        return zip_buffer.getvalue() if ref is zip_ref else b""

    monkeypatch.setattr(api_client, "_download_bytes", fake_download_bytes)
    monkeypatch.setattr(
        api_client,
        "_download_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("zip result must not download middle_json separately")),
    )

    result = _parse_result_from_job(
        {
            "job_id": "job_1",
            "status": "completed",
            "files": [{"output_files": {"zip": zip_ref}}],
        },
        "demo.pdf",
        parser,
    )

    assert download_calls == [zip_ref]
    assert result.pages[0].page_idx == 0
    assert result.images() == {"images/chart.png": b"chart-bytes"}
    assert result._model_output == [[{"raw": "model"}]]


def test_api_client_does_not_read_image_cache_from_zip_unless_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = MinerUApiParser(api_url="http://localhost:8000", tier="standard", include_model_output=True)
    zip_ref = {"file_id": "file-zip", "bytes": 10}
    middle_json = {
        "schema_version": "1.0.0",
        "pages": [
            {
                "page_idx": 0,
                "page_size": [100, 200],
                "para_blocks": [
                    {
                        "index": 0,
                        "type": "image",
                        "bbox": [0, 0, 10, 10],
                        "lines": [
                            {
                                "bbox": [0, 0, 10, 10],
                                "spans": [{"type": "image", "bbox": [0, 0, 10, 10], "image_path": "images/chart.png"}],
                            }
                        ],
                    }
                ],
            }
        ],
    }
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("middle_json.json", json.dumps(middle_json, ensure_ascii=False))
        archive.writestr("images/chart.png", b"chart-bytes")
        archive.writestr("model_output.json", json.dumps([[{"raw": "model"}]], ensure_ascii=False, indent=4))

    monkeypatch.setattr(api_client, "_download_bytes", lambda _parser, ref: zip_buffer.getvalue() if ref is zip_ref else b"")

    result = _parse_result_from_job(
        {
            "job_id": "job_1",
            "status": "completed",
            "files": [{"output_files": {"zip": zip_ref}}],
        },
        "demo.pdf",
        parser,
    )

    assert result.images() == {}
    assert result._model_output == [[{"raw": "model"}]]


def test_api_client_async_include_images_downloads_single_zip(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = MinerUApiParser(
        api_url="http://localhost:8000",
        tier="standard",
        include_images=True,
        include_model_output=True,
    )
    zip_ref = {"file_id": "file-zip", "bytes": 10}
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "middle_json.json",
            json.dumps({"schema_version": "1.0.0", "pages": [{"page_idx": 2, "page_size": [100, 200]}]}),
        )
        archive.writestr("model_output.json", json.dumps([[{"raw": "model"}]], ensure_ascii=False, indent=4))
    download_calls: list[dict[str, object]] = []

    async def fake_download_bytes(_parser: object, ref: dict[str, object]) -> bytes:
        download_calls.append(ref)
        return zip_buffer.getvalue() if ref is zip_ref else b""

    async def fail_download_json(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise AssertionError("zip result must not download middle_json separately")

    monkeypatch.setattr(api_client, "_async_download_bytes", fake_download_bytes)
    monkeypatch.setattr(api_client, "_async_download_json", fail_download_json)

    result = asyncio.run(
        api_client._async_parse_result_from_job(
            {
                "job_id": "job_1",
                "status": "completed",
                "files": [{"output_files": {"zip": zip_ref}}],
            },
            "demo.pdf",
            parser,
        )
    )

    assert download_calls == [zip_ref]
    assert result.pages[0].page_idx == 2
    assert result._model_output == [[{"raw": "model"}]]


def test_api_client_include_images_rejects_unsafe_image_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = MinerUApiParser(api_url="http://localhost:8000", tier="standard", include_images=True)
    zip_ref = {"file_id": "file-zip", "bytes": 10}
    middle_json = {
        "schema_version": "1.0.0",
        "pages": [
            {
                "page_idx": 0,
                "para_blocks": [
                    {
                        "index": 0,
                        "type": "image",
                        "bbox": [0, 0, 10, 10],
                        "lines": [
                            {
                                "bbox": [0, 0, 10, 10],
                                "spans": [{"type": "image", "bbox": [0, 0, 10, 10], "image_path": "../escape.png"}],
                            }
                        ],
                    }
                ],
            }
        ],
    }
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("middle_json.json", json.dumps(middle_json, ensure_ascii=False))

    monkeypatch.setattr(api_client, "_download_bytes", lambda _parser, ref: zip_buffer.getvalue() if ref is zip_ref else b"")

    with pytest.raises(ValueError, match="Unsafe image sidecar path"):
        _parse_result_from_job(
            {
                "job_id": "job_1",
                "status": "completed",
                "files": [{"output_files": {"zip": zip_ref}}],
            },
            "demo.pdf",
            parser,
        )


def test_api_client_accepts_remote_pdf_info_middle_json(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = MinerUApiParser(api_url="https://mineru.net/api", tier="standard")
    middle_json = {
        "_backend": "hybrid",
        "_version_name": "remote",
        "pdf_info": [{"page_idx": 0, "page_size": [100, 200]}],
    }

    monkeypatch.setattr(api_client, "_download_json", lambda _parser, _outputs: middle_json)

    result = _parse_result_from_job(
        {
            "job_id": "job_1",
            "status": "completed",
            "files": [{"output_files": {"middle_json": {"file_id": "file-middle-json", "bytes": 10}}}],
        },
        "demo.pdf",
        parser,
    )

    assert len(result.pages) == 1
    assert result.pages[0].page_idx == 0
    assert result.pages[0].page_size == (100, 200)
    assert result.pages[0]._backend == "hybrid"


def test_async_api_client_accepts_remote_pdf_info_middle_json(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = MinerUApiParser(api_url="https://mineru.net/api", tier="standard")
    middle_json = {
        "_backend": "hybrid",
        "pdf_info": [{"page_idx": 0, "page_size": [100, 200]}],
    }

    async def _download_json(*_args: object, **_kwargs: object) -> dict[str, object]:
        return middle_json

    monkeypatch.setattr(api_client, "_async_download_json", _download_json)

    result = asyncio.run(
        api_client._async_parse_result_from_job(
            {
                "job_id": "job_1",
                "status": "completed",
                "files": [{"output_files": {"middle_json": {"file_id": "file-middle-json", "bytes": 10}}}],
            },
            "demo.pdf",
            parser,
        )
    )

    assert len(result.pages) == 1
    assert result.pages[0].page_idx == 0
    assert result.pages[0].page_size == (100, 200)
    assert result.pages[0]._backend == "hybrid"


def test_api_client_rejects_legacy_json_output_file_key() -> None:
    parser = MinerUApiParser(api_url="https://mineru.net/api", tier="standard")

    with pytest.raises(api_client._V1APIError) as exc_info:
        _parse_result_from_job(
            {
                "job_id": "job_1",
                "status": "completed",
                "files": [{"output_files": {"json": {"file_id": "file-json", "bytes": 10}}}],
            },
            "demo.pdf",
            parser,
        )

    assert exc_info.value.code == "missing_middle_json_output"
    assert "available outputs: json" in exc_info.value.message


def test_api_client_rejects_output_reference_without_file_id() -> None:
    parser = MinerUApiParser(api_url="https://mineru.net/api", tier="standard")

    with pytest.raises(api_client._V1APIError) as exc_info:
        api_client._download_bytes(parser, {"url": "https://example.invalid/output.json", "bytes": 10})

    assert exc_info.value.code == "invalid_response"
    assert exc_info.value.message == "No file_id in output reference"


def test_async_api_client_rejects_output_reference_without_file_id() -> None:
    parser = MinerUApiParser(api_url="https://mineru.net/api", tier="standard")

    async def _run() -> None:
        await api_client._async_download_bytes(parser, {"url": "https://example.invalid/output.json", "bytes": 10})

    with pytest.raises(api_client._V1APIError) as exc_info:
        asyncio.run(_run())

    assert exc_info.value.code == "invalid_response"
    assert exc_info.value.message == "No file_id in output reference"


@pytest.mark.parametrize("image_path", ["../escape.png", "/tmp/escape.png", "\\escape.png", "C:\\escape.png"])
def test_api_client_include_images_rejects_unsafe_zip_image_sidecar_paths(
    monkeypatch: pytest.MonkeyPatch,
    image_path: str,
) -> None:
    parser = MinerUApiParser(api_url="http://localhost:8000", tier="standard", include_images=True)
    zip_ref = {"file_id": "file-zip", "bytes": 10}
    middle_json = {
        "schema_version": "1.0.0",
        "pages": [
            {
                "page_idx": 0,
                "para_blocks": [
                    {
                        "index": 0,
                        "type": "image",
                        "bbox": [0, 0, 10, 10],
                        "lines": [
                            {
                                "bbox": [0, 0, 10, 10],
                                "spans": [{"type": "image", "bbox": [0, 0, 10, 10], "image_path": image_path}],
                            }
                        ],
                    }
                ],
            }
        ],
    }
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("middle_json.json", json.dumps(middle_json, ensure_ascii=False))

    monkeypatch.setattr(api_client, "_download_bytes", lambda _parser, ref: zip_buffer.getvalue() if ref is zip_ref else b"")

    with pytest.raises(ValueError, match="Unsafe image sidecar path"):
        _parse_result_from_job(
            {
                "job_id": "job_1",
                "status": "completed",
                "files": [{"output_files": {"zip": zip_ref}}],
            },
            "demo.pdf",
            parser,
        )


def test_async_api_client_include_images_rejects_unsafe_zip_image_sidecar_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = MinerUApiParser(api_url="http://localhost:8000", tier="standard", include_images=True)
    zip_ref = {"file_id": "file-zip", "bytes": 10}
    middle_json = {
        "schema_version": "1.0.0",
        "pages": [
            {
                "page_idx": 0,
                "para_blocks": [
                    {
                        "index": 0,
                        "type": "image",
                        "bbox": [0, 0, 10, 10],
                        "lines": [
                            {
                                "bbox": [0, 0, 10, 10],
                                "spans": [{"type": "image", "bbox": [0, 0, 10, 10], "image_path": "../escape.png"}],
                            }
                        ],
                    }
                ],
            }
        ],
    }
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("middle_json.json", json.dumps(middle_json, ensure_ascii=False))

    async def _download_bytes(_parser: object, ref: dict[str, object]) -> bytes:
        return zip_buffer.getvalue() if ref is zip_ref else b""

    monkeypatch.setattr(api_client, "_async_download_bytes", _download_bytes)

    with pytest.raises(ValueError, match="Unsafe image sidecar path"):
        asyncio.run(
            api_client._async_parse_result_from_job(
                {
                    "job_id": "job_1",
                    "status": "completed",
                    "files": [{"output_files": {"zip": zip_ref}}],
                },
                "demo.pdf",
                parser,
            )
        )


def test_api_client_disables_env_proxy_for_local_network_urls() -> None:
    assert should_trust_env_for_url("http://localhost:8000") is False
    assert should_trust_env_for_url("http://127.0.0.1:8000") is False
    assert should_trust_env_for_url("http://192.168.1.20:8000/api") is False
    assert should_trust_env_for_url("https://mineru.net/api") is True


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

    MinerUApiParser(api_url="http://127.0.0.1:8000")._do_parse({"files": []})
    MinerUApiParser(api_url="https://mineru.net/api")._do_parse({"files": []})

    assert calls == [False, True]


def test_api_client_poll_uses_fixed_one_second_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证同步 v1 job 轮询使用固定 1 秒间隔，避免指数退避放大本地等待。"""
    parser = MinerUApiParser(api_url="http://localhost:8000", tier="standard")
    statuses = iter(["pending", "processing", "completed"])
    sleep_delays: list[int] = []

    def fake_sleep(delay: int) -> None:
        """记录同步轮询 sleep 参数，避免测试真的等待。"""
        sleep_delays.append(delay)

    def fake_get(url: str, headers: dict[str, str]) -> object:
        """按顺序返回 job 状态，模拟第三次轮询完成。"""
        assert url == "http://localhost:8000/v1/parse/jobs/job_1"
        assert headers == {}
        status = next(statuses)
        return types.SimpleNamespace(
            status_code=200,
            text="",
            json=lambda: {"job_id": "job_1", "status": status},
        )

    monkeypatch.setattr(api_client.time, "sleep", fake_sleep)

    job = parser._poll(types.SimpleNamespace(get=fake_get), "job_1")

    assert job["status"] == "completed"
    assert sleep_delays == [1, 1, 1]


def test_async_api_client_poll_uses_fixed_one_second_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证异步 v1 job 轮询也使用固定 1 秒间隔，保持 Gradio 路径一致。"""
    parser = MinerUApiParser(api_url="http://localhost:8000", tier="standard")
    statuses = iter(["pending", "processing", "completed"])
    sleep_delays: list[int] = []

    async def fake_sleep(delay: int) -> None:
        """记录异步轮询 sleep 参数，避免测试真的等待。"""
        sleep_delays.append(delay)

    async def fake_get(url: str, headers: dict[str, str]) -> object:
        """按顺序返回 job 状态，模拟第三次异步轮询完成。"""
        assert url == "http://localhost:8000/v1/parse/jobs/job_1"
        assert headers == {}
        status = next(statuses)
        return types.SimpleNamespace(
            status_code=200,
            text="",
            json=lambda: {"job_id": "job_1", "status": status},
        )

    monkeypatch.setattr(api_client.asyncio, "sleep", fake_sleep)

    job = asyncio.run(parser._async_poll(types.SimpleNamespace(get=fake_get), "job_1"))

    assert job["status"] == "completed"
    assert sleep_delays == [1, 1, 1]


def test_api_client_poll_timeout_budget_is_one_hour() -> None:
    """验证 v1 job 轮询最大等待预算为 1 小时。"""
    assert api_client._POLL_INTERVAL_SECONDS * api_client._POLL_MAX_ATTEMPTS == 60 * 60


def test_api_client_retries_transient_poll_transport_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = MinerUApiParser(api_url="https://mineru.net/api", tier="standard")
    calls = 0
    sleep_delays: list[float] = []

    def fake_get(_url: str, *, headers: dict[str, str]) -> httpx.Response:
        nonlocal calls
        assert headers == {}
        calls += 1
        if calls < 3:
            raise httpx.ReadError("connection closed")
        return httpx.Response(200, json={"job_id": "job_1", "status": "completed"})

    monkeypatch.setattr(api_client, "_transport_retry_delay", lambda _attempt: 0.0)
    monkeypatch.setattr(api_client.time, "sleep", sleep_delays.append)

    job = parser._poll(types.SimpleNamespace(get=fake_get), "job_1")

    assert job["status"] == "completed"
    assert calls == 3
    assert sleep_delays == [1, 0.0, 0.0]


def test_async_api_client_retries_transient_poll_transport_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = MinerUApiParser(api_url="https://mineru.net/api", tier="standard")
    calls = 0
    sleep_delays: list[float] = []

    async def fake_get(_url: str, *, headers: dict[str, str]) -> httpx.Response:
        nonlocal calls
        assert headers == {}
        calls += 1
        if calls < 3:
            raise httpx.ReadError("connection closed")
        return httpx.Response(200, json={"job_id": "job_1", "status": "completed"})

    async def fake_sleep(delay: float) -> None:
        sleep_delays.append(delay)

    monkeypatch.setattr(api_client, "_transport_retry_delay", lambda _attempt: 0.0)
    monkeypatch.setattr(api_client.asyncio, "sleep", fake_sleep)

    job = asyncio.run(parser._async_poll(types.SimpleNamespace(get=fake_get), "job_1"))

    assert job["status"] == "completed"
    assert calls == 3
    assert sleep_delays == [1, 0.0, 0.0]


def test_api_client_retries_upload_content_put(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    class _Client:
        def put(self, _url: str, *, headers: dict[str, str], content: bytes) -> httpx.Response:
            nonlocal calls
            assert headers == {"Content-Type": "application/pdf"}
            assert content == b"pdf"
            calls += 1
            if calls == 1:
                raise httpx.WriteError("connection closed")
            return httpx.Response(200)

    monkeypatch.setattr(api_client.time, "sleep", lambda _delay: None)

    response = api_client._request_with_retry(
        _Client(),  # type: ignore[arg-type]
        "PUT",
        "https://upload.example/content",
        stage="upload content",
        headers={"Content-Type": "application/pdf"},
        content=b"pdf",
    )

    assert response.status_code == 200
    assert calls == 2


def test_api_client_recovers_completed_upload_after_lost_complete_response() -> None:
    parser = MinerUApiParser(api_url="https://mineru.net/api", tier="standard")
    complete_calls = 0
    status_calls = 0

    class _Client:
        def post(self, _url: str, *, headers: dict[str, str]) -> httpx.Response:
            nonlocal complete_calls
            assert headers == {}
            complete_calls += 1
            raise httpx.ReadError("response lost")

        def get(self, _url: str, *, headers: dict[str, str]) -> httpx.Response:
            nonlocal status_calls
            assert headers == {}
            status_calls += 1
            return httpx.Response(
                200,
                json={"id": "upload_1", "status": "completed", "file": {"id": "file_1"}},
            )

    upload = parser._complete_upload(_Client(), "upload_1")  # type: ignore[arg-type]

    assert upload["file"]["id"] == "file_1"
    assert complete_calls == 1
    assert status_calls == 1


def test_async_api_client_recovers_completed_upload_after_lost_complete_response() -> None:
    parser = MinerUApiParser(api_url="https://mineru.net/api", tier="standard")
    complete_calls = 0
    status_calls = 0

    class _AsyncClient:
        async def post(self, _url: str, *, headers: dict[str, str]) -> httpx.Response:
            nonlocal complete_calls
            assert headers == {}
            complete_calls += 1
            raise httpx.ReadError("response lost")

        async def get(self, _url: str, *, headers: dict[str, str]) -> httpx.Response:
            nonlocal status_calls
            assert headers == {}
            status_calls += 1
            return httpx.Response(
                200,
                json={"id": "upload_1", "status": "completed", "file": {"id": "file_1"}},
            )

    upload = asyncio.run(parser._async_complete_upload(_AsyncClient(), "upload_1"))  # type: ignore[arg-type]

    assert upload["file"]["id"] == "file_1"
    assert complete_calls == 1
    assert status_calls == 1


def test_api_client_does_not_retry_job_submission_transport_error(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    class _Client:
        def __init__(self, **_: object) -> None:
            pass

        def __enter__(self) -> "_Client":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def post(self, _url: str, *, headers: dict[str, str], json: dict[str, object]) -> httpx.Response:
            nonlocal calls
            assert headers == {}
            assert json == {"files": []}
            calls += 1
            raise httpx.ReadError("response lost")

    monkeypatch.setattr(api_client.httpx, "Client", _Client)

    with pytest.raises(api_client._APITransportError) as exc_info:
        MinerUApiParser(api_url="https://mineru.net/api")._do_parse({"files": []})

    assert exc_info.value.stage == "job submission"
    assert exc_info.value.attempts == 1
    assert calls == 1


def test_api_client_omits_tier_when_unspecified(tmp_path: Path) -> None:
    pdf = tmp_path / "demo.pdf"
    pdf.write_bytes(b"%PDF-1.7\n")

    parser = MinerUApiParser(api_url="http://localhost:8000")
    payload = parser._build_payload({"type": "local", "path": str(pdf)}, "")

    assert "tier" not in payload


def test_api_client_constructor_does_not_expose_ocr_or_image_options() -> None:
    parameters = inspect.signature(MinerUApiParser).parameters

    assert "method" not in parameters
    assert "image_analysis" not in parameters


def test_parser_entrypoints_expose_api_server_style_options() -> None:
    for entrypoint in (parse, parse_async, _build_parser):
        parameters = inspect.signature(entrypoint).parameters

        assert "tier" in parameters
        assert "backend" in parameters
        assert "language" in parameters
        assert "ocr_mode" in parameters
        assert "effort" in parameters
        assert "disable_image_analysis" in parameters
        assert _REMOVED_DISABLE_TABLE_PARAM not in parameters
        assert _REMOVED_DISABLE_FORMULA_PARAM not in parameters
        assert _REMOVED_TABLE_ENABLE_PARAM not in parameters
        assert _REMOVED_FORMULA_ENABLE_PARAM not in parameters


def test_pdf_hybrid_parser_constructor_removes_formula_table_switches() -> None:
    """校验 PDF parser 构造链不再公开无效的公式/表格开关。"""
    parameters = inspect.signature(parser_pdf.PdfHybridParser).parameters

    assert _REMOVED_FORMULA_ENABLE_PARAM not in parameters
    assert _REMOVED_TABLE_ENABLE_PARAM not in parameters


def test_api_client_omits_page_range_when_unspecified(tmp_path: Path) -> None:
    pdf = tmp_path / "demo.pdf"
    pdf.write_bytes(b"%PDF-1.7\n")

    parser = MinerUApiParser(api_url="http://localhost:8000", tier="standard")
    payload = parser._build_payload({"type": "local", "path": str(pdf)}, "")

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


def test_api_client_surfaces_failed_file_error_when_job_has_no_top_level_error() -> None:
    parser = MinerUApiParser(api_url="http://localhost:8000", tier="standard")

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


def test_api_client_rejects_json_underscore_output_alias() -> None:
    parser = MinerUApiParser(api_url="http://localhost:8000", tier="standard")

    with pytest.raises(api_client._V1APIError) as exc_info:
        api_client._download_json(parser, {"json_": {"file_id": "file-json", "bytes": 10}})

    assert exc_info.value.code == "missing_middle_json_output"
    assert "available outputs: json_" in exc_info.value.message


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

    unsupported = CreateJobRequest.model_validate(
        {
            "files": [{"source": {"type": "local", "path": "/tmp/demo.pdf"}}],
            "output_formats": ["images"],
        }
    )
    assert unsupported.output_formats == ["images"]

    with pytest.raises(ValidationError):
        CreateJobRequest.model_validate(
            {
                "files": [{"source": {"type": "local", "path": "/tmp/demo.pdf"}}],
                "wait": 5,
            }
        )

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


def test_api_server_rejects_local_source_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_api_server_dependency_preflight(monkeypatch)
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF-1.7\n")
    app = create_app(upload_dir=str(tmp_path / "api"), tier="flash")

    with TestClient(app) as client:
        response = client.post(
            "/v1/parse/jobs",
            json={"tier": "flash", "files": [{"source": {"type": "local", "path": str(source)}}]},
        )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "unsupported_source"
    assert body["error"]["param"] == "files.0.source"
    assert "--allow-local-source" in body["error"]["message"]


def test_api_server_health_sources_reflect_local_source_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_api_server_dependency_preflight(monkeypatch)
    default_app = create_app(upload_dir=str(tmp_path / "default"), tier="flash")
    local_app = create_app(upload_dir=str(tmp_path / "local"), tier="flash", allow_local_source=True)

    with TestClient(default_app) as client:
        default_sources = client.get("/v1/health").json()["features"]["sources"]
    with TestClient(local_app) as client:
        local_sources = client.get("/v1/health").json()["features"]["sources"]

    assert default_sources == ["file_id", "url", "inline"]
    assert local_sources == ["file_id", "url", "inline", "local"]


def test_api_server_rejects_inline_source_over_configured_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_api_server_dependency_preflight(monkeypatch)
    app = create_app(upload_dir=str(tmp_path), tier="flash", max_inline_bytes=3)

    with TestClient(app) as client:
        response = client.post(
            "/v1/parse/jobs",
            json={
                "tier": "flash",
                "files": [
                    {
                        "source": {
                            "type": "inline",
                            "name": "demo.pdf",
                            "data": base64.b64encode(b"1234").decode("ascii"),
                        }
                    }
                ]
            },
        )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "unsupported_source"
    assert body["error"]["param"] == "files.0.source"
    assert "max_inline_bytes" in body["error"]["message"]


def test_api_server_rejects_plain_http_url_source_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_api_server_dependency_preflight(monkeypatch)
    app = create_app(upload_dir=str(tmp_path), tier="flash")

    with TestClient(app) as client:
        response = client.post(
            "/v1/parse/jobs",
            json={"tier": "flash", "files": [{"source": {"type": "url", "url": "http://example.test/demo.pdf"}}]},
        )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "unsupported_source"
    assert body["error"]["param"] == "files.0.source"
    assert "https" in body["error"]["message"]


def test_api_server_rejects_unsupported_output_format_with_specific_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_api_server_dependency_preflight(monkeypatch)
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF-1.7\n")
    app = create_app(upload_dir=str(tmp_path), tier="flash", allow_local_source=True)

    with TestClient(app) as client:
        response = client.post(
            "/v1/parse/jobs",
            json={
                "tier": "flash",
                "files": [{"source": {"type": "local", "path": str(source)}}],
                "output_formats": ["images"],
            },
        )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "unsupported_output_format"
    assert body["error"]["param"] is None
    assert "Unknown output format: images" in body["error"]["message"]


def test_api_server_rejects_unsupported_source_type_with_specific_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_api_server_dependency_preflight(monkeypatch)
    app = create_app(upload_dir=str(tmp_path), tier="flash")

    with TestClient(app) as client:
        response = client.post(
            "/v1/parse/jobs",
            json={
                "tier": "flash",
                "files": [{"source": {"type": "s3", "uri": "s3://bucket/demo.pdf"}}],
            },
        )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "unsupported_source"
    assert body["error"]["param"] == "files.0.source"


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
    body = response.json()
    assert "detail" not in body
    assert body["error"]["code"] == "invalid_request"
    assert "Webhook callback is not supported" in body["error"]["message"]
    assert app.state.job_store._jobs == {}


def test_api_server_validation_errors_use_error_envelope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_api_server_dependency_preflight(monkeypatch)
    app = create_app(upload_dir=str(tmp_path))
    client = TestClient(app)

    response = client.post("/v1/parse/jobs", json={"files": []})

    assert response.status_code == 400
    body = response.json()
    assert "detail" not in body
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["code"] == "invalid_request"
    assert "Invalid request" in body["error"]["message"]


def test_api_server_unhandled_exceptions_use_error_envelope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_api_server_dependency_preflight(monkeypatch)
    app = create_app(upload_dir=str(tmp_path))

    @app.get("/boom")
    async def _boom() -> None:
        raise RuntimeError("boom")

    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/boom")

    assert response.status_code == 500
    body = response.json()
    assert "detail" not in body
    assert body["error"] == {
        "type": "api_error",
        "code": "internal_error",
        "message": "Internal server error",
        "param": None,
    }


def test_create_app_does_not_read_runtime_settings_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_api_server_dependency_preflight(monkeypatch)
    monkeypatch.setenv("MINERU_TIER", "advanced")
    monkeypatch.setenv("MINERU_BACKEND", "hybrid-auto-engine")
    monkeypatch.setenv("MINERU_CONCURRENCY", "9")
    monkeypatch.setenv("MINERU_URL_TIMEOUT", "99")
    monkeypatch.setenv("MINERU_LANGUAGE", "en")
    monkeypatch.setenv("MINERU_OCR_MODE", "ocr")
    monkeypatch.setenv("MINERU_EFFORT", "high")
    monkeypatch.setenv(_REMOVED_TABLE_ENABLE_ENV, "false")
    monkeypatch.setenv(_REMOVED_FORMULA_ENABLE_ENV, "false")
    monkeypatch.setenv("MINERU_IMAGE_ANALYSIS", "false")

    app = create_app(upload_dir=str(tmp_path))

    assert app.state.tier == "standard"
    assert app.state.backend == "hybrid-engine"
    assert app.state.concurrency == 1
    assert app.state.url_timeout == 60
    assert app.state.allow_local_source is False
    assert app.state.max_inline_bytes == 1024 * 1024
    assert app.state.allow_http_source is False
    assert app.state.language == "ch"
    assert app.state.ocr_mode == "auto"
    assert app.state.effort == "high"
    assert app.state.image_analysis is True
    assert not hasattr(app.state, _REMOVED_TABLE_ENABLE_PARAM)
    assert not hasattr(app.state, _REMOVED_FORMULA_ENABLE_PARAM)


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


def test_job_links_do_not_expose_sse_events() -> None:
    links = JobLinks(self="/v1/parse/jobs/job_1", cancel="/v1/parse/jobs/job_1")

    assert links.model_dump() == {
        "self": "/v1/parse/jobs/job_1",
        "cancel": "/v1/parse/jobs/job_1",
    }

    with pytest.raises(ValidationError):
        JobLinks.model_validate(
            {
                "self": "/v1/parse/jobs/job_1",
                "events": "/v1/parse/jobs/job_1/events",
                "cancel": "/v1/parse/jobs/job_1",
            }
        )


@pytest.mark.parametrize(
    ("output_format", "output_attr"),
    [
        ("markdown", "markdown"),
        ("content_list", "content_list"),
        ("structured_content", "structured_content"),
    ],
)
def test_api_server_rendered_outputs_do_not_return_image_sidecars(
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
            "tier": "standard",
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
            image_analysis=True,
            effort="medium",
            allow_local_source=True,
        )
    )

    output_files = rec.files[0].output_files
    assert getattr(output_files, output_attr) is not None
    assert "images" not in output_files.model_dump(by_alias=True, exclude_none=True)


def test_api_server_run_job_normalizes_lightweight_file_tier_to_flash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []

    async def fake_parse_async(*args: object, **kwargs: object) -> ParseResult:
        calls.append({"path": args[0], **kwargs})
        return ParseResult(pages=[PageInfo(page_idx=0, _backend="html")])

    monkeypatch.setattr("mineru.parser.api_server.parse_async", fake_parse_async)
    file_store = FileStore(tmp_path / "api-files")
    source = tmp_path / "demo.html"
    source.write_text("<p>content</p>", encoding="utf-8")
    request = CreateJobRequest.model_validate(
        {
            "files": [{"source": {"type": "local", "path": str(source)}}],
            "tier": "standard",
            "output_formats": ["middle_json"],
        }
    )
    rec = api_server.JobStore().create(request, file_store)

    asyncio.run(
        api_server._run_job(
            rec,
            request,
            file_store,
            server_backend="hybrid-engine",
            language="ch",
            ocr_mode="auto",
            image_analysis=True,
            effort="high",
            allow_local_source=True,
        )
    )

    assert [(call["tier"], call["backend"], call["effort"]) for call in calls] == [("flash", "flash", "medium")]
    assert rec.tier == "standard"
    assert rec.files[0].status == "completed"


@pytest.mark.parametrize(("request_tier", "response_tier"), [("standard", "standard"), (None, "flash")])
def test_api_server_accepts_lightweight_job_without_requested_quality_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    request_tier: str | None,
    response_tier: str,
) -> None:
    async def fake_parse_async(*args: object, **kwargs: object) -> ParseResult:
        return ParseResult(pages=[PageInfo(page_idx=0, _backend="html")])

    monkeypatch.setattr("mineru.parser.api_server.parse_async", fake_parse_async)
    source = tmp_path / "demo.html"
    source.write_text("<p>content</p>", encoding="utf-8")
    app = create_app(upload_dir=str(tmp_path / "api"), tier=["flash"], allow_local_source=True)

    payload: dict[str, object] = {
        "files": [{"source": {"type": "local", "path": str(source)}}],
        "output_formats": ["middle_json"],
    }
    if request_tier is not None:
        payload["tier"] = request_tier

    response = TestClient(app).post("/v1/parse/jobs", json=payload)

    assert response.status_code == 202
    assert response.json()["tier"] == response_tier


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
            "tier": "standard",
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
            image_analysis=True,
            effort="medium",
            allow_local_source=True,
        )
    )

    output_ref = rec.files[0].output_files.middle_json
    payload = json.loads(file_store.read_file_data(output_ref.file_id).decode("utf-8"))
    roundtrip = ParseResult.from_dict(payload)

    assert payload["_backend"] == "hybrid"
    assert roundtrip.content_list()
    assert roundtrip.structured_content()


@pytest.mark.parametrize("model_output", [[{"raw": "model"}], []])
def test_api_server_zip_includes_model_output_when_parse_result_has_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model_output: list[object],
) -> None:
    parse_result = ParseResult(
        pages=[PageInfo(page_idx=0, page_size=(100, 100), _backend="hybrid")],
        _model_output=model_output,
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
            "tier": "standard",
            "output_formats": ["zip"],
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
            image_analysis=True,
            effort="medium",
            allow_local_source=True,
        )
    )

    zip_ref = rec.files[0].output_files.zip
    assert zip_ref is not None
    with zipfile.ZipFile(io.BytesIO(file_store.read_file_data(zip_ref.file_id))) as archive:
        assert "model_output.json" in archive.namelist()
        model_output_text = archive.read("model_output.json").decode("utf-8")
        payload = json.loads(model_output_text)
    assert payload == model_output
    if model_output:
        assert "\n    " in model_output_text


def test_api_server_zip_is_self_contained_when_only_zip_requested(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    img_bytes = b"chart-bytes"
    image_cache = ImagePayloadCache()
    image_path = image_cache.register_bytes(img_bytes, "png", image_path="images/chart.png")
    span = Span(type="image", bbox=(0, 0, 10, 10), image_path=image_path)
    block = Block(index=0, type="image", bbox=(0, 0, 10, 10), lines=[Line(bbox=(0, 0, 10, 10), spans=[span])])
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
        _model_output=[[{"raw": "model"}]],
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
            "tier": "standard",
            "output_formats": ["zip"],
        }
    )
    rec = api_server.JobStore().create(request, file_store)

    asyncio.run(
        api_server._run_job(
            rec,
            request,
            file_store,
            server_backend="hybrid-engine",
            language="ch",
            ocr_mode="auto",
            image_analysis=True,
            effort="medium",
            allow_local_source=True,
        )
    )

    output_files = rec.files[0].output_files
    assert output_files.middle_json is None
    assert output_files.zip is not None
    with zipfile.ZipFile(io.BytesIO(file_store.read_file_data(output_files.zip.file_id))) as archive:
        names = set(archive.namelist())
        assert {
            "markdown.md",
            "middle_json.json",
            "content_list.json",
            "structured_content.json",
            "model_output.json",
            "images/chart.png",
        }.issubset(names)
        assert json.loads(archive.read("middle_json.json").decode("utf-8"))["_backend"] == "hybrid"
        assert json.loads(archive.read("model_output.json").decode("utf-8")) == [[{"raw": "model"}]]
        assert archive.read("images/chart.png") == img_bytes


def test_api_server_zip_rejects_unsafe_image_sidecar_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parse_result = ParseResult(
        pages=[PageInfo(page_idx=0, page_size=(100, 100), _backend="hybrid")],
        _image_cache={"../escape.png": b"bad-image"},
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
            "tier": "standard",
            "output_formats": ["zip"],
        }
    )
    rec = api_server.JobStore().create(request, file_store)

    asyncio.run(
        api_server._run_job(
            rec,
            request,
            file_store,
            server_backend="hybrid-engine",
            language="ch",
            ocr_mode="auto",
            image_analysis=True,
            effort="medium",
            allow_local_source=True,
        )
    )

    assert rec.files[0].status == "failed"
    assert rec.files[0].error is not None
    assert "Unsafe image sidecar path" in rec.files[0].error.message


def test_api_server_zip_skips_model_output_when_parse_result_has_none(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parse_result = ParseResult(
        pages=[PageInfo(page_idx=0, page_size=(100, 100), _backend="hybrid")],
        _model_output=None,
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
            "tier": "standard",
            "output_formats": ["zip"],
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
            image_analysis=True,
            effort="medium",
            allow_local_source=True,
        )
    )

    zip_ref = rec.files[0].output_files.zip
    assert zip_ref is not None
    with zipfile.ZipFile(io.BytesIO(file_store.read_file_data(zip_ref.file_id))) as archive:
        assert "model_output.json" not in archive.namelist()


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
            "tier": "standard",
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
            image_analysis=True,
            effort="medium",
            allow_local_source=True,
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
            "tier": "standard",
            "output_formats": ["middle_json"],
        }
    )
    job_store = api_server.JobStore()
    rec = job_store.create(request, file_store)
    mineru_logger = logging.getLogger("mineru")
    api_logger = logging.getLogger("mineru.parser.api_server")
    monkeypatch.setattr(mineru_logger, "disabled", False)
    monkeypatch.setattr(mineru_logger, "propagate", True)
    monkeypatch.setattr(api_logger, "disabled", False)
    monkeypatch.setattr(api_logger, "propagate", True)

    with caplog.at_level(logging.ERROR, logger="mineru.parser.api_server"):
        asyncio.run(
            api_server._run_job(
                rec,
                request,
                file_store,
                server_backend="hybrid-engine",
                language="ch",
                ocr_mode="auto",
                image_analysis=True,
                effort="medium",
                allow_local_source=True,
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
        "features": {
            "webhook": False,
            "output_formats": ["markdown", "middle_json", "content_list", "structured_content", "zip"],
            "sources": ["file_id", "url", "inline"],
        },
    }

    with pytest.raises(ValidationError):
        FileParseInfo.model_validate({"backend_version": "3.2.1"})

    with pytest.raises(ValidationError):
        HealthResponse.model_validate({"status": "ok", "version": "3.2.1", "backend_version": "3.2.1"})


def test_api_server_tier_selects_compatible_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_api_server_dependency_preflight(monkeypatch)
    flash_app = create_app(upload_dir=str(tmp_path / "flash"), tier="flash")
    basic_app = create_app(upload_dir=str(tmp_path / "basic"), tier="basic")
    standard_app = create_app(upload_dir=str(tmp_path / "standard"), tier="standard")
    advanced_app = create_app(upload_dir=str(tmp_path / "advanced"), tier="advanced")

    assert flash_app.state.tier == "flash"
    assert flash_app.state.backend == "flash"
    assert [tier["id"] for tier in flash_app.state.tiers] == ["flash"]
    assert basic_app.state.tier == "basic"
    assert basic_app.state.backend == "hybrid-engine"
    assert basic_app.state.effort == "medium"
    assert [tier["id"] for tier in basic_app.state.tiers] == ["basic"]
    assert standard_app.state.tier == "standard"
    assert standard_app.state.backend == "hybrid-engine"
    assert standard_app.state.effort == "high"
    assert [tier["id"] for tier in standard_app.state.tiers] == ["standard"]
    assert advanced_app.state.tier == "advanced"
    assert advanced_app.state.backend == "hybrid-engine"
    assert advanced_app.state.effort == "xhigh"
    assert [tier["id"] for tier in advanced_app.state.tiers] == ["advanced"]


def test_api_server_defaults_to_all_quality_tiers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_api_server_dependency_preflight(monkeypatch)
    app = create_app(upload_dir=str(tmp_path))

    assert app.state.tier == "standard"
    assert app.state.default_tier == "standard"
    assert app.state.backend == "hybrid-engine"
    assert app.state.effort == "high"
    assert [tier["id"] for tier in app.state.tiers] == ["flash", "basic", "standard", "advanced"]
    assert app.state.tier_runtime_options["flash"].as_kwargs() == {
        "tier": "flash",
        "backend": "flash",
        "effort": "medium",
    }
    assert app.state.tier_runtime_options["basic"].as_kwargs() == {
        "tier": "basic",
        "backend": "hybrid-engine",
        "effort": "medium",
    }
    assert app.state.tier_runtime_options["standard"].as_kwargs() == {
        "tier": "standard",
        "backend": "hybrid-engine",
        "effort": "high",
    }
    assert app.state.tier_runtime_options["advanced"].as_kwargs() == {
        "tier": "advanced",
        "backend": "hybrid-engine",
        "effort": "xhigh",
    }


def test_api_server_multi_tier_state_and_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """校验同一个 API server 可发布多个 tier，并在包含 standard 时以 standard 作为默认 tier。"""
    _stub_api_server_dependency_preflight(monkeypatch)
    app = create_app(upload_dir=str(tmp_path), tier=["basic", "standard", "advanced"])

    assert app.state.tier == "standard"
    assert app.state.default_tier == "standard"
    assert app.state.backend == "hybrid-engine"
    assert app.state.effort == "high"
    assert [tier["id"] for tier in app.state.tiers] == ["basic", "standard", "advanced"]
    assert app.state.model_ids == ["Hybrid-Basic", "MinerU-HTML", "MinerU2.5-Pro-2605-1.2B"]
    assert app.state.tier_runtime_options["basic"].as_kwargs() == {
        "tier": "basic",
        "backend": "hybrid-engine",
        "effort": "medium",
    }
    assert app.state.tier_runtime_options["standard"].as_kwargs() == {
        "tier": "standard",
        "backend": "hybrid-engine",
        "effort": "high",
    }
    assert app.state.tier_runtime_options["advanced"].as_kwargs() == {
        "tier": "advanced",
        "backend": "hybrid-engine",
        "effort": "xhigh",
    }


def test_api_server_multi_tier_http_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """校验 /v1/tiers 与 /v1/models 返回启动时声明的全部 tier 能力。"""
    _stub_api_server_dependency_preflight(monkeypatch)
    app = create_app(upload_dir=str(tmp_path), tier=["flash", "basic", "standard", "advanced"])

    with TestClient(app) as client:
        tiers_response = client.get("/v1/tiers")
        models_response = client.get("/v1/models")

    assert tiers_response.status_code == 200
    assert [tier["id"] for tier in tiers_response.json()["data"]] == ["flash", "basic", "standard", "advanced"]
    assert models_response.status_code == 200
    assert [model["id"] for model in models_response.json()["data"]] == [
        "MinerU-Flash",
        "Hybrid-Basic",
        "MinerU-HTML",
        "MinerU2.5-Pro-2605-1.2B",
    ]


def test_api_server_multi_tier_jobs_use_requested_tier_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
    app = create_app(upload_dir=str(tmp_path / "api"), tier=["basic", "standard"], allow_local_source=True)
    file_store = FileStore(tmp_path / "api-files")
    job_store = api_server.JobStore()

    async def run_request(payload: dict[str, object]) -> dict[str, object]:
        request = CreateJobRequest.model_validate(payload)
        request.tier = request.tier or app.state.default_tier
        runtime = app.state.tier_runtime_options[request.tier]
        rec = job_store.create(request, file_store)
        await api_server._run_job(
            rec,
            request,
            file_store,
            server_backend=runtime.backend,
            language=app.state.language,
            ocr_mode=app.state.ocr_mode,
            effort=runtime.effort,
            image_analysis=app.state.image_analysis,
            allow_local_source=app.state.allow_local_source,
            max_inline_bytes=app.state.max_inline_bytes,
            allow_http_source=app.state.allow_http_source,
        )
        return job_store.build_response(rec).model_dump(by_alias=True)

    default_response = asyncio.run(
        run_request(
            {
                "files": [{"source": {"type": "local", "path": str(source)}}],
                "output_formats": ["middle_json"],
            }
        )
    )
    basic_response = asyncio.run(
        run_request(
            {
                "files": [{"source": {"type": "local", "path": str(source)}}],
                "tier": "basic",
                "output_formats": ["middle_json"],
            }
        )
    )

    assert default_response["tier"] == "standard"
    assert basic_response["tier"] == "basic"
    assert [(call["tier"], call["backend"], call["effort"]) for call in calls] == [
        ("standard", "hybrid-engine", "high"),
        ("basic", "hybrid-engine", "medium"),
    ]


def test_api_server_single_tier_rejects_unavailable_tier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """校验单 tier server 不接受未启动的 tier，避免请求绕过启动能力边界。"""
    _stub_api_server_dependency_preflight(monkeypatch)
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF-1.7\n")
    app = create_app(upload_dir=str(tmp_path / "api"), tier="standard")

    with TestClient(app) as client:
        response = client.post(
            "/v1/parse/jobs",
            json={
                "files": [{"source": {"type": "local", "path": str(source)}}],
                "tier": "advanced",
            },
        )

    assert response.status_code == 400
    assert "Tier 'advanced' not available in this server" in response.text


def test_api_server_preflights_basic_tier_dependencies(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    imported_modules: list[str] = []

    def fake_import_module(module_name: str):
        imported_modules.append(module_name)
        return object()

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    create_app(upload_dir=str(tmp_path), tier="basic")

    assert imported_modules == [
        "ftfy",
        "shapely",
        "pyclipper",
        "six",
        "torch",
        "torchvision",
        "transformers",
    ]


def test_api_server_preflights_advanced_tier_dependencies_for_platform(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    imported_modules: list[str] = []

    def fake_import_module(module_name: str):
        imported_modules.append(module_name)
        return object()

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    monkeypatch.setattr(parser_tier.sys, "platform", "darwin")
    monkeypatch.setattr(parser_tier.platform, "machine", lambda: "arm64")

    create_app(upload_dir=str(tmp_path), tier="advanced")

    assert imported_modules == [
        "ftfy",
        "shapely",
        "pyclipper",
        "six",
        "torch",
        "torchvision",
        "transformers",
        "accelerate",
        "mlx",
        "mlx_vlm",
    ]


def test_api_server_preflights_advanced_tier_dependencies_skip_mlx_on_intel_macos(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    imported_modules: list[str] = []

    def fake_import_module(module_name: str):
        imported_modules.append(module_name)
        return object()

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    monkeypatch.setattr(parser_tier.sys, "platform", "darwin")
    monkeypatch.setattr(parser_tier.platform, "machine", lambda: "x86_64")

    create_app(upload_dir=str(tmp_path), tier="advanced")

    assert imported_modules == [
        "ftfy",
        "shapely",
        "pyclipper",
        "six",
        "torch",
        "torchvision",
        "transformers",
        "accelerate",
    ]


def test_api_server_preflight_rejects_missing_tier_dependency(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_import_module(module_name: str):
        if module_name == "torch":
            raise ModuleNotFoundError("No module named 'torch'")
        return object()

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    monkeypatch.setattr(parser_tier.importlib_metadata, "packages_distributions", lambda: {"mineru": ["mineru"]})

    with pytest.raises(api_server.ParseServerStartupError, match="tier 'basic'.*torch.*mineru\\[basic\\]"):
        create_app(upload_dir=str(tmp_path), tier="basic")


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


def test_build_parser_maps_legacy_vlm_backend_to_hybrid_xhigh(tmp_path: Path) -> None:
    pdf = tmp_path / "demo.pdf"
    pdf.write_bytes(b"%PDF-1.7\n")

    parser = _build_parser(pdf, backend="vlm-engine", effort="medium")

    assert parser.__class__.__name__ == "PdfHybridParser"
    assert parser.backend == "hybrid-engine"
    assert parser.effort == "xhigh"


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
    """校验 Hybrid basic 直接进入本地分支，不解析 VLM engine。"""
    seen: dict[str, object] = {}
    fake_module = types.ModuleType("mineru.backend.hybrid.hybrid_analyze")

    def fake_doc_analyze(_pdf_bytes: bytes, **kwargs: object) -> tuple[list[PageInfo], list[object], bool]:
        """记录 basic 分支收到的参数，并返回 Hybrid middle-json 形态。"""
        seen.update(kwargs)
        return [PageInfo(page_idx=0, _backend="hybrid")], [], False

    fake_module.doc_analyze = fake_doc_analyze
    monkeypatch.setitem(sys.modules, "mineru.backend.hybrid.hybrid_analyze", fake_module)

    def fail_resolve_backend(*_args: object, **_kwargs: object) -> str:
        """basic 不应触发 VLM backend resolver。"""
        raise AssertionError("medium effort should not resolve VLM backend")

    monkeypatch.setattr(parser_pdf, "_resolve_hybrid_backend", fail_resolve_backend)
    monkeypatch.setenv("MINERU_VLM_FORMULA_ENABLE", "sentinel-formula")
    monkeypatch.setenv("MINERU_VLM_TABLE_ENABLE", "sentinel-table")

    parser = parser_pdf.PdfHybridParser(
        backend="hybrid-engine",
        effort="medium",
        lang="en",
    )
    pages, model_output = parser._run_analysis(b"%PDF-1.7\n")

    assert pages[0]._backend == "hybrid"
    assert model_output == []
    assert seen["backend"] == "hybrid-engine"
    assert seen["effort"] == "medium"
    assert seen["language"] == "en"
    assert _REMOVED_INLINE_FORMULA_PARAM not in seen
    assert os.environ["MINERU_VLM_FORMULA_ENABLE"] == "sentinel-formula"
    assert os.environ["MINERU_VLM_TABLE_ENABLE"] == "sentinel-table"


def test_pdf_hybrid_async_parser_preserves_model_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """校验 v1 server 使用的异步 PDF parser 会把 Hybrid model_list 传入 ParseResult。"""
    model_output = [[{"raw": "model"}]]
    fake_module = types.ModuleType("mineru.backend.hybrid.hybrid_analyze")

    async def fake_aio_doc_analyze(_pdf_bytes: bytes, **kwargs: object) -> tuple[list[PageInfo], list[object], bool]:
        """返回带原始模型输出的 Hybrid 分析结果，避免加载真实模型。"""
        return [PageInfo(page_idx=0, _backend="hybrid")], model_output, False

    fake_module.aio_doc_analyze = fake_aio_doc_analyze
    monkeypatch.setitem(sys.modules, "mineru.backend.hybrid.hybrid_analyze", fake_module)
    monkeypatch.setattr(
        parser_pdf.PdfHybridParser,
        "_prepare_input",
        lambda self, path, page_range="": parser_pdf._PreparedPdfInput(file_name="demo", pdf_bytes=b"%PDF-1.7\n"),
    )

    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF-1.7\n")
    parser = parser_pdf.PdfHybridParser(backend="hybrid-engine", effort="medium")

    result = asyncio.run(parser.parse_async(source))

    assert result._model_output == model_output


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


def test_pdf_vlm_parser_compat_delegates_to_hybrid_xhigh(monkeypatch: pytest.MonkeyPatch) -> None:
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

    pages, model_output = parser._run_analysis(b"%PDF")
    async_pages, async_model_output = asyncio.run(parser._arun_analysis(b"%PDF"))
    assert pages == []
    assert model_output == []
    assert async_pages == []
    assert async_model_output == []
    assert calls == [("hybrid-engine", False), ("hybrid-engine", True)]
    assert backends == ["sync-backend", "async-backend"]
    assert efforts == ["xhigh", "xhigh"]


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

    pages, model_output = parser._run_analysis(b"%PDF")
    async_pages, async_model_output = asyncio.run(parser._arun_analysis(b"%PDF"))
    assert pages == []
    assert model_output == []
    assert async_pages == []
    assert async_model_output == []
    assert calls == [("hybrid-engine", False), ("hybrid-engine", True)]
    assert backends == ["sync-backend", "async-backend"]


def test_api_server_stores_parser_runtime_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_api_server_dependency_preflight(monkeypatch)
    app = create_app(
        upload_dir=str(tmp_path),
        tier="basic",
        language="en",
        ocr_mode="ocr",
        image_analysis=False,
    )

    assert app.state.language == "ch"
    assert app.state.ocr_mode == "ocr"
    assert app.state.effort == "medium"
    assert app.state.image_analysis is False
    assert not hasattr(app.state, _REMOVED_TABLE_ENABLE_PARAM)
    assert not hasattr(app.state, _REMOVED_FORMULA_ENABLE_PARAM)


def test_api_server_rejects_removed_ch_lite_language(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Language ch_lite not supported"):
        create_app(upload_dir=str(tmp_path), language="ch_lite")


def test_api_server_cli_exposes_parser_runtime_options() -> None:
    option_names = {name for param in main.params for name in param.opts}

    assert "--tier" in option_names
    assert "--backend" not in option_names
    assert "--language" in option_names
    assert "--ocr-mode" in option_names
    assert "--allow-local-source" in option_names
    assert "--max-inline-bytes" in option_names
    assert "--allow-http-source" in option_names
    assert "--effort" not in option_names
    assert "--disable-image-analysis" in option_names
    assert _REMOVED_DISABLE_TABLE_OPTION not in option_names
    assert _REMOVED_DISABLE_FORMULA_OPTION not in option_names
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
        "tiers": ["flash", "basic", "standard", "advanced"],
        "default_tier": "standard",
        "effort_by_tier": {"flash": "medium", "basic": "medium", "standard": "high", "advanced": "xhigh"},
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

    result = runner.invoke(main, ["--tier", "basic", "--tier", "advanced", "--host", "0.0.0.0", "--port", "15984"])

    assert result.exit_code == 0
    assert seen == {
        "tiers": ["basic", "advanced"],
        "default_tier": "advanced",
        "effort_by_tier": {"basic": "medium", "advanced": "xhigh"},
    }


def test_api_server_cli_rejects_invalid_tier_names() -> None:
    result = runner.invoke(main, ["--tier", "ultra"])

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
    assert "basic" in normalized_output
    assert "standard" in normalized_output
    assert "advanced" in normalized_output
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
