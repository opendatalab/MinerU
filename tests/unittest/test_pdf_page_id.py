from __future__ import annotations

import asyncio
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfReader
from reportlab.pdfgen import canvas
from typer.testing import CliRunner

from mineru.cli.commands.parse import _validate_page_range_input
from mineru.cli.main import app as cli_app
from mineru.doclib.server import _normalize_content_page_range
from mineru.doclib.services.config_svc import ConfigService
from mineru.doclib.services.parse_svc import default_parse_range
from mineru.errors import InvalidRequestError, MineruError
from mineru.kit.gradio.app import _effective_page_range
from mineru.kit.gradio.artifacts import persist_parse_result
from mineru.kit.gradio.client import V1ArtifactClient
from mineru.kit.main import app as kit_app
from mineru.parser import MinerUParser, api_server
from mineru.parser.api_client import MinerUApiParser
from mineru.parser.page_range import (
    count_pages_in_range,
    expand_page_range,
    format_page_range,
    normalize_page_range_input,
    normalize_result_page_range,
    parse_page_range,
    parse_page_range_set,
)

VALID_RANGES = [
    ("5", [5], "5"),
    ("1-5", [1, 2, 3, 4, 5], "1-5"),
    ("1-3,7,9-10", [1, 2, 3, 7, 9, 10], "1-3,7,9-10"),
    ("3,1-3,2", [1, 2, 3], "1-3"),
    ("all", list(range(1, 11)), "1-10"),
    ("r1", [10], "10"),
    ("r5-r1", [6, 7, 8, 9, 10], "6-10"),
    ("3-r1", list(range(3, 11)), "3-10"),
    (" 1 - 3 , r2 - r1 ", [1, 2, 3, 9, 10], "1-3,9-10"),
    ("8-15", [8, 9, 10], "8-10"),
    ("r20-r8", [1, 2, 3], "1-3"),
    ("1,50", [1], "1"),
    ("5-5", [5], "5"),
]
INVALID_RANGES = [
    "1～5",
    "1~5",
    "-1",
    "-5~-1",
    "0",
    "r0",
    "5-",
    "-5",
    "1,,3",
    ",1",
    "1,",
    "1，3",
    "1:5",
    "5-3",
    "r1-r5",
    "all,1",
    "ALL",
    "R1",
    "r 1",
    "01",
    "+1",
    "１",
    "1--5",
]


@pytest.mark.parametrize(("raw", "page_numbers", "canonical"), VALID_RANGES)
def test_page_selection_contract_across_entrypoints(raw: str, page_numbers: list[int], canonical: str) -> None:
    """同一矩阵覆盖共享解析、Doclib 内容读取、CLI 校验和 Gradio 输入。"""
    _validate_page_range_input(raw)
    assert parse_page_range(raw, 10) == [page - 1 for page in page_numbers]
    assert expand_page_range(raw, 10) == canonical
    assert _normalize_content_page_range(raw, None, {"page_count": 10}) == canonical
    assert expand_page_range(_effective_page_range("demo.pdf", raw), 10) == canonical
    assert parse_page_range_set(canonical) == set(page_numbers)
    assert count_pages_in_range(canonical) == len(page_numbers)
    assert format_page_range(reversed(page_numbers)) == canonical


@pytest.mark.parametrize("raw", INVALID_RANGES)
def test_invalid_and_retired_syntax_is_rejected_everywhere(raw: str) -> None:
    """旧语法与非法端点必须在所有校验入口返回稳定页码错误。"""
    for action in (
        lambda: normalize_page_range_input(raw),
        lambda: parse_page_range(raw, 10),
        lambda: expand_page_range(raw, 10),
        lambda: _normalize_content_page_range(raw, None, {"page_count": 10}),
        lambda: _effective_page_range("demo.pdf", raw),
        lambda: _validate_page_range_input(raw),
    ):
        with pytest.raises(MineruError) as error:
            action()
        assert error.value.code == "page_range_invalid"
        assert error.value.type == "invalid_request_error"


@pytest.mark.parametrize("raw", ["20", "11-20", "r20-r11", "r11", "8-r5", "1,8-r5"])
def test_resolution_rejects_empty_selection_or_reversed_relative_range(raw: str) -> None:
    """依赖总页数的错误在求值时发生；混合有效选项不能掩盖倒序。"""
    normalize_page_range_input(raw)
    with pytest.raises(InvalidRequestError) as error:
        parse_page_range(raw, 10)
    assert error.value.code == "page_range_invalid"


@pytest.mark.parametrize("raw", [None, "", "  \t\n"])
def test_unspecified_range_preserves_entrypoint_default(raw: str | None) -> None:
    """空值统一为未指定，但 Doclib 与直接 Parser 保留各自默认窗口。"""
    assert normalize_page_range_input(raw) == ""
    assert parse_page_range(raw or "", 15) == list(range(15))
    assert _normalize_content_page_range(raw, None, {"page_count": 15}) == "1-10"
    assert default_parse_range(1) == "1"
    assert _normalize_content_page_range("all", None, {"page_count": 15}) == "1-15"


def test_large_ranges_are_clipped_before_expansion() -> None:
    """验证巨大端点只处理实际存在的页面，并以区间计算规范化与统计。"""
    assert parse_page_range("1-999999999999999999", 3) == [0, 1, 2]
    assert expand_page_range("r999999999999999999-r1", 3) == "1-3"
    assert normalize_page_range_input("2,1-999999999999999999") == "1-999999999999999999"
    assert count_pages_in_range("1-999999999999999999,2") == 999999999999999999
    assert parse_page_range_set("") == set()
    assert format_page_range([]) == ""
    assert count_pages_in_range("") == 0
    for raw in ["all", "r1"]:
        with pytest.raises(InvalidRequestError):
            parse_page_range_set(raw)


@pytest.mark.parametrize("raw", ["1~5", "-1", "r1-r5"])
def test_remote_clients_validate_before_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    """同步、异步 API 与 Gradio 客户端在上传或能力发现前拒绝非法表达式。"""
    source = tmp_path / "input.pdf"
    source.write_bytes(b"%PDF-1.7")
    parser = MinerUApiParser(api_url="http://example.test", tier="flash")
    build_source = Mock(side_effect=AssertionError("must not upload"))
    async_build_source = AsyncMock(side_effect=AssertionError("must not upload"))
    monkeypatch.setattr(parser, "_build_source", build_source)
    monkeypatch.setattr(parser, "_async_build_source", async_build_source)
    with pytest.raises(InvalidRequestError):
        parser.parse(source, page_range=raw)
    with pytest.raises(InvalidRequestError):
        asyncio.run(parser.parse_async(source, page_range=raw))
    client = V1ArtifactClient(api_url="http://example.test")
    discover = AsyncMock(side_effect=AssertionError("must not discover"))
    monkeypatch.setattr(client, "discover", discover)
    with pytest.raises(InvalidRequestError):
        asyncio.run(client.parse_file(source, tier="flash", page_range=raw))
    build_source.assert_not_called()
    async_build_source.assert_not_called()
    discover.assert_not_called()


def test_api_rejects_bad_ranges_before_creating_jobs(tmp_path: Path) -> None:
    """直接 HTTP 请求校验共享语法，错误定位到批量请求的具体文件。"""
    app = api_server.create_app(upload_dir=str(tmp_path / "api"), tier="flash", allow_local_source=True)
    with TestClient(app) as client:
        for raw in INVALID_RANGES:
            response = client.post(
                "/v1/parse/jobs",
                json={
                    "tier": "flash",
                    "files": [
                        {"source": {"type": "local", "path": "/unused.pdf"}},
                        {"source": {"type": "local", "path": "/unused.pdf"}, "page_range": raw},
                    ],
                },
            )
            assert response.status_code == 400
            error = response.json()["error"]
            assert error["code"] == "page_range_invalid"
            assert error["param"] == "files.1.page_range"
        schema = client.get("/openapi.json").json()
        assert "r1" in schema["components"]["schemas"]["JobFileEntry"]["properties"]["page_range"]["description"]


def test_cli_rejects_old_syntax_and_kit_preserves_range_error(tmp_path: Path) -> None:
    """两个 CLI 均在执行解析前拒绝旧表达式，主 CLI 返回机器可读错误码。"""
    source = tmp_path / "input.pdf"
    source.write_bytes(b"%PDF-1.7")
    runner = CliRunner()
    for raw in ["1~5", "-1"]:
        result = runner.invoke(cli_app, ["parse", str(source), "--pages", raw, "--json"])
        assert result.exit_code == 1
        assert '"page_range_invalid"' in result.output
        result = runner.invoke(kit_app, ["parse", str(source), "--pages", raw, "-o", str(tmp_path / "out.md")])
        assert result.exit_code == 1
        assert "Invalid page range" in result.output


def test_parsing_rules_validate_without_a_document() -> None:
    """规则保存前校验表达式，允许暂存 rN/all 并拒绝旧语法。"""
    database = SimpleNamespace(execute_insert=AsyncMock(return_value=7))
    service = ConfigService(database)
    for raw, expected in [(" r3 - r1 ", "r3-r1"), ("all", "all"), (" ", None), ("3,1-3", "1-3")]:
        asyncio.run(service.add_rule("pdfs", "parsing_rule", "*.pdf", page_range=raw))
        assert database.execute_insert.call_args.args[1][3] == expected
    database.execute_insert.reset_mock()
    with pytest.raises(InvalidRequestError):
        asyncio.run(service.add_rule("pdfs", "parsing_rule", "*.pdf", page_range="1~5"))
    database.execute_insert.assert_not_called()


def _numbered_pdf(path: Path, page_count: int = 5) -> None:
    """创建每页带不同正文的真实 PDF，供裁剪和原始页号验证。"""
    document = canvas.Canvas(str(path), pagesize=(240, 320))
    for page_no in range(1, page_count + 1):
        document.drawString(30, 250, f"Source page {page_no}")
        document.showPage()
    document.save()


def test_real_pdf_selection_preserves_original_indices_and_artifact_order(tmp_path: Path) -> None:
    """实测非连续选页与末页组合，同时核对解析页号、原始 PDF 和 layout PDF。"""
    source = tmp_path / "numbered.pdf"
    _numbered_pdf(source)
    parser = MinerUParser(tier="flash")
    result = parser.parse(source, page_range="r1,3,1,3")
    assert [page.page_idx for page in result.pages] == [0, 2, 4]
    assert not result.middle_json.is_full_document
    assert "Source page 3" in result.markdown()
    assert "Source page 2" not in result.markdown()
    artifacts = persist_parse_result(result, source, output_root=tmp_path / "review", page_range="r1,3,1,3")
    assert artifacts.page_indices == (0, 2, 4)
    assert artifacts.origin_pdf_path is not None and artifacts.layout_pdf_path is not None
    reader = PdfReader(str(artifacts.origin_pdf_path))
    assert [page.extract_text().strip() for page in reader.pages] == ["Source page 1", "Source page 3", "Source page 5"]
    assert len(PdfReader(str(artifacts.layout_pdf_path)).pages) == 3
    prepared = parser._prepare_input(source, "r1")
    assert prepared.retained_page_indices == [4]
    assert PdfReader(io.BytesIO(prepared.file_bytes)).pages[0].extract_text().strip() == "Source page 5"


def test_async_job_resolves_relative_ranges_and_reports_file_errors(tmp_path: Path) -> None:
    """后台任务完成后返回规范实际范围，依赖总页数的错误保留文件级错误码。"""
    source = tmp_path / "numbered.pdf"
    _numbered_pdf(source)
    store = api_server.FileStore(tmp_path / "files")
    request = api_server.CreateJobRequest.model_validate(
        {
            "tier": "flash",
            "output_formats": ["middle_json"],
            "files": [
                {"source": {"type": "local", "path": str(source)}, "page_range": "1,r2-r1"},
                {"source": {"type": "local", "path": str(source)}, "page_range": "20"},
            ],
        }
    )
    record = api_server.JobStore().create(request, store)
    asyncio.run(api_server._run_job(record, request, store, ocr_mode="auto", image_analysis=True, allow_local_source=True))
    assert record.status == "partial"
    assert record.files[0].page_range == "1,4-5"
    assert record.files[0].status == "completed"
    assert record.files[1].error.code == "page_range_invalid"
    assert record.files[1].error.param == "page_range"


def test_doclib_client_validates_requests_before_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    """Doclib SDK 的解析、查询、导出和规则请求复用校验且不修改调用方模型。"""
    from mineru.doclib.client import DoclibClient
    from mineru.doclib.types import DocContentExportRequest, ParseRequest, ParsingRuleRequest

    client = DoclibClient(base_url="http://example.test")
    transport = Mock()
    monkeypatch.setattr(client, "_request_model", transport)
    actions = (
        lambda: client.ensure_parse(ParseRequest(path="demo.pdf", page_range="1~5")),
        lambda: client.list_parses(page_range="1~5"),
        lambda: client.get_doc_content("a" * 64, tier="flash", page_range="1~5"),
        lambda: client.export_doc_content(
            "a" * 64, DocContentExportRequest(tier="flash", output="out.md", format="markdown", page_range="1~5")
        ),
        lambda: client.add_parsing_rule(ParsingRuleRequest(pattern="*.pdf", page_range="1~5")),
    )
    for action in actions:
        with pytest.raises(InvalidRequestError):
            action()
    transport.assert_not_called()
    request = ParseRequest(path="demo.pdf", page_range=" 3,1-3 ")
    client.ensure_parse(request)
    assert transport.call_args.kwargs["body"].page_range == "1-3"
    assert request.page_range == " 3,1-3 "
    client.close()


def test_kit_preserves_remote_page_range_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """远端返回的页码错误不能在 mineru-kit 中被泛化为 parse_failed。"""
    from mineru.kit.commands import parse as kit_parse
    from mineru.parser.api_client import _V1APIError

    source = tmp_path / "input.pdf"
    source.write_bytes(b"%PDF-1.7")
    parser = SimpleNamespace(parse=Mock(side_effect=_V1APIError("page_range_invalid", "empty selection", "page_range")))
    monkeypatch.setattr(kit_parse, "MinerUApiParser", Mock(return_value=parser))
    fail = Mock(side_effect=RuntimeError("captured error"))
    monkeypatch.setattr(kit_parse, "exit_with_message", fail)
    result = CliRunner().invoke(
        kit_app,
        [
            "parse",
            str(source),
            "--remote-url",
            "http://example.test",
            "--pages",
            "r1",
            "-o",
            str(tmp_path / "out.md"),
        ],
    )
    assert result.exit_code != 0
    assert fail.call_args.args[0] == "page_range_invalid"
    assert fail.call_args.args[2] == "page_range"


def test_doclib_relative_selection_coverage_and_export(tmp_path: Path) -> None:
    """真实 SQLite 缓存的覆盖查询与内容导出必须先按文档总页数求值 rN/all。"""
    import json

    from mineru.doclib.core.db import DatabaseManager
    from mineru.doclib.server import DoclibServer
    from mineru.doclib.services.parse_svc import parse_batch_json_path

    source = tmp_path / "numbered.pdf"
    _numbered_pdf(source)
    result = MinerUParser(tier="flash").parse(source, page_range="1-3,r1")
    sha256 = "a" * 64
    cache = Path(parse_batch_json_path(str(tmp_path), sha256, "flash", "1-3,5", 1000))
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps(result.to_dict()), encoding="utf-8")

    async def verify() -> None:
        """构造实际元数据与缓存记录，验证公开服务读取结果。"""
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        try:
            await db.execute(
                "INSERT INTO docs (sha256, short_id, size_bytes, file_type, page_count, first_seen_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (sha256, "aaaaaaa", source.stat().st_size, "pdf", 5, 1000, 1000),
            )
            await db.execute(
                "INSERT INTO parses (sha256, tier, page_range, status, privacy, done_at, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (sha256, "flash", "1-3,5", "done", "local", 1000, 1000, 1000),
            )
            server = DoclibServer(SimpleNamespace(db=db, data_dir=str(tmp_path)))
            listing = await server.list_parses(doc_ref="aaaaaaa", tier="flash", page_range="r3-r1")
            assert listing.coverage.done_page_range == "3,5"
            assert listing.coverage.missing_page_range == "4"
            text = await server._render_doc_content(sha256, tier="flash", page_range="r1", format="markdown", no_marker=True)
            assert "Source page 5" in text and "Source page 3" not in text
            text = await server._render_doc_content(sha256, tier="flash", page_range="all", format="markdown", no_marker=True)
            assert "Source page 1" in text and "Source page 5" in text
        finally:
            await db.close()

    asyncio.run(verify())


@pytest.mark.parametrize(
    ("raw", "canonical", "pages"),
    [
        ("1~5,8", "1-5,8", {1, 2, 3, 4, 5, 8}),
        ("1~3,5-7", "1-3,5-7", {1, 2, 3, 5, 6, 7}),
        (" 5 , 1 ~ 3 , 2-5 ", "1-5", {1, 2, 3, 4, 5}),
        ("5~5", "5", {5}),
        ("", "", set()),
        ("  ", "", set()),
    ],
)
def test_historical_result_ranges_are_read_without_relaxing_inputs(raw: str, canonical: str, pages: set[int]) -> None:
    """历史半角波浪号仅在已求值结果读取中生效，规范输出仍使用连字符。"""
    assert normalize_result_page_range(raw) == canonical
    assert parse_page_range_set(raw) == pages
    assert count_pages_in_range(raw) == len(pages)
    if "~" in raw:
        with pytest.raises(InvalidRequestError):
            normalize_page_range_input(raw)


@pytest.mark.parametrize("raw", ["1～5", "-1", "-5~-1", "1~-1", "r1", "r5-r1", "all", "5~3", "1~~5", "0~3"])
def test_result_readers_reject_fullwidth_negative_or_unresolved_ranges(raw: str) -> None:
    """历史结果读取不支持全角波浪号、负号倒数页码、符号端点或非法区间。"""
    for reader in (normalize_result_page_range, parse_page_range_set, count_pages_in_range):
        with pytest.raises(InvalidRequestError) as error:
            reader(raw)
        assert error.value.code == "page_range_invalid"
