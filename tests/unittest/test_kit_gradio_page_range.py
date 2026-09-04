"""Gradio 双滑块的页数预检、请求约束和前端状态回归。"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pypdfium2 as pdfium
import pytest
from pypdf import PdfWriter
from typer.testing import CliRunner

from mineru.errors import InvalidRequestError
from mineru.kit.gradio import app as gradio_app
from mineru.kit.gradio import page_range as ranges
from mineru.kit.gradio.client import V1ServerCapabilities
from mineru.kit.main import app
from mineru.model.flash.pdf.pdfium import _pdfium_lock


def _pdf(tmp_path: Path, page_count: int = 100, *, encrypted: bool = False) -> Path:
    """生成测试用 PDF，可选密码保护及空文档。"""
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=200, height=300)
    if encrypted:
        writer.encrypt("required-password")
    path = tmp_path / "report.PDF"
    writer.write(path)
    return path


@pytest.mark.parametrize(("args", "expected"), [([], None), (["--max-pages", "20"], 20)])
def test_command_forwards_max_pages(monkeypatch: pytest.MonkeyPatch, args: list[str], expected: int | None) -> None:
    """验证省略或显式配置上限时的启动参数透传。"""
    launch = Mock()
    monkeypatch.setattr(gradio_app, "launch_gradio", launch)
    result = CliRunner().invoke(app, ["gradio", *args])
    assert result.exit_code == 0, result.output
    assert launch.call_args.kwargs["max_pages"] == expected
    assert "--max-pages" in CliRunner().invoke(app, ["gradio", "--help"]).output


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "invalid"])
def test_command_rejects_invalid_max_pages(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """验证非正整数配置在启动任何服务前报错。"""
    launch = Mock()
    monkeypatch.setattr(gradio_app, "launch_gradio", launch)
    result = CliRunner().invoke(app, ["gradio", "--max-pages", value])
    assert result.exit_code != 0
    launch.assert_not_called()


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "20"])
def test_programmatic_max_pages_validation(value: object) -> None:
    """验证直接构建或启动界面也不能绕过整数配置检查。"""
    with pytest.raises(ValueError, match="positive integer"):
        ranges.validate_max_pages(value)  # type: ignore[arg-type]


def test_launch_forwards_limit_and_versioned_assets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证启动链路向界面传入上限，并使用构建阶段确定的版本兼容资源参数。"""
    client = SimpleNamespace(discover=AsyncMock(return_value=Mock()))
    demo = SimpleNamespace(launch=Mock(), _mineru_kit_launch_kwargs={"css": "test-css", "js": "test-js"})
    build = Mock(return_value=demo)
    monkeypatch.setattr(gradio_app, "V1ArtifactClient", Mock(return_value=client))
    monkeypatch.setattr(gradio_app, "build_gradio_app", build)
    gradio_app.launch_gradio(
        api_url="http://127.0.0.1:1",
        api_key=None,
        server_name="127.0.0.1",
        server_port=None,
        output_dir=str(tmp_path),
        enable_example=False,
        enable_api=True,
        latex_delimiters_type="all",
        api_server_tier="standard",
        api_server_no_flash=False,
        api_server_concurrency=1,
        api_server_language="ch",
        api_server_disable_image_analysis=False,
        api_server_preload_models=False,
        max_pages=20,
    )
    assert build.call_args.kwargs["max_pages"] == 20
    assert demo.launch.call_args.kwargs["css"] == "test-css"
    assert demo.launch.call_args.kwargs["js"] == "test-js"


@pytest.mark.parametrize("count", [1, 12, 20, 100])
def test_pdf_metadata_and_default_range(tmp_path: Path, count: int) -> None:
    """验证真实页数、大小写后缀和不足上限时的默认选区。"""
    source = _pdf(tmp_path, count)
    assert ranges.pdf_page_metadata(str(source)) == {"path": str(source), "page_count": count, "error": ""}
    expected = "1" if count == 1 else f"1-{min(count, 20)}"
    assert ranges.effective_page_range(source, "", tier="standard", max_pages=20) == expected
    assert ranges.effective_page_range(source, "", tier="standard") == ""
    assert ranges.effective_page_range(source, "all", tier="standard") == "all"


@pytest.mark.parametrize(
    ("raw", "expected"), [("20-35", "20-35"), (" 1, 1-20 ", "1-20"), ("r20-r1", "r20-r1"), ("95-200", "95-200")]
)
def test_existing_range_syntax_and_unique_page_count(tmp_path: Path, raw: str, expected: str) -> None:
    """保留既有语法，按求值后的去重页数限制，不按字符串跨度计数。"""
    source = _pdf(tmp_path)
    assert ranges.effective_page_range(source, raw, tier="standard", max_pages=20) == expected


@pytest.mark.parametrize("raw", ["all", "1-21", "r30-r1", "1-10,90-100", "999", "20-1", "~", "0"])
def test_invalid_or_oversized_range_is_rejected_before_submission(tmp_path: Path, raw: str) -> None:
    """验证直接调用 Gradio 转换事件也无法提交超限或非法范围。"""
    source = _pdf(tmp_path)
    client = SimpleNamespace(parse_file=AsyncMock())
    capabilities = V1ServerCapabilities("http://127.0.0.1:1", ("flash", "standard"), ("zip",), ("file_id",))
    demo = gradio_app.build_gradio_app(client, capabilities, output_root=tmp_path, max_pages=20, enable_example=False)
    handler = next(fn.fn for fn in demo.fns.values() if fn.name == "convert_handler")

    async def collect() -> list[tuple[object, ...]]:
        """收集错误事件并确认没有启动实际解析。"""
        return [update async for update in handler(str(source), 1, raw)]

    updates = asyncio.run(collect())
    assert "page_range_invalid" in str(updates[-1][0])
    assert updates[-1][8] is None
    client.parse_file.assert_not_called()


@pytest.mark.parametrize(
    ("path", "tier"), [("missing.pdf", "flash"), ("report.docx", "standard"), ("image.png", "advanced"), (None, "basic")]
)
def test_whole_document_modes_never_read_pdf_metadata(monkeypatch: pytest.MonkeyPatch, path: str | None, tier: str) -> None:
    """Flash、非 PDF 和空输入直接清空范围，不因残留的坏页码而失败。"""
    read_count = Mock(side_effect=AssertionError("unexpected PDF access"))
    monkeypatch.setattr(ranges, "read_pdf_page_count", read_count)
    assert ranges.effective_page_range(path, "bad range", tier=tier, max_pages=1) == ""  # type: ignore[arg-type]
    read_count.assert_not_called()


@pytest.mark.parametrize("kind", ["broken", "encrypted", "empty", "missing"])
def test_unreadable_pdf_is_not_treated_as_all_pages(tmp_path: Path, kind: str) -> None:
    """损坏、加密、空页和丢失文件均反馈错误，不能降级为全部解析。"""
    source = tmp_path / "report.PDF"
    if kind == "broken":
        source.write_bytes(b"not a PDF")
    elif kind != "missing":
        source = _pdf(tmp_path, 0 if kind == "empty" else 1, encrypted=kind == "encrypted")
    metadata = ranges.pdf_page_metadata(str(source))
    assert metadata["path"] == str(source)
    assert metadata["page_count"] == 0
    assert "无法读取 PDF 页数" in metadata["error"]
    with pytest.raises(InvalidRequestError, match="无法读取 PDF 页数"):
        ranges.effective_page_range(source, "", tier="standard", max_pages=20)


@pytest.mark.parametrize("fail", [False, True])
def test_pdfium_access_is_locked_and_document_is_closed(monkeypatch: pytest.MonkeyPatch, fail: bool) -> None:
    """验证页数读取和异常路径都持有共享锁并且恰好关闭文档一次。"""
    closed: list[bool] = []

    class Document:
        """检查文档生命周期内的锁状态。"""

        def __init__(self, _path: str) -> None:
            """确认打开时已持有共享锁。"""
            assert _pdfium_lock._is_owned()

        def __len__(self) -> int:
            """模拟正常页数或读取页数失败。"""
            assert _pdfium_lock._is_owned()
            if fail:
                raise RuntimeError("count failed")
            return 10

        def close(self) -> None:
            """确认清理时仍然持有共享锁。"""
            closed.append(_pdfium_lock._is_owned())

    monkeypatch.setattr(pdfium, "PdfDocument", Document)
    if fail:
        with pytest.raises(InvalidRequestError):
            ranges.read_pdf_page_count("report.pdf")
    else:
        assert ranges.read_pdf_page_count("report.pdf") == 10
    assert closed == [True]


def test_metadata_does_not_read_non_pdf(monkeypatch: pytest.MonkeyPatch) -> None:
    """非 PDF 和清除文件不触发 PDFium 读取。"""
    count = Mock()
    monkeypatch.setattr(ranges, "read_pdf_page_count", count)
    assert ranges.pdf_page_metadata(None) == {"path": "", "page_count": 0, "error": ""}
    assert ranges.pdf_page_metadata("report.docx")["page_count"] == 0
    count.assert_not_called()


def test_native_range_components_and_frontend_only_events(tmp_path: Path) -> None:
    """双滑块联动与档位切换不请求 Python，公开转换输入在页码之后增加 OCR 开关。"""
    capabilities = V1ServerCapabilities("http://127.0.0.1:1", ("flash", "standard"), ("zip",), ("file_id",))
    demo = gradio_app.build_gradio_app(Mock(), capabilities, output_root=tmp_path, max_pages=20, enable_example=False)
    sliders = [block for block in demo.blocks.values() if block.__class__.__name__ == "Slider"]
    assert len(sliders) == 3
    assert sliders[1].elem_classes == ["mineru-page-handle-a"]
    assert sliders[2].elem_classes == ["mineru-page-handle-b"]
    for slider in sliders[1:]:
        assert slider.step == 1 and slider.precision == 0
        events = [event for event in demo.config["dependencies"] if (slider._id, "input") in event["targets"]]
        assert len(events) == 1 and events[0]["backend_fn"] is False and events[0]["queue"] is False
        assert events[0]["trigger_mode"] == "always_last"
        assert len(events[0]["targets"]) == 5
        assert events[0]["outputs"][:2] == [sliders[1]._id, sliders[2]._id]
    handler = next(fn for fn in demo.fns.values() if fn.name == "convert_handler")
    assert len(handler.inputs) == 4
    assert handler.inputs[2].__class__.__name__ == "Textbox" and handler.inputs[2].visible is False
    assert handler.inputs[3].__class__.__name__ == "Checkbox" and handler.inputs[3].value is False
    metadata_handler = next(fn for fn in demo.fns.values() if fn.name == "read_page_metadata")
    assert metadata_handler.outputs[0].__class__.__name__ == "Textbox"
    metadata = json.loads(metadata_handler.fn(str(_pdf(tmp_path, 12))))
    assert metadata["page_count"] == 12 and metadata["error"] == ""
    load_events = [event for event in demo.config["dependencies"] if any(target[1] == "load" for target in event["targets"])]
    assert any(event["js"] == demo._mineru_kit_js and not event["backend_fn"] for event in load_events)
    import gradio as gr

    if gradio_app._gradio_major_version(gr) >= 6:
        assert set(demo._mineru_kit_launch_kwargs) == {"css", "js"}
    else:
        assert demo._mineru_kit_launch_kwargs == {}
        assert demo.css and demo.js


def test_frontend_page_range_state_machine() -> None:
    """通过 Node 执行实际前端脚本，覆盖联动、竞态、键盘等共享状态转换。"""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for frontend state tests")
    result = subprocess.run([node, str(Path(__file__).with_suffix(".cjs"))], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
