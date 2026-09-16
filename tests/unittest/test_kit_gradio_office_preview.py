"""Office 上传即预览、旧版提示文案和转换期间保留 iframe 的回归。"""

from __future__ import annotations

import asyncio
import html
import re
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import parse_qs, quote, urlsplit

import pytest

from mineru.filetypes import OFFICE_EXTENSIONS
from mineru.kit.gradio import app as app_module
from mineru.kit.gradio.client import V1ServerCapabilities
from mineru.kit.gradio.i18n import MESSAGES
from mineru.parser.base import ParseResult
from test_kit_gradio import _middle_json


@pytest.mark.parametrize("suffix", sorted(OFFICE_EXTENSIONS | {ext.upper() for ext in OFFICE_EXTENSIONS}))
def test_office_upload_builds_preview_without_parsing(tmp_path: Path, suffix: str) -> None:
    """所有现有 Office 格式上传后即可预览，请求上下文不增加事件公开输入。"""
    client = SimpleNamespace(parse_file=AsyncMock())
    demo = app_module.build_gradio_app(
        client,
        V1ServerCapabilities("http://localhost:1", ("flash",), ("zip",), ("file_id",)),
        output_root=tmp_path,
        enable_example=False,
    )
    callback = next(fn for fn in demo.fns.values() if fn.name == "update_file_preview")
    assert len(callback.inputs) == 1
    request = SimpleNamespace(headers={"host": "example.test:7860"})
    updates = callback.fn(str(tmp_path / f"document.{suffix}"), request)
    assert updates[0]["visible"] is False and updates[1]["visible"] is False
    assert updates[2]["visible"] is True and updates[3]["visible"] is False
    assert 'class="office-preview-notice"' in updates[2]["value"]
    assert 'class="office-preview-frame"' in updates[2]["value"]
    assert "Office 在线预览" in updates[2]["value"]
    assert "Office 文件将在转换后提供结果" not in updates[2]["value"]
    client.parse_file.assert_not_called()


@pytest.mark.parametrize(
    ("headers", "base"),
    [
        ({}, "http://localhost:7860"),
        ({"host": "example.test:8080"}, "http://example.test:8080"),
        (
            {"host": "internal:7860", "x-forwarded-host": "documents.example.test", "x-forwarded-proto": "https"},
            "https://documents.example.test",
        ),
    ],
)
@pytest.mark.parametrize("name", ["short.docx", "abcdefghijklmnopqrst中文 空格.xlsx", "a<>&\"'.pptx"])
def test_office_short_address_keeps_full_encoded_iframe_url(headers: dict[str, str], base: str, name: str) -> None:
    """短地址只截取主名末尾十二字符，完整预览 URL 仍正确编码并防止 HTML 注入。"""
    source = Path("/uploaded/private") / name
    markup = app_module._build_office_preview_html(source, SimpleNamespace(headers=headers))
    iframe_url = html.unescape(re.search(r'<iframe[^>]+src="([^"]+)"', markup)[1])
    public_url = parse_qs(urlsplit(iframe_url).query)["src"][0]
    assert public_url == f"{base}/gradio_api/file={quote(source.as_posix(), safe='/:')}"
    displayed = html.unescape(re.search(r'<div class="office-preview-source-link">(.*?)</div>', markup)[1])
    assert f"{base}/....{source.stem[-12:]}{source.suffix}" in displayed
    assert "/uploaded/private" not in displayed
    assert "文件链接" in displayed and "File url" in displayed
    assert "<a " not in markup and "<>&" not in markup
    assert markup.count("<iframe") == 1
    if not headers:
        assert markup == app_module._build_office_preview_html(source)


def test_office_windows_file_url_uses_forward_slashes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Windows 上传预览使用正斜杠 URL，保留盘符、中文及空格的正确编码。"""
    source = PureWindowsPath("C:/Users/测试用户/上传 文件/报告.docx")
    monkeypatch.setattr(app_module, "Path", PureWindowsPath)
    markup = app_module._build_office_preview_html(str(source))
    iframe_url = html.unescape(re.search(r'<iframe[^>]+src="([^"]+)"', markup)[1])
    public_url = parse_qs(urlsplit(iframe_url).query)["src"][0]
    assert public_url == "http://localhost:7860/gradio_api/file=" + quote(source.as_posix(), safe="/:")
    assert "%5c" not in public_url.lower()


def test_office_copy_matches_345_release() -> None:
    """固定中英文文案属于本次恢复的用户可见契约。"""
    assert MESSAGES["office_preview_title"] == ("Office online preview", "Office 在线预览")
    assert MESSAGES["office_notice"] == (
        "This preview requires the current file to be reachable by Microsoft Office Online. "
        "Conversion does not depend on this preview.",
        "该预览需要当前文件可被 Microsoft 在线预览服务访问，转换不依赖该预览。",
    )
    assert MESSAGES["office_preview_source_link"] == ("File url", "文件链接")
    assert MESSAGES["ignore_once"] == ("Dismiss", "忽略")
    assert MESSAGES["ignore_forever"] == ("Always dismiss", "不再提示")


@pytest.mark.parametrize("outcome", ["success", "parse_failure", "output_failure"])
def test_office_conversion_never_replaces_uploaded_preview(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    """成功、解析失败及整理失败都不替换 iframe，避免浏览位置或已忽略状态丢失。"""
    source = tmp_path / "document.docx"
    source.write_bytes(b"Office source fixture")
    result = ParseResult(middle_json=_middle_json(with_image=False, file_suffix="docx"))
    parse = AsyncMock(return_value=result)
    if outcome == "parse_failure":
        parse.side_effect = RuntimeError("office parse failed")

    def fail_output(*_args: object, **_kwargs: object) -> None:
        """模拟解析完成后的产物整理失败。"""
        raise OSError("output unavailable")

    if outcome == "output_failure":
        monkeypatch.setattr(app_module, "persist_parse_result", fail_output)
    demo = app_module.build_gradio_app(
        SimpleNamespace(parse_file=parse),
        V1ServerCapabilities("http://localhost:1", ("flash",), ("zip",), ("file_id",)),
        output_root=tmp_path / "output",
        enable_example=False,
    )
    handler = next(fn.fn for fn in demo.fns.values() if fn.name == "convert_handler")

    async def collect() -> list[tuple[object, ...]]:
        """消费完整生成器并检查每一次前端预览更新。"""
        return [update async for update in handler(str(source), 0, "")]

    updates = asyncio.run(collect())
    assert updates and all(update[2:6] == ({"__type__": "update"},) * 4 for update in updates)
    assert bool(updates[-1][6]) is (outcome == "success")
    assert all(item["interactive"] is (outcome == "success") for item in updates[-1][8:15])
    if outcome == "success":
        assert "hello-0" in updates[-1][1]
    else:
        assert "Failed:" in updates[-1][0]
    preview = next(fn.fn for fn in demo.fns.values() if fn.name == "update_file_preview")
    assert "next.pptx" in preview("next.pptx")[2]["value"]
    assert preview("next.pdf")[2]["visible"] is False
    assert preview("next.png")[1]["value"] == "next.png"
    assert preview("next.epub")[2]["visible"] is False
    assert preview(None)[2]["value"] == ""
