"""检查 JSON 标签读取 Structured Content 及其与转换生命周期的一致性。"""

from __future__ import annotations

import asyncio
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import gradio as gr
import pytest

from mineru.kit.gradio.app import build_gradio_app
from mineru.kit.gradio.artifacts import render_download
from mineru.kit.gradio.client import V1ServerCapabilities
from mineru.parser.base import ParseResult
from test_kit_gradio import _middle_json, _pdf_bytes


@pytest.mark.parametrize("with_image", [False, True])
def test_json_view_matches_saved_and_downloaded_structured_content(tmp_path: Path, with_image: bool) -> None:
    """预览逐字复用落盘 JSON，与下载包相同，并只在完成时发送完整内容。"""
    source = tmp_path / "source.pdf"
    source.write_bytes(_pdf_bytes())
    client = SimpleNamespace(parse_file=AsyncMock(return_value=ParseResult(middle_json=_middle_json(with_image=with_image))))
    demo = build_gradio_app(
        client,
        V1ServerCapabilities("http://unused.test", ("flash",), ("zip",), ("file_id",)),
        output_root=tmp_path / "output",
        enable_example=False,
    )
    conversion = next(fn for fn in demo.fns.values() if fn.name == "convert_handler")
    editor = conversion.outputs[-1]
    assert isinstance(editor, gr.Code) and editor.language == "json" and editor.interactive is False
    assert conversion.outputs[2].__class__.__name__ == "File"

    async def collect() -> list[tuple]:
        """完整消费转换流，检查 JSON 不随状态动画重复下发。"""
        return [update async for update in conversion.fn(str(source), 0, "")]

    updates = asyncio.run(collect())
    state = updates[-1][6]
    text = updates[-1][-1]
    assert text == Path(state["structured_content_path"]).read_text(encoding="utf-8")
    payload = json.loads(text)
    assert "pages" in payload and "schema_version" not in payload
    assert all(update[-1] in ("", {"__type__": "update"}) for update in updates[:-1])
    with zipfile.ZipFile(render_download(state, "json", allowed_root=tmp_path / "output")) as archive:
        assert archive.read(f"{state['stem']}.json").decode("utf-8") == text
    if with_image:
        assert "images/" in text and "data:image/" not in text


def test_json_view_clears_on_file_change_clear_and_failed_conversion(tmp_path: Path) -> None:
    """换文件、清除、失败均撤销旧 JSON，重试后展示新结果。"""
    source = tmp_path / "source.pdf"
    source.write_bytes(_pdf_bytes())
    client = SimpleNamespace(parse_file=AsyncMock(return_value=ParseResult(middle_json=_middle_json(with_image=False))))
    demo = build_gradio_app(
        client,
        V1ServerCapabilities("http://unused.test", ("flash",), ("zip",), ("file_id",)),
        output_root=tmp_path / "output",
        enable_example=False,
    )
    conversion = next(fn for fn in demo.fns.values() if fn.name == "convert_handler")
    editor = conversion.outputs[-1]
    for name, arguments in (("update_file_preview", (str(source),)), ("reset_ui", ())):
        callback = next(fn for fn in demo.fns.values() if fn.name == name)
        assert callback.outputs[-1] is editor
        assert callback.fn(*arguments)[-1] == ""

    async def scenario() -> None:
        """按成功、失败、重试顺序验证 JSON 的原子更新和清理。"""
        first = [update async for update in conversion.fn(str(source), 0, "")]
        assert "hello-0" in first[-1][-1]
        client.parse_file.side_effect = RuntimeError("test failure")
        failure = [update async for update in conversion.fn(str(source), 0, "")]
        assert failure[0][-1] == failure[-1][-1] == ""
        assert "Failed:" in failure[-1][0]
        changed = _middle_json(with_image=False)
        changed.pages[0].blocks[0].content[0].content = "更新后的结果"
        client.parse_file.side_effect = None
        client.parse_file.return_value = ParseResult(middle_json=changed)
        retry = [update async for update in conversion.fn(str(source), 0, "")]
        assert retry[0][-1] == ""
        assert "更新后的结果" in retry[-1][-1] and "hello-0" not in retry[-1][-1]

    asyncio.run(scenario())
