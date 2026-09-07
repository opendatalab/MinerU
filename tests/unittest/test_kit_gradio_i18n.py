"""双语初始化、动态文案与转换期间保留预览的行为回归。"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from mineru.kit.gradio.app import build_gradio_app
from mineru.kit.gradio.client import V1ServerCapabilities
from mineru.kit.gradio.i18n import MESSAGES, localized_message, localized_text, translations
from mineru.parser.base import ParseResult
from test_kit_gradio import _middle_json


def test_i18n_launch_and_component_contract(tmp_path: Path) -> None:
    """词典必须进入实际启动参数；OCR 说明在两个语言下都只有指定的一句话。"""
    demo = build_gradio_app(
        SimpleNamespace(),
        V1ServerCapabilities("http://localhost:1", ("flash",), ("zip",), ("file_id",)),
        output_root=tmp_path,
        enable_example=False,
    )
    dictionary = demo._mineru_kit_launch_kwargs["i18n"].translations_dict
    assert dictionary == translations()
    assert dictionary["zh-TW"] == dictionary["zh-CN"]
    # 只注册项目命名空间，原生组件的日语等词典继续由 Gradio 管理。
    assert set(dictionary) == {"en", "zh", "zh-CN", "zh-TW"}
    assert all(key.startswith("mineru.") for locale in dictionary.values() for key in locale)
    checkbox = next(block for block in demo.blocks.values() if block.__class__.__name__ == "Checkbox")
    assert checkbox.info.key == "mineru.force_ocr_info"
    assert dictionary["zh-CN"][checkbox.info.key] == "忽略 PDF 文本层并进行 OCR"
    assert dictionary["en"][checkbox.info.key] == "Ignore the PDF text layer and perform OCR"
    assert "__MINERU_I18N__" not in demo._mineru_kit_js


def test_bilingual_errors_and_html_keep_untrusted_details_as_text() -> None:
    """本地错误附带英文翻译，未知服务端消息和恶意文件名仍是转义后的原文。"""
    value = localized_text("failed", error='<img src=x onerror="alert(1)"> $& 中文.pdf')
    assert "<img" not in value and "&lt;img" in value and "$&amp; 中文.pdf" in value
    for message, expected in [
        ("Failed: input file does not exist", "输入文件不存在"),
        ("Failed: unsupported file type '.exe'", "不支持的文件类型"),
        ("Failed: page_range_invalid: 单次最多解析 20 页，当前选择了 25 页。", "At most 20 pages"),
        ("Failed: <script>unknown detail</script>", "&lt;script&gt;unknown detail&lt;/script&gt;"),
    ]:
        rendered = localized_message(message)
        assert expected in rendered and "<script>" not in rendered


def test_frontend_language_policy_and_dynamic_events() -> None:
    """同一前端脚本在不同浏览器首选语言下覆盖页码、下载与错误提示。"""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for frontend state tests")
    result = subprocess.run(
        [node, str(Path(__file__).with_suffix(".cjs"))],
        input=json.dumps(MESSAGES),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("outcome", ["success", "parse_failure", "output_failure"])
def test_conversion_preserves_all_preview_components_until_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    """暂停真实生成器，确认首帧、处理状态和失败都不改变已有四种预览。"""
    from mineru.kit.gradio import app as app_module

    source = tmp_path / "document.csv"
    source.write_text("a,b\n1,2\n", encoding="utf-8")

    async def scenario() -> None:
        """用事件控制解析完成，模拟前端按 Gradio skip 语义保留已有界面。"""
        started = asyncio.Event()
        finish = asyncio.Event()

        async def parse_file(*_args: object, **kwargs: object) -> ParseResult:
            """先通知处理阶段，待测试放行后返回结果或抛出解析异常。"""
            kwargs["status_callback"]("Processing on server...")
            started.set()
            await finish.wait()
            if outcome == "parse_failure":
                raise RuntimeError("test parse failure")
            return ParseResult(middle_json=_middle_json(with_image=False, file_suffix="csv"))

        def fail_output(*_args: object, **_kwargs: object) -> None:
            """模拟解析已成功但本地输出整理失败。"""
            raise OSError("test output failure")

        if outcome == "output_failure":
            monkeypatch.setattr(app_module, "persist_parse_result", fail_output)
        demo = build_gradio_app(
            SimpleNamespace(parse_file=parse_file),
            V1ServerCapabilities("http://localhost:1", ("flash",), ("zip",), ("file_id",)),
            output_root=tmp_path / "output",
            enable_example=False,
        )
        handler = next(fn.fn for fn in demo.fns.values() if fn.name == "convert_handler")
        updates: list[tuple[object, ...]] = []

        async def collect() -> None:
            """独立消费生成器，避免测试本身阻挡状态通知和任务启动。"""
            async for update in handler(str(source), 0, ""):
                updates.append(update)

        consumer = asyncio.create_task(collect())
        try:
            await asyncio.wait_for(started.wait(), timeout=3)
            assert updates
            assert all(update[2:6] == ({"__type__": "update"},) * 4 for update in updates)
            finish.set()
            await asyncio.wait_for(consumer, timeout=10)
        finally:
            if not consumer.done():
                consumer.cancel()
            await asyncio.gather(consumer, return_exceptions=True)
        assert all(update[2:6] == ({"__type__": "update"},) * 4 for update in updates[:-1])
        if outcome == "success":
            assert updates[-1][5]["visible"] is True
            assert "结果已生成" in updates[-1][5]["value"]
            assert updates[-1][6] is not None
        else:
            assert updates[-1][2:6] == ({"__type__": "update"},) * 4
            assert updates[-1][6] is None
            assert "Failed:" in updates[-1][0]
        # 真正的清除仍会重置预览，文件切换仍按新文件类型展示。
        reset = next(fn.fn for fn in demo.fns.values() if fn.name == "reset_ui")
        preview = next(fn.fn for fn in demo.fns.values() if fn.name == "update_file_preview")
        assert reset()[2]["value"] is None and reset()[5]["visible"] is True
        assert preview("new.pdf")[0]["value"] == "new.pdf"

    asyncio.run(scenario())
