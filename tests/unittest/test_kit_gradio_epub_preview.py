"""验证 Gradio EPUB viewer 的静态资源路由和本地发行文件。"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import gradio as gr
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mineru.kit.gradio.epub_preview import register_epub_preview_resources


def test_epub_preview_static_routes_are_local_and_complete(tmp_path: Path) -> None:
    """viewer、epub.js 和 JSZip 均通过当前 Gradio 静态文件路由提供。"""
    entry = register_epub_preview_resources()
    with gr.Blocks(analytics_enabled=False) as demo:
        gr.HTML("")
    app = gr.mount_gradio_app(FastAPI(), demo, path="/", allowed_paths=[str(tmp_path)])
    with TestClient(app) as client:
        for path, mime in (
            (entry, "text/html"),
            (entry.with_name("epub.min.js"), "text/javascript"),
            (entry.with_name("jszip.min.js"), "text/javascript"),
        ):
            response = client.get(f"/gradio_api/file={quote(str(path), safe='/')}")
            assert response.status_code == 200
            assert response.headers["content-type"].split(";")[0] == mime
            assert response.headers["content-disposition"].startswith("inline")


def test_epub_preview_viewer_contains_script_and_remote_resource_guards() -> None:
    """viewer 默认关闭 EPUB 脚本、弹窗和外部资源访问。"""
    entry = register_epub_preview_resources()
    document = entry.read_text(encoding="utf-8")
    viewer = entry.with_name("viewer.js").read_text(encoding="utf-8")
    assert 'allowScriptedContent: false' in viewer
    assert 'allowPopups: false' in viewer
    assert 'connect-src \'self\'' in document
    assert "[a-z][a-z\\d+.-]*:" in viewer
    assert "node.removeAttribute" in viewer


def test_epub_preview_viewer_uses_continuous_spine_navigation_and_internal_links() -> None:
    """viewer 使用连续管理器、真实 spine 编号和统一的包内链接跳转。"""
    entry = register_epub_preview_resources()
    document = entry.read_text(encoding="utf-8")
    viewer = entry.with_name("viewer.js").read_text(encoding="utf-8")
    assert 'id="page"' in document and 'id="pages"' in document
    assert "#viewer .epub-container" in document and "min-height: 100%" in document
    assert "margin: 0 !important" in document
    assert 'manager: "continuous"' in viewer
    assert 'flow: "scrolled-continuous"' in viewer
    assert 'offset: 1000000000' not in viewer
    assert 'offset: 800' in viewer
    assert 'viewer.querySelector(".epub-container")' in viewer
    assert "pageMetrics" not in viewer and "scrollContainer.scrollHeight" not in viewer
    assert "book.spine?.length" in viewer
    assert "SPINE_SWITCH_RATIO = 0.5" in viewer
    assert "manager.settings.offset" in viewer
    assert "updateActiveSpine" in viewer
    assert "resolveInternalTarget" in viewer
    assert "contents.document?.addEventListener" in viewer
    assert "event.stopImmediatePropagation" in viewer
    assert "book.spine.hooks.content.clear()" in viewer
    assert 'localName === "image"' in viewer
    assert 'stylesheetResource' in viewer
    assert "contents.sectionIndex" in viewer
    assert "book.spine.get(contents.index)" not in viewer
    assert "EPUB internal target not found" in viewer
    assert "if (!section)" in viewer
    assert "MutationObserver" in viewer
    assert "allow-scripts" in viewer
    assert "script-src 'none'" in viewer
    assert viewer.index("if (!section)") < viewer.index("fail(error)")
    assert viewer.index('await book.open(payload, "binary")') < viewer.index("appendContents(book.navigation?.toc || [])")
    assert viewer.index("await book.loaded.navigation") < viewer.index("appendContents(book.navigation?.toc || [])")
