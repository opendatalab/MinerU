"""验证 Gradio EPUB viewer 的静态资源路由和本地发行文件。"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import gradio as gr
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mineru.kit.gradio.epub_preview import register_epub_preview_resources
from mineru.kit.gradio.i18n import MESSAGES


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
            (entry.with_name("viewer.js"), "text/javascript"),
        ):
            response = client.get(f"/gradio_api/file={quote(str(path), safe='/')}")
            assert response.status_code == 200
            assert response.headers["content-type"].split(";")[0] == mime
            assert response.headers["content-disposition"].startswith("inline")


def test_epub_preview_viewer_sanitizes_before_enabling_chapter_events() -> None:
    """章节必须在进入 iframe 前完成清理，Safari 事件兼容也只对已清理 srcdoc 开启。"""
    entry = register_epub_preview_resources()
    document = entry.read_text(encoding="utf-8")
    viewer = entry.with_name("viewer.js").read_text(encoding="utf-8")
    assert 'allowScriptedContent: false' in viewer
    assert 'allowPopups: false' in viewer
    assert 'connect-src \'self\'' in document
    assert "[a-z][a-z\\d+.-]*:" in viewer
    assert "node.removeAttribute" in viewer
    assert 'const SANITIZED_MARKER = "data-mineru-sanitized"' in viewer
    assert "sanitizeDocument(document)" in viewer
    assert "restoreSpineMetadata(doc, section)" in viewer
    assert "installSafeSpineContentHook(book)" in viewer
    assert "SANITIZED_TOKEN = Array.from(crypto.getRandomValues" in viewer
    assert 'srcdoc.includes(`${SANITIZED_MARKER}="${SANITIZED_TOKEN}"`)' in viewer
    assert 'getAttribute(SANITIZED_MARKER) !== SANITIZED_TOKEN' in viewer
    assert 'srcdoc.includes("script-src \'none\'")' in viewer
    assert "sanitizeDocument(contents.document)" not in viewer
    assert viewer.index("installSafeSpineContentHook(book)") < viewer.index("await displayTarget(book.spine.first()?.href")


def test_epub_preview_viewer_accepts_font_obfuscation_and_stream_limits_payload() -> None:
    """encryption.xml 只拒绝未知算法，并在下载过程中限制 EPUB 大小。"""
    entry = register_epub_preview_resources()
    viewer = entry.with_name("viewer.js").read_text(encoding="utf-8")
    assert "MAX_EPUB_BYTES = 128 * 1024 * 1024" in viewer
    assert 'response.headers.get("content-length")' in viewer
    assert "response.body?.cancel?.()" in viewer
    assert "response.body?.getReader?.()" in viewer
    assert "await reader.cancel()" in viewer
    assert "JSZip.loadAsync(payload)" not in viewer
    assert 'currentBook.archive?.getText?.("/META-INF/encryption.xml")' in viewer
    assert "http://www.idpf.org/2008/embedding" in viewer
    assert "http://ns.adobe.com/pdf/enc#RC" in viewer
    assert "archive.file(\"META-INF/encryption.xml\")" not in viewer
    assert 'await book.open(payload, "binary")' in viewer
    assert "await ensureSupportedEncryption(book)" in viewer


def test_epub_preview_viewer_uses_serialized_section_navigation_and_internal_links() -> None:
    """viewer 使用连续管理器、章节语义和串行导航，快速操作不会被旧 display 覆盖。"""
    entry = register_epub_preview_resources()
    document = entry.read_text(encoding="utf-8")
    viewer = entry.with_name("viewer.js").read_text(encoding="utf-8")
    assert 'id="section"' in document and 'id="sections"' in document
    assert 'id="page"' not in document and 'id="pages"' not in document
    assert MESSAGES["epub_previous"] == ("Previous section", "上一节")
    assert MESSAGES["epub_next"] == ("Next section", "下一节")
    assert MESSAGES["epub_spine"] == ("Section", "章节")
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
    assert 'localName === "image"' in viewer
    assert "stylesheetResource" in viewer
    assert "contents.sectionIndex" in viewer
    assert "book.spine.get(contents.index)" not in viewer
    assert "EPUB internal target not found" in viewer
    assert "navigationSequence" in viewer and "navigationQueue" in viewer
    assert "navigationId !== navigationSequence" in viewer
    assert "pendingSpineIndex" in viewer
    assert viewer.count("scrollContainer.scrollTo(") == 1
    assert "return true;" in viewer
    assert 'if (!section) {\n            console.warn("EPUB internal target not found"' in viewer
    assert viewer.index('await book.open(payload, "binary")') < viewer.index("appendContents(book.navigation?.toc || [])")
    assert viewer.index("await book.loaded.navigation") < viewer.index("appendContents(book.navigation?.toc || [])")
