"""MHTML 归档资源、预览回执与原生解析的交叉回归。"""

from __future__ import annotations

import asyncio
import base64
import json
import re
from email import policy
from email.message import EmailMessage
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from mineru.kit.gradio.source_preview import build_mhtml_preview, prepare_source_preview
from mineru.parser import parse


def _part(payload: str | bytes, media_type: str, location: str) -> EmailMessage:
    """构造带来源地址的 MIME 部件，覆盖文本与二进制编码。"""
    part = EmailMessage(policy=policy.SMTP)
    major, minor = media_type.split("/", 1)
    if isinstance(payload, str):
        part.set_content(payload, subtype=minor, charset="utf-8", cte="quoted-printable")
    else:
        part.set_content(payload, maintype=major, subtype=minor, cte="base64")
    part["Content-Location"] = location
    return part


def _archive(*, cycle: bool = False, root_html: str | None = None) -> bytes:
    """生成含 CSS 导入、相对资源、CID 和缺失资源的网页归档。"""
    root = _part(
        root_html
        or (
            '<html><head><title>中文预览</title><link rel="stylesheet" href="style.css" media="screen">'
            '<style>.inline{background:url("pic.png")}</style></head>'
            '<body><iframe src="cid:ad-page"></iframe><img src="pic.png" srcset="pic.png 1x, missing.png 2x">'
            '<picture><source srcset="data:image/png;base64,eA== 1x, pic.png 2x"></picture>'
            '<p style="background:url(pic.png)">正文</p></body></html>'
        ),
        "text/html",
        "https://site.test/article",
    )
    css = _part('@import "nested.css" screen; .main{background:url(pic.png)}', "text/css", "https://site.test/style.css")
    nested_text = '@import "style.css"; ' if cycle else ""
    nested = _part(nested_text + '@font-face{src:url("font.woff2")}', "text/css", "https://site.test/nested.css")
    picture = _part(b"saved-image", "image/png", "https://site.test/pic.png")
    font = _part(b"saved-font", "font/woff2", "https://site.test/font.woff2")
    message = EmailMessage(policy=policy.SMTP)
    message.make_related()
    for part in (root, css, nested, picture, font):
        message.attach(part)
    return message.as_bytes()


def _document(payload: bytes) -> BeautifulSoup:
    """读取 iframe 的 srcdoc，确认归档资源实际交给浏览器的内容。"""
    frame = BeautifulSoup(build_mhtml_preview(payload), "html.parser").iframe
    assert frame is not None and frame.get("sandbox") == ["allow-scripts"]
    return BeautifulSoup(frame["srcdoc"], "html.parser")


@pytest.mark.parametrize("cycle", [False, True])
def test_mhtml_preview_reuses_archived_assets_and_preserves_media(cycle: bool) -> None:
    """图片、字体与嵌套 CSS 从归档加载，循环导入有界终止。"""
    document = _document(_archive(cycle=cycle))
    assert document.title.get_text() == "中文预览"
    assert document.iframe is None
    assert document.img["src"].startswith("data:image/png;base64,")
    assert "https://site.test/missing.png 2x" in document.img["srcset"]
    assert document.source["srcset"].startswith("data:image/png;base64,eA== 1x, data:image/png;base64,")
    assert document.p["style"].count("data:image/png;base64,") == 1
    assert document.find("style", media="screen") is not None
    css = document.find("style", media="screen").get_text()
    encoded = re.search(r"data:text/css;charset=utf-8;base64,([A-Za-z0-9+/=]+)", css)
    assert encoded is not None
    nested = base64.b64decode(encoded.group(1)).decode("utf-8")
    assert "data:font/woff2;base64," in nested
    if cycle:
        assert "@import" not in nested


def test_mhtml_preview_preserves_declared_absolute_base() -> None:
    """归档源地址不能插到已有绝对 base 前，以免未重写的相对脚本指向错误目录。"""
    document = _document(
        _archive(
            root_html='<html><head><base href="https://cdn.site.test/assets/">'
            '<script src="runtime.js"></script></head><body><p>正文</p></body></html>'
        )
    )
    assert [base["href"] for base in document.find_all("base")] == ["https://cdn.site.test/assets/"]
    assert document.find("script", src=True)["src"] == "runtime.js"


@pytest.mark.parametrize("name", ["page.mhtml", "page.MHT"])
def test_mhtml_preview_receipt_and_flash_parse(tmp_path: Path, name: str) -> None:
    """两个后缀共用预览和 Flash 解析，回执仍带请求标识。"""
    source = tmp_path / name
    source.write_bytes(_archive())
    ticket = json.dumps({"id": 7, "path": str(source)})
    receipt = json.loads(asyncio.run(prepare_source_preview(str(source), ticket)))
    assert receipt["id"] == 7 and "中文预览" in receipt["html"]
    result = parse(source, tier="flash")
    assert result.middle_json.metadata.file_suffix == "mhtml"
    assert result.middle_json.extensions["mineru"] == {"tier": "flash", "parse_mode": "txt"}
    assert len(result.middle_json.pages) == 1


def test_mhtml_preview_rejects_invalid_and_oversized_archive(monkeypatch: pytest.MonkeyPatch) -> None:
    """错误归档与预览体积限制返回明确失败，避免构造过大的 srcdoc。"""
    with pytest.raises(Exception):
        build_mhtml_preview(b"not a MIME archive")
    from mineru.kit.gradio import mhtml_preview

    monkeypatch.setattr(mhtml_preview, "MAX_PREVIEW_BYTES", 100)
    with pytest.raises(ValueError, match="64 MiB"):
        build_mhtml_preview(_archive())
