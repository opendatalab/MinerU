from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import pytest

from mineru.types import EquationBlock, MiddleJson, PageInfo, TableBlock, TableBodyBlock


def _data_uri(mime_subtype: str, payload: bytes) -> str:
    """把测试图片字节编码为 data URI。"""
    return f"data:image/{mime_subtype};base64,{base64.b64encode(payload).decode('ascii')}"


def _middle_json_with_table(body: TableBodyBlock) -> MiddleJson:
    """构造只包含一个表格载体的最小 Office MiddleJson。"""
    table = TableBlock(type="table", index=body.index, content=[body])
    return MiddleJson(
        pages=[PageInfo(page_idx=3, blocks=[table])],
        is_full_document=True,
        file_suffix="docx",
        effort="flash",
        parse_mode="txt",
        mineru_version="test",
    )


def test_full_serialization_round_trip_and_recursive_field_exclusion() -> None:
    """验证完整 dump 保留图片，而递归排除不遗漏 visual 子块。"""
    jpeg_uri = _data_uri("jpeg", b"\xff\xd8\xffpayload\xff\xd9")
    middle_json = _middle_json_with_table(
        TableBodyBlock(type="table_body", index=0, content="<table></table>", image_base64=jpeg_uri)
    )

    full_json = middle_json.to_json(indent=None)
    excluded = middle_json.to_dict(exclude_block_fields={"image_base64"})

    assert jpeg_uri in full_json
    assert MiddleJson.model_validate_json(full_json) == middle_json
    assert "image_base64" not in excluded["pages"][0]["blocks"][0]["content"][0]


def test_equation_export_uses_canonical_sidecar_name(tmp_path: Path) -> None:
    """验证行间公式图片使用 equation discriminator 生成确定性 sidecar 名称。"""
    jpeg_payload = b"\xff\xd8\xffequation\xff\xd9"
    equation = EquationBlock(
        type="equation",
        index=6,
        content="x+1",
        image_base64=_data_uri("jpeg", jpeg_payload),
    )
    middle_json = MiddleJson(
        pages=[PageInfo(page_idx=3, blocks=[equation])],
        is_full_document=True,
        file_suffix="docx",
        effort="flash",
        parse_mode="txt",
        mineru_version="test",
    )

    result = middle_json.export(tmp_path)
    exported_equation = result.middle_json.pages[0].blocks[0]

    assert [path.name for path in result.image_paths] == ["page_3_equation_6.jpg"]
    assert (tmp_path / "images/page_3_equation_6.jpg").read_bytes() == jpeg_payload
    assert exported_equation.image_path == "images/page_3_equation_6.jpg"
    assert exported_equation.image_base64 is None
    assert equation.image_base64 is not None


def test_export_writes_direct_and_multiple_html_images_without_mutating_source(tmp_path: Path) -> None:
    """验证直接图片与 HTML 多图采用确定性命名，导出副本不污染原对象。"""
    jpeg_payload = b"\xff\xd8\xffjpeg\xff\xd9"
    gif_payload = b"GIF89agif"
    png_payload = b"\x89PNG\r\n\x1a\npng"
    jpeg_uri = _data_uri("jpeg", jpeg_payload)
    gif_uri = _data_uri("gif", gif_payload)
    png_uri = _data_uri("png", png_payload)
    html = f'<table><img src="{gif_uri}"><img src="{png_uri}"></table>'
    middle_json = _middle_json_with_table(
        TableBodyBlock(type="table_body", index=4, content=html, image_base64=jpeg_uri)
    )

    result = middle_json.export(tmp_path)
    exported_json = result.json_path.read_text()
    exported_body = result.middle_json.pages[0].blocks[0].content[0]

    assert {path.name for path in result.image_paths} == {
        "page_3_table_body_4.jpg",
        "page_3_table_body_4_1.gif",
        "page_3_table_body_4_2.png",
    }
    assert (tmp_path / "images/page_3_table_body_4.jpg").read_bytes() == jpeg_payload
    assert (tmp_path / "images/page_3_table_body_4_1.gif").read_bytes() == gif_payload
    assert (tmp_path / "images/page_3_table_body_4_2.png").read_bytes() == png_payload
    assert exported_body.image_path == "images/page_3_table_body_4.jpg"
    assert exported_body.image_base64 is None
    assert "images/page_3_table_body_4_1.gif" in exported_body.content
    assert "images/page_3_table_body_4_2.png" in exported_body.content
    assert "image_base64" not in exported_json
    assert "data:image/" not in exported_json
    assert middle_json.pages[0].blocks[0].content[0].image_base64 == jpeg_uri
    assert "data:image/" in middle_json.pages[0].blocks[0].content[0].content


def test_export_supports_strict_svg_payload(tmp_path: Path) -> None:
    """验证 PPTX 可能产出的 SVG data URI 可按 XML 根元素严格校验并外置。"""
    svg_payload = b'<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"></svg>'
    middle_json = _middle_json_with_table(
        TableBodyBlock(
            type="table_body",
            index=2,
            content="<table></table>",
            image_base64=_data_uri("svg+xml", svg_payload),
        )
    )

    result = middle_json.export(tmp_path)

    assert (tmp_path / "images/page_3_table_body_2.svg").read_bytes() == svg_payload
    assert result.middle_json.pages[0].blocks[0].content[0].image_path.endswith(".svg")


def test_export_supports_direct_png_and_html_jpeg(tmp_path: Path) -> None:
    """验证直接 PNG 与 Office 表格 HTML 内嵌 JPEG 都能按各自格式外置。"""
    png_payload = b"\x89PNG\r\n\x1a\npng"
    jpeg_payload = b"\xff\xd8\xffjpeg\xff\xd9"
    jpeg_uri = _data_uri("jpeg", jpeg_payload)
    middle_json = _middle_json_with_table(
        TableBodyBlock(
            type="table_body",
            index=5,
            content=f'<table><img src="{jpeg_uri}"></table>',
            image_base64=_data_uri("png", png_payload),
        )
    )

    result = middle_json.export(tmp_path)

    assert (tmp_path / "images/page_3_table_body_5.png").read_bytes() == png_payload
    assert (tmp_path / "images/page_3_table_body_5_1.jpg").read_bytes() == jpeg_payload
    assert "images/page_3_table_body_5_1.jpg" in result.middle_json.pages[0].blocks[0].content[0].content


@pytest.mark.parametrize(
    "data_uri",
    [
        "data:image/jpeg;base64,not-base64!",
        _data_uri("png", b"\xff\xd8\xffwrong-mime\xff\xd9"),
        _data_uri("avif", b"unsupported-format"),
        _data_uri("svg+xml", b"<html></html>"),
    ],
)
def test_export_rejects_invalid_payload_before_writing(tmp_path: Path, data_uri: str) -> None:
    """验证非法 base64、MIME 签名不符和不支持格式均不会产生半成品。"""
    output_dir = tmp_path / "output"
    middle_json = _middle_json_with_table(
        TableBodyBlock(type="table_body", index=0, content="<table></table>", image_base64=data_uri)
    )

    with pytest.raises(ValueError):
        middle_json.export(output_dir)

    assert not output_dir.exists()


def test_export_rejects_unparsed_inline_data_uri_before_writing(tmp_path: Path) -> None:
    """验证不符合 base64 data URI 语法的 HTML 图片不会原样泄漏到导出 JSON。"""
    output_dir = tmp_path / "output"
    middle_json = _middle_json_with_table(
        TableBodyBlock(
            type="table_body",
            index=0,
            content='<table><img src="data:image/jpeg,raw-payload"></table>',
        )
    )

    with pytest.raises(ValueError, match="inline image data URI"):
        middle_json.export(output_dir)

    assert not output_dir.exists()


def test_export_preflights_conflicts_and_supports_explicit_overwrite(tmp_path: Path) -> None:
    """验证同名同内容可复用、不同内容默认报错且 overwrite 可替换。"""
    first_payload = b"\xff\xd8\xfffirst\xff\xd9"
    second_payload = b"\xff\xd8\xffsecond\xff\xd9"
    first = _middle_json_with_table(
        TableBodyBlock(
            type="table_body",
            index=0,
            content="<table></table>",
            image_base64=_data_uri("jpeg", first_payload),
        )
    )
    second = _middle_json_with_table(
        TableBodyBlock(
            type="table_body",
            index=0,
            content="<table></table>",
            image_base64=_data_uri("jpeg", second_payload),
        )
    )

    first.export(tmp_path)
    first.export(tmp_path)
    image_path = tmp_path / "images/page_3_table_body_0.jpg"
    with pytest.raises(FileExistsError):
        second.export(tmp_path)
    assert image_path.read_bytes() == first_payload

    second.export(tmp_path, overwrite=True)
    assert image_path.read_bytes() == second_payload


def test_export_conflicting_json_rolls_back_before_any_image_write(tmp_path: Path) -> None:
    """验证 JSON 冲突在提交前被发现，不会先留下图片 sidecar。"""
    (tmp_path / "middle_json.json").write_text("occupied")
    middle_json = _middle_json_with_table(
        TableBodyBlock(
            type="table_body",
            index=9,
            content="<table></table>",
            image_base64=_data_uri("jpeg", b"\xff\xd8\xffpayload\xff\xd9"),
        )
    )

    with pytest.raises(FileExistsError):
        middle_json.export(tmp_path)

    assert not (tmp_path / "images/page_3_table_body_9.jpg").exists()
    assert (tmp_path / "middle_json.json").read_text() == "occupied"


def test_export_restores_existing_files_after_commit_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证提交中途失败时会恢复已覆盖文件，并清理全部临时文件。"""
    image_path = tmp_path / "images/page_3_table_body_0.jpg"
    image_path.parent.mkdir()
    image_path.write_bytes(b"old-image")
    json_path = tmp_path / "middle_json.json"
    json_path.write_bytes(b"old-json")
    middle_json = _middle_json_with_table(
        TableBodyBlock(
            type="table_body",
            index=0,
            content="<table></table>",
            image_base64=_data_uri("jpeg", b"\xff\xd8\xffnew-image\xff\xd9"),
        )
    )
    original_replace = os.replace
    replace_count = 0

    def fail_second_replace(source: str | Path, target: str | Path) -> None:
        """模拟第二个文件提交失败，以覆盖事务回滚分支。"""
        nonlocal replace_count
        replace_count += 1
        if replace_count == 2:
            raise OSError("simulated commit failure")
        original_replace(source, target)

    monkeypatch.setattr("mineru.types.os.replace", fail_second_replace)

    with pytest.raises(OSError, match="simulated commit failure"):
        middle_json.export(tmp_path, overwrite=True)

    assert image_path.read_bytes() == b"old-image"
    assert json_path.read_bytes() == b"old-json"
    assert not list(tmp_path.rglob(".mineru-export-*"))


@pytest.mark.parametrize("json_name", ["../middle.json", "/tmp/middle.json", "..\\middle.json"])
def test_export_rejects_path_escape(tmp_path: Path, json_name: str) -> None:
    """验证 JSON 输出名不能使用绝对路径或逃逸文档目录。"""
    middle_json = _middle_json_with_table(
        TableBodyBlock(type="table_body", index=0, content="<table></table>")
    )
    with pytest.raises(ValueError):
        middle_json.export(tmp_path, json_name=json_name)


def test_export_rejects_file_and_directory_path_collision(tmp_path: Path) -> None:
    """验证 JSON 文件名不能占用图片 sidecar 所需的 images 目录。"""
    output_dir = tmp_path / "output"
    middle_json = _middle_json_with_table(
        TableBodyBlock(
            type="table_body",
            index=0,
            content="<table></table>",
            image_base64=_data_uri("jpeg", b"\xff\xd8\xffpayload\xff\xd9"),
        )
    )

    with pytest.raises(ValueError, match="required directory"):
        middle_json.export(output_dir, json_name="images")

    assert not output_dir.exists()


def test_export_rejects_symlink_output_directory(tmp_path: Path) -> None:
    """验证导出根目录是符号链接时直接拒绝写入。"""
    actual_dir = tmp_path / "actual"
    actual_dir.mkdir()
    link_dir = tmp_path / "link"
    os.symlink(actual_dir, link_dir)
    middle_json = _middle_json_with_table(
        TableBodyBlock(type="table_body", index=0, content="<table></table>")
    )

    with pytest.raises(ValueError, match="symlink"):
        middle_json.export(link_dir)

    assert list(actual_dir.iterdir()) == []


def test_export_rejects_symlink_sidecar_directory(tmp_path: Path) -> None:
    """验证图片子目录是符号链接时不会跟随链接写到文档目录之外。"""
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    os.symlink(outside_dir, output_dir / "images")
    middle_json = _middle_json_with_table(
        TableBodyBlock(
            type="table_body",
            index=0,
            content="<table></table>",
            image_base64=_data_uri("jpeg", b"\xff\xd8\xffpayload\xff\xd9"),
        )
    )

    with pytest.raises(ValueError, match="symlink"):
        middle_json.export(output_dir)

    assert list(outside_dir.iterdir()) == []


def test_image_path_is_validated_during_deserialization() -> None:
    """验证对象边界拒绝绝对路径、目录逃逸和 Windows 反斜杠路径。"""
    for image_path in ("/tmp/image.jpg", "../image.jpg", "images\\image.jpg"):
        with pytest.raises(ValueError):
            TableBodyBlock(
                type="table_body",
                index=0,
                content="<table></table>",
                image_path=image_path,
            )


def test_exported_json_is_a_pure_middle_json_object(tmp_path: Path) -> None:
    """验证导出 JSON 可独立严格反序列化且不依赖导出结果包装对象。"""
    middle_json = _middle_json_with_table(
        TableBodyBlock(type="table_body", index=0, content="<table></table>")
    )
    result = middle_json.export(tmp_path)

    payload = json.loads(result.json_path.read_text())
    restored = MiddleJson.model_validate(payload)

    assert restored == result.middle_json
