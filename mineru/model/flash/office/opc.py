# Copyright (c) Opendatalab. All rights reserved.
"""OOXML 格式复用的 Open Packaging Conventions 基础能力。"""

from __future__ import annotations

from collections.abc import Sequence
from io import BytesIO
import posixpath
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


def write_zip_package(members: Sequence[tuple[ZipInfo, bytes]]) -> bytes:
    """把规范化后的成员重新写成 ZIP 包，并由标准库重新计算 CRC。"""
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as target:
        for info, member_data in members:
            target.writestr(info, member_data)
    return output.getvalue()


def relationship_source_base_dir(rels_filename: str) -> str | None:
    """从规范 OPC relationship 成员路径推导源 part 所在目录。"""
    if rels_filename == "_rels/.rels":
        return ""

    marker = "/_rels/"
    if marker not in rels_filename:
        return None

    prefix, rels_basename = rels_filename.rsplit(marker, 1)
    if not rels_basename.endswith(".rels"):
        return None

    source_part_name = rels_basename[: -len(".rels")]
    source_part_path = posixpath.normpath(posixpath.join(prefix, source_part_name))
    return posixpath.dirname(source_part_path)


__all__ = ["relationship_source_base_dir", "write_zip_package"]
