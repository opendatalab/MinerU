# Copyright (c) Opendatalab. All rights reserved.
"""从 OFD.xml 与 Document.xml 提取 Doclib 基础元数据。"""

from __future__ import annotations

from typing import BinaryIO

from .constants import MAX_PAGE_COUNT, MAX_TOTAL_BYTES
from .errors import OfdResourceLimitError
from .package import OfdPackage, first_descendant, local_name


def extract_ofd_metadata(file_binary: BinaryIO) -> dict[str, object | None]:
    """返回全部 DocBody 的总页数和首个非空文档元数据。"""
    file_bytes = file_binary.read(MAX_TOTAL_BYTES + 1)
    if len(file_bytes) > MAX_TOTAL_BYTES:
        raise OfdResourceLimitError(f"OFD resource limit exceeded: max_total_bytes={MAX_TOTAL_BYTES}")
    with OfdPackage(file_bytes) as package:
        refs = package.document_refs()
        page_count = 0
        metadata: dict[str, str | None] = {
            "title": None,
            "author": None,
            "subject": None,
            "keywords": None,
        }
        key_map = {
            "Title": "title",
            "Author": "author",
            "Subject": "subject",
            "Keywords": "keywords",
        }
        for ref in refs:
            document_root = package.xml_part(ref.document_part, required=True)
            assert document_root is not None
            pages = first_descendant(document_root, "Pages")
            if pages is not None:
                for child in pages:
                    if local_name(child.tag) != "Page":
                        continue
                    page_count += 1
                    if page_count > MAX_PAGE_COUNT:
                        raise OfdResourceLimitError(f"OFD resource limit exceeded: max_page_count={MAX_PAGE_COUNT}")
            for source_key, target_key in key_map.items():
                if metadata[target_key] is None and ref.metadata.get(source_key):
                    metadata[target_key] = ref.metadata[source_key]
        return {
            "page_count": page_count,
            **metadata,
            "is_image_based": 0,
        }


__all__ = ["extract_ofd_metadata"]
