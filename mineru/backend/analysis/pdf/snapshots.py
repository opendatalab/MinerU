# Copyright (c) Opendatalab. All rights reserved.
"""仅在当前真实 PDF 窗口内持有自有文字快照，按需补齐矢量和链接。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from weakref import ReferenceType, ref

from docvortex.document.pdf import PDFPage, PDFPageTextGeometry, PDFPageVectorGeometry

if TYPE_CHECKING:
    from docvortex.document.pdf import PDFPageSnapshot


@dataclass
class _PageSnapshotEntry:
    """文字 owner 跨阶段共享，页面只保留弱引用以免延长文档生命周期。"""

    page_ref: ReferenceType[PDFPage]
    text_owner: Any
    full_snapshot: PDFPageSnapshot | None = None

    def validate_page(self, page: PDFPage) -> None:
        """拒绝将当前窗口条目误用于其他页面对象。"""
        if self.page_ref() is not page:
            raise ValueError("window snapshot belongs to a different PDF page")

    def get_full_snapshot(self, page: PDFPage) -> PDFPageSnapshot:
        """仅在表格或正文确实需要时补齐矢量和链接，并复用已持有的文字 owner。"""
        self.validate_page(page)
        if self.full_snapshot is None:
            self.full_snapshot = page.get_snapshot()
        return self.full_snapshot

    def close(self) -> None:
        """清除强引用；快照不持有需手工关闭的 PDFium 页面句柄。"""
        self.full_snapshot = None
        self.text_owner = None


PageSnapshotCache = list[_PageSnapshotEntry | None]


def create_page_snapshot_cache(pages: list[PDFPage], *, enabled: bool) -> PageSnapshotCache | None:
    """只为真实页面开启新通道，旧 fake 页面及不提供新公开接口的环境保持原调用。"""
    if not enabled or not pages or not callable(getattr(PDFPage, "get_text_snapshot", None)):
        return None
    # MagicMock(spec=PDFPage) 可伪造 isinstance，必须检查真实类型。
    return [None] * len(pages) if all(type(page) is PDFPage for page in pages) else None


def get_page_snapshot_entry(
    cache: PageSnapshotCache | None,
    index: int,
    page: PDFPage,
    *,
    geometry: PDFPageTextGeometry | None = None,
    vector_geometry: PDFPageVectorGeometry | None = None,
) -> _PageSnapshotEntry | None:
    """显式几何始终优先；只对空缓存的真实页面惰性读取一次 owned 文字。"""
    if cache is None or type(page) is not PDFPage or geometry is not None or vector_geometry is not None:
        return None
    entry = cache[index]
    if entry is None:
        entry = _PageSnapshotEntry(ref(page), page.get_text_snapshot())
        cache[index] = entry
    entry.validate_page(page)
    return entry if entry.text_owner is not None else None


def clear_page_snapshot_cache(cache: PageSnapshotCache | None) -> None:
    """成功、失败与取消均释放完整窗口的 owned 数据，即使外层仍持有条目对象。"""
    if cache is not None:
        for entry in cache:
            if entry is not None:
                entry.close()
        cache.clear()
