# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator


def _make_getitem(cls_name: str):
    """Generate dict-access compatibility methods."""

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None

    def __setitem__(self, key: str, value: Any) -> None:
        if not hasattr(self, key):
            raise KeyError(key)
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            val = getattr(self, key)
        except AttributeError:
            return default
        if val is None and default is not None:
            return default
        return val

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    __getitem__.__qualname__ = f"{cls_name}.__getitem__"
    __getitem__.__name__ = "__getitem__"
    __setitem__.__qualname__ = f"{cls_name}.__setitem__"
    __setitem__.__name__ = "__setitem__"
    get.__qualname__ = f"{cls_name}.get"
    get.__name__ = "get"
    __contains__.__qualname__ = f"{cls_name}.__contains__"
    __contains__.__name__ = "__contains__"
    return __getitem__, __setitem__, get, __contains__


@dataclass
class Span:
    """Leaf node of the block tree.  Holds text, formula, image, or table content."""

    type: str
    bbox: tuple[float, float, float, float] | None = None
    content: str = ""
    score: float = 0.0
    image_path: str = ""
    image_base64: str = ""
    html: str = ""
    latex: str = ""

    __getitem__, __setitem__, get, __contains__ = _make_getitem("Span")  # type: ignore[assignment]

    @classmethod
    def from_dict(cls, d: dict) -> Span:
        bbox = d.get("bbox")
        return cls(
            type=d["type"],
            bbox=tuple(bbox) if bbox else None,
            content=d.get("content", ""),
            score=d.get("score", 0.0),
            image_path=d.get("image_path", ""),
            image_base64=d.get("image_base64", ""),
            html=d.get("html", ""),
            latex=d.get("latex", ""),
        )


@dataclass
class Line:
    """A line within a block, containing one or more spans."""

    spans: list[Span] = field(default_factory=list)

    __getitem__, __setitem__, get, __contains__ = _make_getitem("Line")  # type: ignore[assignment]

    @classmethod
    def from_dict(cls, d: dict) -> Line:
        return cls(spans=[Span.from_dict(s) for s in d.get("spans", [])])


@dataclass
class Block:
    """A layout block on a page.  May nest child blocks (e.g. list items, image body)."""

    type: str
    bbox: tuple[float, float, float, float] | None = None
    lines: list[Line] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)
    level: int | None = None
    section_number: str = ""
    sub_type: str = ""
    html: str = ""
    merge_prev: bool = False

    __getitem__, __setitem__, get, __contains__ = _make_getitem("Block")  # type: ignore[assignment]

    @classmethod
    def from_dict(cls, d: dict) -> Block:
        bbox = d.get("bbox")
        return cls(
            type=d["type"],
            bbox=tuple(bbox) if bbox else None,
            lines=[Line.from_dict(line) for line in d.get("lines", [])],
            blocks=[Block.from_dict(b) for b in d.get("blocks", [])],
            level=d.get("level"),
            section_number=d.get("section_number", ""),
            sub_type=d.get("sub_type", ""),
            html=d.get("html", ""),
            merge_prev=d.get("merge_prev", False),
        )

    def all_spans(self) -> Iterator[Span]:
        """Depth-first yield every span in this block tree."""
        for line in self.lines:
            yield from line.spans
        for child in self.blocks:
            yield from child.all_spans()


@dataclass
class PageInfo:
    """Parsed content of a single page."""

    page_idx: int
    page_size: tuple[int, int] | None = None
    preproc_blocks: list[Block] = field(default_factory=list)
    para_blocks: list[Block] = field(default_factory=list)
    discarded_blocks: list[Block] = field(default_factory=list)

    __getitem__, __setitem__, get, __contains__ = _make_getitem("PageInfo")  # type: ignore[assignment]

    @classmethod
    def from_dict(cls, d: dict) -> PageInfo:
        ps = d.get("page_size")
        return cls(
            page_idx=d["page_idx"],
            page_size=tuple(ps) if ps else None,
            preproc_blocks=[Block.from_dict(b) for b in d.get("preproc_blocks", [])],
            para_blocks=[Block.from_dict(b) for b in d.get("para_blocks", [])],
            discarded_blocks=[Block.from_dict(b) for b in d.get("discarded_blocks", [])],
        )
