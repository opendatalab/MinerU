# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

# ── dict-compatibility for dataclass-backed model objects ──────────
#
# Block / Line / Span / PageInfo are dataclasses whose canonical fields
# are declared below.  Backend code (MagicModel, para_split, union_make)
# also attaches *internal processing fields* (``index``, ``page_num``,
# ``_ocr_det_lines``, etc.) via ``block["key"] = value``.  Those
# non-canonical keys are stored in ``_extra`` so that the dataclass
# signature stays clean while still providing full dict compatibility
# during migration.


def _extra_getitem(self, key: str) -> Any:
    try:
        return getattr(self, key)
    except AttributeError:
        pass
    try:
        return self._extra[key]
    except (AttributeError, KeyError):
        raise KeyError(key) from None


def _extra_setitem(self, key: str, value: Any) -> None:
    if hasattr(self, key):
        setattr(self, key, value)
    else:
        self._extra[key] = value


def _extra_get(self, key: str, default: Any = None) -> Any:
    try:
        val = getattr(self, key)
    except AttributeError:
        return self._extra.get(key, default)
    return default if val is None and default is not None else val


def _extra_contains(self, key: str) -> bool:
    return hasattr(self, key) or key in self._extra


def _extra_delitem(self, key: str) -> None:
    if hasattr(self, key):
        raise KeyError(f"Cannot delete canonical field '{key}'")
    del self._extra[key]


def _install_dict_compat(*classes) -> None:
    for cls in classes:
        cls.__getitem__ = _extra_getitem
        cls.__setitem__ = _extra_setitem
        cls.__delitem__ = _extra_delitem
        cls.get = _extra_get
        cls.__contains__ = _extra_contains


# ── conversion helpers (dict → typed) ──────────────────────────────


def block_from_dict(d: dict | Block) -> Block:
    if isinstance(d, Block):
        return d
    return Block(
        **{k: v for k, v in d.items()
           if k in Block.__dataclass_fields__},
        _extra={k: v for k, v in d.items()
                if k not in Block.__dataclass_fields__},
    )


def line_from_dict(d: dict) -> Line:
    return Line(
        spans=[span_from_dict(s) for s in d.get("spans", [])],
        _extra={k: v for k, v in d.items() if k != "spans"},
    )


def span_from_dict(d: dict) -> Span:
    return Span(
        **{k: v for k, v in d.items()
           if k in Span.__dataclass_fields__},
        _extra={k: v for k, v in d.items()
                if k not in Span.__dataclass_fields__},
    )


# ── model types ─────────────────────────────────────────────────────


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
    _extra: dict = field(default_factory=dict)

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
    _extra: dict = field(default_factory=dict)

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
    _extra: dict = field(default_factory=dict)

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
    page_size: tuple[int, int] | list[int, int] | None = None
    preproc_blocks: list[Block] = field(default_factory=list)
    para_blocks: list[Block] = field(default_factory=list)
    discarded_blocks: list[Block] = field(default_factory=list)
    _extra: dict = field(default_factory=dict)

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


_install_dict_compat(Span, Line, Block, PageInfo)
