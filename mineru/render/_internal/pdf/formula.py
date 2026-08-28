# Copyright (c) Opendatalab. All rights reserved.
"""ZiaMath LaTeX 到受控 ReportLab 矢量路径的转换。"""

from __future__ import annotations

from dataclasses import dataclass
import re
from threading import RLock
from typing import Any
from xml.etree import ElementTree

from fontTools.pens.basePen import BasePen
from fontTools.svgLib.path import parse_path
from reportlab.graphics.shapes import Drawing, Group, Path, Rect
from reportlab.lib.colors import Color, toColor
from reportlab.platypus import Flowable
import ziamath

MAX_FORMULA_CHARACTERS = 20_000
MAX_CACHED_FORMULAS = 512
_MAX_SVG_NODES = 20_000
_MAX_SVG_PATH_CHARACTERS = 20_000_000
_SVG_NAMESPACE = "http://www.w3.org/2000/svg"
_ZIAMATH_LOCK = RLock()
_SVG_PATH_COMMAND_RE = re.compile(r"[A-DF-Za-df-z]")


class PdfFormulaError(ValueError):
    """表示 LaTeX 或 ZiaMath SVG 无法安全转换为 PDF 矢量对象。"""


@dataclass(frozen=True, slots=True)
class FormulaVector:
    """保存一个公式的 ReportLab Drawing 与基线几何。"""

    drawing: Drawing
    width: float
    height: float
    ascent: float
    descent: float

    def scaled(self, factor: float) -> FormulaVector:
        """返回仅调整展示几何、不复制 Drawing 的等比公式对象。"""
        return FormulaVector(
            drawing=self.drawing,
            width=self.width * factor,
            height=self.height * factor,
            ascent=self.ascent * factor,
            descent=self.descent * factor,
        )


@dataclass(frozen=True, slots=True)
class InlineFormulaImage:
    """作为 ReportLab Paragraph 行内图片占位的矢量公式代理。"""

    vector: FormulaVector


class FormulaRenderer:
    """维护单份 PDF 文档内有界且无跨文档状态的公式缓存。"""

    def __init__(self) -> None:
        """初始化最多缓存 512 个唯一公式的文档级缓存。"""
        self._cache: dict[tuple[str, bool, float, str], FormulaVector] = {}

    def render(
        self,
        latex: str,
        *,
        inline: bool,
        font_size: float,
        color: str = "#1f2937",
    ) -> FormulaVector:
        """把一个裸 LaTeX 公式转换为行内或行间矢量对象。"""
        if not isinstance(latex, str) or not latex.strip():
            raise PdfFormulaError("formula must contain non-blank LaTeX")
        if not isinstance(font_size, (int, float)) or isinstance(font_size, bool) or font_size <= 0:
            raise PdfFormulaError("font_size must be a positive number")

        key = (latex, inline, float(font_size), color)
        if len(latex) <= MAX_FORMULA_CHARACTERS and key in self._cache:
            return self._cache[key]

        vector = _render_ziamath_formula(latex, inline=inline, font_size=float(font_size), color=color)
        if len(latex) <= MAX_FORMULA_CHARACTERS and len(self._cache) < MAX_CACHED_FORMULAS:
            self._cache[key] = vector
        return vector


class DisplayFormulaFlowable(Flowable):
    """在可用行宽内居中绘制公式，并把可选编号贴到右边界。"""

    def __init__(self, formula: FormulaVector, tag: FormulaVector | None = None) -> None:
        """保存公式、可选编号以及延迟到 wrap 阶段计算的缩放参数。"""
        super().__init__()
        self.formula = formula
        self.tag = tag
        self._available_width = formula.width
        self._formula_scale = 1.0
        self._tag_scale = 1.0
        self.width = formula.width
        self.height = formula.height

    def wrap(self, avail_width: float, _avail_height: float) -> tuple[float, float]:
        """按正文宽度缩放主公式和编号，避免二者重叠或水平裁切。"""
        self._available_width = max(1.0, avail_width)
        tag_width = self.tag.width if self.tag is not None else 0.0
        tag_gap = 10.0 if self.tag is not None else 0.0
        main_limit = self._available_width if self.tag is None else max(1.0, self._available_width - tag_width - tag_gap)
        self._formula_scale = min(1.0, main_limit / max(self.formula.width, 1.0))
        if self.tag is not None and tag_width > self._available_width * 0.25:
            self._tag_scale = min(1.0, self._available_width * 0.25 / max(tag_width, 1.0))
            main_limit = max(1.0, self._available_width - tag_width * self._tag_scale - tag_gap)
            self._formula_scale = min(self._formula_scale, main_limit / max(self.formula.width, 1.0))
        formula_height = self.formula.height * self._formula_scale
        tag_height = self.tag.height * self._tag_scale if self.tag is not None else 0.0
        self.width = self._available_width
        self.height = max(formula_height, tag_height, 1.0)
        return self.width, self.height

    def draw(self) -> None:
        """把缩放后的主公式居中，并在同一垂直中心绘制右侧编号。"""
        formula_width = self.formula.width * self._formula_scale
        formula_height = self.formula.height * self._formula_scale
        formula_x = max(0.0, (self._available_width - formula_width) / 2)
        formula_y = max(0.0, (self.height - formula_height) / 2)
        _draw_vector(self.canv, self.formula, formula_x, formula_y, self._formula_scale)
        if self.tag is not None:
            tag_width = self.tag.width * self._tag_scale
            tag_height = self.tag.height * self._tag_scale
            tag_x = max(0.0, self._available_width - tag_width)
            tag_y = max(0.0, (self.height - tag_height) / 2)
            _draw_vector(self.canv, self.tag, tag_x, tag_y, self._tag_scale)


class _ReportLabPathPen(BasePen):
    """把 FontTools SVG path 回调写入 ReportLab Path。"""

    def __init__(self) -> None:
        """创建不依赖 glyphSet 的空 ReportLab 路径。"""
        super().__init__(None)
        self.path = Path()

    def _moveTo(self, point: tuple[float, float]) -> None:
        """把 SVG move 命令写入目标路径。"""
        self.path.moveTo(*point)

    def _lineTo(self, point: tuple[float, float]) -> None:
        """把 SVG line 命令写入目标路径。"""
        self.path.lineTo(*point)

    def _curveToOne(
        self,
        point1: tuple[float, float],
        point2: tuple[float, float],
        point3: tuple[float, float],
    ) -> None:
        """把三次曲线写入目标路径，二次曲线由 BasePen 自动转换。"""
        self.path.curveTo(*point1, *point2, *point3)

    def _closePath(self) -> None:
        """闭合当前 ReportLab 子路径。"""
        self.path.closePath()

    def _endPath(self) -> None:
        """结束不闭合的 SVG 子路径。"""


def split_formula_tag(content: str) -> tuple[str, str | None]:
    """剥离公式末尾括号平衡的 ``\\tag{...}``，并返回正文与编号。"""
    stripped_end = len(content.rstrip())
    if stripped_end == 0 or content[stripped_end - 1] != "}":
        return content, None
    search_end = stripped_end
    while (tag_start := content.rfind(r"\tag", 0, search_end)) >= 0:
        if not _is_escaped_character(content, tag_start):
            opening_brace = _find_tag_opening_brace(content, tag_start, stripped_end)
            if opening_brace is not None:
                closing_brace = _find_balanced_closing_brace(content, opening_brace, stripped_end)
                if closing_brace == stripped_end - 1:
                    return content[:tag_start].rstrip(), content[opening_brace + 1 : closing_brace].strip()
        search_end = tag_start
    return content, None


def draw_inline_formula(
    canvas: Any,
    image: InlineFormulaImage,
    x: float,
    y: float,
    width: float,
    height: float,
) -> tuple[float, float]:
    """由自定义 Canvas 在 Paragraph 计算的位置绘制一个行内矢量公式。"""
    vector = image.vector
    scale = min(width / max(vector.width, 1.0), height / max(vector.height, 1.0))
    _draw_vector(canvas, vector, x, y, scale)
    return width, height


def _render_ziamath_formula(latex: str, *, inline: bool, font_size: float, color: str) -> FormulaVector:
    """在全局锁内临时关闭 SVG2 symbols，并转换单个 ZiaMath 结果。"""
    try:
        with _ZIAMATH_LOCK:
            previous_svg2 = ziamath.config.svg2
            ziamath.config.svg2 = False
            try:
                formula = ziamath.Latex(latex, inline=inline, size=font_size, color=color, margin=0)
                root = formula.svgxml()
            finally:
                ziamath.config.svg2 = previous_svg2
        return _svg_root_to_vector(root)
    except PdfFormulaError:
        raise
    except Exception as exc:
        raise PdfFormulaError(f"LaTeX formula cannot be rendered: {_formula_preview(latex)!r}") from exc


def _svg_root_to_vector(root: ElementTree.Element) -> FormulaVector:
    """把 ZiaMath 的固定 SVG 子集转换为坐标已翻转的 ReportLab Drawing。"""
    namespace = root.get("xmlns") if root.tag == "svg" else root.tag.removeprefix("{").split("}", 1)[0]
    if _local_name(root.tag) != "svg" or namespace != _SVG_NAMESPACE:
        raise PdfFormulaError("ZiaMath output must contain an SVG root")
    view_box = _parse_number_list(root.get("viewBox"), count=4, field="viewBox")
    min_x, min_y, width, height = view_box
    if width <= 0 or height <= 0:
        raise PdfFormulaError("ZiaMath SVG dimensions must be positive")
    drawing = Drawing(width, height)
    root_group = Group()
    root_group.transform = (1, 0, 0, -1, -min_x, min_y + height)
    node_counter = [0]
    path_budget = [0]
    for child in root:
        _append_svg_element(
            root_group,
            child,
            inherited={},
            node_counter=node_counter,
            path_budget=path_budget,
        )
    drawing.add(root_group)
    ascent = max(0.0, -min_y)
    descent = max(0.0, min_y + height)
    return FormulaVector(drawing=drawing, width=width, height=height, ascent=ascent, descent=descent)


def _append_svg_element(
    parent: Group,
    element: ElementTree.Element,
    *,
    inherited: dict[str, str],
    node_counter: list[int],
    path_budget: list[int],
) -> None:
    """递归转换 ZiaMath 允许的 group、path 与 rect 节点。"""
    node_counter[0] += 1
    if node_counter[0] > _MAX_SVG_NODES:
        raise PdfFormulaError("ZiaMath SVG exceeds its node limit")
    tag = _local_name(element.tag)
    if tag == "g":
        _reject_unknown_attributes(element, {"fill", "stroke", "stroke-width"})
        group_style = {**inherited, **element.attrib}
        group = Group()
        for child in element:
            _append_svg_element(
                group,
                child,
                inherited=group_style,
                node_counter=node_counter,
                path_budget=path_budget,
            )
        parent.add(group)
        return
    if tag == "path":
        _reject_unknown_attributes(element, {"d", "fill", "stroke", "stroke-width"})
        path_data = element.get("d", "")
        path_budget[0] += len(path_data)
        if not path_data or path_budget[0] > _MAX_SVG_PATH_CHARACTERS:
            raise PdfFormulaError("ZiaMath SVG path data is empty or exceeds its limit")
        commands = set(_SVG_PATH_COMMAND_RE.findall(path_data))
        if not commands.issubset({"M", "L", "Q", "Z"}):
            raise PdfFormulaError(f"Unsupported ZiaMath SVG path commands: {', '.join(sorted(commands))}")
        pen = _ReportLabPathPen()
        try:
            parse_path(path_data, pen)
        except Exception as exc:
            raise PdfFormulaError("ZiaMath SVG path data is invalid") from exc
        _apply_paint(pen.path, element.attrib, inherited)
        parent.add(pen.path)
        return
    if tag == "rect":
        _reject_unknown_attributes(element, {"x", "y", "width", "height", "fill", "stroke", "stroke-width"})
        x, y, width, height = (_parse_number(element.get(name, "0"), field=name) for name in ("x", "y", "width", "height"))
        if width < 0 or height < 0:
            raise PdfFormulaError("ZiaMath SVG rect dimensions must not be negative")
        rectangle = Rect(x, y, width, height)
        _apply_paint(rectangle, element.attrib, inherited)
        parent.add(rectangle)
        return
    raise PdfFormulaError(f"Unsupported ZiaMath SVG element: {tag}")


def _apply_paint(shape: Any, attributes: dict[str, str], inherited: dict[str, str]) -> None:
    """把受控 SVG fill、stroke 与 stroke-width 映射到 ReportLab shape。"""
    fill = attributes.get("fill", inherited.get("fill", "black"))
    stroke = attributes.get("stroke", inherited.get("stroke", "none"))
    shape.fillColor = _parse_color(fill)
    shape.strokeColor = _parse_color(stroke)
    stroke_width = attributes.get("stroke-width", inherited.get("stroke-width", "1"))
    shape.strokeWidth = _parse_number(stroke_width, field="stroke-width")


def _parse_color(value: str) -> Color | None:
    """解析 ZiaMath 生成的静态颜色，none 映射为透明。"""
    if value.strip().casefold() == "none":
        return None
    try:
        return toColor(value)
    except Exception as exc:
        raise PdfFormulaError(f"Unsupported ZiaMath SVG color: {value!r}") from exc


def _parse_number(value: str, *, field: str) -> float:
    """严格读取不含 CSS 单位的有限 SVG 数值。"""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PdfFormulaError(f"Invalid ZiaMath SVG {field}: {value!r}") from exc
    if number != number or number in (float("inf"), float("-inf")):
        raise PdfFormulaError(f"Invalid ZiaMath SVG {field}: {value!r}")
    return number


def _parse_number_list(value: str | None, *, count: int, field: str) -> tuple[float, ...]:
    """读取固定长度的空白或逗号分隔 SVG 数值列表。"""
    if value is None:
        raise PdfFormulaError(f"ZiaMath SVG is missing {field}")
    values = tuple(_parse_number(item, field=field) for item in value.replace(",", " ").split())
    if len(values) != count:
        raise PdfFormulaError(f"ZiaMath SVG {field} must contain {count} numbers")
    return values


def _reject_unknown_attributes(element: ElementTree.Element, allowed: set[str]) -> None:
    """拒绝 ZiaMath 固定子集之外的 SVG 属性。"""
    unexpected = set(element.attrib) - allowed
    if unexpected:
        raise PdfFormulaError(f"Unsupported ZiaMath SVG attributes: {', '.join(sorted(unexpected))}")


def _local_name(tag: str) -> str:
    """返回可带 XML namespace 的元素本地名称。"""
    return tag.rsplit("}", 1)[-1]


def _formula_preview(latex: str) -> str:
    """为告警生成有界 LaTeX 摘要，避免超长公式污染日志。"""
    return latex if len(latex) <= 200 else f"{latex[:197]}..."


def _draw_vector(canvas: Any, vector: FormulaVector, x: float, y: float, scale: float) -> None:
    """在 canvas 上按给定位置和比例绘制公式 Drawing。"""
    canvas.saveState()
    try:
        canvas.translate(x, y)
        canvas.scale(scale, scale)
        vector.drawing.drawOn(canvas, 0, 0)
    finally:
        canvas.restoreState()


def _find_tag_opening_brace(content: str, tag_start: int, content_end: int) -> int | None:
    """查找 tag 命令允许空白后的左花括号。"""
    cursor = tag_start + len(r"\tag")
    while cursor < content_end and content[cursor].isspace():
        cursor += 1
    return cursor if cursor < content_end and content[cursor] == "{" else None


def _find_balanced_closing_brace(content: str, opening_brace: int, content_end: int) -> int | None:
    """查找与 tag 左花括号配对的右花括号，并忽略转义花括号。"""
    depth = 0
    for cursor in range(opening_brace, content_end):
        character = content[cursor]
        if character not in "{}" or _is_escaped_character(content, cursor):
            continue
        depth += 1 if character == "{" else -1
        if depth == 0:
            return cursor
        if depth < 0:
            return None
    return None


def _is_escaped_character(content: str, position: int) -> bool:
    """判断指定字符前是否存在奇数个连续反斜杠。"""
    preceding_backslashes = 0
    cursor = position - 1
    while cursor >= 0 and content[cursor] == "\\":
        preceding_backslashes += 1
        cursor -= 1
    return preceding_backslashes % 2 == 1


__all__ = [
    "DisplayFormulaFlowable",
    "FormulaRenderer",
    "FormulaVector",
    "InlineFormulaImage",
    "MAX_CACHED_FORMULAS",
    "MAX_FORMULA_CHARACTERS",
    "PdfFormulaError",
    "draw_inline_formula",
    "split_formula_tag",
]
