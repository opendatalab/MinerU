# Copyright (c) Opendatalab. All rights reserved.
"""WMF/EMF 回放状态、绘图 IR 与结果模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias


MetafileSourceFormat: TypeAlias = Literal["wmf", "emf"]
MetafileOutputFormat: TypeAlias = Literal["png", "jpeg", "svg"]
EmfPlusMode: TypeAlias = Literal["none", "dual", "only"]
DiagnosticLevel: TypeAlias = Literal["info", "warning", "error"]
Point: TypeAlias = tuple[float, float]


class MetafileError(ValueError):
    """WMF/EMF 渲染错误基类，并携带稳定错误码。"""

    code = "metafile_error"


class MetafileMalformedError(MetafileError):
    """WMF/EMF 头部或记录流不满足格式边界。"""

    code = "malformed"


class MetafileResourceLimitError(MetafileError):
    """WMF/EMF 输入或输出超过固定安全预算。"""

    code = "resource_limit"


class MetafileUnsupportedError(MetafileError):
    """输入使用无法安全降级的 WMF/EMF 能力。"""

    code = "unsupported"


@dataclass(frozen=True, slots=True)
class MetafileDiagnostic:
    """记录单个可定位、可聚合的 WMF/EMF 渲染诊断。"""

    code: str
    level: DiagnosticLevel
    message: str
    record_type: int | None = None
    record_index: int | None = None
    offset: int | None = None


@dataclass(frozen=True, slots=True)
class Color:
    """保存标准 RGBA 颜色分量。"""

    red: int
    green: int
    blue: int
    alpha: int = 255

    def rgba(self) -> tuple[int, int, int, int]:
        """返回 Pillow 可直接消费的 RGBA 元组。"""
        return self.red, self.green, self.blue, self.alpha

    def svg(self) -> str:
        """返回不含透明度的 SVG 十六进制颜色。"""
        return f"#{self.red:02x}{self.green:02x}{self.blue:02x}"


BLACK = Color(0, 0, 0)
WHITE = Color(255, 255, 255)
TRANSPARENT = Color(0, 0, 0, 0)


@dataclass(frozen=True, slots=True)
class Rect:
    """保存规范化前后均可使用的浮点矩形。"""

    left: float
    top: float
    right: float
    bottom: float

    @property
    def width(self) -> float:
        """返回矩形的有符号宽度。"""
        return self.right - self.left

    @property
    def height(self) -> float:
        """返回矩形的有符号高度。"""
        return self.bottom - self.top

    def normalized(self) -> Rect:
        """返回左右、上下均按升序排列的矩形。"""
        return Rect(
            min(self.left, self.right),
            min(self.top, self.bottom),
            max(self.left, self.right),
            max(self.top, self.bottom),
        )


@dataclass(frozen=True, slots=True)
class Matrix:
    """保存二维仿射变换，采用 SVG/Pillow 常用的六参数表示。"""

    a: float = 1.0
    b: float = 0.0
    c: float = 0.0
    d: float = 1.0
    e: float = 0.0
    f: float = 0.0

    def transform_point(self, point: Point) -> Point:
        """把单点应用到当前仿射变换。"""
        x, y = point
        return self.a * x + self.c * y + self.e, self.b * x + self.d * y + self.f

    def transform_vector(self, vector: Point) -> Point:
        """只应用线性部分，避免平移污染长度和方向。"""
        x, y = vector
        return self.a * x + self.c * y, self.b * x + self.d * y

    def then(self, following: Matrix) -> Matrix:
        """返回先应用当前矩阵、再应用 following 的组合矩阵。"""
        return Matrix(
            a=following.a * self.a + following.c * self.b,
            b=following.b * self.a + following.d * self.b,
            c=following.a * self.c + following.c * self.d,
            d=following.b * self.c + following.d * self.d,
            e=following.a * self.e + following.c * self.f + following.e,
            f=following.b * self.e + following.d * self.f + following.f,
        )


@dataclass(frozen=True, slots=True)
class Pen:
    """保存已经解析的 GDI 画笔。"""

    color: Color = BLACK
    width: float = 1.0
    style: int = 0
    cosmetic: bool = True
    null: bool = False
    cap: Literal["round", "square", "flat"] = "round"
    join: Literal["round", "bevel", "miter"] = "round"
    dashes: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class Brush:
    """保存已经解析的 GDI 画刷。"""

    kind: Literal["solid", "null", "hatch", "pattern"] = "solid"
    color: Color = WHITE
    hatch: int = 0
    pattern: bytes | None = None


@dataclass(frozen=True, slots=True)
class Font:
    """保存 GDI LOGFONT 中与跨平台绘制有关的字段。"""

    face_name: str = "Arial"
    height: float = -12.0
    width: float = 0.0
    weight: int = 400
    italic: bool = False
    underline: bool = False
    strikeout: bool = False
    charset: int = 1
    escapement: float = 0.0
    orientation: float = 0.0


@dataclass(frozen=True, slots=True)
class PathSegment:
    """保存单个 move/line/cubic/close 路径片段。"""

    verb: Literal["M", "L", "C", "Z"]
    points: tuple[Point, ...] = ()


@dataclass(frozen=True, slots=True)
class GraphicsPath:
    """保存不可变的跨后端绘图路径。"""

    segments: tuple[PathSegment, ...]


@dataclass(frozen=True, slots=True)
class ClipOperation:
    """保存按 GDI 顺序应用的单次裁剪区域操作。"""

    path: GraphicsPath
    mode: Literal["and", "or", "xor", "diff", "copy"]
    fill_rule: Literal["evenodd", "nonzero"] = "evenodd"


ClipStack: TypeAlias = tuple[ClipOperation, ...]


@dataclass(frozen=True, slots=True)
class DrawPathCommand:
    """保存一次路径填充和描边操作。"""

    path: GraphicsPath
    pen: Pen
    brush: Brush
    stroke: bool
    fill: bool
    fill_rule: Literal["evenodd", "nonzero"]
    clip: ClipStack
    rop2: int
    miter_limit: float = 10.0


@dataclass(frozen=True, slots=True)
class DrawTextCommand:
    """保存一次 GDI 文字输出及其显式字符位置。"""

    text: str
    origin: Point
    positions: tuple[Point, ...]
    font: Font
    font_height: float
    rotation: float
    text_align: int
    color: Color
    background_color: Color
    opaque: bool
    bounds: Rect | None
    clip: ClipStack
    advance_end: Point | None = None


@dataclass(frozen=True, slots=True)
class DrawImageCommand:
    """保存一次 DIB/位图绘制及其 GDI 合成参数。"""

    dib_header: bytes
    bits: bytes
    destination: tuple[Point, Point, Point, Point]
    source: Rect | None
    rop: int
    stretch_mode: int = 3
    constant_alpha: int = 255
    use_source_alpha: bool = False
    clip: ClipStack = ()


DrawCommand: TypeAlias = DrawPathCommand | DrawTextCommand | DrawImageCommand


@dataclass(frozen=True, slots=True)
class MetafileDocument:
    """保存解析完成、可被多个后端消费的统一图元文档。"""

    source_format: MetafileSourceFormat
    emfplus_mode: EmfPlusMode
    bounds: Rect
    width: int
    height: int
    commands: tuple[DrawCommand, ...]
    diagnostics: tuple[MetafileDiagnostic, ...]
    partial: bool


@dataclass(frozen=True, slots=True)
class MetafileRenderResult:
    """保存最终图片字节、尺寸与解析诊断。"""

    data: bytes
    output_format: MetafileOutputFormat
    media_type: str
    width: int
    height: int
    source_format: MetafileSourceFormat
    emfplus_mode: EmfPlusMode
    partial: bool
    diagnostics: tuple[MetafileDiagnostic, ...]


@dataclass(slots=True)
class GdiState:
    """保存 WMF/EMF 当前 playback device context。"""

    pen: Pen = field(default_factory=Pen)
    brush: Brush = field(default_factory=Brush)
    font: Font = field(default_factory=Font)
    text_color: Color = BLACK
    background_color: Color = WHITE
    background_mode: int = 2
    text_align: int = 0
    map_mode: int = 1
    window_origin: Point = (0.0, 0.0)
    window_extent: Point = (1.0, 1.0)
    viewport_origin: Point = (0.0, 0.0)
    viewport_extent: Point = (1.0, 1.0)
    world_transform: Matrix = field(default_factory=Matrix)
    current_position: Point = (0.0, 0.0)
    polygon_fill_mode: int = 1
    rop2: int = 13
    stretch_mode: int = 1
    arc_direction: int = 2
    miter_limit: float = 10.0
    brush_origin: Point = (0.0, 0.0)
    clip: ClipStack = ()
    device_pixels_per_mm: Point = (96.0 / 25.4, 96.0 / 25.4)


__all__ = [
    "BLACK",
    "WHITE",
    "Brush",
    "ClipOperation",
    "Color",
    "DrawCommand",
    "DrawImageCommand",
    "DrawPathCommand",
    "DrawTextCommand",
    "EmfPlusMode",
    "Font",
    "GdiState",
    "GraphicsPath",
    "Matrix",
    "MetafileDiagnostic",
    "MetafileDocument",
    "MetafileError",
    "MetafileMalformedError",
    "MetafileOutputFormat",
    "MetafileRenderResult",
    "MetafileResourceLimitError",
    "MetafileSourceFormat",
    "MetafileUnsupportedError",
    "PathSegment",
    "Pen",
    "Point",
    "Rect",
]
