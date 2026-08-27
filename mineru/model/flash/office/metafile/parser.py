# Copyright (c) Opendatalab. All rights reserved.
"""WMF/EMF 有界解析与 GDI 状态回放。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import atan2, cos, degrees, isfinite, radians, sin
import struct

from .binary import BoundedReader
from .font import measure_text_advance
from .geometry import (
    PathBuilder,
    arc_path,
    close_open_subpaths,
    colorref_to_rgb,
    ellipse_path,
    path_bounds,
    rectangle_path,
    round_rectangle_path,
    transform_path,
    union_rectangles,
    vector_length,
)
from .limits import (
    MAX_CANVAS_DIMENSION,
    MAX_CANVAS_PIXELS,
    MAX_CLIP_OPERATIONS,
    MAX_COMMANDS,
    MAX_DIAGNOSTICS,
    MAX_EMBEDDED_BITMAP_BYTES,
    MAX_METAFILE_BYTES,
    MAX_OBJECTS,
    MAX_POINTS_PER_RECORD,
    MAX_RECORDS,
    MAX_STATE_DEPTH,
    MAX_TOTAL_CLIP_OPERATIONS,
    MAX_TOTAL_POINTS,
)
from .models import (
    BLACK,
    WHITE,
    Brush,
    ClipOperation,
    Color,
    DrawCommand,
    DrawImageCommand,
    DrawPathCommand,
    DrawTextCommand,
    EmfPlusMode,
    Font,
    GdiState,
    GraphicsPath,
    Matrix,
    MetafileDiagnostic,
    MetafileDocument,
    MetafileMalformedError,
    MetafileResourceLimitError,
    MetafileSourceFormat,
    MetafileUnsupportedError,
    Pen,
    Point,
    Rect,
)


_EMF_SIGNATURE = 0x464D4520
_EMFPLUS_SIGNATURE = 0x2B464D45
_PLACEABLE_WMF_KEY = 0x9AC6CDD7

_RGN_AND = 1
_RGN_OR = 2
_RGN_XOR = 3
_RGN_DIFF = 4
_RGN_COPY = 5

_ETO_OPAQUE = 0x0002
_ETO_CLIPPED = 0x0004
_ETO_GLYPH_INDEX = 0x0010
_ETO_PDY = 0x2000
_TA_UPDATECP = 0x0001
_TA_RIGHT = 0x0002
_TA_CENTER = 0x0006

_SRCCOPY = 0x00CC0020
_SUPPORTED_ROP3 = {
    _SRCCOPY,
    0x008800C6,
    0x00EE0086,
    0x00660046,
    0x00550009,
    0x00000042,
    0x00FF0062,
}


@dataclass(frozen=True, slots=True)
class _HeaderInfo:
    """保存格式头部解析出的记录边界、坐标边界和输出尺寸。"""

    source_format: MetafileSourceFormat
    record_start: int
    record_end: int
    bounds: Rect | None
    pixel_size: tuple[int, int] | None
    emfplus_mode: EmfPlusMode = "none"
    device_pixels_per_mm: Point = (96.0 / 25.4, 96.0 / 25.4)
    wmf_object_count: int = 0


class _DiagnosticSink:
    """收集有界诊断并聚合超出预算的重复项目。"""

    def __init__(self) -> None:
        """创建空诊断列表和截断计数器。"""
        self.items: list[MetafileDiagnostic] = []
        self.omitted = 0

    def add(
        self,
        code: str,
        level: str,
        message: str,
        *,
        record_type: int | None = None,
        record_index: int | None = None,
        offset: int | None = None,
    ) -> None:
        """在预算内追加诊断，超出时只累计省略数量。"""
        if len(self.items) >= MAX_DIAGNOSTICS:
            self.omitted += 1
            return
        normalized_level = level if level in {"info", "warning", "error"} else "warning"
        self.items.append(
            MetafileDiagnostic(
                code=code,
                level=normalized_level,  # type: ignore[arg-type]
                message=message,
                record_type=record_type,
                record_index=record_index,
                offset=offset,
            )
        )

    def freeze(self) -> tuple[MetafileDiagnostic, ...]:
        """冻结诊断，并在需要时追加一条截断摘要。"""
        if self.omitted:
            self.items.append(
                MetafileDiagnostic(
                    code="diagnostics_truncated",
                    level="warning",
                    message=f"omitted {self.omitted} additional metafile diagnostics",
                )
            )
        return tuple(self.items)


class _Playback:
    """把 WMF/EMF records 回放为统一绘图命令。"""

    def __init__(self, header: _HeaderInfo, diagnostics: _DiagnosticSink) -> None:
        """初始化 GDI 状态、对象表、路径与资源计数。"""
        self.header = header
        self.diagnostics = diagnostics
        self.state = GdiState(device_pixels_per_mm=header.device_pixels_per_mm)
        self.state_stack: list[GdiState] = []
        self.emf_objects: dict[int, Pen | Brush | Font] = {}
        self.wmf_objects: list[Pen | Brush | Font | None] = [None] * min(header.wmf_object_count, MAX_OBJECTS)
        self.commands: list[DrawCommand] = []
        self.path_builder = PathBuilder()
        self.path_active = False
        self.path_ready = False
        self.partial = False
        self.total_points = 0
        self.total_clip_operations = 0
        self.record_type: int | None = None
        self.record_index: int | None = None
        self.record_offset: int | None = None

    def set_record_context(self, record_type: int, record_index: int, offset: int) -> None:
        """更新后续诊断所使用的当前记录定位信息。"""
        self.record_type = record_type
        self.record_index = record_index
        self.record_offset = offset

    def warn(self, code: str, message: str, *, partial: bool = True) -> None:
        """为当前记录追加警告，并按需标记输出不完整。"""
        self.partial = self.partial or partial
        self.diagnostics.add(
            code,
            "warning",
            message,
            record_type=self.record_type,
            record_index=self.record_index,
            offset=self.record_offset,
        )

    def charge_points(self, count: int) -> None:
        """累计点数并执行单记录与全文件预算。"""
        if count < 0 or count > MAX_POINTS_PER_RECORD:
            raise MetafileResourceLimitError(f"metafile record exceeds max_points_per_record={MAX_POINTS_PER_RECORD}")
        self.total_points += count
        if self.total_points > MAX_TOTAL_POINTS:
            raise MetafileResourceLimitError(f"metafile exceeds max_total_points={MAX_TOTAL_POINTS}")

    def append_command(self, command: DrawCommand) -> None:
        """在命令预算内追加一条统一绘图命令。"""
        if len(self.commands) >= MAX_COMMANDS:
            raise MetafileResourceLimitError(f"metafile exceeds max_commands={MAX_COMMANDS}")
        self.total_clip_operations += len(command.clip)
        if self.total_clip_operations > MAX_TOTAL_CLIP_OPERATIONS:
            raise MetafileResourceLimitError(f"metafile exceeds max_total_clip_operations={MAX_TOTAL_CLIP_OPERATIONS}")
        self.commands.append(command)

    def copy_state(self) -> GdiState:
        """复制当前可由 SaveDC/RestoreDC 恢复的 GDI 状态。"""
        return replace(self.state)

    def save_dc(self) -> None:
        """把当前 GDI 状态压入有界栈。"""
        if len(self.state_stack) >= MAX_STATE_DEPTH:
            raise MetafileResourceLimitError(f"metafile exceeds max_state_depth={MAX_STATE_DEPTH}")
        self.state_stack.append(self.copy_state())

    def restore_dc(self, level: int) -> None:
        """按 GDI 相对或绝对层级恢复已保存状态。"""
        if not self.state_stack:
            self.warn("restore_dc_underflow", "RestoreDC ignored because the state stack is empty")
            return
        if level < 0:
            target = len(self.state_stack) + level
        elif level > 0:
            target = level - 1
        else:
            self.warn("restore_dc_zero", "RestoreDC level 0 is invalid")
            return
        if target < 0 or target >= len(self.state_stack):
            self.warn("restore_dc_out_of_range", f"RestoreDC level is out of range: {level}")
            return
        self.state = self.state_stack[target]
        del self.state_stack[target:]

    def _mapping_matrix(self) -> Matrix:
        """根据 mapping mode、window 和 viewport 计算 page-to-device 变换。"""
        mode = self.state.map_mode
        window_x, window_y = self.state.window_origin
        viewport_x, viewport_y = self.state.viewport_origin
        if mode in {7, 8}:
            window_width, window_height = self.state.window_extent
            viewport_width, viewport_height = self.state.viewport_extent
            if window_width == 0.0 or window_height == 0.0:
                self.warn("zero_window_extent", "zero window extent was replaced by identity mapping")
                scale_x = scale_y = 1.0
            else:
                scale_x = viewport_width / window_width
                scale_y = viewport_height / window_height
        elif mode == 2:
            scale_x = self.state.device_pixels_per_mm[0] / 10.0
            scale_y = -self.state.device_pixels_per_mm[1] / 10.0
        elif mode == 3:
            scale_x = self.state.device_pixels_per_mm[0] / 100.0
            scale_y = -self.state.device_pixels_per_mm[1] / 100.0
        elif mode == 4:
            scale_x = self.state.device_pixels_per_mm[0] * 25.4 / 100.0
            scale_y = -self.state.device_pixels_per_mm[1] * 25.4 / 100.0
        elif mode == 5:
            scale_x = self.state.device_pixels_per_mm[0] * 25.4 / 1000.0
            scale_y = -self.state.device_pixels_per_mm[1] * 25.4 / 1000.0
        elif mode == 6:
            scale_x = self.state.device_pixels_per_mm[0] * 25.4 / 1440.0
            scale_y = -self.state.device_pixels_per_mm[1] * 25.4 / 1440.0
        else:
            scale_x = scale_y = 1.0
        return Matrix(
            a=scale_x,
            d=scale_y,
            e=viewport_x - window_x * scale_x,
            f=viewport_y - window_y * scale_y,
        )

    def logical_matrix(self) -> Matrix:
        """返回 world-to-page 与 page-to-device 的组合变换。"""
        return self.state.world_transform.then(self._mapping_matrix())

    def map_point(self, point: Point) -> Point:
        """把 GDI 逻辑点转换为统一 device 坐标。"""
        mapped = self.logical_matrix().transform_point(point)
        if not all(isfinite(value) and abs(value) <= 1e12 for value in mapped):
            raise MetafileResourceLimitError("metafile coordinate is non-finite or unreasonably large")
        return mapped

    def map_vector(self, vector: Point) -> Point:
        """把不含平移的 GDI 逻辑向量转换为 device 向量。"""
        mapped = self.logical_matrix().transform_vector(vector)
        if not all(isfinite(value) and abs(value) <= 1e12 for value in mapped):
            raise MetafileResourceLimitError("metafile vector is non-finite or unreasonably large")
        return mapped

    def resolved_pen(self) -> Pen:
        """把当前画笔宽度解析为 device 单位并保留 cosmetic 语义。"""
        pen = self.state.pen
        if pen.null:
            return pen
        if pen.cosmetic:
            return replace(pen, width=max(1.0, pen.width))
        width = vector_length(self.map_vector((pen.width or 1.0, 0.0)))
        return replace(pen, width=max(width, 1.0))

    def emit_path(self, path: GraphicsPath, *, stroke: bool, fill: bool) -> None:
        """把路径追加到当前 path bracket 或直接生成绘图命令。"""
        if self.path_active:
            self.path_builder.extend(path)
            return
        if not path.segments:
            return
        self.append_command(
            DrawPathCommand(
                path=path,
                pen=self.resolved_pen(),
                brush=self.state.brush,
                stroke=stroke and not self.state.pen.null,
                fill=fill and self.state.brush.kind != "null",
                fill_rule="evenodd" if self.state.polygon_fill_mode == 1 else "nonzero",
                clip=self.state.clip,
                rop2=self.state.rop2,
                miter_limit=self.state.miter_limit,
            )
        )

    def emit_logical_path(self, path: GraphicsPath, *, stroke: bool, fill: bool) -> None:
        """把逻辑坐标路径变换后交给统一路径输出。"""
        self.emit_path(transform_path(path, self.logical_matrix()), stroke=stroke, fill=fill)

    def move_to(self, point: Point) -> None:
        """更新 GDI current position，并在活动 path 中开始新 figure。"""
        self.state.current_position = point
        if self.path_active:
            self.path_builder.move_to(self.map_point(point))

    def _ensure_active_path_current(self) -> None:
        """保证活动 path 的下一条 To 记录从 GDI current position 起笔。"""
        mapped = self.map_point(self.state.current_position)
        if self.path_builder.current != mapped:
            self.path_builder.move_to(mapped)

    def line_to(self, endpoint: Point) -> None:
        """按 LineTo 语义追加直线并更新 GDI current position。"""
        if self.path_active:
            self._ensure_active_path_current()
            self.path_builder.line_to(self.map_point(endpoint))
        else:
            self.emit_path(
                _path_from_device_points(
                    [self.map_point(self.state.current_position), self.map_point(endpoint)],
                    close=False,
                ),
                stroke=True,
                fill=False,
            )
        self.state.current_position = endpoint

    def polyline_to(self, points: list[Point]) -> None:
        """按 PolylineTo 语义连续追加折线并更新 GDI current position。"""
        if not points:
            return
        if self.path_active:
            self._ensure_active_path_current()
            for point in points:
                self.path_builder.line_to(self.map_point(point))
        else:
            mapped = [self.map_point(self.state.current_position), *(self.map_point(point) for point in points)]
            self.emit_path(_path_from_device_points(mapped, close=False), stroke=True, fill=False)
        self.state.current_position = points[-1]

    def polybezier_to(self, points: list[Point]) -> None:
        """按 PolyBezierTo 语义连续追加三次曲线并更新 GDI current position。"""
        if not points:
            return
        remaining = points
        if len(remaining) % 3:
            self.warn("invalid_bezier_points", "PolyBezierTo point count is not a multiple of three")
            remaining = remaining[: len(remaining) - len(remaining) % 3]
        if not remaining:
            return
        builder = self.path_builder if self.path_active else PathBuilder()
        if self.path_active:
            self._ensure_active_path_current()
        else:
            builder.move_to(self.map_point(self.state.current_position))
        for index in range(0, len(remaining), 3):
            builder.cubic_to(
                self.map_point(remaining[index]),
                self.map_point(remaining[index + 1]),
                self.map_point(remaining[index + 2]),
            )
        if not self.path_active:
            self.emit_path(builder.build(), stroke=True, fill=False)
        self.state.current_position = remaining[-1]

    def connected_arc(self, path: GraphicsPath) -> None:
        """绘制 current-to-projected-start 连线和圆弧，并更新到 projected endpoint。"""
        if not path.segments:
            return
        mapped_path = transform_path(path, self.logical_matrix())
        mapped_start = mapped_path.segments[0].points[0]
        tail = GraphicsPath(mapped_path.segments[1:])
        if self.path_active:
            self._ensure_active_path_current()
            self.path_builder.line_to(mapped_start)
            self.path_builder.extend(tail)
        else:
            builder = PathBuilder()
            builder.move_to(self.map_point(self.state.current_position))
            builder.line_to(mapped_start)
            builder.extend(tail)
            self.emit_path(builder.build(), stroke=True, fill=False)
        self.state.current_position = path.segments[-1].points[-1]

    def begin_path(self) -> None:
        """开始新的 GDI path bracket 并清除旧路径。"""
        if self.path_active:
            self.warn("begin_path_while_active", "BeginPath replaced an already active path bracket")
        self.path_builder.clear()
        self.path_active = True
        self.path_ready = False

    def end_path(self) -> None:
        """结束 path bracket 并保留路径供后续 fill/stroke/clip。"""
        if not self.path_active:
            self.warn("end_path_without_begin", "EndPath ignored without a matching BeginPath")
            return
        self.path_active = False
        self.path_ready = True

    def consume_path(self) -> GraphicsPath:
        """返回并清除已经结束的当前路径。"""
        if not self.path_ready:
            self.warn("path_not_ready", "path operation ignored because no completed path exists")
            return GraphicsPath(())
        path = self.path_builder.build()
        self.path_builder.clear()
        self.path_ready = False
        return path

    def add_clip(self, path: GraphicsPath, mode: int) -> None:
        """把 GDI region combine mode 转换为不可变裁剪操作。"""
        mode_name = _clip_mode_name(mode)
        if mode_name is None:
            self.warn("unsupported_clip_mode", f"unsupported clip combine mode: {mode}")
            return
        operation = ClipOperation(
            path=path,
            mode=mode_name,
            fill_rule="evenodd" if self.state.polygon_fill_mode == 1 else "nonzero",
        )
        if mode == _RGN_COPY:
            self.state.clip = (operation,)
        else:
            self.state.clip = self.clip_with_operation(operation)

    def clip_with_operation(self, operation: ClipOperation) -> tuple[ClipOperation, ...]:
        """在固定 clip 深度预算内返回追加单次操作后的不可变栈。"""
        if len(self.state.clip) >= MAX_CLIP_OPERATIONS:
            raise MetafileResourceLimitError(f"metafile exceeds max_clip_operations={MAX_CLIP_OPERATIONS}")
        return (*self.state.clip, operation)

    def reset_clip(self) -> None:
        """恢复为没有额外裁剪区域的初始状态。"""
        self.state.clip = ()

    def create_wmf_object(self, value: Pen | Brush | Font) -> None:
        """按 WMF 首个空槽规则登记图形对象。"""
        for index, existing in enumerate(self.wmf_objects):
            if existing is None:
                self.wmf_objects[index] = value
                return
        if len(self.wmf_objects) >= MAX_OBJECTS:
            raise MetafileResourceLimitError(f"WMF object table exceeds max_objects={MAX_OBJECTS}")
        self.wmf_objects.append(value)

    def select_object(self, value: Pen | Brush | Font) -> None:
        """按对象实际类型更新当前选中画笔、画刷或字体。"""
        if isinstance(value, Pen):
            self.state.pen = value
        elif isinstance(value, Brush):
            self.state.brush = value
        else:
            self.state.font = value

    def select_emf_handle(self, handle: int) -> None:
        """选择 EMF 显式 handle 或 stock object。"""
        if handle & 0x80000000:
            stock = _stock_object(handle & 0x7FFFFFFF)
            if stock is None:
                self.warn("unknown_stock_object", f"unknown EMF stock object: {handle:#x}")
                return
            self.select_object(stock)
            return
        value = self.emf_objects.get(handle)
        if value is None:
            self.warn("missing_object", f"EMF object handle does not exist: {handle}")
            return
        self.select_object(value)

    def select_wmf_handle(self, handle: int) -> None:
        """选择 WMF 对象槽或兼容的 stock object。"""
        if handle & 0x8000:
            stock = _stock_object(handle & 0x7FFF)
            if stock is None:
                self.warn("unknown_stock_object", f"unknown WMF stock object: {handle:#x}")
                return
            self.select_object(stock)
            return
        if handle >= len(self.wmf_objects) or self.wmf_objects[handle] is None:
            self.warn("missing_object", f"WMF object handle does not exist: {handle}")
            return
        self.select_object(self.wmf_objects[handle])  # type: ignore[arg-type]

    def finalize(self, size_hint: tuple[int, int] | None) -> MetafileDocument:
        """解析最终边界、输出尺寸并冻结统一图元文档。"""
        command_bounds = _commands_bounds(self.commands)
        bounds = self.header.bounds
        if bounds is None or abs(bounds.width) < 1e-9 or abs(bounds.height) < 1e-9:
            bounds = command_bounds
        if bounds is None or abs(bounds.width) < 1e-9 or abs(bounds.height) < 1e-9:
            raise MetafileUnsupportedError("metafile contains no visible output")
        bounds = bounds.normalized()
        requested = size_hint or self.header.pixel_size
        if requested is None:
            requested = max(1, round(bounds.width)), max(1, round(bounds.height))
        width, height, downscaled = _bounded_canvas_size(requested)
        if downscaled:
            self.diagnostics.add(
                "canvas_downscaled",
                "warning",
                f"metafile canvas was downscaled from {requested[0]}x{requested[1]} to {width}x{height}",
            )
        return MetafileDocument(
            source_format=self.header.source_format,
            emfplus_mode=self.header.emfplus_mode,
            bounds=bounds,
            width=width,
            height=height,
            commands=tuple(self.commands),
            diagnostics=self.diagnostics.freeze(),
            partial=self.partial,
        )


def _clip_mode_name(mode: int) -> str | None:
    """把 GDI RegionMode 数值转换为内部裁剪操作名称。"""
    return {
        _RGN_AND: "and",
        _RGN_OR: "or",
        _RGN_XOR: "xor",
        _RGN_DIFF: "diff",
        _RGN_COPY: "copy",
    }.get(mode)


def _stock_object(index: int) -> Pen | Brush | Font | None:
    """返回常见 GDI stock object 的确定性跨平台表示。"""
    brushes: dict[int, Brush] = {
        0: Brush(color=WHITE),
        1: Brush(color=Color(192, 192, 192)),
        2: Brush(color=Color(128, 128, 128)),
        3: Brush(color=Color(64, 64, 64)),
        4: Brush(color=BLACK),
        5: Brush(kind="null"),
        6: Brush(kind="null"),
        18: Brush(color=WHITE),
    }
    if index in brushes:
        return brushes[index]
    pens: dict[int, Pen] = {
        7: Pen(color=WHITE),
        8: Pen(color=BLACK),
        9: Pen(null=True),
        19: Pen(color=BLACK),
    }
    if index in pens:
        return pens[index]
    if 10 <= index <= 17:
        return Font(face_name="Courier New" if index in {10, 11, 16} else "Arial")
    return None


def _horizontal_text_align_factor(text_align: int) -> float:
    """返回 LEFT/CENTER/RIGHT 文本 bounds 相对 origin 的左移比例。"""
    if text_align & _TA_CENTER == _TA_CENTER:
        return 0.5
    if text_align & _TA_RIGHT:
        return 1.0
    return 0.0


def _commands_bounds(commands: list[DrawCommand]) -> Rect | None:
    """计算全部绘图命令的保守可见包围盒。"""
    rectangles: list[Rect | None] = []
    for command in commands:
        if isinstance(command, DrawPathCommand):
            rectangles.append(path_bounds(command.path))
        elif isinstance(command, DrawTextCommand):
            if command.bounds is not None and abs(command.bounds.width) > 0 and abs(command.bounds.height) > 0:
                rectangles.append(command.bounds)
            else:
                estimated_width = max(command.font_height * 0.6 * len(command.text), command.font_height)
                if command.advance_end is not None:
                    estimated_width = max(estimated_width, abs(command.advance_end[0] - command.origin[0]))
                align_factor = _horizontal_text_align_factor(command.text_align)
                left = command.origin[0] - estimated_width * align_factor
                rectangles.append(
                    Rect(
                        left,
                        command.origin[1] - command.font_height,
                        left + estimated_width,
                        command.origin[1] + command.font_height * 0.25,
                    )
                )
        else:
            xs = [point[0] for point in command.destination]
            ys = [point[1] for point in command.destination]
            rectangles.append(Rect(min(xs), min(ys), max(xs), max(ys)))
    return union_rectangles(rectangles)


def _bounded_canvas_size(requested: tuple[int, int]) -> tuple[int, int, bool]:
    """按单边和总像素预算等比收缩画布尺寸。"""
    width, height = requested
    if width <= 0 or height <= 0:
        raise MetafileMalformedError(f"metafile canvas must be positive: {width}x{height}")
    scale = min(1.0, MAX_CANVAS_DIMENSION / width, MAX_CANVAS_DIMENSION / height)
    if width * height * scale * scale > MAX_CANVAS_PIXELS:
        scale = min(scale, (MAX_CANVAS_PIXELS / (width * height)) ** 0.5)
    bounded_width = max(1, round(width * scale))
    bounded_height = max(1, round(height * scale))
    return bounded_width, bounded_height, scale < 1.0


def _parse_rect_i32(reader: BoundedReader, offset: int) -> Rect:
    """读取四个有符号 32 位坐标组成的 RectL。"""
    return Rect(reader.i32(offset), reader.i32(offset + 4), reader.i32(offset + 8), reader.i32(offset + 12))


def _parse_rect_i16(reader: BoundedReader, offset: int) -> Rect:
    """读取四个有符号 16 位坐标组成的 WMF 矩形。"""
    return Rect(reader.i16(offset), reader.i16(offset + 2), reader.i16(offset + 4), reader.i16(offset + 6))


def _parse_xform(reader: BoundedReader, offset: int) -> Matrix:
    """读取 EMF XForm 六个浮点参数并拒绝非有限值。"""
    values = tuple(reader.f32(offset + index * 4) for index in range(6))
    if not all(isfinite(value) and abs(value) <= 1e12 for value in values):
        raise MetafileMalformedError("EMF XForm contains non-finite or unreasonably large values")
    return Matrix(a=values[0], b=values[1], c=values[2], d=values[3], e=values[4], f=values[5])


def _parse_color(value: int, *, alpha: int = 255) -> Color:
    """把 GDI COLORREF 转换为带指定透明度的内部颜色。"""
    red, green, blue = colorref_to_rgb(value)
    return Color(red, green, blue, alpha)


def _detect_source_format(data: bytes) -> MetafileSourceFormat:
    """只依据实际文件签名区分 WMF 与 EMF。"""
    if (
        len(data) >= 44
        and struct.unpack_from("<I", data, 0)[0] == 1
        and struct.unpack_from("<I", data, 40)[0] == _EMF_SIGNATURE
    ):
        return "emf"
    wmf_offset = 22 if len(data) >= 22 and struct.unpack_from("<I", data, 0)[0] == _PLACEABLE_WMF_KEY else 0
    if len(data) >= wmf_offset + 18:
        file_type, header_size = struct.unpack_from("<HH", data, wmf_offset)
        if file_type in {1, 2} and header_size == 9:
            return "wmf"
    raise MetafileMalformedError("input does not contain a valid WMF or EMF signature")


def _scan_emfplus_mode(data: bytes, start: int, end: int) -> EmfPlusMode:
    """预扫描 EMR_COMMENT 中的 EMF+ header 并识别 Only/Dual。"""
    offset = start
    found_header = False
    found_dual = False
    record_count = 0
    while offset + 8 <= end:
        record_type, record_size = struct.unpack_from("<II", data, offset)
        record_count += 1
        if record_count > MAX_RECORDS:
            raise MetafileResourceLimitError(f"EMF exceeds max_records={MAX_RECORDS}")
        if record_size < 8 or record_size % 4 or record_size > end - offset:
            raise MetafileMalformedError(f"invalid EMF record while scanning comments: offset={offset}, size={record_size}")
        if record_type == 70 and record_size >= 28:
            data_size = struct.unpack_from("<I", data, offset + 8)[0]
            if data_size <= record_size - 12 and struct.unpack_from("<I", data, offset + 12)[0] == _EMFPLUS_SIGNATURE:
                payload_offset = offset + 16
                payload_end = offset + 12 + data_size
                while payload_offset + 12 <= payload_end:
                    plus_type, flags, plus_size, plus_data_size = struct.unpack_from("<HHII", data, payload_offset)
                    if plus_size < 12 or plus_size > payload_end - payload_offset or plus_data_size > plus_size - 12:
                        break
                    if plus_type == 0x4001:
                        found_header = True
                        found_dual = found_dual or bool(flags & 1)
                    payload_offset += plus_size
        offset += record_size
        if record_type == 14:
            break
    if not found_header:
        return "none"
    return "dual" if found_dual else "only"


def _parse_emf_header(data: bytes, dpi: int, size_hint: tuple[int, int] | None) -> _HeaderInfo:
    """解析 EMR_HEADER、声明文件边界和物理输出尺寸。"""
    reader = BoundedReader(data)
    if len(reader) < 88 or reader.u32(0) != 1:
        raise MetafileMalformedError("EMF header is truncated or has the wrong record type")
    header_size = reader.u32(4)
    if header_size < 88 or header_size % 4 or header_size > len(reader):
        raise MetafileMalformedError(f"invalid EMF header size: {header_size}")
    if reader.u32(40) != _EMF_SIGNATURE:
        raise MetafileMalformedError("invalid EMF signature")
    declared_bytes = reader.u32(48)
    declared_records = reader.u32(52)
    if declared_bytes < header_size or declared_bytes > len(reader):
        raise MetafileMalformedError(f"invalid EMF declared byte size: {declared_bytes}")
    if declared_records == 0 or declared_records > MAX_RECORDS:
        raise MetafileResourceLimitError(f"invalid or excessive EMF record count: {declared_records}")
    bounds = _parse_rect_i32(reader, 8)
    frame = _parse_rect_i32(reader, 24)
    device_x, device_y = reader.i32(72), reader.i32(76)
    millimeter_x, millimeter_y = reader.i32(80), reader.i32(84)
    if millimeter_x > 0 and millimeter_y > 0 and device_x > 0 and device_y > 0:
        device_scale = device_x / millimeter_x, device_y / millimeter_y
    else:
        device_scale = dpi / 25.4, dpi / 25.4
    pixel_size = size_hint
    if pixel_size is None and frame.width != 0 and frame.height != 0:
        pixel_size = max(1, round(abs(frame.width) * dpi / 2540.0)), max(1, round(abs(frame.height) * dpi / 2540.0))
    if pixel_size is None and bounds.width != 0 and bounds.height != 0:
        pixel_size = max(1, round(abs(bounds.width))), max(1, round(abs(bounds.height)))
    emfplus_mode = _scan_emfplus_mode(data, 0, declared_bytes)
    return _HeaderInfo(
        source_format="emf",
        record_start=0,
        record_end=declared_bytes,
        bounds=bounds,
        pixel_size=pixel_size,
        emfplus_mode=emfplus_mode,
        device_pixels_per_mm=device_scale,
    )


def _parse_wmf_header(data: bytes, dpi: int, size_hint: tuple[int, int] | None) -> _HeaderInfo:
    """解析 placeable/standard WMF 头部、对象表大小与输出尺寸。"""
    reader = BoundedReader(data)
    record_start = 0
    bounds: Rect | None = None
    pixel_size = size_hint
    if len(reader) >= 22 and reader.u32(0) == _PLACEABLE_WMF_KEY:
        checksum = 0
        for offset in range(0, 20, 2):
            checksum ^= reader.u16(offset)
        if checksum != reader.u16(20):
            raise MetafileMalformedError("placeable WMF checksum does not match")
        bounds = Rect(reader.i16(6), reader.i16(8), reader.i16(10), reader.i16(12))
        units_per_inch = reader.u16(14)
        if units_per_inch == 0:
            raise MetafileMalformedError("placeable WMF units-per-inch must be nonzero")
        if pixel_size is None:
            pixel_size = (
                max(1, round(abs(bounds.width) * dpi / units_per_inch)),
                max(1, round(abs(bounds.height) * dpi / units_per_inch)),
            )
        record_start = 22
    if len(reader) < record_start + 18:
        raise MetafileMalformedError("standard WMF header is truncated")
    header = reader.subreader(record_start, 18)
    if header.u16(0) not in {1, 2} or header.u16(2) != 9:
        raise MetafileMalformedError("invalid standard WMF header")
    declared_bytes = header.u32(6) * 2
    if declared_bytes < 18 or declared_bytes > len(reader) - record_start:
        raise MetafileMalformedError(f"invalid WMF declared byte size: {declared_bytes}")
    object_count = header.u16(10)
    if object_count > MAX_OBJECTS:
        raise MetafileResourceLimitError(f"WMF object table exceeds max_objects={MAX_OBJECTS}")
    return _HeaderInfo(
        source_format="wmf",
        record_start=record_start + 18,
        record_end=record_start + declared_bytes,
        bounds=bounds,
        pixel_size=pixel_size,
        device_pixels_per_mm=(dpi / 25.4, dpi / 25.4),
        wmf_object_count=object_count,
    )


def parse_metafile(
    data: bytes,
    *,
    dpi: int = 144,
    size_hint: tuple[int, int] | None = None,
) -> MetafileDocument:
    """检测并解析 WMF/EMF，返回跨后端统一图元文档。"""
    if not isinstance(data, bytes):
        raise TypeError("metafile data must be bytes")
    if not data:
        raise MetafileMalformedError("metafile data must not be empty")
    if len(data) > MAX_METAFILE_BYTES:
        raise MetafileResourceLimitError(f"metafile exceeds max_metafile_bytes={MAX_METAFILE_BYTES}")
    if not isinstance(dpi, int) or dpi < 1 or dpi > 1200:
        raise ValueError("dpi must be an integer between 1 and 1200")
    if size_hint is not None and (
        len(size_hint) != 2 or not all(isinstance(value, int) for value in size_hint) or size_hint[0] <= 0 or size_hint[1] <= 0
    ):
        raise ValueError("size_hint must contain two positive integers")
    source_format = _detect_source_format(data)
    header = _parse_emf_header(data, dpi, size_hint) if source_format == "emf" else _parse_wmf_header(data, dpi, size_hint)
    if header.emfplus_mode == "only":
        raise MetafileUnsupportedError("EMF+ Only metafiles are not supported")
    diagnostics = _DiagnosticSink()
    playback = _Playback(header, diagnostics)
    if source_format == "emf":
        _play_emf(data, header, playback)
    else:
        _play_wmf(data, header, playback)
    return playback.finalize(size_hint)


def _pen_dash_pattern(style: int, width: float) -> tuple[float, ...]:
    """把常见 GDI pen style 转换为 device 单位虚线模式。"""
    unit = max(width, 1.0)
    return {
        1: (6.0 * unit, 3.0 * unit),
        2: (unit, 2.0 * unit),
        3: (6.0 * unit, 2.0 * unit, unit, 2.0 * unit),
        4: (6.0 * unit, 2.0 * unit, unit, 2.0 * unit, unit, 2.0 * unit),
        8: (unit, unit),
    }.get(style & 0xF, ())


def _make_pen(style: int, width: float, color_value: int, *, extended: bool = False) -> Pen:
    """从 LOGPEN/EXTLOGPEN 字段构造统一画笔。"""
    basic_style = style & 0xF
    cap = "square" if style & 0x100 else "flat" if style & 0x200 else "round"
    join = "bevel" if style & 0x1000 else "miter" if style & 0x2000 else "round"
    geometric = bool(style & 0x10000) if extended else abs(width) > 1.0
    normalized_width = max(abs(width), 1.0)
    return Pen(
        color=_parse_color(color_value),
        width=normalized_width,
        style=style,
        cosmetic=not geometric,
        null=basic_style == 5,
        cap=cap,
        join=join,
        dashes=_pen_dash_pattern(style, normalized_width),
    )


def _make_brush(style: int, color_value: int, hatch: int = 0, pattern: bytes | None = None) -> Brush:
    """从 LOGBRUSH 字段构造统一画刷。"""
    if style == 1:
        return Brush(kind="null")
    if style == 2:
        return Brush(kind="hatch", color=_parse_color(color_value), hatch=hatch)
    if style in {3, 5, 6, 7, 8}:
        return Brush(kind="pattern", color=_parse_color(color_value), hatch=hatch, pattern=pattern)
    return Brush(kind="solid", color=_parse_color(color_value), hatch=hatch)


def _decode_font_name(raw: bytes, *, wide: bool) -> str:
    """解码 LOGFONT face name 并移除结尾 NUL。"""
    if wide:
        if len(raw) % 2:
            raw = raw[:-1]
        text = raw.decode("utf-16le", errors="replace")
    else:
        text = raw.decode("cp1252", errors="replace")
    return text.split("\x00", 1)[0].strip() or "Arial"


def _parse_emf_font(record: BoundedReader) -> Font:
    """解析 EMR_EXTCREATEFONTINDIRECTW 的 LOGFONTW 首部。"""
    return Font(
        face_name=_decode_font_name(record.bytes(40, 64), wide=True),
        height=float(record.i32(12)),
        width=float(record.i32(16)),
        escapement=float(record.i32(20)),
        orientation=float(record.i32(24)),
        weight=record.i32(28),
        italic=bool(record.u8(32)),
        underline=bool(record.u8(33)),
        strikeout=bool(record.u8(34)),
        charset=record.u8(35),
    )


def _parse_wmf_font(payload: BoundedReader) -> Font:
    """解析 META_CREATEFONTINDIRECT 的 16 位 LOGFONT。"""
    if len(payload) < 18:
        raise MetafileMalformedError("WMF LOGFONT is truncated")
    face_size = min(32, max(0, len(payload) - 18))
    return Font(
        face_name=_decode_font_name(payload.bytes(18, face_size), wide=False),
        height=float(payload.i16(0)),
        width=float(payload.i16(2)),
        escapement=float(payload.i16(4)),
        orientation=float(payload.i16(6)),
        weight=payload.i16(8),
        italic=bool(payload.u8(10)),
        underline=bool(payload.u8(11)),
        strikeout=bool(payload.u8(12)),
        charset=payload.u8(13),
    )


def _charset_codec(charset: int) -> str:
    """把常见 Windows LOGFONT charset 映射为 Python codec。"""
    return {
        0: "cp1252",
        1: "cp1252",
        2: "cp1252",
        128: "shift_jis",
        129: "cp949",
        134: "gb18030",
        136: "big5",
        161: "cp1253",
        162: "cp1254",
        163: "cp1258",
        177: "cp1255",
        178: "cp1256",
        186: "cp1257",
        204: "cp1251",
        222: "cp874",
        238: "cp1250",
        255: "cp437",
    }.get(charset, "cp1252")


def _decode_ansi_text(raw: bytes, charset: int) -> str:
    """按当前 LOGFONT charset 尽力解码 GDI ANSI 文本。"""
    return raw.decode(_charset_codec(charset), errors="replace")


def _path_from_device_points(points: list[Point], *, close: bool) -> GraphicsPath:
    """从 device 坐标点列构造单个折线路径。"""
    if not points:
        return GraphicsPath(())
    builder = PathBuilder()
    builder.move_to(points[0])
    for point in points[1:]:
        builder.line_to(point)
    if close:
        builder.close()
    return builder.build()


def _read_emf_points(record: BoundedReader, count: int, offset: int, *, compact: bool) -> list[Point]:
    """读取 EMF PointL/PointS 数组并保留逻辑坐标。"""
    stride = 4 if compact else 8
    if count < 0 or count > MAX_POINTS_PER_RECORD or count > record.remaining(offset) // stride:
        raise MetafileMalformedError(f"EMF point array exceeds record boundary: count={count}")
    if compact:
        return [(float(record.i16(offset + index * 4)), float(record.i16(offset + index * 4 + 2))) for index in range(count)]
    return [(float(record.i32(offset + index * 8)), float(record.i32(offset + index * 8 + 4))) for index in range(count)]


def _read_wmf_points(payload: BoundedReader, count: int, offset: int) -> list[Point]:
    """读取 WMF PointS 数组并保留逻辑坐标。"""
    if count < 0 or count > MAX_POINTS_PER_RECORD or count > payload.remaining(offset) // 4:
        raise MetafileMalformedError(f"WMF point array exceeds record boundary: count={count}")
    return [(float(payload.i16(offset + index * 4)), float(payload.i16(offset + index * 4 + 2))) for index in range(count)]


def _emit_polybezier(playback: _Playback, points: list[Point], *, to: bool) -> None:
    """把 PolyBezier/PolyBezierTo 点列追加为三次路径。"""
    if not points:
        return
    if to:
        playback.polybezier_to(points)
        return
    builder = PathBuilder()
    builder.move_to(playback.map_point(points[0]))
    remaining = points[1:]
    if len(remaining) % 3:
        playback.warn("invalid_bezier_points", "PolyBezier point count is not a multiple of three")
        remaining = remaining[: len(remaining) - len(remaining) % 3]
    for index in range(0, len(remaining), 3):
        builder.cubic_to(
            playback.map_point(remaining[index]),
            playback.map_point(remaining[index + 1]),
            playback.map_point(remaining[index + 2]),
        )
    playback.emit_path(builder.build(), stroke=True, fill=False)


def _emit_text(
    playback: _Playback,
    *,
    text: str,
    reference: Point,
    options: int,
    bounds: Rect | None,
    advances: list[Point] | None,
) -> None:
    """把 GDI 文字和显式 advance 转换为统一文字命令。"""
    if not text:
        return
    if options & _ETO_GLYPH_INDEX:
        playback.warn("glyph_index_text", "glyph-index text was replaced with visible placeholder glyphs")
        text = "□" * len(text)
    update_current = bool(playback.state.text_align & _TA_UPDATECP)
    logical_origin = playback.state.current_position if update_current else reference
    origin = playback.map_point(logical_origin)
    positions: list[Point] = []
    advance_end: Point | None = None
    font = playback.state.font
    if advances:
        cursor_x, cursor_y = logical_origin
        for index, _character in enumerate(text):
            positions.append(playback.map_point((cursor_x, cursor_y)))
            if index < len(advances):
                cursor_x += advances[index][0]
                cursor_y += advances[index][1]
        advance_end = playback.map_point((cursor_x, cursor_y))
        if update_current:
            playback.state.current_position = (cursor_x, cursor_y)
    elif update_current:
        advance = measure_text_advance(font, text)
        angle = radians(font.escapement / 10.0)
        playback.state.current_position = (
            logical_origin[0] + advance * cos(angle),
            logical_origin[1] - advance * sin(angle),
        )
    mapped_height = vector_length(playback.map_vector((0.0, font.height or -12.0)))
    baseline_vector = playback.map_vector((1.0, 0.0))
    rotation = degrees(atan2(baseline_vector[1], baseline_vector[0])) + font.escapement / 10.0
    mapped_bounds: Rect | None = None
    if bounds is not None and (options & (_ETO_OPAQUE | _ETO_CLIPPED)):
        corners = [
            playback.map_point((bounds.left, bounds.top)),
            playback.map_point((bounds.right, bounds.top)),
            playback.map_point((bounds.right, bounds.bottom)),
            playback.map_point((bounds.left, bounds.bottom)),
        ]
        mapped_bounds = Rect(
            min(point[0] for point in corners),
            min(point[1] for point in corners),
            max(point[0] for point in corners),
            max(point[1] for point in corners),
        )
    clip = playback.state.clip
    if mapped_bounds is not None and options & _ETO_CLIPPED:
        clip = playback.clip_with_operation(ClipOperation(rectangle_path(mapped_bounds), "and"))
    playback.append_command(
        DrawTextCommand(
            text=text,
            origin=origin,
            positions=tuple(positions),
            font=font,
            font_height=max(mapped_height, 1.0),
            rotation=rotation,
            text_align=playback.state.text_align,
            color=playback.state.text_color,
            background_color=playback.state.background_color,
            opaque=bool(options & _ETO_OPAQUE) or playback.state.background_mode == 2,
            bounds=mapped_bounds,
            clip=clip,
            advance_end=advance_end,
        )
    )


def _append_image_command(
    playback: _Playback,
    record: BoundedReader,
    *,
    off_bmi: int,
    cb_bmi: int,
    off_bits: int,
    cb_bits: int,
    destination: Rect,
    source: Rect | None,
    rop: int,
    constant_alpha: int = 255,
    use_source_alpha: bool = False,
) -> None:
    """校验 DIB 子区间并追加统一位图命令。"""
    if cb_bmi <= 0 or cb_bits <= 0:
        playback.warn("empty_dib", "bitmap record does not contain a DIB header and pixel payload")
        return
    if rop not in _SUPPORTED_ROP3:
        playback.warn("unsupported_rop3", f"unsupported ROP3 will use source-over approximation: {rop:#x}")
    if (
        cb_bmi > MAX_EMBEDDED_BITMAP_BYTES
        or cb_bits > MAX_EMBEDDED_BITMAP_BYTES
        or cb_bmi + cb_bits > MAX_EMBEDDED_BITMAP_BYTES
    ):
        raise MetafileResourceLimitError(f"embedded bitmap exceeds max_embedded_bitmap_bytes={MAX_EMBEDDED_BITMAP_BYTES}")
    dib_header = record.bytes(off_bmi, cb_bmi)
    bits = record.bytes(off_bits, cb_bits)
    corners = (
        playback.map_point((destination.left, destination.top)),
        playback.map_point((destination.right, destination.top)),
        playback.map_point((destination.right, destination.bottom)),
        playback.map_point((destination.left, destination.bottom)),
    )
    playback.append_command(
        DrawImageCommand(
            dib_header=dib_header,
            bits=bits,
            destination=corners,
            source=source,
            rop=rop,
            stretch_mode=playback.state.stretch_mode,
            constant_alpha=max(0, min(constant_alpha, 255)),
            use_source_alpha=use_source_alpha,
            clip=playback.state.clip,
        )
    )


def _play_emf(data: bytes, header: _HeaderInfo, playback: _Playback) -> None:
    """按文件顺序验证并回放全部 EMF records。"""
    offset = header.record_start
    record_index = 0
    saw_eof = False
    while offset < header.record_end:
        if offset + 8 > header.record_end:
            raise MetafileMalformedError(f"truncated EMF record header at offset={offset}")
        record_type, record_size = struct.unpack_from("<II", data, offset)
        record_index += 1
        if record_index > MAX_RECORDS:
            raise MetafileResourceLimitError(f"EMF exceeds max_records={MAX_RECORDS}")
        if record_size < 8 or record_size % 4 or record_size > header.record_end - offset:
            raise MetafileMalformedError(f"invalid EMF record size at offset={offset}: {record_size}")
        playback.set_record_context(record_type, record_index, offset)
        record = BoundedReader(memoryview(data)[offset : offset + record_size], base_offset=offset)
        _handle_emf_record(record_type, record, playback)
        offset += record_size
        if record_type == 14:
            saw_eof = True
            break
    if not saw_eof:
        raise MetafileMalformedError("EMF record stream does not contain EMR_EOF")


def _play_wmf(data: bytes, header: _HeaderInfo, playback: _Playback) -> None:
    """按文件顺序验证并回放全部 WMF records。"""
    offset = header.record_start
    record_index = 0
    saw_eof = False
    while offset < header.record_end:
        if offset + 6 > header.record_end:
            raise MetafileMalformedError(f"truncated WMF record header at offset={offset}")
        size_words = struct.unpack_from("<I", data, offset)[0]
        function = struct.unpack_from("<H", data, offset + 4)[0]
        record_size = size_words * 2
        record_index += 1
        if record_index > MAX_RECORDS:
            raise MetafileResourceLimitError(f"WMF exceeds max_records={MAX_RECORDS}")
        if size_words < 3 or record_size > header.record_end - offset:
            raise MetafileMalformedError(f"invalid WMF record size at offset={offset}: words={size_words}")
        playback.set_record_context(function, record_index, offset)
        payload = BoundedReader(memoryview(data)[offset + 6 : offset + record_size], base_offset=offset + 6)
        _handle_wmf_record(function, payload, playback)
        offset += record_size
        if function == 0:
            saw_eof = True
            break
    if not saw_eof:
        raise MetafileMalformedError("WMF record stream does not contain META_EOF")


def _handle_emf_poly(record_type: int, record: BoundedReader, playback: _Playback) -> bool:
    """处理 EMF 32/16 位折线、多边形与贝塞尔记录。"""
    simple_types = {2, 3, 4, 5, 6, 85, 86, 87, 88, 89}
    poly_types = {7, 8, 90, 91}
    if record_type not in simple_types | poly_types:
        return False
    compact = record_type >= 85
    if record_type in simple_types:
        count = record.u32(24)
        playback.charge_points(count)
        points = _read_emf_points(record, count, 28, compact=compact)
        if record_type in {2, 5, 85, 88}:
            _emit_polybezier(playback, points, to=record_type in {5, 88})
            return True
        mapped = [playback.map_point(point) for point in points]
        if record_type in {6, 89}:
            playback.polyline_to(points)
            return True
        playback.emit_path(
            _path_from_device_points(mapped, close=record_type in {3, 86}),
            stroke=True,
            fill=record_type in {3, 86},
        )
        return True

    polygon_count = record.u32(24)
    total_count = record.u32(28)
    if polygon_count > MAX_POINTS_PER_RECORD or polygon_count > record.remaining(32) // 4:
        raise MetafileMalformedError(f"EMF polygon count array exceeds record boundary: count={polygon_count}")
    counts = [record.u32(32 + index * 4) for index in range(polygon_count)]
    if sum(counts) != total_count:
        raise MetafileMalformedError("EMF poly-polygon counts do not equal total point count")
    playback.charge_points(total_count)
    points_offset = 32 + polygon_count * 4
    points = _read_emf_points(record, total_count, points_offset, compact=compact)
    builder = PathBuilder()
    cursor = 0
    close = record_type in {8, 91}
    for count in counts:
        mapped = [playback.map_point(point) for point in points[cursor : cursor + count]]
        builder.extend(_path_from_device_points(mapped, close=close))
        cursor += count
    playback.emit_path(builder.build(), stroke=True, fill=close)
    return True


def _handle_emf_record(record_type: int, record: BoundedReader, playback: _Playback) -> None:
    """分派并执行单条 EMF record，未知绘图记录按规范跳过。"""
    if record_type in {1, 14, 70}:
        return
    if _handle_emf_poly(record_type, record, playback):
        return

    if record_type == 9:
        playback.state.window_extent = float(record.i32(8)), float(record.i32(12))
        return
    if record_type == 10:
        playback.state.window_origin = float(record.i32(8)), float(record.i32(12))
        return
    if record_type == 11:
        playback.state.viewport_extent = float(record.i32(8)), float(record.i32(12))
        return
    if record_type == 12:
        playback.state.viewport_origin = float(record.i32(8)), float(record.i32(12))
        return
    if record_type == 13:
        playback.state.brush_origin = float(record.i32(8)), float(record.i32(12))
        return
    if record_type == 17:
        playback.state.map_mode = record.i32(8)
        return
    if record_type == 18:
        playback.state.background_mode = record.u32(8)
        return
    if record_type == 19:
        playback.state.polygon_fill_mode = record.u32(8)
        return
    if record_type == 20:
        rop2 = record.u32(8)
        if not 1 <= rop2 <= 16:
            playback.warn("unsupported_rop2", f"invalid ROP2 will use COPYPEN approximation: {rop2}")
            rop2 = 13
        playback.state.rop2 = rop2
        return
    if record_type == 21:
        playback.state.stretch_mode = record.u32(8)
        return
    if record_type == 22:
        playback.state.text_align = record.u32(8)
        return
    if record_type == 24:
        playback.state.text_color = _parse_color(record.u32(8))
        return
    if record_type == 25:
        playback.state.background_color = _parse_color(record.u32(8))
        return
    if record_type == 26:
        offset = playback.map_vector((float(record.i32(8)), float(record.i32(12))))
        translation = Matrix(e=offset[0], f=offset[1])
        playback.state.clip = tuple(
            replace(operation, path=transform_path(operation.path, translation)) for operation in playback.state.clip
        )
        return
    if record_type == 27:
        point = float(record.i32(8)), float(record.i32(12))
        playback.move_to(point)
        return
    if record_type == 28:
        playback.reset_clip()
        return
    if record_type in {29, 30}:
        logical = _parse_rect_i32(record, 8)
        mapped = transform_path(rectangle_path(logical), playback.logical_matrix())
        playback.add_clip(mapped, _RGN_DIFF if record_type == 29 else _RGN_AND)
        return
    if record_type in {31, 32}:
        x_num, x_den = record.i32(8), record.i32(12)
        y_num, y_den = record.i32(16), record.i32(20)
        if x_den == 0 or y_den == 0:
            playback.warn("zero_scale_denominator", "viewport/window scale record has a zero denominator")
            return
        if record_type == 31:
            width, height = playback.state.viewport_extent
            playback.state.viewport_extent = width * x_num / x_den, height * y_num / y_den
        else:
            width, height = playback.state.window_extent
            playback.state.window_extent = width * x_num / x_den, height * y_num / y_den
        return
    if record_type == 33:
        playback.save_dc()
        return
    if record_type == 34:
        playback.restore_dc(record.i32(8))
        return
    if record_type == 35:
        playback.state.world_transform = _parse_xform(record, 8)
        return
    if record_type == 36:
        transform = _parse_xform(record, 8)
        mode = record.u32(32)
        if mode == 1:
            playback.state.world_transform = Matrix()
        elif mode == 2:
            playback.state.world_transform = playback.state.world_transform.then(transform)
        elif mode == 3:
            playback.state.world_transform = transform.then(playback.state.world_transform)
        elif mode == 4:
            playback.state.world_transform = transform
        else:
            playback.warn("unsupported_transform_mode", f"unsupported ModifyWorldTransform mode: {mode}")
        return
    if record_type == 37:
        playback.select_emf_handle(record.u32(8))
        return
    if record_type == 38:
        handle = record.u32(8)
        _validate_emf_handle(handle)
        playback.emf_objects[handle] = _make_pen(record.u32(12), float(record.i32(16)), record.u32(24))
        return
    if record_type == 39:
        handle = record.u32(8)
        _validate_emf_handle(handle)
        playback.emf_objects[handle] = _make_brush(record.u32(12), record.u32(16), record.u32(20))
        return
    if record_type == 40:
        playback.emf_objects.pop(record.u32(8), None)
        return
    if record_type == 57:
        playback.state.arc_direction = record.u32(8)
        return
    if record_type == 58:
        value = record.f32(8)
        if isfinite(value) and value > 0:
            playback.state.miter_limit = value
        else:
            playback.warn("invalid_miter_limit", f"invalid miter limit: {value}")
        return

    if record_type == 15:
        point = playback.map_point((float(record.i32(8)), float(record.i32(12))))
        color = _parse_color(record.u32(16))
        playback.append_command(
            DrawPathCommand(
                path=rectangle_path(Rect(point[0], point[1], point[0] + 1.0, point[1] + 1.0)),
                pen=Pen(null=True),
                brush=Brush(color=color),
                stroke=False,
                fill=True,
                fill_rule="evenodd",
                clip=playback.state.clip,
                rop2=13,
            )
        )
        return
    if record_type == 54:
        endpoint = float(record.i32(8)), float(record.i32(12))
        playback.line_to(endpoint)
        return
    if record_type in {42, 43, 44}:
        rect = _parse_rect_i32(record, 8)
        if record_type == 42:
            path = ellipse_path(rect)
        elif record_type == 44:
            path = round_rectangle_path(rect, abs(record.i32(24)) / 2.0, abs(record.i32(28)) / 2.0)
        else:
            path = rectangle_path(rect)
        playback.emit_logical_path(path, stroke=True, fill=True)
        return
    if record_type in {45, 46, 47, 55}:
        rect = _parse_rect_i32(record, 8)
        start = float(record.i32(24)), float(record.i32(28))
        end = float(record.i32(32)), float(record.i32(36))
        close_mode = "chord" if record_type == 46 else "pie" if record_type == 47 else "open"
        path = arc_path(rect, start, end, direction=playback.state.arc_direction, close_mode=close_mode)
        if record_type == 55:
            playback.connected_arc(path)
        else:
            playback.emit_logical_path(path, stroke=True, fill=close_mode != "open")
        return
    if record_type == 41:
        center = float(record.i32(8)), float(record.i32(12))
        radius = abs(float(record.u32(16)))
        start_angle = record.f32(20)
        sweep_angle = record.f32(24)
        from math import cos, radians, sin

        start = (
            center[0] + radius * cos(radians(start_angle)),
            center[1] - radius * sin(radians(start_angle)),
        )
        end_angle = start_angle + sweep_angle
        end = (
            center[0] + radius * cos(radians(end_angle)),
            center[1] - radius * sin(radians(end_angle)),
        )
        direction = 1 if sweep_angle >= 0 else 2
        playback.connected_arc(
            arc_path(
                Rect(center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius),
                start,
                end,
                direction=direction,
            ),
        )
        return

    if record_type == 59:
        playback.begin_path()
        return
    if record_type == 60:
        playback.end_path()
        return
    if record_type == 61:
        if playback.path_active and playback.path_builder.figure_open:
            playback.path_builder.close()
        else:
            playback.warn("close_figure_without_path", "CloseFigure ignored without an active path")
        return
    if record_type in {62, 63, 64}:
        path = playback.consume_path()
        if record_type == 63:
            path = close_open_subpaths(path)
        playback.emit_path(path, stroke=record_type in {63, 64}, fill=record_type in {62, 63})
        return
    if record_type == 67:
        path = playback.consume_path()
        if path.segments:
            playback.add_clip(path, record.u32(8))
        return
    if record_type == 68:
        playback.path_builder.clear()
        playback.path_active = False
        playback.path_ready = False
        return

    if record_type == 75:
        _handle_emf_region_clip(record, playback)
        return
    if record_type in {76, 77, 80, 81, 114}:
        _handle_emf_bitmap(record_type, record, playback)
        return
    if record_type == 82:
        handle = record.u32(8)
        _validate_emf_handle(handle)
        playback.emf_objects[handle] = _parse_emf_font(record)
        return
    if record_type in {83, 84}:
        _handle_emf_text(record_type, record, playback)
        return
    if record_type == 94:
        handle = record.u32(8)
        _validate_emf_handle(handle)
        cb_bmi, cb_bits = record.u32(20), record.u32(28)
        pattern = b""
        if cb_bmi + cb_bits <= MAX_EMBEDDED_BITMAP_BYTES:
            pattern = record.bytes(record.u32(16), cb_bmi) + record.bytes(record.u32(24), cb_bits)
        playback.emf_objects[handle] = Brush(kind="pattern", pattern=pattern)
        playback.warn("pattern_brush_approximation", "DIB pattern brush will use a solid-color approximation")
        return
    if record_type == 95:
        handle = record.u32(8)
        _validate_emf_handle(handle)
        style = record.u32(28)
        width = float(record.u32(32))
        pen = _make_pen(style, width, record.u32(40), extended=True)
        style_count = record.u32(48)
        if style_count and style_count <= record.remaining(52) // 4:
            dashes = tuple(max(1.0, float(record.u32(52 + index * 4))) for index in range(style_count))
            pen = replace(pen, dashes=dashes)
        playback.emf_objects[handle] = pen
        return
    if record_type == 120:
        playback.state.brush = replace(playback.state.brush, color=_parse_color(record.u32(8)))
        return
    if record_type == 121:
        playback.state.pen = replace(playback.state.pen, color=_parse_color(record.u32(8)))
        return

    harmless_state_records = {
        16,
        23,
        48,
        49,
        50,
        51,
        52,
        98,
        99,
        100,
        101,
        104,
        107,
        109,
        110,
        111,
        112,
        113,
        115,
        117,
        119,
    }
    if record_type in harmless_state_records:
        playback.warn("ignored_state_record", f"EMF state/control record was ignored: {record_type}", partial=False)
        return
    playback.warn("unsupported_emf_record", f"unsupported EMF record was skipped: {record_type}")


def _validate_emf_handle(handle: int) -> None:
    """拒绝保留 handle、stock handle 和超出固定对象预算的值。"""
    if handle == 0 or handle & 0x80000000 or handle > MAX_OBJECTS:
        raise MetafileResourceLimitError(f"invalid or excessive EMF object handle: {handle}")


def _handle_emf_region_clip(record: BoundedReader, playback: _Playback) -> None:
    """解析 EMR_EXTSELECTCLIPRGN 的矩形 region 数据。"""
    data_size = record.u32(8)
    mode = record.u32(12)
    if data_size == 0:
        if mode == _RGN_COPY:
            playback.reset_clip()
        else:
            playback.warn("empty_clip_region", f"empty clip region used with combine mode {mode}")
        return
    if data_size > record.remaining(16) or data_size < 32:
        raise MetafileMalformedError(f"EMF region data exceeds record boundary: size={data_size}")
    region = record.subreader(16, data_size)
    header_size = region.u32(0)
    region_type = region.u32(4)
    rectangle_count = region.u32(8)
    region_bytes = region.u32(12)
    if header_size < 32 or region_type != 1:
        raise MetafileMalformedError("EMF region header is invalid")
    if rectangle_count > MAX_POINTS_PER_RECORD or region_bytes > region.remaining(header_size):
        raise MetafileMalformedError("EMF region rectangle data exceeds its declared boundary")
    if rectangle_count > region.remaining(header_size) // 16:
        raise MetafileMalformedError("EMF region rectangle count exceeds record boundary")
    playback.charge_points(rectangle_count * 4)
    builder = PathBuilder()
    for index in range(rectangle_count):
        rect = _parse_rect_i32(region, header_size + index * 16)
        builder.extend(transform_path(rectangle_path(rect), playback.logical_matrix()))
    path = builder.build()
    if path.segments:
        playback.add_clip(path, mode)


def _handle_emf_bitmap(record_type: int, record: BoundedReader, playback: _Playback) -> None:
    """解析常见 EMF DIB、BitBlt、StretchBlt 与 AlphaBlend 记录。"""
    if record_type in {76, 77}:
        destination = Rect(
            float(record.i32(24)),
            float(record.i32(28)),
            float(record.i32(24) + record.i32(32)),
            float(record.i32(28) + record.i32(36)),
        )
        source_width = record.i32(100) if record_type == 77 and len(record) >= 108 else record.i32(32)
        source_height = record.i32(104) if record_type == 77 and len(record) >= 108 else record.i32(36)
        source = Rect(
            float(record.i32(44)),
            float(record.i32(48)),
            float(record.i32(44) + source_width),
            float(record.i32(48) + source_height),
        )
        _append_image_command(
            playback,
            record,
            off_bmi=record.u32(84),
            cb_bmi=record.u32(88),
            off_bits=record.u32(92),
            cb_bits=record.u32(96),
            destination=destination,
            source=source,
            rop=record.u32(40),
        )
        return
    if record_type == 81:
        destination = Rect(
            float(record.i32(24)),
            float(record.i32(28)),
            float(record.i32(24) + record.i32(72)),
            float(record.i32(28) + record.i32(76)),
        )
        source = Rect(
            float(record.i32(32)),
            float(record.i32(36)),
            float(record.i32(32) + record.i32(40)),
            float(record.i32(36) + record.i32(44)),
        )
        _append_image_command(
            playback,
            record,
            off_bmi=record.u32(48),
            cb_bmi=record.u32(52),
            off_bits=record.u32(56),
            cb_bits=record.u32(60),
            destination=destination,
            source=source,
            rop=record.u32(68),
        )
        return
    if record_type == 80:
        width, height = record.i32(40), record.i32(44)
        destination = Rect(
            float(record.i32(24)),
            float(record.i32(28)),
            float(record.i32(24) + width),
            float(record.i32(28) + height),
        )
        source = Rect(
            float(record.i32(32)),
            float(record.i32(36)),
            float(record.i32(32) + width),
            float(record.i32(36) + height),
        )
        _append_image_command(
            playback,
            record,
            off_bmi=record.u32(48),
            cb_bmi=record.u32(52),
            off_bits=record.u32(56),
            cb_bits=record.u32(60),
            destination=destination,
            source=source,
            rop=_SRCCOPY,
        )
        return
    destination = Rect(
        float(record.i32(24)),
        float(record.i32(28)),
        float(record.i32(24) + record.i32(32)),
        float(record.i32(28) + record.i32(36)),
    )
    source = Rect(
        float(record.i32(44)),
        float(record.i32(48)),
        float(record.i32(44) + record.i32(100)),
        float(record.i32(48) + record.i32(104)),
    )
    _append_image_command(
        playback,
        record,
        off_bmi=record.u32(84),
        cb_bmi=record.u32(88),
        off_bits=record.u32(92),
        cb_bits=record.u32(96),
        destination=destination,
        source=source,
        rop=_SRCCOPY,
        constant_alpha=record.u8(42),
        use_source_alpha=bool(record.u8(43) & 1),
    )


def _handle_emf_text(record_type: int, record: BoundedReader, playback: _Playback) -> None:
    """解析 EMR_EXTTEXTOUTA/W 文字、bounds 与显式 advance。"""
    if len(record) < 76:
        raise MetafileMalformedError("EMF ExtTextOut record is truncated")
    reference = float(record.i32(36)), float(record.i32(40))
    character_count = record.u32(44)
    if character_count > MAX_POINTS_PER_RECORD:
        raise MetafileResourceLimitError(f"EMF text exceeds max characters={MAX_POINTS_PER_RECORD}")
    string_offset = record.u32(48)
    options = record.u32(52)
    bounds = _parse_rect_i32(record, 56)
    dx_offset = record.u32(72)
    char_bytes = 2 if record_type == 84 else 1
    raw = record.bytes(string_offset, character_count * char_bytes)
    text = (
        raw.decode("utf-16le", errors="replace") if record_type == 84 else _decode_ansi_text(raw, playback.state.font.charset)
    )
    advances: list[Point] | None = None
    if dx_offset:
        values_per_character = 2 if options & _ETO_PDY else 1
        value_count = character_count * values_per_character
        if value_count > record.remaining(dx_offset) // 4:
            raise MetafileMalformedError("EMF ExtTextOut advance array exceeds record boundary")
        advances = []
        for index in range(character_count):
            dx = float(record.i32(dx_offset + index * values_per_character * 4))
            dy = float(record.i32(dx_offset + (index * values_per_character + 1) * 4)) if values_per_character == 2 else 0.0
            advances.append((dx, dy))
    playback.charge_points(character_count)
    _emit_text(
        playback,
        text=text,
        reference=reference,
        options=options,
        bounds=bounds,
        advances=advances,
    )


def _handle_wmf_record(function: int, payload: BoundedReader, playback: _Playback) -> None:
    """分派并执行单条 WMF record，未知绘图记录按规范跳过。"""
    if function == 0:
        return
    if function == 0x0201:
        playback.state.background_color = _parse_color(payload.u32(0))
        return
    if function == 0x0102:
        playback.state.background_mode = payload.u16(0)
        return
    if function == 0x0103:
        playback.state.map_mode = payload.i16(0)
        return
    if function == 0x0104:
        rop2 = payload.u16(0)
        if not 1 <= rop2 <= 16:
            playback.warn("unsupported_rop2", f"invalid WMF ROP2 will use COPYPEN approximation: {rop2}")
            rop2 = 13
        playback.state.rop2 = rop2
        return
    if function == 0x0106:
        playback.state.polygon_fill_mode = payload.u16(0)
        return
    if function == 0x0107:
        playback.state.stretch_mode = payload.u16(0)
        return
    if function == 0x0209:
        playback.state.text_color = _parse_color(payload.u32(0))
        return
    if function == 0x020B:
        playback.state.window_origin = float(payload.i16(2)), float(payload.i16(0))
        return
    if function == 0x020C:
        playback.state.window_extent = float(payload.i16(2)), float(payload.i16(0))
        return
    if function == 0x020D:
        playback.state.viewport_origin = float(payload.i16(2)), float(payload.i16(0))
        return
    if function == 0x020E:
        playback.state.viewport_extent = float(payload.i16(2)), float(payload.i16(0))
        return
    if function == 0x020F:
        x, y = playback.state.window_origin
        playback.state.window_origin = x + payload.i16(2), y + payload.i16(0)
        return
    if function == 0x0211:
        x, y = playback.state.viewport_origin
        playback.state.viewport_origin = x + payload.i16(2), y + payload.i16(0)
        return
    if function in {0x0410, 0x0412}:
        y_num, y_den, x_num, x_den = payload.i16(0), payload.i16(2), payload.i16(4), payload.i16(6)
        if x_den == 0 or y_den == 0:
            playback.warn("zero_scale_denominator", "WMF viewport/window scale record has a zero denominator")
            return
        if function == 0x0410:
            width, height = playback.state.window_extent
            playback.state.window_extent = width * x_num / x_den, height * y_num / y_den
        else:
            width, height = playback.state.viewport_extent
            playback.state.viewport_extent = width * x_num / x_den, height * y_num / y_den
        return
    if function == 0x001E:
        playback.save_dc()
        return
    if function == 0x0127:
        playback.restore_dc(payload.i16(0))
        return
    if function == 0x012E:
        playback.state.text_align = payload.u16(0)
        return
    if function == 0x012D:
        playback.select_wmf_handle(payload.u16(0))
        return
    if function == 0x01F0:
        handle = payload.u16(0)
        if handle < len(playback.wmf_objects):
            playback.wmf_objects[handle] = None
        return
    if function == 0x02FA:
        playback.create_wmf_object(_make_pen(payload.u16(0), float(payload.i16(2)), payload.u32(6)))
        return
    if function == 0x02FC:
        playback.create_wmf_object(_make_brush(payload.u16(0), payload.u32(2), payload.u16(6)))
        return
    if function == 0x02FB:
        playback.create_wmf_object(_parse_wmf_font(payload))
        return

    if function == 0x0214:
        point = float(payload.i16(2)), float(payload.i16(0))
        playback.move_to(point)
        return
    if function == 0x0213:
        endpoint = float(payload.i16(2)), float(payload.i16(0))
        playback.line_to(endpoint)
        return
    if function in {0x0324, 0x0325}:
        count = payload.u16(0)
        playback.charge_points(count)
        points = _read_wmf_points(payload, count, 2)
        playback.emit_path(
            _path_from_device_points([playback.map_point(point) for point in points], close=function == 0x0324),
            stroke=True,
            fill=function == 0x0324,
        )
        return
    if function == 0x0538:
        polygon_count = payload.u16(0)
        if polygon_count > MAX_POINTS_PER_RECORD or polygon_count > payload.remaining(2) // 2:
            raise MetafileMalformedError("WMF PolyPolygon count array exceeds record boundary")
        counts = [payload.u16(2 + index * 2) for index in range(polygon_count)]
        total_count = sum(counts)
        playback.charge_points(total_count)
        points = _read_wmf_points(payload, total_count, 2 + polygon_count * 2)
        builder = PathBuilder()
        cursor = 0
        for count in counts:
            builder.extend(
                _path_from_device_points(
                    [playback.map_point(point) for point in points[cursor : cursor + count]],
                    close=True,
                )
            )
            cursor += count
        playback.emit_path(builder.build(), stroke=True, fill=True)
        return
    if function == 0x061C:
        rect = Rect(float(payload.i16(10)), float(payload.i16(8)), float(payload.i16(6)), float(payload.i16(4)))
        path = round_rectangle_path(rect, abs(payload.i16(2)) / 2.0, abs(payload.i16(0)) / 2.0)
        playback.emit_logical_path(path, stroke=True, fill=True)
        return
    if function in {0x0418, 0x041B}:
        rect = Rect(float(payload.i16(6)), float(payload.i16(4)), float(payload.i16(2)), float(payload.i16(0)))
        if function == 0x0418:
            path = ellipse_path(rect)
        else:
            path = rectangle_path(rect)
        playback.emit_logical_path(path, stroke=True, fill=True)
        return
    if function in {0x0817, 0x081A, 0x0830}:
        end = float(payload.i16(2)), float(payload.i16(0))
        start = float(payload.i16(6)), float(payload.i16(4))
        rect = Rect(float(payload.i16(14)), float(payload.i16(12)), float(payload.i16(10)), float(payload.i16(8)))
        close_mode = "pie" if function == 0x081A else "chord" if function == 0x0830 else "open"
        playback.emit_logical_path(
            arc_path(rect, start, end, direction=2, close_mode=close_mode),
            stroke=True,
            fill=close_mode != "open",
        )
        return
    if function == 0x041F:
        point = playback.map_point((float(payload.i16(6)), float(payload.i16(4))))
        color = _parse_color(payload.u32(0))
        playback.append_command(
            DrawPathCommand(
                path=rectangle_path(Rect(point[0], point[1], point[0] + 1.0, point[1] + 1.0)),
                pen=Pen(null=True),
                brush=Brush(color=color),
                stroke=False,
                fill=True,
                fill_rule="evenodd",
                clip=playback.state.clip,
                rop2=13,
            )
        )
        return
    if function in {0x0415, 0x0416}:
        rect = Rect(float(payload.i16(6)), float(payload.i16(4)), float(payload.i16(2)), float(payload.i16(0)))
        playback.add_clip(
            transform_path(rectangle_path(rect), playback.logical_matrix()),
            _RGN_DIFF if function == 0x0415 else _RGN_AND,
        )
        return

    if function in {0x0521, 0x0A32}:
        _handle_wmf_text(function, payload, playback)
        return
    if function == 0x0F43:
        _handle_wmf_stretchdib(payload, playback)
        return
    if function in {0x0940, 0x0B41}:
        playback.warn("unsupported_wmf_dib_record", f"WMF DIB record will be skipped: {function:#x}")
        return

    harmless_records = {
        0x0108,
        0x020A,
        0x0231,
        0x0626,
        0x0234,
        0x0035,
        0x0436,
        0x0037,
        0x0139,
    }
    if function in harmless_records:
        playback.warn("ignored_wmf_state_record", f"WMF state/control record was ignored: {function:#x}", partial=False)
        return
    playback.warn("unsupported_wmf_record", f"unsupported WMF record was skipped: {function:#x}")


def _handle_wmf_text(function: int, payload: BoundedReader, playback: _Playback) -> None:
    """解析 WMF TextOut/ExtTextOut 字符串、矩形和 advance。"""
    if function == 0x0521:
        count = payload.u16(0)
        if count > payload.remaining(2):
            raise MetafileMalformedError("WMF TextOut string exceeds record boundary")
        raw = payload.bytes(2, count)
        coordinate_offset = 2 + count + (count & 1)
        reference = float(payload.i16(coordinate_offset + 2)), float(payload.i16(coordinate_offset))
        playback.charge_points(count)
        _emit_text(
            playback,
            text=_decode_ansi_text(raw, playback.state.font.charset),
            reference=reference,
            options=0,
            bounds=None,
            advances=None,
        )
        return
    reference = float(payload.i16(2)), float(payload.i16(0))
    count = payload.u16(4)
    options = payload.u16(6)
    offset = 8
    bounds: Rect | None = None
    if options & (_ETO_OPAQUE | _ETO_CLIPPED):
        bounds = _parse_rect_i16(payload, offset)
        offset += 8
    if count > payload.remaining(offset):
        raise MetafileMalformedError("WMF ExtTextOut string exceeds record boundary")
    raw = payload.bytes(offset, count)
    offset += count + (count & 1)
    advances: list[Point] | None = None
    if payload.remaining(offset) >= count * 2:
        advances = [(float(payload.i16(offset + index * 2)), 0.0) for index in range(count)]
    playback.charge_points(count)
    _emit_text(
        playback,
        text=_decode_ansi_text(raw, playback.state.font.charset),
        reference=reference,
        options=options,
        bounds=bounds,
        advances=advances,
    )


def _handle_wmf_stretchdib(payload: BoundedReader, playback: _Playback) -> None:
    """解析 META_STRETCHDIB 的内联 DIB 与源/目标矩形。"""
    if len(payload) < 22:
        raise MetafileMalformedError("WMF StretchDIB record is truncated")
    rop = payload.u32(0)
    source_height, source_width = payload.i16(6), payload.i16(8)
    source_y, source_x = payload.i16(10), payload.i16(12)
    destination_height, destination_width = payload.i16(14), payload.i16(16)
    destination_y, destination_x = payload.i16(18), payload.i16(20)
    dib = payload.bytes(22, payload.remaining(22))
    if len(dib) < 12:
        raise MetafileMalformedError("WMF StretchDIB payload does not contain a DIB header")
    header_size = struct.unpack_from("<I", dib, 0)[0]
    if header_size < 12 or header_size > len(dib):
        raise MetafileMalformedError(f"invalid WMF DIB header size: {header_size}")
    bits_offset = _dib_bits_offset(dib)
    if bits_offset > len(dib):
        raise MetafileMalformedError("WMF DIB pixel offset exceeds record boundary")
    header = dib[:bits_offset]
    bits = dib[bits_offset:]
    corners = (
        playback.map_point((float(destination_x), float(destination_y))),
        playback.map_point((float(destination_x + destination_width), float(destination_y))),
        playback.map_point((float(destination_x + destination_width), float(destination_y + destination_height))),
        playback.map_point((float(destination_x), float(destination_y + destination_height))),
    )
    playback.append_command(
        DrawImageCommand(
            dib_header=header,
            bits=bits,
            destination=corners,
            source=Rect(
                float(source_x),
                float(source_y),
                float(source_x + source_width),
                float(source_y + source_height),
            ),
            rop=rop,
            stretch_mode=playback.state.stretch_mode,
            clip=playback.state.clip,
        )
    )


def _dib_bits_offset(dib: bytes) -> int:
    """根据 DIB header、bit depth 和 palette 推导像素起始偏移。"""
    reader = BoundedReader(dib)
    header_size = reader.u32(0)
    if header_size == 12:
        bit_count = reader.u16(10)
        palette_entries = 1 << bit_count if bit_count <= 8 else 0
        return header_size + palette_entries * 3
    if header_size < 40:
        return header_size
    bit_count = reader.u16(14)
    compression = reader.u32(16)
    colors_used = reader.u32(32)
    palette_entries = colors_used or ((1 << bit_count) if bit_count <= 8 else 0)
    bitfields = 12 if compression == 3 and header_size == 40 else 16 if compression == 6 and header_size == 40 else 0
    return header_size + bitfields + palette_entries * 4


__all__ = ["parse_metafile"]
