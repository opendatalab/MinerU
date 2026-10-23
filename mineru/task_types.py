"""Tasks 客户端与服务端共用的轻量格式定义，不导入解析或渲染实现。"""

from typing import Literal

TaskOutputFormat = Literal["default", "docx", "html", "latex"]
TASK_OUTPUT_FORMATS: tuple[TaskOutputFormat, ...] = ("default", "docx", "html", "latex")

__all__ = ["TASK_OUTPUT_FORMATS", "TaskOutputFormat"]
