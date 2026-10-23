"""Tasks 共用的轻量格式与查询路由，不导入解析或渲染实现。"""

from typing import Literal
from urllib.parse import quote

TaskOutputFormat = Literal["default", "docx", "html", "latex"]
TASK_OUTPUT_FORMATS: tuple[TaskOutputFormat, ...] = ("default", "docx", "html", "latex")
TaskStatus = Literal["waiting-file", "pending", "running", "completed", "partial", "failed", "canceled"]
TaskFileState = Literal["waiting-file", "pending", "running", "completed", "failed", "canceled"]


def task_status_url(api_url: str, task_id: str) -> str:
    """保留 API 部署前缀并编码任务标识，拒绝会改变固定路由的点路径。"""
    if not task_id or task_id in {".", ".."}:
        raise ValueError("Invalid task_id")
    return api_url.rstrip("/") + "/v1/tasks/" + quote(task_id, safe="")


__all__ = ["TASK_OUTPUT_FORMATS", "TaskFileState", "TaskOutputFormat", "TaskStatus", "task_status_url"]
