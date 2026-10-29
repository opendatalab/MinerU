"""Tasks 与本地执行服务共用的结构化错误，不依赖 HTTP 框架。"""

from __future__ import annotations

from typing import NoReturn

from .contracts import ErrorDetail


class ApiServerError(Exception):
    """在共享任务业务边界中携带 HTTP 状态和结构化错误内容。"""

    def __init__(self, status_code: int, error: ErrorDetail) -> None:
        """保存调用方的状态与错误字段，异常文本沿用公开错误信息。"""
        super().__init__(error.message)
        self.status_code = status_code
        self.error = error


def raise_api_error(
    status_code: int,
    *,
    code: str,
    message: str,
    param: str | None = None,
) -> NoReturn:
    """使用稳定错误码、说明和可选参数路径构造公共错误，保留 HTTP 状态。"""
    raise ApiServerError(
        status_code,
        ErrorDetail(
            code=code,
            message=message,
            param=param,
        ),
    )


def invalid_task_request(
    message: str, *, code: str = "invalid_request", status_code: int = 400, param: str | None = None
) -> NoReturn:
    """把协议校验错误转换成两类服务共用的 V1 error envelope。"""
    raise_api_error(
        status_code,
        code=code,
        message=message,
        param=param,
    )


__all__ = ["ApiServerError", "invalid_task_request", "raise_api_error"]
