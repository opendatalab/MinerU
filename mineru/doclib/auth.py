"""Doclib 本机 HTTP 接口统一 Bearer 凭证校验。"""

from __future__ import annotations

import secrets

from pydantic import SecretStr
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from ..config import validate_doclib_token
from ..errors import MineruError, error_response


class DoclibAuthMiddleware:
    """在路由和业务依赖执行前校验凭证，覆盖状态、文档和关闭等全部 HTTP 接口。"""

    def __init__(self, app: ASGIApp, *, token: SecretStr) -> None:
        """保存当前实例的凭证快照，配置更新须重启后生效。"""
        self.app = app
        self._token = validate_doclib_token(token.get_secret_value()).encode("ascii")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """拒绝缺失、重复或不匹配的授权头，认证失败不消费请求正文。"""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request = Request(scope)
        authorization = request.headers.getlist("authorization")
        scheme, _, value = authorization[0].partition(" ") if len(authorization) == 1 else ("", "", "")
        if scheme.lower() != "bearer" or not secrets.compare_digest(value.encode("utf-8"), self._token):
            response = JSONResponse(
                error_response(MineruError("invalid_doclib_token", "A valid Doclib Bearer token is required.")),
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


__all__ = ["DoclibAuthMiddleware"]
