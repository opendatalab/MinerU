"""启动只调用共同 Tasks 契约的同步包装服务。"""

from __future__ import annotations

from pathlib import Path

import typer

from ...utils.i18n import t
from ..errors import exit_with_message


def file_parse_server_cmd(
    api_url: str = typer.Option(..., "--api-url", envvar="MINERU_API_URL", help=t("Upstream Tasks API base URL")),
    api_key: str | None = typer.Option(None, "--api-key", envvar="MINERU_API_KEY", help=t("Upstream Bearer API key")),
    host: str = typer.Option("127.0.0.1", "--host", help=t("Server host")),
    port: int = typer.Option(8003, "--port", min=1, max=65535, help=t("Server port")),
    state_dir: Path | None = typer.Option(None, "--state-dir", help=t("File parse recovery state directory")),
    retention_seconds: int = typer.Option(
        86400, "--retention-seconds", min=0, help=t("Terminal retention seconds (default 86400; 0 disables collection)")
    ),
) -> None:
    """惰性加载包装应用，不启动本地解析模型或继承调用者的认证头。"""
    import uvicorn
    from filelock import Timeout

    from ..file_parse import FileParseSettings, create_app

    try:
        settings = FileParseSettings(api_url, api_key, state_dir, retention_seconds)
    except ValueError as exc:
        exit_with_message("invalid_request", str(exc))
    try:
        uvicorn.run(create_app(settings), host=host, port=port)
    except Timeout:
        exit_with_message("invalid_request", "Another file-parse-server is using this state directory")


__all__ = ["file_parse_server_cmd"]
