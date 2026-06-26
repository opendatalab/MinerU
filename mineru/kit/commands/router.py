from __future__ import annotations

from typing import Any

import typer


def _load_old_router() -> Any:
    """懒加载旧 router 实现，避免 mineru-kit 其他子命令提前加载重依赖。"""
    from ...cli_old import router as old_router

    return old_router


def router_cmd(
    *,
    ctx: typer.Context | None = None,
    host: str = "127.0.0.1",
    port: int = 8002,
    reload: bool = False,
    allow_public_http_client: bool = False,
    upstream_urls: list[str] | None = None,
    local_gpus: str = "auto",
    worker_host: str = "127.0.0.1",
    enable_vlm_preload: bool = False,
) -> None:
    """通过 mineru-kit 入口转发 router 参数，避免当前入口继续依赖旧脚本名。"""
    upstream_args = [
        item
        for upstream_url in upstream_urls or []
        for item in ("--upstream-url", upstream_url)
    ]
    args = [
        "--host",
        host,
        "--port",
        str(port),
        *(["--reload"] if reload else []),
        *(["--allow-public-http-client"] if allow_public_http_client else []),
        *upstream_args,
        "--local-gpus",
        local_gpus,
        "--worker-host",
        worker_host,
        *(["--enable-vlm-preload", "true"] if enable_vlm_preload else []),
        *(list(ctx.args) if ctx is not None else []),
    ]
    try:
        old_router = _load_old_router()
        old_router.main.main(
            args=args,
            prog_name="mineru-kit router",
            standalone_mode=False,
        )
    except SystemExit as exc:
        assert isinstance(exc.code, int)
        raise typer.Exit(exc.code) from None
