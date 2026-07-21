from __future__ import annotations

from typing import Any

import typer


def _load_old_router() -> Any:
    """懒加载旧 router 实现，避免 mineru-kit 其他子命令提前加载重依赖。"""
    from ...cli_old import router as old_router

    return old_router


def router_cmd(
    ctx: typer.Context,
    host: str = typer.Option("127.0.0.1", "--host", help="Server host"),
    port: int = typer.Option(8002, "--port", help="Server port"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
    allow_public_http_client: bool = typer.Option(
        False,
        "--allow-public-http-client",
        help="Allow *-http-client backends when binding to a public host",
    ),
    upstream_url: list[str] | None = typer.Option(
        None,
        "--upstream-url",
        help="Existing MinerU FastAPI base URL; repeat to add multiple upstreams",
    ),
    local_gpus: str = typer.Option("auto", "--local-gpus", help="Local GPU workers: auto, none, or CSV such as 0,1,2"),
    worker_host: str = typer.Option("127.0.0.1", "--worker-host", help="Host for router-managed API workers"),
    enable_vlm_preload: bool = typer.Option(
        False,
        "--enable-vlm-preload",
        help="Preload the local VLM model in router-managed API workers",
    ),
) -> None:
    """Start the MinerU router service."""
    upstream_args = [item for url in upstream_url or [] for item in ("--upstream-url", url)]
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
        *list(ctx.args),
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
