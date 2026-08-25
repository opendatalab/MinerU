# Copyright (c) Opendatalab. All rights reserved.
"""`mineru-kit router` 与过渡 `mineru-router` 的统一 Typer 入口。"""

from __future__ import annotations

from typing import cast

import typer
import uvicorn

from ...types import SERVER_TIERS, ServerTier
from ...utils.stdio import configure_standard_streams
from ..errors import exit_with_message
from .app import create_app
from .workers import RouterSettings


def router_cmd(
    host: str = typer.Option("127.0.0.1", "--host", help="Server host"),
    port: int = typer.Option(8002, "--port", help="Server port"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
    upstream_url: list[str] | None = typer.Option(
        None,
        "--upstream-url",
        help="Existing MinerU V1 API base URL; repeat to add upstreams",
    ),
    local_gpus: str = typer.Option("auto", "--local-gpus", help="Local workers: auto, none, or GPU CSV"),
    worker_host: str = typer.Option("127.0.0.1", "--worker-host", help="Host for managed api-server workers"),
    worker_tier: str = typer.Option("standard", "--worker-tier", help="Managed worker tier: flash, basic, standard"),
    worker_concurrency: int = typer.Option(1, "--worker-concurrency", help="Concurrency per managed worker"),
    preload_models: bool = typer.Option(False, "--preload-models", help="Preload models in managed workers"),
) -> None:
    """启动只暴露 MinerU V1 API 的独立 Router 服务。"""
    if worker_tier not in SERVER_TIERS:
        exit_with_message(
            "invalid_request",
            f"Unsupported worker tier '{worker_tier}'. Supported tiers: {', '.join(SERVER_TIERS)}",
            "worker_tier",
        )
    try:
        settings = RouterSettings(
            upstream_urls=tuple(upstream_url or ()),
            local_gpus=local_gpus,
            worker_host=worker_host,
            worker_tier=cast(ServerTier, worker_tier),
            worker_concurrency=worker_concurrency,
            preload_models=preload_models,
        )
    except ValueError as exc:
        exit_with_message("invalid_request", str(exc))
    settings.apply_to_env()
    if reload:
        uvicorn.run(
            "mineru.kit.router.app:create_app_from_env",
            host=host,
            port=port,
            reload=True,
            factory=True,
        )
        return
    uvicorn.run(create_app(settings), host=host, port=port, reload=False)


def main() -> None:
    """配置标准流后运行过渡 `mineru-router` 单命令入口。"""
    configure_standard_streams()
    typer.run(router_cmd)


__all__ = ["main", "router_cmd"]
