# Copyright (c) Opendatalab. All rights reserved.
"""`mineru-kit router` 与过渡 `mineru-router` 的统一 Typer 入口。"""

from __future__ import annotations

from typing import cast

import typer
import uvicorn

from ...types import SERVER_TIERS, ServerTier
from ...utils.i18n import t
from ...utils.logger import configure_global_log_level
from ...utils.stdio import configure_standard_streams
from ..errors import exit_with_message
from .app import create_app
from .workers import RouterSettings


def router_cmd(
    host: str = typer.Option("127.0.0.1", "--host", help=t("Server host")),
    port: int = typer.Option(8002, "--port", help=t("Server port")),
    reload: bool = typer.Option(False, "--reload", help=t("Enable auto-reload")),
    upstream_url: list[str] | None = typer.Option(
        None,
        "--upstream-url",
        help=t("Existing MinerU V1 API base URL; repeat to add upstreams"),
    ),
    local_gpus: str = typer.Option("auto", "--local-gpus", help=t("Local workers: auto, none, or GPU CSV")),
    worker_host: str = typer.Option("127.0.0.1", "--worker-host", help=t("Host for managed api-server workers")),
    worker_tier: str = typer.Option("standard", "--worker-tier", help=t("Managed worker tier: flash, basic, standard")),
    worker_concurrency: int = typer.Option(1, "--worker-concurrency", help=t("Concurrency per managed worker")),
    preload_models: bool = typer.Option(False, "--preload-models", help=t("Preload models in managed workers")),
) -> None:
    """启动只暴露 MinerU V1 API 的独立 Router 服务。"""
    if worker_tier not in SERVER_TIERS:
        exit_with_message(
            "invalid_request",
            t(
                "Unsupported worker tier '{tier}'. Supported tiers: {tiers}",
                tier=worker_tier,
                tiers=", ".join(SERVER_TIERS),
            ),
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
    configure_global_log_level()
    app = typer.Typer(add_completion=False)
    app.command(help=t("Start a standalone Router service exposing only the MinerU V1 API."))(router_cmd)
    app()


__all__ = ["main", "router_cmd"]
