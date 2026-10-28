from __future__ import annotations

import click
import typer

from ...parser import api_server as parser_api_server
from ...utils.i18n import t
from ...utils.logger import configure_global_log_level
from ...utils.stdio import configure_standard_streams
from ..errors import exit_with_message


def api_server_cmd(
    host: str = typer.Option("127.0.0.1", "--host", help=t("Server host")),
    port: int = typer.Option(8000, "--port", help=t("Server port")),
    upload_dir: str = typer.Option("", "--upload-dir", help=t("Upload directory")),
    tier: str = typer.Option(
        "auto", "--tier", help=t("PDF/image tier ceiling: flash, basic, standard, advanced; or auto")
    ),
    retention_seconds: int | None = typer.Option(
        None,
        "--retention-seconds",
        min=0,
        envvar="MINERU_API_RETENTION_SECONDS",
        help=t("Terminal retention seconds (default 86400; 0 disables collection)"),
    ),
    concurrency: int = typer.Option(1, "--concurrency", help=t("Maximum concurrent parse jobs")),
    url_timeout: int = typer.Option(60, "--url-timeout", help=t("Timeout for URL source downloads")),
    allow_http_source: bool = typer.Option(False, "--allow-http-source", help=t("Allow URL sources to use plain HTTP")),
    disable_image_analysis: bool = typer.Option(False, "--disable-image-analysis", help=t("Disable image analysis")),
    preload_models: bool = typer.Option(
        False, "--preload-models", help=t("Initialize VLM client and local Hybrid models at startup")
    ),
    api_key: str | None = typer.Option(None, "--api-key", help=t("Optional fixed API key")),
    vlm_server_url: str | None = typer.Option(
        None, "--vlm-server-url", help=t("Remote VLM URL; empty value selects local VLM")
    ),
    vlm_api_key: str | None = typer.Option(None, "--vlm-api-key", help=t("Bearer key for the remote VLM server")),
    vlm_model: str | None = typer.Option(None, "--vlm-model", help=t("Remote VLM model name; empty value enables discovery")),
    vlm_http_timeout: int | None = typer.Option(
        None, "--vlm-http-timeout", min=1, help=t("VLM HTTP timeout in seconds (default: 600)")
    ),
    vlm_max_concurrency: int | None = typer.Option(
        None, "--vlm-max-concurrency", min=1, help=t("VLM inference concurrency (default: 100)")
    ),
    log_level: str | None = typer.Option(
        None,
        "--log-level",
        help=t(
            "API service log level: critical, error, warning, info, debug, trace; default: global log.level. "
            "Server-scope only (Uvicorn startup, HTTP access, ASGI, and service logs); "
            "the Loguru default sink always follows global log.level"
        ),
    ),
) -> None:
    """转发显式启动参数，启动 self-hosted MinerU 解析 API 服务。"""
    vlm_args: list[str] = []
    for option, value in (
        ("--vlm-server-url", vlm_server_url),
        ("--vlm-api-key", vlm_api_key),
        ("--vlm-model", vlm_model),
        ("--vlm-http-timeout", vlm_http_timeout),
        ("--vlm-max-concurrency", vlm_max_concurrency),
    ):
        if value is not None:
            vlm_args.extend([option, str(value)])
    try:
        parser_api_server.main.main(
            args=[
                "--host",
                host,
                "--port",
                str(port),
                *(["--log-level", log_level] if log_level is not None else []),
                *(["--retention-seconds", str(retention_seconds)] if retention_seconds is not None else []),
                "--concurrency",
                str(concurrency),
                "--url-timeout",
                str(url_timeout),
                "--tier",
                tier,
                *vlm_args,
                *(["--allow-http-source"] if allow_http_source else []),
                *(["--upload-dir", upload_dir] if upload_dir else []),
                *(["--disable-image-analysis"] if disable_image_analysis else []),
                *(["--preload-models"] if preload_models else []),
                *(["--api-key", api_key] if api_key else []),
            ],
            prog_name="mineru-kit api-server",
            standalone_mode=False,
        )
    except SystemExit as exc:
        assert isinstance(exc.code, int)
        raise typer.Exit(exc.code) from None
    except click.ClickException as exc:
        exit_with_message("invalid_request", exc.format_message())


def main() -> None:
    """配置标准流后，以独立命令运行新版 API 服务入口。"""
    configure_standard_streams()
    configure_global_log_level()
    app = typer.Typer(add_completion=False)
    app.command(help=t("Start the self-hosted MinerU parsing API service."))(api_server_cmd)
    app()


__all__ = ["api_server_cmd", "main"]
