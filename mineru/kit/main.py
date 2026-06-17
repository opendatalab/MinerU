"""mineru-kit CLI — parsing and service tools."""

from __future__ import annotations

import typer
from click.core import Context
from typer.core import TyperGroup

from .commands import api_server, models, parse, vlm_server

TOP_LEVEL_COMMAND_ORDER = [
    "parse",
    "api-server",
    "vlm-server",
    "models",
]


class OrderedRootGroup(TyperGroup):
    def list_commands(self, ctx: Context) -> list[str]:
        ordered = [name for name in TOP_LEVEL_COMMAND_ORDER if name in self.commands]
        return ordered + [name for name in self.commands if name not in TOP_LEVEL_COMMAND_ORDER]


app = typer.Typer(
    name="mineru-kit",
    cls=OrderedRootGroup,
    help="MinerU Kit — parsing and service tools",
    no_args_is_help=True,
)

app.add_typer(models.app, name="models")


@app.command("parse")
def parse_command(
    inputs: list[str] = typer.Argument(..., help="Input files or directories"),
    output: str = typer.Option(..., "-o", "--output", help="Output path; required"),
    pages: str | None = typer.Option(None, "-p", "--pages", help="Page range, e.g. '1~5' or 'all'"),
    format: str = typer.Option("markdown", "-f", "--format", help="Output format: markdown, middle_json, zip"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose output"),
    tier: str | None = typer.Option(None, "--tier", help="Parse tier: flash, standard, pro"),
    backend: str | None = typer.Option(None, "--backend", help="Expert backend override"),
    remote: bool = typer.Option(False, "--remote", help="Use mineru.net official remote parse service"),
    remote_url: str | None = typer.Option(None, "--remote-url", help="Use a custom remote parse service URL"),
    api_key: str | None = typer.Option(None, "--api-key", help="API key for remote parse service"),
    language: str = typer.Option("ch", "--language", help="Parser language hint"),
    ocr_mode: str = typer.Option("auto", "--ocr-mode", help="OCR mode: auto, txt, ocr"),
    disable_table: bool = typer.Option(False, "--disable-table", help="Disable table recognition"),
    disable_formula: bool = typer.Option(False, "--disable-formula", help="Disable formula recognition"),
    disable_image_analysis: bool = typer.Option(False, "--disable-image-analysis", help="Disable image analysis"),
) -> None:
    """Parse files or directories into markdown, middle JSON, or zip outputs."""
    parse.parse_cmd(
        inputs=inputs,
        output=output,
        pages=pages,
        format=format,  # type: ignore[arg-type]
        verbose=verbose,
        tier=tier,  # type: ignore[arg-type]
        backend=backend,
        remote=remote,
        remote_url=remote_url,
        api_key=api_key,
        language=language,
        ocr_mode=ocr_mode,  # type: ignore[arg-type]
        disable_table=disable_table,
        disable_formula=disable_formula,
        disable_image_analysis=disable_image_analysis,
    )


@app.command("api-server")
def api_server_command(
    host: str = typer.Option("127.0.0.1", "--host", help="Server host"),
    port: int = typer.Option(8000, "--port", help="Server port"),
    upload_dir: str = typer.Option("", "--upload-dir", help="Upload directory"),
    backend: str | None = typer.Option(None, "--backend", help="Expert backend override"),
    tier: str | None = typer.Option(None, "--tier", help="Server tier: standard or pro (default: standard)"),
    concurrency: int = typer.Option(1, "--concurrency", help="Maximum concurrent parse jobs"),
    url_timeout: int = typer.Option(60, "--url-timeout", help="Timeout for URL source downloads"),
    max_wait: int = typer.Option(600, "--max-wait", help="Maximum seconds for wait parameter"),
    language: str = typer.Option("ch", "--language", help="Parser language hint"),
    ocr_mode: str = typer.Option("auto", "--ocr-mode", help="OCR mode: auto, txt, ocr"),
    disable_table: bool = typer.Option(False, "--disable-table", help="Disable table recognition"),
    disable_formula: bool = typer.Option(False, "--disable-formula", help="Disable formula recognition"),
    disable_image_analysis: bool = typer.Option(False, "--disable-image-analysis", help="Disable image analysis"),
    api_key: str | None = typer.Option(None, "--api-key", help="Optional fixed API key"),
) -> None:
    """Start the self-hosted MinerU parse API server."""
    api_server.api_server_cmd(
        host=host,
        port=port,
        upload_dir=upload_dir,
        backend=backend,
        tier=tier,  # type: ignore[arg-type]
        concurrency=concurrency,
        url_timeout=url_timeout,
        max_wait=max_wait,
        language=language,
        ocr_mode=ocr_mode,  # type: ignore[arg-type]
        disable_table=disable_table,
        disable_formula=disable_formula,
        disable_image_analysis=disable_image_analysis,
        api_key=api_key,
    )


@app.command(
    "vlm-server",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
def vlm_server_command(
    ctx: typer.Context,
    engine: str = typer.Option("auto", "--engine", help="VLM serving engine: auto, vllm, lmdeploy, sglang, mlx"),
) -> None:
    """Start the local VLM server with OpenAI-compatible chat completions."""
    vlm_server.vlm_server_cmd(engine=engine, ctx=ctx)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
