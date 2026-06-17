from __future__ import annotations

import typer

from ...cli_old import vlm_server as old_vlm_server
from ..errors import exit_with_message


def vlm_server_cmd(
    engine: str = "auto",
    ctx: typer.Context | None = None,
) -> None:
    if engine not in {"auto", "vllm", "lmdeploy", "sglang", "mlx"}:
        exit_with_message("invalid_request", f"Unsupported engine '{engine}'.", "engine")
    if engine in {"sglang", "mlx"}:
        exit_with_message("invalid_request", f"Engine '{engine}' is not implemented yet in mineru-kit vlm-server.", "engine")
    extra_args = list(ctx.args) if ctx is not None else []
    try:
        old_vlm_server.openai_server.main(
            args=["--engine", engine, *extra_args],
            prog_name="mineru-kit vlm-server",
            standalone_mode=False,
        )
    except SystemExit as exc:
        assert isinstance(exc.code, int)
        raise typer.Exit(exc.code) from None
