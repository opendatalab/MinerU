"""mineru-kit CLI — parsing and service tools."""

from __future__ import annotations

import typer
from click.core import Context
from typer.core import TyperGroup

from ..cli.version_command import show_version, version_cmd
from ..utils.i18n import t
from ..utils.logger import configure_global_log_level
from ..utils.stdio import configure_standard_streams
from .commands import api_server, models, parse, router, vlm_server, webui

TOP_LEVEL_COMMAND_ORDER = [
    "parse",
    "webui",
    "api-server",
    "vlm-server",
    "router",
    "models",
    "version",
]


class OrderedRootGroup(TyperGroup):
    def list_commands(self, ctx: Context) -> list[str]:
        ordered = [name for name in TOP_LEVEL_COMMAND_ORDER if name in self.commands]
        return ordered + [name for name in self.commands if name not in TOP_LEVEL_COMMAND_ORDER]


app = typer.Typer(
    name="mineru-kit",
    cls=OrderedRootGroup,
    help=t("MinerU Kit — parsing and service tools"),
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def root(
    _version_requested: bool = typer.Option(
        False,
        "--version",
        callback=show_version,
        is_eager=True,
        help=t("Show the version and exit."),
    ),
) -> None:
    pass


app.add_typer(models.app, name="models")
app.command("parse", help=t("Parse files or directories into markdown, middle JSON, or zip outputs."))(parse.parse_cmd)
app.command("webui", help=t("Start the Gradio document parsing web UI backed by the MinerU V1 API."))(webui.webui_cmd)
app.command("api-server", help=t("Forward explicit startup options and launch the self-hosted MinerU parsing API service."))(
    api_server.api_server_cmd
)
app.command(
    "vlm-server",
    context_settings=vlm_server.FORWARD_CONTEXT_SETTINGS,
    help=t("Start the local VLM server with OpenAI-compatible chat completions."),
)(vlm_server.vlm_server_cmd)
app.command("router", help=t("Start a standalone Router service exposing only the MinerU V1 API."))(router.router_cmd)
app.command("version", help=t("Print MinerU and Python versions."))(version_cmd)


def main() -> None:
    configure_standard_streams()
    configure_global_log_level()
    app()


if __name__ == "__main__":
    main()
