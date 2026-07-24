"""mineru-kit CLI — parsing and service tools."""

from __future__ import annotations

import typer
from click.core import Context
from typer.core import TyperGroup

from ..cli.version_command import show_version, version_cmd
from ..utils.stdio import configure_standard_streams
from .commands import api_server, models, parse, router, vlm_server

TOP_LEVEL_COMMAND_ORDER = [
    "parse",
    "api-server",
    "vlm-server",
    "router",
    "models",
    "version",
]

FORWARD_CONTEXT_SETTINGS = {
    "ignore_unknown_options": True,
    "allow_extra_args": True,
}


class OrderedRootGroup(TyperGroup):
    def list_commands(self, ctx: Context) -> list[str]:
        ordered = [name for name in TOP_LEVEL_COMMAND_ORDER if name in self.commands]
        return ordered + [name for name in self.commands if name not in TOP_LEVEL_COMMAND_ORDER]


app = typer.Typer(
    name="mineru-kit",
    cls=OrderedRootGroup,
    help="MinerU Kit — parsing and service tools",
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
        help="Show the version and exit.",
    ),
) -> None:
    pass


app.add_typer(models.app, name="models")
app.command("parse")(parse.parse_cmd)
app.command("api-server")(api_server.api_server_cmd)
app.command("vlm-server", context_settings=FORWARD_CONTEXT_SETTINGS)(vlm_server.vlm_server_cmd)
app.command("router", context_settings=FORWARD_CONTEXT_SETTINGS)(router.router_cmd)
app.command("version")(version_cmd)


def main() -> None:
    configure_standard_streams()
    app()


if __name__ == "__main__":
    main()
