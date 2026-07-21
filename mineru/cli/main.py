"""mineru CLI — personal document center, built for agents."""

from __future__ import annotations

import typer
from click.core import Context
from typer.core import TyperGroup

from .commands import cleanup, config, list_resources, server, show, telemetry, usage, watch
from .commands.forget import forget_cmd
from .commands.invalidate import invalidate_cmd
from .commands.parse import parse_cmd
from .commands.read import read_cmd
from .commands.scan import scan_cmd
from .commands.search import find_cmd, search_cmd
from .telemetry import prepare_cli_telemetry
from .version_command import show_version, version_cmd


# Typer stores commands and command groups separately before building the Click
# command tree, so source registration order alone does not preserve the mixed
# top-level help order.
TOP_LEVEL_COMMAND_ORDER = [
    "parse",
    "read",
    "scan",
    "watch",
    "search",
    "find",
    "usage",
    "list",
    "show",
    "telemetry",
    "server",
    "config",
    "invalidate",
    "forget",
    "cleanup",
    "version",
]


class OrderedRootGroup(TyperGroup):
    def list_commands(self, ctx: Context) -> list[str]:
        ordered = [name for name in TOP_LEVEL_COMMAND_ORDER if name in self.commands]
        return ordered + [name for name in self.commands if name not in TOP_LEVEL_COMMAND_ORDER]

    def parse_args(self, ctx: Context, args: list[str]) -> list[str]:
        ctx.meta["mineru_raw_args"] = list(args)
        return super().parse_args(ctx, args)


app = typer.Typer(
    name="mineru",
    cls=OrderedRootGroup,
    help="MinerU — your personal document center, built for agents",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def root(
    ctx: typer.Context,
    _version_requested: bool = typer.Option(
        False,
        "--version",
        callback=show_version,
        is_eager=True,
        help="Show the version and exit.",
    ),
) -> None:
    prepare_cli_telemetry(ctx)


app.command("parse")(parse_cmd)
app.command("read")(read_cmd)
app.command("scan")(scan_cmd)
app.add_typer(watch.app, name="watch")
app.command("search")(search_cmd)
app.command("find")(find_cmd)
app.command("usage")(usage.usage_cmd)
app.add_typer(list_resources.app, name="list")
app.add_typer(show.app, name="show")
app.add_typer(telemetry.app, name="telemetry")
app.add_typer(server.app, name="server")
app.add_typer(config.app, name="config")
app.command("invalidate")(invalidate_cmd)
app.command("forget")(forget_cmd)
app.add_typer(cleanup.app, name="cleanup")
app.command("version")(version_cmd)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
