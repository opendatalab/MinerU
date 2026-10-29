"""mineru-kit CLI — parsing and service tools."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module

import typer
from typer import Context
from typer.core import TyperCommand, TyperGroup
from typer.main import get_command

from ..cli.version_command import show_version
from ..utils.i18n import t
from ..utils.stdio import configure_standard_streams
from .commands._base import KitContext, RuntimeCommand


@dataclass(frozen=True)
class CommandSpec:
    """显式声明命令入口和帮助文案，展示根帮助时无需导入命令实现。"""

    module: str
    attribute: str
    help: str
    group: bool = False
    context_settings_attribute: str | None = None


COMMANDS: dict[str, CommandSpec] = {
    "parse": CommandSpec(
        ".commands.parse", "parse_cmd", "Parse files or directories into markdown, middle JSON, or zip outputs."
    ),
    "webui": CommandSpec(
        ".commands.webui", "webui_cmd", "Start the Gradio document parsing web UI backed by the MinerU V1 API."
    ),
    "api-server": CommandSpec(".commands.api_server", "api_server_cmd", "Start the self-hosted MinerU parsing API service."),
    "file-parse-server": CommandSpec(
        ".commands.file_parse_server", "file_parse_server_cmd", "Start the synchronous single-file Tasks wrapper service."
    ),
    "vlm-server": CommandSpec(
        ".commands.vlm_server",
        "vlm_server_cmd",
        "Start the local VLM server with OpenAI-compatible chat completions.",
        context_settings_attribute="FORWARD_CONTEXT_SETTINGS",
    ),
    "router": CommandSpec(
        ".commands.router", "router_cmd", "Start a standalone Router service exposing only the MinerU V1 API."
    ),
    "models": CommandSpec(".commands.models", "app", "Download, inspect, and verify local MinerU models.", group=True),
    "hardware": CommandSpec(
        ".commands.hardware", "app", "Detect recommended tiers or inspect hardware and runtime information.", group=True
    ),
    "version": CommandSpec("..cli.version_command", "version_cmd", "Print MinerU and Python versions."),
}
TOP_LEVEL_COMMAND_ORDER = list(COMMANDS)


class OrderedRootGroup(TyperGroup):
    """根帮助使用轻量命令描述，仅在解析选中的命令时加载完整参数定义。"""

    def list_commands(self, ctx: Context) -> list[str]:
        """按产品顺序返回命令名称，不触发服务或模型模块导入。"""
        return list(TOP_LEVEL_COMMAND_ORDER)

    def get_command(self, ctx: Context, cmd_name: str) -> TyperCommand | None:
        """为帮助、补全和命令名称校验提供轻量描述。"""
        spec = COMMANDS.get(cmd_name)
        return TyperCommand(cmd_name, help=t(spec.help)) if spec is not None else None

    def resolve_command(self, ctx: Context, args: list[str]) -> tuple[str | None, TyperCommand | TyperGroup | None, list[str]]:
        """仅加载实际选中的入口，保留原始函数签名和转发参数设置。"""
        name, command, remaining = super().resolve_command(ctx, args)
        if name is None or command is None:
            return name, command, remaining
        spec = COMMANDS[name]
        module = import_module(spec.module, package=__package__)
        target = getattr(module, spec.attribute)
        if spec.group:
            return name, get_command(target), remaining
        selected = typer.Typer(add_completion=False)
        settings = getattr(module, spec.context_settings_attribute) if spec.context_settings_attribute else None
        selected.command(name, help=t(spec.help), cls=RuntimeCommand, context_settings=settings)(target)
        return name, get_command(selected), remaining


app = typer.Typer(
    name="mineru-kit",
    cls=OrderedRootGroup,
    help=t("MinerU Kit — batch parsing and service tools"),
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
    """只声明根参数，帮助和版本查询不初始化运行环境。"""


def main() -> None:
    """先配置输出编码，再由实际执行的叶子命令初始化日志。"""
    configure_standard_streams()
    app(obj=KitContext(configure_logging=True))


__all__ = ["app", "main"]

if __name__ == "__main__":
    main()
