"""Kit 命令的运行阶段初始化，不影响帮助参数解析。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from typer import Context
from typer.core import TyperCommand


@dataclass(frozen=True)
class KitContext:
    """由控制台入口显式启用进程日志初始化，嵌入应用时保留宿主日志配置。"""

    configure_logging: bool = False


class RuntimeCommand(TyperCommand):
    """仅在叶子命令真正执行时初始化日志，帮助无需读取用户配置。"""

    def invoke(self, ctx: Context) -> Any:
        """参数与帮助处理完成后配置日志，然后调用原始命令。"""
        options = ctx.find_object(KitContext)
        if options is not None and options.configure_logging:
            from ...utils.logger import configure_global_log_level

            configure_global_log_level()
        return super().invoke(ctx)


__all__ = ["KitContext", "RuntimeCommand"]
