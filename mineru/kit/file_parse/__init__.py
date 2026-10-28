"""独立单文件同步解析服务的显式入口。"""

from .app import create_app
from .settings import FileParseSettings

__all__ = ["FileParseSettings", "create_app"]
