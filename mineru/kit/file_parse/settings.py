"""同步包装服务的显式启动配置。"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit


def origin(url: str) -> tuple[str, str, int]:
    """严格比较协议、主机和端口，并拒绝 URL 内嵌凭据。"""
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password:
        raise ValueError("Expected an HTTP(S) URL without embedded credentials")
    return parts.scheme.lower(), parts.hostname.lower(), parts.port or (443 if parts.scheme == "https" else 80)


@dataclass(frozen=True)
class FileParseSettings:
    """只在创建配置实例时读取默认目录，不在模块导入时读取环境。"""

    api_url: str
    api_key: str | None = None
    state_dir: Path | None = None
    retention_seconds: int = 86400
    poll_interval: float = 2.0
    connect_timeout: float = 10.0
    request_timeout: float = 30.0
    transfer_timeout: float = 600.0

    def __post_init__(self) -> None:
        """校验地址和预算，禁止静默选择官方远程上游。"""
        normalized = self.api_url.rstrip("/")
        origin(normalized)
        parts = urlsplit(normalized)
        if parts.query or parts.fragment:
            raise ValueError("api_url must not contain a query or fragment")
        if self.api_key and any(character in self.api_key for character in "\r\n"):
            raise ValueError("api_key must not contain line breaks")
        object.__setattr__(self, "api_url", normalized)
        if self.state_dir is None:
            home = Path(os.environ.get("MINERU_HOME") or "~/.mineru").expanduser()
            object.__setattr__(self, "state_dir", home / "file-parse")
        else:
            object.__setattr__(self, "state_dir", Path(self.state_dir).expanduser().resolve())
        if type(self.retention_seconds) is not int or self.retention_seconds < 0:
            raise ValueError("retention_seconds must be a non-negative integer")
        for name in ("poll_interval", "connect_timeout", "request_timeout", "transfer_timeout"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")

    @property
    def namespace(self) -> str:
        """按新任务协议、上游和认证隔离记录，保留但不加载旧协议状态。"""
        return "tasks-v5-" + hashlib.sha256(f"{self.api_url}\0{self.api_key or ''}".encode()).hexdigest()


__all__ = ["FileParseSettings", "origin"]
