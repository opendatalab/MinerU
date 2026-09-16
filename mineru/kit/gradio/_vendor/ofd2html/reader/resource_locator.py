"""ResourceLocator: resolve OFD ``ST_Loc`` paths against the virtual container.

Supports cd/pwd semantics:
  * absolute path (``"/Doc_0/..."``)         -> resolved from container root.
  * relative path (``"Pages/Page_0/..."``)   -> resolved from the current
    working directory.
"""

from __future__ import annotations

import posixpath

from ..pkg.container import OFDContainer


class ResourceLocator:
    """Stateful path resolver bound to one ``OFDContainer``."""

    # 初始化当前转换对象及资源。
    def __init__(self, container: OFDContainer, cwd: str = "/") -> None:
        self._container = container
        self._cwd = self._normalize(cwd or "/")

    # ----- working-directory management -------------------------------------

    @property
    # 返回当前虚拟目录。
    def cwd(self) -> str:
        return self._cwd

    # 切换虚拟目录。
    def cd(self, path: str) -> "ResourceLocator":
        """Change working directory; ``path`` may be absolute or relative."""
        self._cwd = self._normalize(self._join(path))
        return self

    # 保存当前虚拟目录。
    def push(self) -> str:
        """Snapshot current cwd; pair with :meth:`restore`."""
        return self._cwd

    # 恢复已保存的虚拟目录。
    def restore(self, snapshot: str) -> None:
        self._cwd = snapshot

    # ----- path resolution --------------------------------------------------

    # 解析资源的绝对虚拟路径。
    def resolve(self, loc: str) -> str:
        """Return absolute virtual path for ``loc`` (no leading slash)."""
        return self._join(loc).lstrip("/")

    # 按虚拟路径读取压缩包条目。
    def read(self, loc: str) -> bytes:
        return self._container.read(self.resolve(loc))

    # 检查解析后的资源路径是否存在。
    def exists(self, loc: str) -> bool:
        return self._container.has(self.resolve(loc))

    # ----- internals --------------------------------------------------------

    # 拼接并规范化虚拟路径。
    def _join(self, path: str) -> str:
        if not path:
            return self._cwd
        if path.startswith("/"):
            joined = path
        else:
            joined = posixpath.join(self._cwd, path)
        return self._normalize(joined)

    @staticmethod
    # 消除路径中的点段。
    def _normalize(path: str) -> str:
        # posixpath.normpath collapses "..", "." and duplicate slashes.
        norm = posixpath.normpath(path.replace("\\", "/"))
        if not norm.startswith("/"):
            norm = "/" + norm
        return norm
