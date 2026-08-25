"""Standard stream configuration helpers."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping

_STDIO_ENCODING = "utf-8"
_STDIO_ERRORS = "backslashreplace"


def configure_standard_streams() -> None:
    """Configure the current process standard streams for UTF-8 output."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding=_STDIO_ENCODING, errors=_STDIO_ERRORS)


def utf8_subprocess_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return a child-process environment with UTF-8 standard streams."""
    env = dict(os.environ if base is None else base)
    env["PYTHONIOENCODING"] = f"{_STDIO_ENCODING}:{_STDIO_ERRORS}"
    return env
