"""Primitive output functions for MinerU CLI runtimes."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any

from pydantic import BaseModel
from pydantic_core import to_json
from rich.console import Console
from rich.text import Text

from .contracts import RenderableObject

console = Console()
stderr_console = Console(stderr=True)


def print_rich(*objects: RenderableObject) -> None:
    console.print(*objects)


def print_success(msg: str) -> None:
    if console:
        console.print(msg, style="green")
    else:
        print(msg)


def print_info(msg: str) -> None:
    if console:
        console.print(msg, style="dim")
    else:
        print(msg)


def print_notice(msg: str) -> None:
    stderr_console.print(msg, style="dim")


def print_error(msg: str) -> None:
    text = Text("Error:", style="red")
    text.append(f" {msg}")
    stderr_console.print(text)


def print_json(data: Any) -> None:
    if isinstance(data, BaseModel):
        print(to_json(data, indent=2).decode("utf-8"))
        return
    if is_dataclass(data) and not isinstance(data, type):
        print(json.dumps(asdict(data), ensure_ascii=False, indent=2))
        return
    print(json.dumps(data, ensure_ascii=False, indent=2))
