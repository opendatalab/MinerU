"""Primitive output functions for MinerU CLI runtimes."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, is_dataclass
from typing import Any

from pydantic import BaseModel
from pydantic_core import to_json
from rich.console import Console
from rich.text import Text

from .contracts import RenderableObject

console = Console(markup=False)
stderr_console = Console(stderr=True, markup=False)


def print_rich(*objects: RenderableObject) -> None:
    console.print(*objects)


def print_text(value: object) -> None:
    sys.stdout.write(f"{value}\n")


def print_success(msg: str) -> None:
    if console:
        console.print(msg, style="green")
    else:
        print_text(msg)


def print_info(msg: str) -> None:
    if console:
        console.print(msg, style="dim")
    else:
        print_text(msg)


def print_notice(msg: str) -> None:
    stderr_console.print(msg, style="dim")


def print_error(msg: str) -> None:
    text = Text("Error:", style="red")
    text.append(f" {msg}")
    stderr_console.print(text)


def print_json(data: Any) -> None:
    if isinstance(data, BaseModel):
        print_text(to_json(data, indent=2).decode("utf-8"))
        return
    if is_dataclass(data) and not isinstance(data, type):
        print_text(json.dumps(asdict(data), ensure_ascii=False, indent=2))
        return
    print_text(json.dumps(data, ensure_ascii=False, indent=2))
