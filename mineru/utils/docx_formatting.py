from enum import Enum

from pydantic import BaseModel


class Script(str, Enum):
    """Text script position."""

    BASELINE = "baseline"
    SUB = "sub"
    SUPER = "super"


class Formatting(BaseModel):
    """Formatting."""

    bold: bool = False
    italic: bool = False
    underline: bool = False
    strikethrough: bool = False
    script: Script = Script.BASELINE
