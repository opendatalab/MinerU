# Copyright (c) Opendatalab. All rights reserved.
from collections.abc import Sequence

import click


def _coerce_cli_value(raw_value: str) -> bool | float | int | str:
    lowered = raw_value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    try:
        return int(raw_value)
    except ValueError:
        pass

    try:
        return float(raw_value)
    except ValueError:
        return raw_value


def parse_unknown_args(args: Sequence[str]) -> dict:
    """Parse unknown click args into keyword arguments."""
    extra_kwargs = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if not arg.startswith("--"):
            i += 1
            continue

        raw_option = arg[2:]
        if "=" in raw_option:
            param_name, raw_value = raw_option.split("=", 1)
            extra_kwargs[param_name.replace("-", "_")] = _coerce_cli_value(raw_value)
            i += 1
            continue

        param_name = raw_option.replace("-", "_")
        next_index = i + 1
        if next_index < len(args) and not args[next_index].startswith("--"):
            extra_kwargs[param_name] = _coerce_cli_value(args[next_index])
            i += 2
            continue

        extra_kwargs[param_name] = True
        i += 1

    return extra_kwargs


def arg_parse(ctx: "click.Context") -> dict:
    return parse_unknown_args(ctx.args)
