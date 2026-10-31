"""Explicit VLM CLI boolean choices take precedence over hardware defaults."""

from __future__ import annotations

import argparse

import pytest

from mineru.model.vlm.engine_utils import _apply_server_config


@pytest.mark.parametrize("name", ["enable-chunked-prefill", "enable-prefix-caching"])
def test_explicit_positive_boolean_flag_is_not_overridden(name: str) -> None:
    """Do not append a negative flag that reverses an explicit command-line choice."""
    args = ["--model", "local-model", f"--{name}"]
    original = args.copy()
    _apply_server_config(args, {name.replace("-", "_"): False})
    assert args == original


@pytest.mark.parametrize("name", ["enable-chunked-prefill", "enable-prefix-caching"])
def test_explicit_negative_boolean_flag_is_not_duplicated(name: str) -> None:
    """A caller already selecting the device default keeps one flag."""
    args = [f"--no-{name}"]
    _apply_server_config(args, {name.replace("-", "_"): False})
    assert args == [f"--no-{name}"]


@pytest.mark.parametrize("enabled", [False, True])
def test_boolean_defaults_use_boolean_flags_and_preserve_explicit_choices(enabled: bool) -> None:
    """Device defaults are valid BooleanOptionalAction arguments for both truth values."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction)
    args: list[str] = []
    _apply_server_config(args, {"enable_prefix_caching": enabled})
    assert parser.parse_args(args).enable_prefix_caching is enabled
    assert args == ["--enable-prefix-caching" if enabled else "--no-enable-prefix-caching"]


def test_explicit_negative_choice_survives_positive_default() -> None:
    """The override rule is symmetric when a future device recommends enabling a flag."""
    args = ["--no-enable-chunked-prefill"]
    _apply_server_config(args, {"enable_chunked_prefill": True})
    assert args == ["--no-enable-chunked-prefill"]


def test_other_server_values_keep_their_existing_precedence() -> None:
    """Non-boolean arguments still retain caller values and receive absent defaults."""
    args = ["--block-size=32"]
    _apply_server_config(args, {"block_size": 128, "dtype": "float16"})
    assert args == ["--block-size=32", "--dtype", "float16"]
