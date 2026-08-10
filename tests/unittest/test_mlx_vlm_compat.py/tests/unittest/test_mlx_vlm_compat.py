# Copyright (c) Opendatalab. All rights reserved.
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from mineru.utils import mlx_vlm_compat


class FakeDetokenizer:
    _byte_decoder = {"x": 0x8D, "s": 32}

    def __init__(self) -> None:
        self.tokenmap = ["x", "s"]
        self._unflushed = ""
        self.text = ""
        self.trim_space = False

    def add_token(self, token: int, skip_special_token_ids: list[int] = []) -> None:
        if token in skip_special_token_ids:
            return
        value = self.tokenmap[token]
        if self._byte_decoder[value[0]] == 32:
            current_text = bytearray(
                self._byte_decoder[char] for char in self._unflushed
            ).decode("utf-8")
            self.text += current_text
            self._unflushed = value
        else:
            self._unflushed += value


def test_backports_mlx_vlm_utf8_replacement_decoder() -> None:
    tokenizer_utils = SimpleNamespace(
        BPEStreamingDetokenizer=FakeDetokenizer,
        _remove_space=lambda value: value.removeprefix(" "),
    )
    mlx_vlm = ModuleType("mlx_vlm")
    mlx_vlm.tokenizer_utils = tokenizer_utils

    with patch.dict(sys.modules, {"mlx_vlm": mlx_vlm}):
        with patch.object(mlx_vlm_compat.inspect, "getsource", return_value="strict"):
            assert mlx_vlm_compat.patch_mlx_vlm_utf8_decoder()

    decoder = FakeDetokenizer()
    decoder.add_token(0)
    decoder.add_token(1)

    assert decoder.text == "�"


def test_skips_mlx_vlm_versions_with_the_upstream_fix() -> None:
    tokenizer_utils = SimpleNamespace(BPEStreamingDetokenizer=FakeDetokenizer)
    mlx_vlm = ModuleType("mlx_vlm")
    mlx_vlm.tokenizer_utils = tokenizer_utils
    original_add_token = FakeDetokenizer.add_token

    with patch.dict(sys.modules, {"mlx_vlm": mlx_vlm}):
        with patch.object(
            mlx_vlm_compat.inspect,
            "getsource",
            return_value="errors=" + chr(34) + "replace" + chr(34),
        ):
            assert not mlx_vlm_compat.patch_mlx_vlm_utf8_decoder()

    assert FakeDetokenizer.add_token is original_add_token
