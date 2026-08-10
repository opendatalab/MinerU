"""Compatibility patches for optional MLX VLM dependencies."""

import inspect
from typing import Any


def patch_mlx_vlm_utf8_decoder() -> bool:
    """Backport the MLX VLM UTF-8 streaming fix for older releases."""
    try:
        from mlx_vlm import tokenizer_utils
    except ImportError:
        return False

    detokenizer = tokenizer_utils.BPEStreamingDetokenizer
    try:
        source = inspect.getsource(detokenizer.add_token)
    except (OSError, TypeError):
        return False

    if "errors=\"replace\"" in source or "errors=\x27replace\x27" in source:
        return False

    def add_token(
        self: Any,
        token: int,
        skip_special_token_ids: list[int] = [],
    ) -> None:
        if token in skip_special_token_ids:
            return
        value = self.tokenmap[token]
        if self._byte_decoder[value[0]] == 32:
            current_text = bytearray(
                self._byte_decoder[char] for char in self._unflushed
            ).decode("utf-8", errors="replace")
            if self.text or not self.trim_space:
                self.text += current_text
            else:
                self.text += tokenizer_utils._remove_space(current_text)
            self._unflushed = value
        else:
            self._unflushed += value

    detokenizer.add_token = add_token
    return True
