import numpy as np
import sys
import types

sys.modules.setdefault("torch", types.SimpleNamespace(Tensor=object))

from mineru.model.utils.pytorchocr.postprocess.rec_postprocess import (
    BaseRecLabelDecode,
    CTCLabelDecode,
)


def test_ctc_decode_strips_word_joiner_without_shifting_indices(tmp_path):
    dict_path = tmp_path / "dict.txt"
    dict_path.write_text("A\n\u2060\nB\n", encoding="utf-8")

    decoder = CTCLabelDecode(str(dict_path), use_space_char=False)
    text_index = np.array([[1, 2, 3]])
    decoded = decoder.decode(text_index)

    assert decoded == [("AB", 1.0)]


def test_sanitize_decoded_text_preserves_visible_characters():
    assert BaseRecLabelDecode.sanitize_decoded_text("A\u2060B中") == "AB中"
