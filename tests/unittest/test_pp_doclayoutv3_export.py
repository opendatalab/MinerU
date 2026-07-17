# Copyright (c) Opendatalab. All rights reserved.
"""PP-DocLayoutV3导出脚本的契约测试：其产物必须能被MinerU侧的layout doc体系消费。

导出脚本运行在独立Paddle环境、不依赖mineru包；本测试验证两侧对
layout.json schema 的理解一致（跨环境契约），无需安装paddle。
"""
import importlib.util
import sys
from pathlib import Path

from mineru.backend.vlm.stages import (
    PrecomputedLayoutDetector,
    blocks_from_jsonable,
    load_layout_doc,
)

# demo目录不是包，按路径加载模块（且不得触发paddle导入）
_SCRIPT = Path(__file__).parent.parent.parent / "demo" / "pp_doclayoutv3_layout.py"
_spec = importlib.util.spec_from_file_location("pp_doclayoutv3_layout", _SCRIPT)
v3 = importlib.util.module_from_spec(_spec)
sys.modules["pp_doclayoutv3_layout"] = v3
_spec.loader.exec_module(v3)


def _paddle_box(label, coordinate, score=0.9):
    return {"cls_id": 0, "label": label, "score": score, "coordinate": coordinate}


def test_box_to_block_mapping_and_normalization():
    block = v3.box_to_block(_paddle_box("paragraph_title", [80, 40, 720, 90]), 800, 1000)
    assert block["type"] == "title"
    assert block["bbox"] == [0.1, 0.04, 0.9, 0.09]
    assert block["angle"] == 0

    assert v3.box_to_block(_paddle_box("display_formula", [10, 10, 100, 50]), 800, 1000)["type"] == "equation"
    assert v3.box_to_block(_paddle_box("seal", [10, 10, 100, 50]), 800, 1000)["sub_type"] == "seal"
    # 未知标签与非法框都返回None
    assert v3.box_to_block(_paddle_box("nonexistent_label", [10, 10, 100, 50]), 800, 1000) is None
    assert v3.box_to_block(_paddle_box("text", [100, 50, 100, 50]), 800, 1000) is None


def test_label_mapping_consistent_with_hybrid():
    """静态复制的映射表必须与hybrid的单一事实来源保持一致。"""
    from mineru.backend.hybrid.hybrid_analyze import MEDIUM_EFFORT_LAYOUT_LABEL_TO_VLM_TYPE

    assert v3.LABEL_TO_BLOCK_TYPE == {
        label: str(block_type)
        for label, block_type in MEDIUM_EFFORT_LAYOUT_LABEL_TO_VLM_TYPE.items()
    }


def test_exported_doc_consumable_by_mineru(tmp_path):
    boxes = [
        _paddle_box("doc_title", [80, 40, 720, 90]),
        _paddle_box("text", [80, 120, 720, 400]),
        _paddle_box("table", [80, 420, 720, 700]),
        _paddle_box("formula_number", [650, 720, 700, 760]),
    ]
    blocks = [b for b in (v3.box_to_block(box, 800, 1000) for box in boxes) if b]
    doc = v3.build_layout_doc([
        {"page_idx": 0, "page_size": [800, 1000], "blocks": blocks},
    ])

    # MinerU侧schema校验通过
    loaded = load_layout_doc(doc)
    assert loaded["layout_backend"] == "pp_doclayout_v3"
    assert loaded["emits_formula_number"] is True

    # 反序列化为合法ContentBlock并可被PrecomputedLayoutDetector消费
    restored = blocks_from_jsonable(loaded["pages"][0]["blocks"])
    assert len(restored) == len(blocks)
    detector = PrecomputedLayoutDetector(doc)
    assert detector.emits_formula_number is True
    page_blocks = detector.batch_detect([None], start_page_idx=0)[0]
    assert [b["type"] for b in page_blocks] == ["title", "text", "table", "formula_number"]
