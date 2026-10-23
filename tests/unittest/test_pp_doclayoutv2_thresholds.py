"""验证模型分类阈值与阅读顺序有效候选在两种推理后端保持一致。"""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

from mineru.model.layout.pp_doclayout_v2_base import DEFAULT_CLASS_THRESHOLDS, PP_DOCLAYOUT_V2_LABELS


def _torch_predictions(
    probabilities: np.ndarray,
    *,
    class_thresholds: Sequence[float] = DEFAULT_CLASS_THRESHOLDS,
) -> list[dict[str, Any]]:
    """以受控检测分数和排序张量调用真实 Torch 后处理，不加载模型权重。"""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers", minversion="5.10.1")
    from mineru.model.layout.pp_doclayoutv2 import PPDocLayoutV2LayoutModel

    model = object.__new__(PPDocLayoutV2LayoutModel)
    model.config = SimpleNamespace(class_thresholds=list(class_thresholds))
    batch_size, query_count, _ = probabilities.shape
    outputs = SimpleNamespace(
        logits=torch.logit(torch.as_tensor(probabilities, dtype=torch.float32)),
        pred_boxes=torch.tensor([[[0.5, 0.5, 0.2, 0.1]]]).expand(batch_size, query_count, 4),
        order_logits=torch.full((batch_size, query_count, query_count), 20.0),
    )
    return model._post_process_object_detection(outputs, [(200, 100)] * batch_size)


def _onnx_predictions(rows: np.ndarray, counts: Sequence[int]) -> list[dict[str, np.ndarray]]:
    """通过模拟 ORT 输出验证逐页拆分、分类筛选和阅读顺序的完整路径。"""
    pytest.importorskip("onnxruntime")
    from mineru.model.layout.pp_doclayout_v2_onnx import PPDocLayoutV2LayoutModelONNX

    model = object.__new__(PPDocLayoutV2LayoutModelONNX)
    model.imgsz = (800, 800)
    model._input_names = ["image", "im_shape", "scale_factor"]
    model.session = SimpleNamespace(run=Mock(return_value=[rows, np.asarray(counts, dtype=np.int32)]))
    return model._run_session(np.zeros((len(counts), 3, 4, 4), dtype=np.float32), [(200, 100)] * len(counts))


def _predict(backend: str, probabilities: np.ndarray) -> dict[str, Any]:
    """将同一组多标签候选输入两种后端，反转 ONNX 行序以排除顺序依赖。"""
    if backend == "torch":
        return _torch_predictions(probabilities[None])[0]
    rows = np.asarray(
        [
            [cls_id, score, query * 10, 2, query * 10 + 4, 8, query, query]
            for query, scores in enumerate(probabilities)
            for cls_id, score in enumerate(scores)
        ],
        dtype=np.float32,
    ).reshape(-1, 8)
    return _onnx_predictions(rows[::-1], [len(rows)])[0]


@pytest.mark.parametrize("backend", [pytest.param("torch", marks=pytest.mark.torch), "onnx"])
@pytest.mark.parametrize("cls_id", range(len(PP_DOCLAYOUT_V2_LABELS)), ids=PP_DOCLAYOUT_V2_LABELS)
def test_class_threshold_boundaries(backend: str, cls_id: int) -> None:
    """逐类验证低于阈值拒绝，等于阈值和高于阈值均可保留。"""
    threshold = DEFAULT_CLASS_THRESHOLDS[cls_id]
    probabilities = np.full((3, len(PP_DOCLAYOUT_V2_LABELS)), 0.001, dtype=np.float32)
    probabilities[:, cls_id] = [threshold - 0.001, threshold, threshold + 0.001]

    result = _predict(backend, probabilities)

    assert result["labels"].tolist() == [cls_id, cls_id]
    np.testing.assert_allclose(result["scores"], [threshold, threshold + 0.001], rtol=0, atol=1e-7)


@pytest.mark.parametrize("backend", [pytest.param("torch", marks=pytest.mark.torch), "onnx"])
def test_inline_formula_uses_045_without_changing_other_class_thresholds(backend: str) -> None:
    """行内公式低于 0.45 时拒绝，行间公式和正文仍保留 0.4 以上的候选。"""
    probabilities = np.full((7, len(PP_DOCLAYOUT_V2_LABELS)), 0.001, dtype=np.float32)
    for query, (cls_id, score) in enumerate(
        [(15, 0.425), (15, 0.449), (15, 0.45), (15, 0.451), (5, 0.425), (22, 0.425), (0, 0.475)]
    ):
        probabilities[query, cls_id] = score

    result = _predict(backend, probabilities)

    assert result["labels"].tolist() == [15, 15, 5, 22]
    np.testing.assert_allclose(result["scores"], [0.45, 0.451, 0.425, 0.425], rtol=0, atol=1e-7)


@pytest.mark.parametrize("backend", [pytest.param("torch", marks=pytest.mark.torch), "onnx"])
def test_invalid_primary_class_cannot_be_reintroduced_by_secondary_label(backend: str) -> None:
    """摘要主类别未达标时，即使次高正文类别达标也不能重新进入输出。"""
    probabilities = np.full((4, len(PP_DOCLAYOUT_V2_LABELS)), 0.001, dtype=np.float32)
    probabilities[0, 17] = 0.9
    probabilities[1, 0] = 0.49
    probabilities[1, 22] = 0.44

    result = _predict(backend, probabilities)

    assert result["labels"].tolist() == [17]


@pytest.mark.parametrize("backend", [pytest.param("torch", marks=pytest.mark.torch), "onnx"])
def test_summary_page_scores_restore_title_and_four_paragraphs(backend: str) -> None:
    """用第八页实测分数复现大摘要框抢序和低分第二段丢失，验证分类筛选修复。"""
    probabilities = np.full((10, len(PP_DOCLAYOUT_V2_LABELS)), 0.001, dtype=np.float32)
    for query, (cls_id, score) in enumerate(
        [(0, 0.4740), (17, 0.8968), (22, 0.4426), (22, 0.5669), (22, 0.6167), (22, 0.5063), (8, 0.6086), (8, 0.5270)]
    ):
        probabilities[query, cls_id] = score
    probabilities[6, 9] = 0.4692

    result = _predict(backend, probabilities)

    assert result["labels"].tolist() == [17, 22, 22, 22, 22, 8, 8]
    np.testing.assert_allclose(result["scores"][1], 0.4426, rtol=0, atol=1e-7)


@pytest.mark.parametrize("backend", [pytest.param("torch", marks=pytest.mark.torch), "onnx"])
def test_valid_primary_keeps_only_secondary_labels_above_their_own_threshold(backend: str) -> None:
    """有效候选保留达到自身阈值的次高正文标签，拒绝未达标的页眉标签。"""
    probabilities = np.full((4, len(PP_DOCLAYOUT_V2_LABELS)), 0.001, dtype=np.float32)
    probabilities[0, 0] = 0.6
    probabilities[0, 22] = 0.44
    probabilities[0, 12] = 0.49

    result = _predict(backend, probabilities)

    assert set(result["labels"].tolist()) == {0, 22}


@pytest.mark.parametrize("backend", [pytest.param("torch", marks=pytest.mark.torch), "onnx"])
@pytest.mark.parametrize(("first_cls", "second_cls", "expected"), [(0, 22, []), (6, 8, [6])])
def test_primary_class_ties_use_the_smallest_class_id(
    backend: str, first_cls: int, second_cls: int, expected: list[int]
) -> None:
    """最高分同分时按较小类别编号判断有效性，与模型 argmax 规则一致。"""
    probabilities = np.full((4, len(PP_DOCLAYOUT_V2_LABELS)), 0.001, dtype=np.float32)
    probabilities[0, first_cls] = probabilities[0, second_cls] = 0.45

    result = _predict(backend, probabilities)

    assert result["labels"].tolist() == expected


@pytest.mark.torch
def test_torch_uses_loaded_model_thresholds() -> None:
    """Torch 按已加载模型配置筛选，不以共享默认常量覆盖模型自带阈值。"""
    probabilities = np.full((1, 1, len(PP_DOCLAYOUT_V2_LABELS)), 0.001, dtype=np.float32)
    probabilities[0, 0, 22] = 0.6
    thresholds = list(DEFAULT_CLASS_THRESHOLDS)
    thresholds[22] = 0.7

    result = _torch_predictions(probabilities, class_thresholds=thresholds)[0]

    assert result["labels"].tolist() == []


@pytest.mark.torch
@pytest.mark.parametrize("query_count", [0, 2])
def test_torch_empty_and_all_rejected_candidates(query_count: int) -> None:
    """无候选和全部未达标的批次均保留逐页空结果与框数组形状。"""
    probabilities = np.full((2, query_count, len(PP_DOCLAYOUT_V2_LABELS)), 0.001, dtype=np.float32)

    results = _torch_predictions(probabilities)

    assert len(results) == 2
    assert all(result["labels"].numel() == 0 and result["boxes"].shape == (0, 4) for result in results)


def test_onnx_groups_candidates_with_coordinates_and_both_order_keys() -> None:
    """坐标相同而排序键不同或排序键相同而坐标不同的候选必须独立判断。"""
    rows = np.asarray(
        [
            [0, 0.49, 1, 2, 3, 4, 0, 0],
            [22, 0.44, 1, 2, 3, 4, 0, 0],
            [22, 0.44, 1, 2, 3, 4, 0, 1],
            [22, 0.44, 10, 2, 30, 4, 0, 0],
        ],
        dtype=np.float32,
    )

    result = _onnx_predictions(rows, [4])[0]

    assert result["labels"].tolist() == [22, 22]
    np.testing.assert_array_equal(result["boxes"], [[1, 2, 3, 4], [10, 2, 30, 4]])


def test_onnx_candidate_validity_is_isolated_by_page() -> None:
    """相同框和排序键跨页出现时独立筛选，同时保留零框页和全过滤页。"""
    rows = np.asarray(
        [[0, 0.49, 1, 2, 3, 4, 0, 0], [22, 0.44, 1, 2, 3, 4, 0, 0], [22, 0.44, 1, 2, 3, 4, 0, 0]],
        dtype=np.float32,
    )

    results = _onnx_predictions(rows, [2, 0, 1])

    assert [result["labels"].tolist() for result in results] == [[], [], [22]]
    assert all(result["boxes"].shape == (0, 4) for result in results[:2])


@pytest.mark.parametrize("backend", [pytest.param("torch", marks=pytest.mark.torch), "onnx"])
def test_layout_constructor_rejects_removed_uniform_conf(backend: str) -> None:
    """包装类明确移除统一 conf 参数，禁止调用方再次覆盖分类阈值。"""
    if backend == "torch":
        pytest.importorskip("torch")
        pytest.importorskip("transformers", minversion="5.10.1")
        from mineru.model.layout.pp_doclayoutv2 import PPDocLayoutV2LayoutModel as model_type
    else:
        pytest.importorskip("onnxruntime")
        from mineru.model.layout.pp_doclayout_v2_onnx import PPDocLayoutV2LayoutModelONNX as model_type

    with pytest.raises(TypeError, match="conf"):
        model_type("unused-checkpoint", conf=0.45)
