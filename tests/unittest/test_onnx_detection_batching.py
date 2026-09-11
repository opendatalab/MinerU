"""ONNX 版面与文字检测真批处理的输入、输出及生命周期约束。"""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from mineru.model.layout.pp_doclayout_v2_onnx import PPDocLayoutV2LayoutModelONNX
from mineru.model.ocr.pp_ocr_v6_onnx import TextDetectorONNX


def layout_model(counts: np.ndarray) -> PPDocLayoutV2LayoutModelONNX:
    """构造含零框页、低置信框和阅读顺序的受控版面模型。"""
    model = object.__new__(PPDocLayoutV2LayoutModelONNX)
    model.imgsz = (800, 800)
    model.conf = 0.5
    model._input_names = ["image", "im_shape", "scale_factor"]
    rows = np.array(
        [[1, 0.9, 1, 2, 3, 4, 2, 0], [2, 0.8, 5, 6, 7, 8, 1, 0], [9, 0.1, 1, 1, 2, 2, 0, 0], [3, 0.7, 10, 11, 12, 13, 0, 0]],
        np.float32,
    )
    model.session = SimpleNamespace(run=Mock(return_value=[rows, counts]))
    return model


def test_layout_counts_split_zero_page_and_preserve_coordinates() -> None:
    """逐页框数量必须先拆分，再独立过滤排序；缩放参数按原页尺寸构造。"""
    model = layout_model(np.array([3, 0, 1], np.int32))
    result = model._run_session(np.zeros((3, 3, 800, 800), np.float32), [(100, 200), (200, 400), (400, 800)])
    assert [row["labels"].tolist() for row in result] == [[2, 1], [], [3]]
    np.testing.assert_array_equal(result[2]["boxes"], [[10, 11, 12, 13]])
    feed = model.session.run.call_args.args[1]
    np.testing.assert_array_equal(feed["scale_factor"], [[8, 4], [4, 2], [2, 1]])
    np.testing.assert_array_equal(feed["im_shape"], [[800, 800]] * 3)


@pytest.mark.parametrize(
    "counts", [np.array([4]), np.array([2, -1, 3]), np.array([1, 1, 1]), np.array([3.0, 0.0, 1.0]), np.array([[3, 0, 1]])]
)
def test_layout_rejects_invalid_counts(counts: np.ndarray) -> None:
    """禁止错误的数量、维度或类型造成页面预测错配。"""
    with pytest.raises(ValueError, match="counts"):
        layout_model(counts)._run_session(np.zeros((3, 3, 800, 800), np.float32), [(100, 100)] * 3)


def detector_model(monkeypatch: pytest.MonkeyPatch) -> tuple[TextDetectorONNX, list[list[int]]]:
    """以输入标记模拟概率图，记录真实送入 ORT 的同尺寸批次。"""
    model = object.__new__(TextDetectorONNX)
    model.input_name = "image"
    calls = []

    def preprocess(img: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """保留尺寸和图像标记，模拟逐图缩放元数据。"""
        if not img.size:
            return None
        return img.transpose(2, 0, 1)[None], np.array([[*img.shape[:2], 1, 1]])

    def infer(outputs: object, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        """同批形状必须一致，返回独立的单通道概率图。"""
        pixels = feed["image"]
        calls.append(pixels[:, 0, 0, 0].astype(int).tolist())
        return [pixels[:, :1]]

    def postprocess(pred: np.ndarray, shape: np.ndarray) -> list[np.ndarray]:
        """模拟印章变长多边形，确保概率图和原图信息按同一索引对应。"""
        assert tuple(pred.shape[2:]) == tuple(shape[:2])
        marker = int(pred[0, 0, 0, 0])
        return [np.full((marker + 4, 2), marker)]

    monkeypatch.setattr(model, "_preprocess", preprocess)
    monkeypatch.setattr(model, "_postprocess", postprocess)
    model.session = SimpleNamespace(run=Mock(side_effect=infer))
    return model, calls


def test_detector_buckets_tails_empty_images_and_external_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    """交错尺寸按桶合批，空图不入图，结果及进度恢复原输入顺序。"""
    model, calls = detector_model(monkeypatch)
    images = [np.full((h, h, 3), i, np.float32) for i, h in [(1, 2), (2, 3), (3, 2), (4, 3), (5, 2)]]
    images.insert(2, np.empty((0, 0, 3)))
    progress = Mock()
    results = model.batch_predict(images, max_batch_size=2, tqdm_progress_bar=progress)
    assert calls == [[1, 3], [2, 4], [5]]
    assert [None if boxes is None else int(boxes[0][0, 0]) for boxes, _ in results] == [1, 2, None, 3, 4, 5]
    assert results[2] == (None, 0.0)
    assert all(elapsed >= 0 for _, elapsed in results)
    assert sum(call.args[0] for call in progress.update.call_args_list) == 6
    progress.close.assert_not_called()
    calls.clear()
    assert int(model(images[0])[0][0][0, 0]) == 1
    assert calls == [[1]]


def test_detector_closes_only_owned_progress_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """异常传播时关闭自建进度条，不关闭调用方拥有的进度条。"""
    from mineru.model.ocr import pp_ocr_v6_onnx

    model, _ = detector_model(monkeypatch)
    model.session.run.side_effect = RuntimeError("inference failed")
    progress = Mock()
    monkeypatch.setattr(pp_ocr_v6_onnx, "tqdm", Mock(return_value=progress))
    with pytest.raises(RuntimeError, match="inference failed"):
        model.batch_predict([np.zeros((2, 2, 3))])
    progress.close.assert_called_once()
    progress.reset_mock()
    with pytest.raises(RuntimeError):
        model.batch_predict([np.zeros((2, 2, 3))], tqdm_progress_bar=progress)
    progress.close.assert_not_called()


@pytest.mark.parametrize("size", [0, -1])
def test_detection_batch_sizes_must_be_positive(size: int) -> None:
    """即使输入为空，也应明确拒绝非法批次上限。"""
    with pytest.raises(ValueError, match="batch_size"):
        object.__new__(TextDetectorONNX).batch_predict([], max_batch_size=size)
    with pytest.raises(ValueError, match="batch_size"):
        object.__new__(PPDocLayoutV2LayoutModelONNX).batch_predict([], batch_size=size)


def test_detector_rejects_wrong_output_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """模型输出少于输入批次时必须报错，不能通过 zip 静默丢图。"""
    model, _ = detector_model(monkeypatch)
    model.session.run.side_effect = None
    model.session.run.return_value = [np.zeros((1, 1, 2, 2))]
    with pytest.raises(ValueError, match="output batch"):
        model.batch_predict([np.zeros((2, 2, 3))] * 2, max_batch_size=2)


@pytest.mark.parametrize("outputs", [[np.zeros((0, 8))], [np.zeros((1, 7)), np.array([1, 0, 0])]])
def test_layout_rejects_missing_counts_or_wrong_box_shape(outputs: list[np.ndarray]) -> None:
    """缺少数量输出或框形状不符时拒绝继续拆分。"""
    model = layout_model(np.array([3, 0, 1]))
    model.session.run.return_value = outputs
    with pytest.raises(ValueError, match="Layout"):
        model._run_session(np.zeros((3, 3, 800, 800)), [(100, 100)] * 3)
