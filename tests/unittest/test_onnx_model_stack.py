"""聚合模型仓库、CPU ONNX 与跨后端结果契约回归。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from mineru.model import download, registry
from mineru.model.ocr.resources import PPOCRV6_DICT_PATH


@pytest.mark.parametrize(
    ("small_backend", "vlm_engine", "tier", "names"),
    [
        ("torch", "vllm", "basic", ["MinerU-4_models_torch"]),
        ("torch", "vllm", "standard", ["MinerU-4_models_torch", "MinerU2.5-Pro-2605-1.2B"]),
        ("onnx", "llama-cpp", "basic", ["MinerU-4_models_onnx"]),
        ("onnx", "llama-cpp", "standard", ["MinerU-4_models_onnx", "MinerU2.5-Pro-2605-1.2B-GGUF"]),
    ],
)
def test_tier_resource_ownership(small_backend: str, vlm_engine: str, tier: str, names: list[str]) -> None:
    """档位资源与实际模型栈对应，旧仓库不再出现在下载集合。"""
    assert [
        repo.name for repo in registry.model_repos_for_tier(tier, small_backend=small_backend, vlm_engine=vlm_engine)
    ] == names
    assert registry.small_model_repo(small_backend).name == names[0]
    assert "PDF-Extract-Kit-1.0" not in registry.model_repo_names()


@pytest.mark.parametrize("whole_repo", [False, True])
def test_unpublished_source_does_not_invalidate_existing_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    whole_repo: bool,
) -> None:
    """未发布来源明确报错，不能删除现有完整标记或误下载旧仓库。"""
    monkeypatch.setattr(download.config.model, "base_dir", str(tmp_path))
    repo = registry.MINERU_4_MODELS_ONNX
    repo.local_dir().mkdir()
    marker = repo.local_dir() / download.MODEL_COMPLETE_MARKER
    marker.touch()
    remote = Mock(side_effect=AssertionError("Unexpected network request"))
    monkeypatch.setattr(download, "hf_snapshot_download", remote)
    monkeypatch.setattr(download, "ms_snapshot_download", remote)
    with pytest.raises(ValueError, match="not available from modelscope"):
        if whole_repo:
            download.download_model_repo(repo, source="modelscope")
        else:
            download.download_model_files(repo, [repo.ocr_det], source="modelscope")
    assert marker.is_file()
    remote.assert_not_called()


def test_onnx_required_paths_cannot_be_satisfied_by_legacy_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """旧缓存与根标记均不能替代新 ONNX 仓库的实际文件。"""
    monkeypatch.setattr(download.config.model, "base_dir", str(tmp_path))
    (tmp_path / "PDF-Extract-Kit-1.0").mkdir()
    repo = registry.MINERU_4_MODELS_ONNX
    repo.local_dir().mkdir()
    (repo.local_dir() / download.MODEL_COMPLETE_MARKER).touch()
    assert not download.verify_model_repo(repo).ready
    for resource in repo.required_paths():
        path = resource.local_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"payload")
    assert download.verify_model_repo(repo).ready
    repo.seal_det.local_path().unlink()
    assert download.verify_model_repo(repo).missing_paths == [repo.seal_det.relative_path]


def test_ort_providers_remain_cpu_on_accelerated_hosts(monkeypatch: pytest.MonkeyPatch) -> None:
    """即使宿主安装其他 providers，模型会话仍明确使用 CPU。"""
    from mineru.model.runtime import onnx

    monkeypatch.setattr(onnx.ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "OpenVINOExecutionProvider"])
    factory = Mock()
    monkeypatch.setattr(onnx.ort, "InferenceSession", factory)
    onnx.ort_session("model.onnx", device="cuda")
    assert factory.call_args.kwargs["providers"] == [("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})]


def test_ocr_txt_dictionary_ctc_space_and_shape_validation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """字符顺序、CTC blank 和尾部空格不得重复或错位。"""
    from mineru.model.ocr import pp_ocr_v6_onnx as ocr

    dictionary = tmp_path / "dict.txt"
    dictionary.write_text("中\nA\n", encoding="utf-8")
    session = SimpleNamespace(
        get_inputs=lambda: [SimpleNamespace(name="image")],
        get_outputs=lambda: [SimpleNamespace(shape=["batch", "width", 4])],
    )
    monkeypatch.setattr(ocr, "ort_session", lambda *args: session)
    recognizer = ocr.TextRecognizerONNX("rec.onnx", str(dictionary))
    logits = np.eye(4, dtype=np.float32)[[1, 1, 0, 2, 3, 3, 0, 1]]
    assert recognizer._decode(logits) == ("中A 中", 1.0)
    with pytest.raises(ValueError, match="CTC output shape"):
        recognizer._decode(np.zeros((3, 5)))
    dictionary.write_text("中\nA\nB\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dictionary requires"):
        ocr.TextRecognizerONNX("rec.onnx", str(dictionary))
    assert len(PPOCRV6_DICT_PATH.read_text().splitlines()) == 18708


def test_formula_invalid_boxes_do_not_shift_latex_between_items(monkeypatch: pytest.MonkeyPatch) -> None:
    """无效框保留空公式，后续有效公式与跨页结果必须回填到原目标。"""
    from mineru.model.mfr.pp_formulanet_plus_m_onnx import PPFormulaNetPlusMONNX

    model = object.__new__(PPFormulaNetPlusMONNX)
    monkeypatch.setattr(model, "_infer_batch", lambda crops, batch_size: ["first", "second"])
    source = [
        [{"label": "display_formula", "bbox": [-8, 0, -1, 5]}, {"label": "inline_formula", "bbox": [1, 1, 8, 8]}],
        [{"label": "display_formula", "bbox": [2, 2, 9, 9]}],
    ]
    results = model.batch_predict(source, [np.zeros((10, 10, 3), dtype=np.uint8)] * 2)
    assert [[item["latex"] for item in page] for page in results] == [["", "first"], ["second"]]
    assert all("latex" not in item for page in source for item in page)


def test_recognition_preprocess_handles_thin_and_long_crops() -> None:
    """短边至少保留 16 像素，超长行宽度有界，填充区域保持零。"""
    from mineru.model.ocr.image import resize_text_recognition_image

    narrow = resize_text_recognition_image(np.full((100, 1, 3), 255, dtype=np.uint8), 0.01, (3, 48, 320))
    assert narrow.shape == (3, 48, 320)
    assert narrow.dtype == np.float32
    np.testing.assert_array_equal(narrow[:, :, :16], 1)
    np.testing.assert_array_equal(narrow[:, :, 16:], 0)
    wide = resize_text_recognition_image(np.zeros((10, 1000, 3), dtype=np.uint8), 100, (3, 48, 320))
    assert wide.shape == (3, 48, 2560)
    np.testing.assert_array_equal(wide, -1)


def test_detection_boxes_can_be_cropped_without_box_merging() -> None:
    """未经过框合并的检测点仍须满足 OpenCV 透视变换的 float32 契约。"""
    from mineru.model.ocr.image import get_rotate_crop_image_for_text_rec
    from mineru.model.ocr.pp_ocr_v6_onnx import TextDetectorONNX

    detector = object.__new__(TextDetectorONNX)
    boxes = detector._filter_det_res(np.array([[[2, 2], [15, 2], [15, 10], [2, 10]]], dtype=np.int16), (20, 20))
    assert boxes.dtype == np.float32
    crop = get_rotate_crop_image_for_text_rec(np.zeros((20, 20, 3), dtype=np.uint8), boxes[0])
    assert crop is not None and crop.shape[:2] == (8, 13)


def test_formula_eos_stops_before_trailing_tokens() -> None:
    """ONNX 图 EOS 后的填充或残留 token 不进入共享解码器。"""
    from mineru.model.mfr.pp_formulanet_plus_m_onnx import PPFormulaNetPlusMONNX

    model = object.__new__(PPFormulaNetPlusMONNX)
    model.decoder = Mock(return_value=["x"])
    assert model._decode_tokens(np.array([[0, 17, 2, 99, 1]])) == "x"
    np.testing.assert_array_equal(model.decoder.call_args.args[0], [[0, 17, 2]])


def test_layout_batch_keeps_independent_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    """外部批量接口真正合批，并保持每页独立预测及尾批。"""
    from mineru.model.layout.pp_doclayout_v2_onnx import PPDocLayoutV2LayoutModelONNX

    model = object.__new__(PPDocLayoutV2LayoutModelONNX)
    model.use_paddlex_filter_boxes = False
    monkeypatch.setattr(model, "_preprocess_single_image", lambda image: (image, (10, 10)))
    calls = []

    def infer(pixels: np.ndarray, sizes: list[tuple[int, int]]) -> list[dict]:
        """用图像标记模拟各页不同的模型预测。"""
        assert pixels.shape[0] == len(sizes)
        calls.append(len(sizes))
        return [{"page": int(page.mean())} for page in pixels]

    monkeypatch.setattr(model, "_run_session", infer)
    monkeypatch.setattr(model, "_parse_prediction", lambda prediction, size: [prediction])
    monkeypatch.setattr(model, "_apply_layout_post_process", lambda result, image_size: result)
    assert model.batch_predict([np.full((3, 8, 8), i) for i in (1, 2, 3)], batch_size=2) == [
        [{"page": 1}],
        [{"page": 2}],
        [{"page": 3}],
    ]
    assert calls == [2, 1]


def test_seal_uses_poly_crops_and_keeps_variable_vertex_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    """印章检测参数独立，变长多边形不得被四边形合并或 NumPy 堆叠破坏。"""
    from mineru.model.ocr import pp_ocr_v6_onnx as ocr

    detector = Mock(return_value=([np.zeros((5, 2)), np.ones((7, 2))], 0.0))
    detector_factory = Mock(return_value=detector)
    monkeypatch.setattr(ocr, "TextDetectorONNX", detector_factory)
    recognizer = Mock(return_value=([("印章", 0.1), ("测试", 0.0)], 0.0))
    monkeypatch.setattr(ocr, "TextRecognizerONNX", Mock(return_value=recognizer))
    model = ocr.PPOCRv6ONNX("seal.onnx", "rec.onnx", lang="seal")
    assert model.is_seal and model.drop_score == 0
    assert detector_factory.call_args.kwargs["box_type"] == "poly"
    assert detector_factory.call_args.kwargs["box_thresh"] == 0.6
    assert detector_factory.call_args.kwargs["limit_side_len"] == 736
    crops = [np.zeros((10, 30, 3), dtype=np.uint8)] * 2
    monkeypatch.setattr(model, "_seal_crop_by_polys", Mock(return_value=crops))
    boxes, texts = model._det_rec(np.zeros((50, 50, 3), dtype=np.uint8))
    assert [len(box) for box in boxes] == [5, 7]
    assert [text for text, _ in texts] == ["印章", "测试"]
    assert recognizer.call_args.args[0] is crops


def test_atom_and_context_caches_are_isolated_by_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    """同一 CPU 上的 full/light 原子模型与上下文必须分别缓存。"""
    from mineru.model.runtime import hybrid

    monkeypatch.setattr(hybrid.AtomModelSingleton, "_models", {})
    monkeypatch.setattr(hybrid.HybridLocalModelContextSingleton, "_models", {})
    factory = Mock(side_effect=lambda **kwargs: object())
    monkeypatch.setattr(hybrid, "atom_model_init", factory)
    manager = hybrid.AtomModelSingleton()
    full = manager.get_atom_model("ocr", small_backend="torch", device="cpu")
    light = manager.get_atom_model("ocr", small_backend="onnx", device="mps")
    assert full is not light
    assert manager.get_atom_model("ocr", small_backend="onnx", device="cpu") is light
    monkeypatch.setattr(hybrid, "HybridLocalModelContext", factory)
    monkeypatch.setattr(hybrid, "get_device", lambda: "cpu")
    context_manager = hybrid.HybridLocalModelContextSingleton()
    monkeypatch.setattr(hybrid, "resolve_small_model_backend", lambda: "torch")
    full_context = context_manager.get_model()
    monkeypatch.setattr(hybrid, "resolve_small_model_backend", lambda: "onnx")
    assert context_manager.get_model() is not full_context
    assert factory.call_args.kwargs == {"small_backend": "onnx", "device": "cpu"}


def test_light_standard_does_not_probe_platform_engines(monkeypatch: pytest.MonkeyPatch) -> None:
    """CPU 自动选择 GGUF，不加载平台 VLM 引擎。"""
    from mineru.model.vlm import selector

    monkeypatch.setattr(selector, "get_device", lambda: "cpu")
    monkeypatch.setattr(selector, "module_available", Mock(side_effect=AssertionError("engine probe")))
    assert selector.get_vlm_engine("auto") == "llama-cpp-engine"


def test_light_table_models_use_only_the_onnx_bundle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """三种表格模型从 Light 仓库取文件，并把相同 small_backend 传给共享 OCR。"""
    from mineru.model.runtime import hybrid

    resources = []

    def ensure(resource: download.ModelPath) -> Path:
        """拒绝跨仓库下载并记录表格文件。"""
        assert resource.repo is registry.MINERU_4_MODELS_ONNX
        resources.append(resource.name)
        return tmp_path / resource.relative_path

    monkeypatch.setattr(download.ModelPath, "ensure", ensure)
    ocr_factory = Mock(return_value=object())
    monkeypatch.setattr(hybrid.AtomModelSingleton, "get_atom_model", ocr_factory)
    monkeypatch.setattr(hybrid, "PaddleTableModel", Mock())
    monkeypatch.setattr(hybrid, "UnetTableModel", Mock())
    monkeypatch.setattr(hybrid, "PaddleTableClsModel", Mock())
    hybrid.wireless_table_model_init(small_backend="onnx", device="cpu")
    hybrid.wired_table_model_init(small_backend="onnx", device="cpu")
    hybrid.table_cls_model_init(small_backend="onnx")
    assert resources == ["slanet_plus", "unet_structure", "paddle_table_cls"]
    assert all(call.kwargs["small_backend"] == "onnx" for call in ocr_factory.call_args_list)


def test_light_seal_uses_dedicated_model_and_packaged_dictionary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Seal 不能退回普通检测图，也不能通过加载字典触发 Torch 仓库下载。"""
    from mineru.model.ocr import pp_ocr_v6_onnx
    from mineru.model.runtime.hybrid import atom_model_init

    resources = []

    def ensure(resource: download.ModelPath) -> Path:
        """记录 Light 印章初始化所需的精确资源。"""
        assert resource.repo is registry.MINERU_4_MODELS_ONNX
        resources.append(resource.name)
        return tmp_path / resource.relative_path

    monkeypatch.setattr(download.ModelPath, "ensure", ensure)
    factory = Mock()
    monkeypatch.setattr(pp_ocr_v6_onnx, "PPOCRv6ONNX", factory)
    atom_model_init("ocr", small_backend="onnx", lang="seal")
    assert resources == ["seal_det", "ocr_rec"]
    assert factory.call_args.kwargs["lang"] == "seal"
    assert factory.call_args.kwargs["dict_path"] == str(PPOCRV6_DICT_PATH)


def test_light_model_modules_import_without_torch() -> None:
    """共享字典、DB、印章裁剪及公式处理器均不能强制导入 Torch。"""
    code = """
import importlib.abc
import os
import sys
os.environ["MINERU_MODEL_SMALL_BACKEND"] = "onnx"
class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        # 阻断推理重依赖，验证共享资源边界。
        if fullname.split(".")[0] in {"torch", "transformers"}:
            raise ModuleNotFoundError(fullname, name=fullname)
        return None
sys.meta_path.insert(0, BlockTorch())
from mineru.model.ocr.pp_ocr_v6_onnx import PPOCRv6ONNX
from mineru.model.mfr.pp_formulanet_plus_m_onnx import PPFormulaNetPlusMONNX
from mineru.model.runtime.hybrid import ocr_det_batch_setting
assert ocr_det_batch_setting()
assert "torch" not in sys.modules
print("ok")
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_formula_cpu_batch_eight_sorts_and_restores_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证真实合批上限、尾批、空裁剪、面积排序及回填顺序。"""
    from mineru.model.mfr.pp_formulanet_plus_m_onnx import PPFormulaNetPlusMONNX

    model = object.__new__(PPFormulaNetPlusMONNX)
    model.input_name = "image"
    calls = []

    def infer(outputs: object, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        """用每张图的标记作为 token，记录送入 ORT 的真正 batch。"""
        values = feed["image"][:, 0, 0, 0].astype(np.int64)
        calls.append(values.tolist())
        return [values[:, None]]

    model.session = SimpleNamespace(run=infer)
    monkeypatch.setattr(model, "_preprocess", lambda crop: None if crop.size == 0 else np.full((1, 1, 2, 2), crop[0, 0, 0]))
    monkeypatch.setattr(model, "_decode_tokens", lambda tokens: str(tokens[0]))
    crops = [np.full((i, i, 3), i, dtype=np.uint8) for i in range(10, 0, -1)]
    crops.insert(3, np.empty((0, 0, 3), dtype=np.uint8))
    assert model._infer_batch(crops, batch_size=64) == ["10", "9", "8", "", "7", "6", "5", "4", "3", "2", "1"]
    assert calls == [list(range(1, 9)), [9, 10]]
    calls.clear()
    model._infer_batch(crops[:3], batch_size=2)
    assert calls == [[8, 9], [10]]
