"""守卫无需 OpenCV 的实际路径及模型输入替代操作的数值行为。"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

from mineru.model.ocr.image import alpha_to_color, check_img, get_rotate_crop_image, rgb_to_bgr
from mineru.model.table.cls.mineru_table_ori_cls import MineruTableOrientationClsModel

_ROOT = Path(__file__).resolve().parents[2]
_REFERENCE = json.loads((_ROOT / "tests/fixtures/model_image_reference.json").read_text())
_BLOCK_CV2 = '''
import importlib.abc
import sys

class BlockCv2(importlib.abc.MetaPathFinder):
    """在实际入口执行前拒绝直接或间接的 OpenCV 导入。"""
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "cv2" or fullname.startswith("cv2."):
            raise ImportError("OpenCV must not load on this path")

sys.meta_path.insert(0, BlockCv2())
'''


def _run_blocked(script: str, *args: str) -> None:
    """在独立进程验证真实加载边界，避免其他测试已加载 cv2 掩盖泄漏。"""
    result = subprocess.run(
        [sys.executable, "-c", _BLOCK_CV2 + script, *args],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_pdf_and_model_helpers_import_without_opencv() -> None:
    """混合编排和模型模块的导入不提前加载 OpenCV 或实际表格模型。"""
    _run_blocked("""
import importlib
for module in (
    "mineru.parser", "mineru.render", "mineru.model.vlm.client",
    "mineru.backend.analysis.pdf.pipeline", "mineru.backend.analysis.pdf.window",
    "mineru.backend.analysis.pdf.formulas", "mineru.backend.analysis.pdf.ocr",
    "mineru.backend.analysis.pdf.tables", "mineru.backend.analysis.pdf.text.content",
    "mineru.model.runtime.hybrid", "mineru.model.ocr.image",
    "mineru.model.ocr.db_postprocess", "mineru.model.ocr.seal_crop",
    "mineru.model.table.cls.paddle_table_cls", "mineru.model.table.rec.unet_table.utils",
    "mineru.model.table.rec.unet_table.utils_table_line_rec",
):
    importlib.import_module(module)
assert "cv2" not in sys.modules
assert "mineru.model.table.rec.unet_table.main" not in sys.modules
assert "mineru.model.table.rec.slanet_plus.main" not in sys.modules
""")


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("ocr_mode", ["txt", "auto"])
def test_real_flash_pdf_parse_and_all_renderers_without_opencv(tmp_path: Path, asynchronous: bool, ocr_mode: str) -> None:
    """真实选页、自动分类、原生补图和九种渲染均不加载 OpenCV。"""
    source = tmp_path / "source.pdf"
    canvas = Canvas(str(source))
    for page in range(3):
        for line in range(12):
            canvas.drawString(50, 760 - line * 18, f"Page {page + 1}: Native text evidence line {line}.")
        with Image.new("RGB", (180, 90), (30, 80, 180)) as picture:
            canvas.drawImage(ImageReader(picture), 50, 350, width=180, height=90)
        canvas.showPage()
    canvas.save()
    _run_blocked(
        """
import asyncio
from pathlib import Path
from typing import Any
from mineru.config import get_config
from mineru.parser import parse, parse_async
from mineru.parser.writer import FileBasedDataWriter
from mineru.render import RenderFormat, render

get_config().llm_aided.features.title_leveling = False
get_config().llm_aided.features.cross_page_table_cell_merge = False
options = dict(tier="flash", ocr_mode=sys.argv[2], page_range="1,3")
result = asyncio.run(parse_async(sys.argv[1], **options)) if sys.argv[3] == "True" else parse(sys.argv[1], **options)
assert len(result.pages) == 2
assert result.middle_json.extensions["mineru"]["parse_mode"] == "txt"
assert "Page 1" in result.markdown() and "Page 3" in result.markdown()
assert "Page 2" not in result.markdown()
assert "image" in result.middle_json.model_dump_json()
assert "mineru.model.runtime.hybrid" not in sys.modules
before = result.middle_json.model_dump_json()
for output_format in RenderFormat:
    assert render(result.middle_json, output_format)
result.save(FileBasedDataWriter(str(Path(sys.argv[1]).parent / "export")))
assert before == result.middle_json.model_dump_json()
assert "cv2" not in sys.modules
""",
        str(source),
        ocr_mode,
        str(asynchronous),
    )


def test_simple_image_operations_without_opencv() -> None:
    """颜色转换、alpha 合成、直角旋转和轴对齐裁剪不进入 OpenCV。"""
    _run_blocked("""
import numpy as np
from mineru.model.ocr.image import alpha_to_color, check_img, get_rotate_crop_image, rgb_to_bgr
from mineru.model.table.cls.mineru_table_ori_cls import MineruTableOrientationClsModel
image = np.zeros((8, 10, 4), dtype=np.uint8)
assert rgb_to_bgr(image).shape == (8, 10, 3)
assert alpha_to_color(image).shape == (8, 10, 3)
assert check_img(image[:, :, 0]).shape == (8, 10, 3)
points = np.array([[1, 1], [7, 1], [7, 6], [1, 6]], dtype=np.float32)
assert get_rotate_crop_image(image, points).shape == (5, 6, 4)
assert MineruTableOrientationClsModel._rotate_image_by_label(image, "90").shape == (10, 8, 4)
assert "cv2" not in sys.modules
""")


def _assert_array_reference(actual: np.ndarray, reference: dict[str, object]) -> None:
    """按形状、位深和全部逻辑像素验证冻结参考，避免候选代码自证。"""
    assert list(actual.shape) == reference["shape"]
    assert str(actual.dtype) == reference["dtype"]
    assert hashlib.sha256(actual.tobytes()).hexdigest() == reference["sha256"]


def _assert_model_reference(actual: np.ndarray, key: str, index: int = 0) -> None:
    """读取明确命名的既有模型参考结果。"""
    _assert_array_reference(actual, _REFERENCE["model_arrays"][key][index])


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_model_channel_conversion_matches_reference(dtype: type, channels: int) -> None:
    """各种位深和非连续输入保留历史像素以及独立连续存储。"""
    image = np.arange(8 * 10 * channels).reshape(8, 10, channels).astype(dtype)[::2, ::2]
    actual = rgb_to_bgr(image)
    _assert_model_reference(actual, f"rgb:{np.dtype(dtype)}:{channels}")
    assert actual.flags.c_contiguous and not np.shares_memory(actual, image)


def test_grayscale_and_alpha_conversion_keep_rounding() -> None:
    """灰度复制及非白色背景 alpha 合成沿用冻结的逐通道截断结果。"""
    image = np.random.default_rng(7).integers(0, 256, (17, 13, 4), dtype=np.uint8)
    _assert_model_reference(check_img(image[:, :, 0]), "gray_alpha", 0)
    _assert_model_reference(alpha_to_color(image, (10, 20, 30)), "gray_alpha", 1)


@pytest.mark.parametrize("label", ["90", "270", "0"])
def test_orientation_rotation_keeps_pixels_and_ownership(label: str) -> None:
    """方向评分保持历史旋转像素和零角度副本行为。"""
    image = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)[::2, ::2]
    actual = MineruTableOrientationClsModel._rotate_image_by_label(image, label)
    _assert_model_reference(actual, f"rotation:{label}")
    assert actual.flags.c_contiguous and not np.shares_memory(actual, image)


def test_perspective_crop_keeps_numeric_contract() -> None:
    """高位深倾斜裁图保留历史透视矩阵、三次采样和边界复制结果。"""
    image = np.arange(20 * 30 * 3, dtype=np.uint16).reshape(20, 30, 3)
    points = np.array([[2, 2], [21, 4], [22, 15], [1, 14]], dtype=np.float32)
    _assert_model_reference(get_rotate_crop_image(image, points), "perspective")


@pytest.mark.parametrize("paddle_compatible", [False, True])
def test_formula_tensor_keeps_original_gray_and_channel_values(paddle_compatible: bool) -> None:
    """两种公式预处理保持冻结的灰度舍入和归一化模型张量。"""
    from mineru.model.mfr.pp_formulanet.processors import UniMERNetTestTransform

    image = np.random.default_rng(11).integers(0, 256, (17, 19, 3), dtype=np.uint8)
    actual = UniMERNetTestTransform(paddle_compatible=paddle_compatible).transform(image)
    _assert_model_reference(actual, f"formula:{paddle_compatible}")


@pytest.mark.parametrize("shape", [(1, 1), (1, 17), (19, 1), (23, 27)])
def test_formula_margin_crop_keeps_threshold_and_bounds(shape: tuple[int, int]) -> None:
    """空白图、单行单列和边界像素保持冻结裁边及原对象语义。"""
    from mineru.model.mfr.pp_formulanet.processors import UniMERNetImgDecode

    processor = UniMERNetImgDecode((192, 672))
    for blank in (True, False):
        array = np.full(shape, 255, dtype=np.uint8)
        if not blank:
            array[0, 0] = 0
            array[-1, -1] = 30
        with Image.fromarray(array) as image:
            actual = processor.crop_margin(image)
            if array.max() == array.min():
                assert actual is image
                continue
            _assert_model_reference(np.array(actual), f"margin:{shape[0]}:{shape[1]}")
            actual.close()


def test_table_classifier_tensor_keeps_channel_calculation_order() -> None:
    """单张和批量表格分类使用同一独立冻结张量验证浮点计算顺序。"""
    from mineru.model.table.cls.paddle_table_cls import PaddleTableClsModel

    classifier = object.__new__(PaddleTableClsModel)
    image = np.random.default_rng(13).integers(0, 256, (79, 137, 3), dtype=np.uint8)
    _assert_model_reference(classifier.preprocess(image), "table_classifier", 0)
    _assert_model_reference(classifier.batch_preprocess([image]), "table_classifier", 1)


def test_unet_input_tensor_keeps_inplace_channel_exchange() -> None:
    """UNet 预处理使用独立冻结输入，验证通道交换和浮点减乘顺序。"""
    from mineru.model.table.rec.unet_table.table_structure_unet import TSRUnet

    model = object.__new__(TSRUnet)
    model.mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    model.std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    model.inp_height, model.inp_width = 32, 64
    image = np.random.default_rng(17).integers(0, 256, (29, 47, 3), dtype=np.uint8)
    _assert_model_reference(model.preprocess(image)["img"], "unet")


@pytest.mark.parametrize("case", _REFERENCE["seal_cases"], ids=lambda case: case["name"] + "-" + case["interpolation"])
def test_seal_homography_keeps_frozen_pixels(case: dict[str, Any]) -> None:
    """原生内核验证横竖曲线、四边形、短多边形和退化处理的既有像素。"""
    from mineru.model.ocr.seal_det_warp import AutoRectifier

    image = np.random.default_rng(29).integers(0, 256, (96, 96, 3), dtype=np.uint8)
    original = image.copy()
    points = np.float32(case["points"])
    rectifier = AutoRectifier()
    actual = rectifier(image, points, interpolation=case["interpolation"])
    _assert_array_reference(actual, case["crop"])
    outputs, visual = rectifier.run(image, [points.reshape(-1).tolist()], interpolation=case["interpolation"])
    _assert_array_reference(outputs[0], case["crop"])
    _assert_array_reference(visual, case["visual"])
    np.testing.assert_array_equal(image, original)
    assert not np.shares_memory(actual, image)


def test_default_seal_rectification_runs_without_opencv() -> None:
    """独立阻断进程验证默认单张、批量和曲线入口均执行原 homography 结果。"""
    _run_blocked(
        """
import hashlib
import json
from pathlib import Path
import numpy as np
from mineru.model.ocr.seal_det_warp import AutoRectifier, CurveTextRectifier

reference = json.loads(Path(sys.argv[1]).read_text())
image = np.random.default_rng(29).integers(0, 256, (96, 96, 3), dtype=np.uint8)
for case in reference['seal_cases']:
    if case['interpolation'] != 'linear' or case['name'] not in ('horizontal', 'vertical'):
        continue
    points = np.float32(case['points'])
    expected = case['crop']['sha256']
    rectifier = AutoRectifier()
    single = rectifier(image, points)
    outputs, _ = rectifier.run(image, [points.reshape(-1).tolist()])
    curve, _ = CurveTextRectifier()(image, points)
    for actual in (single, outputs[0], curve):
        assert hashlib.sha256(actual.tobytes()).hexdigest() == expected
assert 'cv2' not in sys.modules
""",
        str(_ROOT / "tests/fixtures/model_image_reference.json"),
    )


@pytest.mark.parametrize("mode", ["calibration", "unknown"])
def test_seal_rejects_removed_mode(mode: str) -> None:
    """旧模式和拼写错误必须明确失败，不静默改变矫正算法。"""
    from mineru.model.ocr.seal_det_warp import AutoRectifier, CurveTextRectifier

    image = np.zeros((96, 96, 3), np.uint8)
    points = np.float32(_REFERENCE["seal_cases"][0]["points"])
    with pytest.raises(ValueError, match="homography"):
        CurveTextRectifier()(image, points, mode=mode)
    with pytest.raises(ValueError, match="homography"):
        AutoRectifier()(image, points, mode=mode)
    with pytest.raises(ValueError, match="homography"):
        AutoRectifier().run(image, [points.reshape(-1).tolist()], mode=mode)


def test_base_dependencies_exclude_opencv_and_keep_product_extras() -> None:
    """基础声明移除 OpenCV，ONNX、llama.cpp、Torch/full 和苹果自动 extra 保持原样。"""
    import tomllib
    from packaging.requirements import Requirement

    project = tomllib.loads((_ROOT / "pyproject.toml").read_text())["project"]
    requirements = [Requirement(value) for value in project["dependencies"]]
    assert not any(requirement.name.lower().replace("_", "-").startswith("opencv") for requirement in requirements)
    names = {requirement.name for requirement in requirements}
    assert {"onnxruntime", "mineru-llama-cpp", "docvortex"} <= names
    assert "mineru[torch] ; sys_platform == 'darwin' and platform_machine == 'arm64'" in project["dependencies"]
    extras = project["optional-dependencies"]
    assert {Requirement(value).name for value in extras["torch"]} == {
        "torch",
        "torchvision",
        "transformers",
        "accelerate",
        "safetensors",
    }
    assert {Requirement(value).name for value in extras["full"]} == {"mineru", "vllm", "lmdeploy", "qwen-vl-utils"}
    full_mineru = next(Requirement(value) for value in extras["full"] if Requirement(value).name == "mineru")
    assert full_mineru.extras == {"torch"}


def test_production_does_not_reference_opencv() -> None:
    """生产代码不允许导入、访问或动态请求 OpenCV，遗留白名单已移除。"""
    import ast

    root = _ROOT / "mineru"
    errors = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            is_import = isinstance(node, ast.Import) and any(alias.name.split(".")[0] == "cv2" for alias in node.names)
            is_from = isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "cv2"
            is_name = isinstance(node, ast.Name) and node.id == "cv2"
            is_string = (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and (node.value == "cv2" or node.value.startswith("cv2."))
            )
            if is_import or is_from or is_name or is_string:
                errors.append(f"{path.relative_to(root)}:{node.lineno}")
    assert not errors, "\n".join(errors)


def test_model_preprocessors_and_geometry_run_without_opencv() -> None:
    """独立阻断进程执行检测后处理、识别缩放、表格线和公式图像处理。"""
    _run_blocked("""
import numpy as np
from mineru.model.ocr.db_postprocess import DBPostProcess
from mineru.model.ocr.image import get_rotate_crop_image, resize_text_recognition_image
from mineru.model.table.rec.unet_table.utils import LoadImage
from mineru.model.table.rec.unet_table.utils_table_line_rec import draw_lines, _iter_connected_component_coords
from mineru.model.mfr.pp_formulanet.processors import UniMERNetTestTransform

pred = np.zeros((1, 1, 32, 48), np.float32)
pred[0, 0, 5:12, 8:30] = .85
pred[0, 0, 19:27, 23:42] = .95
result = DBPostProcess()({"maps": pred}, np.float32([[96, 144, 1, 1]]))
assert result[0]["points"].tolist() == [[[54, 42], [138, 42], [138, 93], [54, 93]], [[9, 0], [102, 0], [102, 48], [9, 48]]]
image = np.arange(40 * 80 * 3, dtype=np.uint8).reshape(40, 80, 3)
points = np.float32([[3, 5], [74, 7], [70, 30], [1, 28]])
crop = get_rotate_crop_image(image, points)
assert crop.size and crop.dtype == np.uint8 and crop.flags.c_contiguous
assert resize_text_recognition_image(crop, 8., (3, 48, 320)).dtype == np.float32
mask = draw_lines(np.zeros((60, 90), np.uint8), [[2, 30, 87, 30]], color=255, lineW=2)
assert len(list(_iter_connected_component_coords(mask))) == 1
rgba = np.zeros((10, 20, 4), np.uint8)
assert np.all(LoadImage.cvt_four_to_three(rgba) == 255)
assert "cv2" not in sys.modules
""")
