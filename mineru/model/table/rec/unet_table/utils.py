# Copyright (c) Opendatalab. All rights reserved.

from docvortex.image import resize_image
import traceback
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import loguru
import numpy as np
from onnxruntime import GraphOptimizationLevel, SessionOptions
from PIL import Image, ImageDraw, UnidentifiedImageError

from ....runtime.onnx import configure_ort_threads, table_ort_session

root_dir = Path(__file__).resolve().parent
InputType = Union[str, np.ndarray, bytes, Path]


class OrtInferSession:
    def __init__(self, config: Dict[str, Any]):
        """保留原有会话选项，并使用统一的表格设备与回退策略。"""
        self.logger = loguru.logger
        model_path = config.get("model_path", None)

        self.session = table_ort_session(
            model_path,
            sess_options=self._init_sess_opts(config),
            small_backend=config.get("small_backend"),
            model_kind="unet",
        )

    @staticmethod
    def _init_sess_opts(config: Dict[str, Any]) -> SessionOptions:
        """保留表格内存选项，线程数交给共享策略处理。"""
        sess_opt = SessionOptions()
        sess_opt.log_severity_level = 4
        sess_opt.enable_cpu_mem_arena = False
        sess_opt.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        configure_ort_threads(
            sess_opt,
            intra_op_num_threads=config.get("intra_op_num_threads", 0),
            inter_op_num_threads=config.get("inter_op_num_threads", 0),
        )

        return sess_opt

    def __call__(self, input_content: List[np.ndarray]) -> np.ndarray:
        input_dict = dict(zip(self.get_input_names(), input_content))
        try:
            return self.session.run(None, input_dict)
        except Exception as e:
            error_info = traceback.format_exc()
            raise ONNXRuntimeError(error_info) from e

    def get_input_names(self) -> List[str]:
        return [v.name for v in self.session.get_inputs()]


class ONNXRuntimeError(Exception):
    pass


class LoadImage:
    def __init__(
        self,
    ):
        pass

    def __call__(self, img: InputType) -> np.ndarray:
        if not isinstance(img, InputType.__args__):
            raise LoadImageError(f"The img type {type(img)} does not in {InputType.__args__}")

        img = self.load_img(img)
        img = self.convert_img(img)
        return img

    def load_img(self, img: InputType) -> np.ndarray:
        if isinstance(img, (str, Path)):
            self.verify_exist(img)
            try:
                img = np.array(Image.open(img))
            except UnidentifiedImageError as e:
                raise LoadImageError(f"cannot identify image file {img}") from e
            return img

        if isinstance(img, bytes):
            img = np.array(Image.open(BytesIO(img)))
            return img

        if isinstance(img, np.ndarray):
            return img

        raise LoadImageError(f"{type(img)} is not supported!")

    def convert_img(self, img: np.ndarray):
        if img.ndim == 2:
            return np.repeat(img[:, :, None], 3, axis=2)

        if img.ndim == 3:
            channel = img.shape[2]
            if channel == 1:
                return np.repeat(img, 3, axis=2)

            if channel == 2:
                return self.cvt_two_to_three(img)

            if channel == 4:
                return self.cvt_four_to_three(img)

            if channel == 3:
                return img[:, :, ::-1].copy(order="C")

            raise LoadImageError(f"The channel({channel}) of the img is not in [1, 2, 3, 4]")

        raise LoadImageError(f"The ndim({img.ndim}) of the img is not in [2, 3]")

    @staticmethod
    def cvt_four_to_three(img: np.ndarray) -> np.ndarray:
        """RGBA → BGR"""

        r, g, b, a = (img[:, :, index] for index in range(4))
        new_img = np.stack((b, g, r), axis=2)

        not_a = np.bitwise_not(a)
        not_a = np.repeat(not_a[:, :, None], 3, axis=2)

        new_img = np.where(a[:, :, None] != 0, new_img, 0)
        new_img = np.minimum(new_img.astype(np.uint16) + not_a, 255).astype(np.uint8)
        return new_img

    @staticmethod
    def cvt_two_to_three(img: np.ndarray) -> np.ndarray:
        """gray + alpha → BGR"""

        img_gray = img[..., 0]
        img_bgr = np.repeat(img_gray[:, :, None], 3, axis=2)

        img_alpha = img[..., 1]
        not_a = np.bitwise_not(img_alpha)
        not_a = np.repeat(not_a[:, :, None], 3, axis=2)

        new_img = np.where(img_alpha[:, :, None] != 0, img_bgr, 0)
        new_img = np.minimum(new_img.astype(np.uint16) + not_a, 255).astype(np.uint8)
        return new_img

    @staticmethod
    def verify_exist(file_path: Union[str, Path]):
        if not Path(file_path).exists():
            raise LoadImageError(f"{file_path} does not exist.")


class LoadImageError(Exception):
    pass


# Pillow >=v9.1.0 use a slightly different naming scheme for filters.
# Set pillow_interp_codes according to the naming scheme used.
if Image is not None:
    if hasattr(Image, "Resampling"):
        pillow_interp_codes = {
            "nearest": Image.Resampling.NEAREST,
            "bilinear": Image.Resampling.BILINEAR,
            "bicubic": Image.Resampling.BICUBIC,
            "box": Image.Resampling.BOX,
            "lanczos": Image.Resampling.LANCZOS,
            "hamming": Image.Resampling.HAMMING,
        }
    else:
        pillow_interp_codes = {
            "nearest": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "box": Image.BOX,
            "lanczos": Image.LANCZOS,
            "hamming": Image.HAMMING,
        }


def resize_img(img, scale, keep_ratio=True):
    if keep_ratio:
        # 缩小使用area更保真
        if min(img.shape[:2]) > min(scale):
            interpolation = "area"
        else:
            interpolation = "bicubic"  # bilinear
        img_new, scale_factor = imrescale(img, scale, return_scale=True, interpolation=interpolation)
        # the w_scale and h_scale has minor difference
        # a real fix should be done in the mmcv.imrescale in the future
        new_h, new_w = img_new.shape[:2]
        h, w = img.shape[:2]
        w_scale = new_w / w
        h_scale = new_h / h
    else:
        img_new, w_scale, h_scale = imresize(img, scale, return_scale=True)
    return img_new, w_scale, h_scale


def imrescale(img, scale, return_scale=False, interpolation="bilinear", backend=None):
    """Resize image while keeping the aspect ratio.

    Args:
        img (ndarray): The input image.
        scale (float | tuple[int]): The scaling factor or maximum size.
            If it is a float number, then the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, then the image will
            be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image.
        interpolation (str): Same as :func:`resize`.
        backend (str | None): Same as :func:`resize`.

    Returns:
        ndarray: The rescaled image.
    """
    h, w = img.shape[:2]
    new_size, scale_factor = rescale_size((w, h), scale, return_scale=True)
    rescaled_img = imresize(img, new_size, interpolation=interpolation, backend=backend)
    if return_scale:
        return rescaled_img, scale_factor
    else:
        return rescaled_img


def imresize(img, size, return_scale=False, interpolation="bilinear", out=None, backend=None):
    """Resize image to a given size.

    Args:
        img (ndarray): The input image.
        size (tuple[int]): Target size (w, h).
        return_scale (bool): Whether to return `w_scale` and `h_scale`.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'numeric'
            backend, "nearest", "bilinear" for 'pillow' backend.
        out (ndarray): The output destination.
        backend (str | None): The image resize backend type. Options are
            `numeric`, `pillow`, `None`. None selects the shared numeric kernel.

    Returns:
        tuple | ndarray: (`resized_img`, `w_scale`, `h_scale`) or
        `resized_img`.
    """
    h, w = img.shape[:2]
    if backend is None:
        backend = "numeric"
    if backend not in ["numeric", "pillow"]:
        raise ValueError(f"backend: {backend} is not supported for resize.Supported backends are 'numeric', 'pillow'")

    if backend == "pillow":
        assert img.dtype == np.uint8, "Pillow backend only support uint8 type"
        pil_image = Image.fromarray(img)
        pil_image = pil_image.resize(size, pillow_interp_codes[interpolation])
        resized_img = np.array(pil_image)
    else:

        interpolation_names = {
            "nearest": 'nearest',
            "bilinear": 'linear',
            "bicubic": 'cubic',
            "area": 'area',
            "lanczos": 'lanczos4',
        }
        resized_img = resize_image(img, size, interpolation=interpolation_names[interpolation])
        if out is not None and out.shape == resized_img.shape and out.dtype == resized_img.dtype:
            np.copyto(out, resized_img)
            resized_img = out
    if not return_scale:
        return resized_img
    else:
        w_scale = size[0] / w
        h_scale = size[1] / h
        return resized_img, w_scale, h_scale


def rescale_size(old_size, scale, return_scale=False):
    """Calculate the new size to be rescaled to.

    Args:
        old_size (tuple[int]): The old size (w, h) of image.
        scale (float | tuple[int]): The scaling factor or maximum size.
            If it is a float number, then the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, then the image will
            be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image size.

    Returns:
        tuple[int]: The new rescaled image size.
    """
    w, h = old_size
    if isinstance(scale, (float, int)):
        if scale <= 0:
            raise ValueError(f"Invalid scale {scale}, must be positive.")
        scale_factor = scale
    elif isinstance(scale, tuple):
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge / max(h, w), max_short_edge / min(h, w))
    else:
        raise TypeError(f"Scale must be a number or tuple of int, but got {type(scale)}")

    new_size = _scale_size((w, h), scale_factor)

    if return_scale:
        return new_size, scale_factor
    else:
        return new_size


def _scale_size(size, scale):
    """Rescale a size by a ratio.

    Args:
        size (tuple[int]): (w, h).
        scale (float | tuple(float)): Scaling factor.

    Returns:
        tuple[int]: scaled size.
    """
    if isinstance(scale, (float, int)):
        scale = (scale, scale)
    w, h = size
    return int(w * float(scale[0]) + 0.5), int(h * float(scale[1]) + 0.5)


class VisTable:
    def __init__(self):
        self.load_img = LoadImage()

    def __call__(
        self,
        img_path: Union[str, Path],
        table_results,
        save_html_path: Optional[Union[str, Path]] = None,
        save_drawed_path: Optional[Union[str, Path]] = None,
        save_logic_path: Optional[Union[str, Path]] = None,
    ):
        if save_html_path:
            html_with_border = self.insert_border_style(table_results.pred_html)
            self.save_html(save_html_path, html_with_border)

        table_cell_bboxes = table_results.cell_bboxes
        table_logic_points = table_results.logic_points
        if table_cell_bboxes is None:
            return None

        img = self.load_img(img_path)

        dims_bboxes = table_cell_bboxes.shape[1]
        if dims_bboxes == 4:
            drawed_img = self.draw_rectangle(img, table_cell_bboxes)
        elif dims_bboxes == 8:
            drawed_img = self.draw_polylines(img, table_cell_bboxes)
        else:
            raise ValueError("Shape of table bounding boxes is not between in 4 or 8.")

        if save_drawed_path:
            self.save_img(save_drawed_path, drawed_img)

        if save_logic_path:
            polygons = [[box[0], box[1], box[4], box[5]] for box in table_cell_bboxes]
            self.plot_rec_box_with_logic_info(img, save_logic_path, table_logic_points, polygons)
        return drawed_img

    def insert_border_style(self, table_html_str: str):
        style_res = """<meta charset="UTF-8"><style>
        table {
            border-collapse: collapse;
            width: 100%;
        }
        th, td {
            border: 1px solid black;
            padding: 8px;
            text-align: center;
        }
        th {
            background-color: #f2f2f2;
        }
                    </style>"""

        prefix_table, suffix_table = table_html_str.split("<body>")
        html_with_border = f"{prefix_table}{style_res}<body>{suffix_table}"
        return html_with_border

    def plot_rec_box_with_logic_info(self, img, output_path, logic_points, sorted_polygons):
        """在表格诊断图中标注逻辑行列，Pillow 画布只用于调试输出。"""
        padded = np.pad(img, ((0, 0), (0, 100), (0, 0)), constant_values=255)
        with Image.fromarray(padded[:, :, ::-1]) as canvas:
            draw = ImageDraw.Draw(canvas)
            for polygon, logic in zip(sorted_polygons, logic_points):
                x0, y0, x1, y1 = map(round, polygon[:4])
                draw.rectangle((x0, y0, x1, y1), outline=(255, 0, 0), width=1)
                draw.text((x0 + 3, y0 + 8), f"row: {logic[0]}-{logic[1]}", fill=(255, 0, 0))
                draw.text((x0 + 3, y0 + 18), f"col: {logic[2]}-{logic[3]}", fill=(255, 0, 0))
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            canvas.save(output_path)

    @staticmethod
    def draw_rectangle(img: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """在独立 BGR 诊断图上绘制矩形，保持原数组所有权。"""
        with Image.fromarray(img[:, :, ::-1]) as canvas:
            draw = ImageDraw.Draw(canvas)
            for box in boxes.astype(int):
                draw.rectangle(tuple(map(int, box)), outline=(0, 0, 255), width=2)
            return np.asarray(canvas)[:, :, ::-1].copy()

    @staticmethod
    def draw_polylines(img: np.ndarray, points) -> np.ndarray:
        """用 Pillow 标注表格多边形，点序及蓝色诊断语义保持不变。"""
        with Image.fromarray(img[:, :, ::-1]) as canvas:
            draw = ImageDraw.Draw(canvas)
            for point in points.astype(int):
                vertices = [tuple(map(int, xy)) for xy in point.reshape(4, 2)]
                draw.line(vertices + vertices[:1], fill=(0, 0, 255), width=2)
            return np.asarray(canvas)[:, :, ::-1].copy()

    @staticmethod
    def save_img(save_path: Union[str, Path], img: np.ndarray) -> None:
        """物化 BGR 诊断图后交由 Pillow 编码保存。"""
        with Image.fromarray(img[:, :, ::-1] if img.ndim == 3 else img) as canvas:
            canvas.save(save_path)

    @staticmethod
    def save_html(save_path: Union[str, Path], html: str):
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(html)
