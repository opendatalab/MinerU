# Copyright (c) Opendatalab. All rights reserved.
"""Hybrid 与本地小模型共享的运行时初始化模块。"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from functools import cached_property
from typing import TYPE_CHECKING, Any

from loguru import logger

from ..ocr.language import normalize_ocr_model_lang
from ..table.cls.mineru_table_ori_cls import MineruTableOrientationClsModel
from ..table.cls.paddle_table_cls import PaddleTableClsModel
from ..table.rec.slanet_plus.main import PaddleTableModel
from ..table.rec.unet_table.main import UnetTableModel
from .contracts import AtomicModelName

if TYPE_CHECKING:
    from ..layout.pp_doclayoutv2 import PPDocLayoutV2LayoutModel
    from ..mfr.pp_formulanet_plus_m.predict_formula import FormulaRecognizer
    from ..mfr.unimernet.Unimernet import UnimernetModel
    from ..ocr.pytorch_paddle import PytorchPaddleOCR

from ..registry import PDF_EXTRACT_KIT, ModelPath
from .device import get_device, get_model_stack

LOCAL_MODEL_INIT_LOCK = threading.RLock()
# 这些锁保护 Hybrid medium/high/xhigh 共享的 atom model/native 模型推理调用，避免多线程同时进入同一个模型对象。
LOCAL_MODEL_LAYOUT_INFERENCE_LOCK = threading.RLock()
LOCAL_MODEL_MFR_INFERENCE_LOCK = threading.RLock()
LOCAL_MODEL_OCR_INFERENCE_LOCK = threading.RLock()


def _read_bool_env(primary_name: str, fallback_name: str | None = None, default: bool = False) -> bool:
    """读取布尔环境变量；新变量未配置时回退到旧变量，保持已有部署兼容。"""
    raw_value = os.getenv(primary_name)
    if raw_value is None and fallback_name is not None:
        raw_value = os.getenv(fallback_name)
    if raw_value is None:
        return default
    return raw_value.lower() in ["true", "1", "yes"]


# 临时关闭共享推理阶段锁；旧 PIPELINE 变量仅作兼容回退，新的 Hybrid 本地变量优先生效。
LOCAL_MODEL_INFERENCE_LOCKS_ENABLED = _read_bool_env(
    "MINERU_ENABLE_LOCAL_MODEL_INFERENCE_LOCKS",
    fallback_name="MINERU_ENABLE_PIPELINE_INFERENCE_LOCKS",
    default=False,
)


def _run_with_inference_lock(
    inference_lock: threading.RLock, inference_callable: Callable[..., Any], *args: Any, **kwargs: Any
) -> object:
    """按实验开关决定是否在指定推理锁内执行真实 native 模型调用。"""
    if not LOCAL_MODEL_INFERENCE_LOCKS_ENABLED:
        return inference_callable(*args, **kwargs)

    with inference_lock:
        return inference_callable(*args, **kwargs)


def run_layout_inference(inference_callable: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """按实验开关执行共享 Layout 模型调用。"""
    return _run_with_inference_lock(LOCAL_MODEL_LAYOUT_INFERENCE_LOCK, inference_callable, *args, **kwargs)


def run_mfr_inference(inference_callable: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """按实验开关执行共享 MFR 模型调用。"""
    return _run_with_inference_lock(LOCAL_MODEL_MFR_INFERENCE_LOCK, inference_callable, *args, **kwargs)


def run_ocr_inference(inference_callable: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """按实验开关执行共享 OCR native 模型调用。"""
    return _run_with_inference_lock(LOCAL_MODEL_OCR_INFERENCE_LOCK, inference_callable, *args, **kwargs)


def _resolve_mfr_model_path() -> ModelPath:
    """解析公式识别模型路径。"""
    return PDF_EXTRACT_KIT.unimernet_small


def table_orientation_cls_model_init() -> MineruTableOrientationClsModel:
    """初始化表格方向分类包装器，并注入适配方向检测的 OCR 引擎。"""
    atom_model_manager = AtomModelSingleton()
    ocr_engine = atom_model_manager.get_atom_model(
        atom_model_name=AtomicModelName.OCR,
        det_db_box_thresh=0.5,
        det_db_unclip_ratio=1.6,
        lang="ch",
        enable_merge_det_boxes=False,
    )
    cls_model = MineruTableOrientationClsModel(ocr_engine)
    return cls_model


def table_cls_model_init() -> PaddleTableClsModel:
    """初始化有线与无线表格类型分类模型。"""
    return PaddleTableClsModel()


def wired_table_model_init(lang: str | None = None) -> UnetTableModel:
    """初始化有线表格识别模型，并注入指定语言的 OCR 引擎。"""
    atom_model_manager = AtomModelSingleton()
    ocr_engine = atom_model_manager.get_atom_model(
        atom_model_name=AtomicModelName.OCR,
        det_db_box_thresh=0.5,
        det_db_unclip_ratio=1.6,
        lang=lang,
        enable_merge_det_boxes=False,
    )
    table_model = UnetTableModel(ocr_engine)
    return table_model


def wireless_table_model_init(lang: str | None = None) -> PaddleTableModel:
    """初始化无线表格识别模型，并注入指定语言的 OCR 引擎。"""
    atom_model_manager = AtomModelSingleton()
    ocr_engine = atom_model_manager.get_atom_model(
        atom_model_name=AtomicModelName.OCR,
        det_db_box_thresh=0.5,
        det_db_unclip_ratio=1.6,
        lang=lang,
        enable_merge_det_boxes=False,
    )
    table_model = PaddleTableModel(ocr_engine)
    return table_model


def mfr_model_init(weight_dir: str, device: str = "cpu") -> "UnimernetModel":
    from ..mfr.unimernet.Unimernet import UnimernetModel

    return UnimernetModel(weight_dir, device)


def pp_doclayout_v2_model_init(weight: str, device: str = "cpu") -> "PPDocLayoutV2LayoutModel":
    """在指定设备上初始化 PP-DocLayoutV2 版面分析模型。"""

    from ..layout.pp_doclayoutv2 import PPDocLayoutV2LayoutModel

    if str(device).startswith("npu"):
        import torch

        device = torch.device(device)
    model = PPDocLayoutV2LayoutModel(weight, device)
    return model


def ocr_model_init(
    det_db_box_thresh: float = 0.5,
    lang: str | None = None,
    det_db_unclip_ratio: float = 1.5,
    enable_merge_det_boxes: bool = True,
) -> "PytorchPaddleOCR":
    """按语言和检测阈值初始化本地 Paddle OCR 模型。"""

    from ..ocr.pytorch_paddle import PytorchPaddleOCR

    ocr_kwargs = {
        "lang": normalize_ocr_model_lang(lang),
        "det_db_box_thresh": det_db_box_thresh,
        "det_db_unclip_ratio": det_db_unclip_ratio,
        "enable_merge_det_boxes": enable_merge_det_boxes,
    }
    return PytorchPaddleOCR(**ocr_kwargs)


class AtomModelSingleton:
    """按模型配置缓存并复用本地原子模型实例。"""

    _instance: AtomModelSingleton | None = None
    _models: dict[object, object] = {}
    _lock: threading.RLock = LOCAL_MODEL_INIT_LOCK

    def __new__(cls, *args: Any, **kwargs: Any) -> AtomModelSingleton:
        """在线程锁保护下创建或返回唯一的原子模型管理器。"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def get_atom_model(self, atom_model_name: str, **kwargs: Any) -> Any:
        """根据模型名称和关键配置生成缓存键，并获取对应原子模型。"""
        lang = kwargs.get("lang", None)
        ocr_singleton_lang = normalize_ocr_model_lang(lang)

        if atom_model_name in [AtomicModelName.WiredTable, AtomicModelName.WirelessTable]:
            key = (atom_model_name, ocr_singleton_lang)
        elif atom_model_name in [AtomicModelName.OCR]:
            key = (
                atom_model_name,
                kwargs.get("det_db_box_thresh", 0.5),
                ocr_singleton_lang,
                kwargs.get("det_db_unclip_ratio", 1.5),
                kwargs.get("enable_merge_det_boxes", True),
            )
        elif atom_model_name in [AtomicModelName.Layout, AtomicModelName.MFR]:
            key = (
                atom_model_name,
                kwargs.get("device"),
            )
        else:
            key = atom_model_name

        with self._lock:
            if key not in self._models:
                self._models[key] = atom_model_init(model_name=atom_model_name, **kwargs)
        return self._models[key]


def atom_model_init(model_name: str, **kwargs: Any) -> Any:
    """将原子模型名称分派到具体初始化函数，并校验初始化结果。"""
    stack = get_model_stack()
    atom_model = None
    if model_name == AtomicModelName.Layout:
        if stack == "light":
            from ..layout.pp_doclayout_v2_onnx import PPDocLayoutV2LayoutModelONNX
            from ..registry import PP_DOCLAYOUT_V2_ONNX

            atom_model = PPDocLayoutV2LayoutModelONNX(
                weight=str(PP_DOCLAYOUT_V2_ONNX.onnx.ensure()),
                device=kwargs.get("device"),
            )
        else:
            atom_model = pp_doclayout_v2_model_init(kwargs.get("pp_doclayout_v2_weights"), kwargs.get("device"))
    elif model_name == AtomicModelName.MFR:
        if stack == "light":
            from ..mfr.pp_formulanet_plus_m_onnx import PPFormulaNetPlusMONNX
            from ..registry import PP_FORMULANET_PLUS_M_ONNX

            atom_model = PPFormulaNetPlusMONNX(
                model_path=str(PP_FORMULANET_PLUS_M_ONNX.onnx.ensure()),
                config_path=str(PP_FORMULANET_PLUS_M_ONNX.config.ensure()),
                device=kwargs.get("device"),
            )
        else:
            atom_model = mfr_model_init(kwargs.get("mfr_weight_dir"), kwargs.get("device"))
    elif model_name == AtomicModelName.OCR:
        if stack == "light":
            from ..ocr.pp_ocr_v6_onnx import PPOCRv6ONNX
            from ..registry import PP_OCR_V6_SMALL_DET_ONNX, PP_OCR_V6_SMALL_REC_ONNX

            atom_model = PPOCRv6ONNX(
                det_model_path=str(PP_OCR_V6_SMALL_DET_ONNX.onnx.ensure()),
                rec_model_path=str(PP_OCR_V6_SMALL_REC_ONNX.onnx.ensure()),
                dict_path=str(PP_OCR_V6_SMALL_REC_ONNX.config.ensure()),
                device=kwargs.get("device"),
                det_db_box_thresh=kwargs.get("det_db_box_thresh", 0.5),
                det_db_unclip_ratio=kwargs.get("det_db_unclip_ratio", 1.5),
                enable_merge_det_boxes=kwargs.get("enable_merge_det_boxes", True),
            )
        else:
            atom_model = ocr_model_init(
                kwargs.get("det_db_box_thresh", 0.5),
                kwargs.get("lang"),
                kwargs.get("det_db_unclip_ratio", 1.5),
                kwargs.get("enable_merge_det_boxes", True),
            )
    elif model_name == AtomicModelName.WirelessTable:
        atom_model = wireless_table_model_init(
            kwargs.get("lang"),
        )
    elif model_name == AtomicModelName.WiredTable:
        atom_model = wired_table_model_init(
            kwargs.get("lang"),
        )
    elif model_name == AtomicModelName.TableCls:
        atom_model = table_cls_model_init()
    elif model_name == AtomicModelName.TableOrientationCls:
        atom_model = table_orientation_cls_model_init()
    else:
        logger.error("model name not allow")
        exit(1)

    if atom_model is None:
        logger.error("model init failed")
        exit(1)
    else:
        return atom_model


class HybridLocalModelContextSingleton:
    """全局缓存并复用 Hybrid 本地模型上下文。"""

    _instance: HybridLocalModelContextSingleton | None = None
    _model: HybridLocalModelContext | None = None
    _lock: threading.RLock = LOCAL_MODEL_INIT_LOCK

    def __new__(cls, *args: Any, **kwargs: Any) -> HybridLocalModelContextSingleton:
        """在线程锁保护下创建或返回唯一的上下文管理器。"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def get_model(
        self,
    ) -> HybridLocalModelContext:
        """获取并复用唯一的 Hybrid 本地模型上下文。"""
        with self._lock:
            if self._model is None:
                self._model = HybridLocalModelContext()
        return self._model


def ocr_det_batch_setting() -> bool:
    """根据运行设备和 PyTorch 版本确定是否启用 OCR 检测批处理。"""

    try:
        import torch as _torch
        from packaging import version
    except ImportError:
        return True

    device_type = os.getenv("MINERU_LMDEPLOY_DEVICE", "")
    if device_type.lower() in ["corex"]:
        enable_ocr_det_batch = False
    else:
        if version.parse(_torch.__version__) >= version.parse("2.8.0"):
            os.environ["TORCH_CUDNN_V8_API_DISABLED"] = "1"
        enable_ocr_det_batch = True

    return enable_ocr_det_batch


class HybridLocalModelContext:
    """集中管理 Hybrid 分析所需本地模型及其惰性加载生命周期。"""

    def __init__(
        self,
        device: str | None = None,
    ) -> None:
        """初始化 Hybrid 基础运行时，其他模型在首次访问对应属性时加载。"""
        if device is not None:
            self.device: str = device
        else:
            self.device: str = get_device()

        self.enable_ocr_det_batch: bool = ocr_det_batch_setting()

        if str(self.device).startswith("npu"):
            try:
                import torch_npu

                if torch_npu.npu.is_available():
                    torch_npu.npu.set_compile_mode(jit_compile=False)
            except Exception as e:
                raise RuntimeError(
                    "NPU is selected as device, but torch_npu is not available. "
                    "Please ensure that the torch_npu package is installed correctly."
                ) from e

        self.atom_model_manager = AtomModelSingleton()

        # 初始化OCR模型
        self.ocr_model = self.get_ocr_model()

        # 初始化layout模型，用于提供行内公式检测框和Hybrid标题拆分
        self.layout_model = self.get_layout_model()

    @cached_property
    def mfr_model(self) -> UnimernetModel | FormulaRecognizer:
        """首次访问时加载公式识别模型，并在当前 Context 内复用。"""
        return self.get_mfr_model()

    @cached_property
    def wireless_table_model(self) -> PaddleTableModel:
        """首次访问时加载无线表格识别模型，并在当前 Context 内复用。"""
        return self.get_wireless_table_model()

    @cached_property
    def wired_table_model(self) -> UnetTableModel:
        """首次访问时加载有线表格识别模型，并在当前 Context 内复用。"""
        return self.get_wired_table_model()

    @cached_property
    def table_cls_model(self) -> PaddleTableClsModel:
        """首次访问时加载表格类型分类模型，并在当前 Context 内复用。"""
        return self.get_table_cls_model()

    @cached_property
    def table_orientation_cls_model(self) -> MineruTableOrientationClsModel:
        """首次访问时加载基于 OCR 的表格方向包装器，并在当前 Context 内复用。"""
        return self.get_table_orientation_cls_model()

    @cached_property
    def seal_model(self) -> PytorchPaddleOCR:
        """首次访问时加载印章 OCR 模型，并在当前 Context 内复用。"""
        return self.get_seal_ocr_model()

    def get_ocr_model(
        self,
        det_db_box_thresh: float = 0.5,
        det_db_unclip_ratio: float = 1.5,
        enable_merge_det_boxes: bool = True,
    ) -> "PytorchPaddleOCR":
        """获取 OCR 原子模型，默认使用当前 Hybrid 本地上下文语言并复用 singleton 缓存。"""
        return self.atom_model_manager.get_atom_model(
            atom_model_name=AtomicModelName.OCR,
            det_db_box_thresh=det_db_box_thresh,
            lang="ch",
            det_db_unclip_ratio=det_db_unclip_ratio,
            enable_merge_det_boxes=enable_merge_det_boxes,
        )

    def get_layout_model(self) -> "PPDocLayoutV2LayoutModel":
        """获取 Layout 原子模型，供 Hybrid 本地 layout、标题拆分和公式框检测复用。"""
        return self.atom_model_manager.get_atom_model(
            atom_model_name=AtomicModelName.Layout,
            pp_doclayout_v2_weights=str(PDF_EXTRACT_KIT.pp_doclayout_v2.ensure()),
            device=self.device,
        )

    def get_mfr_model(self) -> "UnimernetModel":
        """获取公式识别原子模型，统一复用当前公式模型配置和设备。"""
        mfr_model_path = _resolve_mfr_model_path()
        return self.atom_model_manager.get_atom_model(
            atom_model_name=AtomicModelName.MFR,
            mfr_weight_dir=str(mfr_model_path.ensure()),
            device=self.device,
        )

    def get_wireless_table_model(self) -> PaddleTableModel:
        """获取无线表格识别原子模型。"""
        return self.atom_model_manager.get_atom_model(
            atom_model_name=AtomicModelName.WirelessTable,
            lang="ch",
        )

    def get_wired_table_model(self) -> UnetTableModel:
        """获取有线表格识别原子模型。"""
        return self.atom_model_manager.get_atom_model(
            atom_model_name=AtomicModelName.WiredTable,
            lang="ch",
        )

    def get_table_cls_model(self) -> PaddleTableClsModel:
        """获取表格分类原子模型。"""
        return self.atom_model_manager.get_atom_model(
            atom_model_name=AtomicModelName.TableCls,
        )

    def get_table_orientation_cls_model(self) -> MineruTableOrientationClsModel:
        """获取表格方向分类原子模型。"""
        return self.atom_model_manager.get_atom_model(
            atom_model_name=AtomicModelName.TableOrientationCls,
        )

    def get_seal_ocr_model(self) -> PytorchPaddleOCR:
        """获取印章识别 OCR 原子模型"""
        return self.atom_model_manager.get_atom_model(
            atom_model_name=AtomicModelName.OCR,
            lang="seal",
        )
