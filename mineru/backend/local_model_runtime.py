"""Hybrid 与本地小模型共享的运行时初始化模块。"""

# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import os
import threading
from collections.abc import Callable
from typing import Any

import torch
from loguru import logger

from ..model.layout.pp_doclayoutv2 import PPDocLayoutV2LayoutModel
from ..model.mfr.pp_formulanet_plus_m.predict_formula import FormulaRecognizer
from ..model.mfr.unimernet.Unimernet import UnimernetModel
from ..model.ocr.pytorch_paddle import PytorchPaddleOCR
from ..model.table.cls.mineru_table_ori_cls import MineruTableOrientationClsModel
from ..model.table.cls.paddle_table_cls import PaddleTableClsModel
from ..model.table.rec.slanet_plus.main import PaddleTableModel
from ..model.table.rec.unet_table.main import UnetTableModel
from ..utils.config_reader import get_device
from ..utils.model_registry import PDF_EXTRACT_KIT, ModelPath
from ..utils.ocr_language import normalize_ocr_model_lang


class AtomicModel:
    """本地原子模型名称集合，供 Hybrid medium 和共享模型单例统一索引。"""

    Layout = "layout"
    MFD = "mfd"
    MFR = "mfr"
    OCR = "ocr"
    WirelessTable = "wireless_table"
    WiredTable = "wired_table"
    TableCls = "table_cls"
    TableOrientationCls = "table_ori_cls"

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


MFR_MODEL = os.getenv("MINERU_FORMULA_CH_SUPPORT", "False")
if MFR_MODEL.lower() in ["true", "1", "yes"]:
    MFR_MODEL = "pp_formulanet_plus_m"
elif MFR_MODEL.lower() in ["false", "0", "no"]:
    MFR_MODEL = "unimernet_small"
else:
    logger.warning(f"Invalid MINERU_FORMULA_CH_SUPPORT value: {MFR_MODEL}, set to default 'False'")
    MFR_MODEL = "unimernet_small"


def _resolve_mfr_model_path() -> ModelPath:
    """解析当前公式识别模型路径，集中维护 MFR_MODEL 到模型目录枚举的映射关系。"""
    if MFR_MODEL == "unimernet_small":
        return PDF_EXTRACT_KIT.unimernet_small
    if MFR_MODEL == "pp_formulanet_plus_m":
        return PDF_EXTRACT_KIT.pp_formulanet_plus_m
    logger.error("MFR model name not allow")
    exit(1)


def table_orientation_cls_model_init() -> MineruTableOrientationClsModel:
    atom_model_manager = AtomModelSingleton()
    ocr_engine = atom_model_manager.get_atom_model(
        atom_model_name=AtomicModel.OCR,
        det_db_box_thresh=0.5,
        det_db_unclip_ratio=1.6,
        lang="ch",
        enable_merge_det_boxes=False,
    )
    cls_model = MineruTableOrientationClsModel(ocr_engine)
    return cls_model


def table_cls_model_init() -> PaddleTableClsModel:
    return PaddleTableClsModel()


def wired_table_model_init(lang: str | None = None) -> UnetTableModel:
    atom_model_manager = AtomModelSingleton()
    ocr_engine = atom_model_manager.get_atom_model(
        atom_model_name=AtomicModel.OCR, det_db_box_thresh=0.5, det_db_unclip_ratio=1.6, lang=lang, enable_merge_det_boxes=False
    )
    table_model = UnetTableModel(ocr_engine)
    return table_model


def wireless_table_model_init(lang: str | None = None) -> PaddleTableModel:
    atom_model_manager = AtomModelSingleton()
    ocr_engine = atom_model_manager.get_atom_model(
        atom_model_name=AtomicModel.OCR, det_db_box_thresh=0.5, det_db_unclip_ratio=1.6, lang=lang, enable_merge_det_boxes=False
    )
    table_model = PaddleTableModel(ocr_engine)
    return table_model


def mfr_model_init(weight_dir: str, device: str | torch.device = "cpu") -> UnimernetModel | FormulaRecognizer:
    if MFR_MODEL == "unimernet_small":
        mfr_model = UnimernetModel(weight_dir, device)
    elif MFR_MODEL == "pp_formulanet_plus_m":
        mfr_model = FormulaRecognizer(weight_dir, device)
    else:
        logger.error("MFR model name not allow")
        exit(1)
    return mfr_model


def pp_doclayout_v2_model_init(weight: str, device: str | torch.device = "cpu") -> PPDocLayoutV2LayoutModel:
    if str(device).startswith("npu"):
        device = torch.device(device)
    model = PPDocLayoutV2LayoutModel(weight, device)
    return model


def ocr_model_init(
    det_db_box_thresh: float = 0.5,
    lang: str | None = None,
    det_db_unclip_ratio: float = 1.5,
    enable_merge_det_boxes: bool = True,
) -> PytorchPaddleOCR:
    ocr_kwargs = {
        "lang": normalize_ocr_model_lang(lang),
        "det_db_box_thresh": det_db_box_thresh,
        "det_db_unclip_ratio": det_db_unclip_ratio,
        "enable_merge_det_boxes": enable_merge_det_boxes,
    }
    return PytorchPaddleOCR(**ocr_kwargs)


class AtomModelSingleton:
    _instance: AtomModelSingleton | None = None
    _models: dict[object, object] = {}
    _lock: threading.RLock = LOCAL_MODEL_INIT_LOCK

    def __new__(cls, *args: Any, **kwargs: Any) -> AtomModelSingleton:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def get_atom_model(self, atom_model_name: str, **kwargs: Any) -> Any:
        lang = kwargs.get("lang", None)
        ocr_singleton_lang = normalize_ocr_model_lang(lang)

        if atom_model_name in [AtomicModel.WiredTable, AtomicModel.WirelessTable]:
            key = (atom_model_name, ocr_singleton_lang)
        elif atom_model_name in [AtomicModel.OCR]:
            key = (
                atom_model_name,
                kwargs.get("det_db_box_thresh", 0.5),
                ocr_singleton_lang,
                kwargs.get("det_db_unclip_ratio", 1.5),
                kwargs.get("enable_merge_det_boxes", True),
            )
        elif atom_model_name in [AtomicModel.Layout, AtomicModel.MFR]:
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
    atom_model = None
    if model_name == AtomicModel.Layout:
        atom_model = pp_doclayout_v2_model_init(kwargs.get("pp_doclayout_v2_weights"), kwargs.get("device"))
    elif model_name == AtomicModel.MFR:
        atom_model = mfr_model_init(kwargs.get("mfr_weight_dir"), kwargs.get("device"))
    elif model_name == AtomicModel.OCR:
        atom_model = ocr_model_init(
            kwargs.get("det_db_box_thresh", 0.5),
            kwargs.get("lang"),
            kwargs.get("det_db_unclip_ratio", 1.5),
            kwargs.get("enable_merge_det_boxes", True),
        )
    elif model_name == AtomicModel.WirelessTable:
        atom_model = wireless_table_model_init(
            kwargs.get("lang"),
        )
    elif model_name == AtomicModel.WiredTable:
        atom_model = wired_table_model_init(
            kwargs.get("lang"),
        )
    elif model_name == AtomicModel.TableCls:
        atom_model = table_cls_model_init()
    elif model_name == AtomicModel.TableOrientationCls:
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
    _instance: HybridLocalModelContextSingleton | None = None
    _models: dict[object, object] = {}
    _lock: threading.RLock = LOCAL_MODEL_INIT_LOCK

    def __new__(cls, *args: Any, **kwargs: Any) -> HybridLocalModelContextSingleton:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def get_model(
        self,
        lang: str | None = None,
        formula_enable: bool | None = None,
    ) -> HybridLocalModelContext:
        key = (lang, formula_enable)
        with self._lock:
            if key not in self._models:
                self._models[key] = HybridLocalModelContext(
                    lang=lang,
                    formula_enable=formula_enable,
                )
        return self._models[key]


def ocr_det_batch_setting() -> bool:
    import torch as _torch
    from packaging import version

    device_type = os.getenv("MINERU_LMDEPLOY_DEVICE", "")
    if device_type.lower() in ["corex"]:
        enable_ocr_det_batch = False
    else:
        if version.parse(_torch.__version__) >= version.parse("2.8.0"):
            os.environ["TORCH_CUDNN_V8_API_DISABLED"] = "1"
        enable_ocr_det_batch = True

    return enable_ocr_det_batch


class HybridLocalModelContext:
    def __init__(
        self,
        device: str | None = None,
        lang: str | None = None,
        formula_enable: bool = True,
    ) -> None:
        if device is not None:
            self.device: str = device
        else:
            self.device: str = get_device()

        self.lang: str | None = lang

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

        if formula_enable:
            # 初始化公式解析模型
            self.mfr_model = self.get_mfr_model()

    def get_ocr_model(
        self,
        *,
        lang: str | None = None,
        det_db_box_thresh: float = 0.5,
        det_db_unclip_ratio: float = 1.5,
        enable_merge_det_boxes: bool = True,
    ) -> PytorchPaddleOCR:
        """获取 OCR 原子模型，默认使用当前 Hybrid 本地上下文语言并复用 singleton 缓存。"""
        return self.atom_model_manager.get_atom_model(
            atom_model_name=AtomicModel.OCR,
            det_db_box_thresh=det_db_box_thresh,
            lang=self.lang if lang is None else lang,
            det_db_unclip_ratio=det_db_unclip_ratio,
            enable_merge_det_boxes=enable_merge_det_boxes,
        )

    def get_layout_model(self) -> PPDocLayoutV2LayoutModel:
        """获取 Layout 原子模型，供 Hybrid 本地 layout、标题拆分和公式框检测复用。"""
        return self.atom_model_manager.get_atom_model(
            atom_model_name=AtomicModel.Layout,
            pp_doclayout_v2_weights=str(PDF_EXTRACT_KIT.pp_doclayout_v2.ensure()),
            device=self.device,
        )

    def get_mfr_model(self) -> UnimernetModel | FormulaRecognizer:
        """获取公式识别原子模型，统一复用当前公式模型配置和设备。"""
        mfr_model_path = _resolve_mfr_model_path()
        return self.atom_model_manager.get_atom_model(
            atom_model_name=AtomicModel.MFR,
            mfr_weight_dir=str(mfr_model_path.ensure()),
            device=self.device,
        )
