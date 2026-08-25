# Copyright (c) Opendatalab. All rights reserved.
"""模型仓库目录、具名资源路径与部署档位映射。"""

from __future__ import annotations

from typing import Literal

from ..types import DEPLOYMENT_TIERS, DeploymentTier
from .download import MODEL_COMPLETE_MARKER, DownloadMode, ModelPath, ModelRepo, model_path_exists


PDF_EXTRACT_KIT = ModelRepo(
    name="PDF-Extract-Kit-1.0",
    download_mode="required_paths",
    repos={
        "huggingface": "opendatalab/PDF-Extract-Kit-1.0",
        "modelscope": "OpenDataLab/PDF-Extract-Kit-1.0",
    },
    paths={
        "pp_doclayout_v2": "models/Layout/PP-DocLayoutV2",
        "unimernet_small": "models/MFR/unimernet_hf_small_2503",
        "pytorch_paddle": "models/OCR/paddleocr_torch",
        "slanet_plus": "models/TabRec/SlanetPlus/slanet-plus.onnx",
        "unet_structure": "models/TabRec/UnetStructure/unet.onnx",
        "paddle_table_cls": "models/TabCls/paddle_table_cls/PP-LCNet_x1_0_table_cls.onnx",
    },
)

MINERU_2_5_PRO_2605_1_2B = ModelRepo(
    name="MinerU2.5-Pro-2605-1.2B",
    repos={
        "huggingface": "opendatalab/MinerU2.5-Pro-2605-1.2B",
        "modelscope": "OpenDataLab/MinerU2.5-Pro-2605-1.2B",
    },
)

# PaddlePaddle 官方 ONNX 模型，与 PDF_EXTRACT_KIT 中 transformers/torch 版本等价但更轻量。
# 暂不与任何 tier 关联，仅供实验性 ONNX 后端使用。
PP_DOCLAYOUT_V2_ONNX = ModelRepo(
    name="PP-DocLayoutV2_onnx",
    stack="light",
    repos={
        "huggingface": "PaddlePaddle/PP-DocLayoutV2_onnx",
        "modelscope": "PaddlePaddle/PP-DocLayoutV2_onnx",
    },
    paths={
        "onnx": "inference.onnx",
        "config": "inference.yml",
    },
)

PP_OCR_V6_SMALL_DET_ONNX = ModelRepo(
    name="PP-OCRv6_small_det_onnx",
    stack="light",
    repos={
        "huggingface": "PaddlePaddle/PP-OCRv6_small_det_onnx",
        "modelscope": "PaddlePaddle/PP-OCRv6_small_det_onnx",
    },
    paths={
        "onnx": "inference.onnx",
        "config": "inference.yml",
    },
)

PP_OCR_V6_SMALL_REC_ONNX = ModelRepo(
    name="PP-OCRv6_small_rec_onnx",
    stack="light",
    repos={
        "huggingface": "PaddlePaddle/PP-OCRv6_small_rec_onnx",
        "modelscope": "PaddlePaddle/PP-OCRv6_small_rec_onnx",
    },
    paths={
        "onnx": "inference.onnx",
        "config": "inference.yml",
    },
)

PP_OCR_V6_MEDIUM_DET_ONNX = ModelRepo(
    name="PP-OCRv6_medium_det_onnx",
    stack="light",
    repos={
        "huggingface": "PaddlePaddle/PP-OCRv6_medium_det_onnx",
        "modelscope": "PaddlePaddle/PP-OCRv6_medium_det_onnx",
    },
    paths={
        "onnx": "inference.onnx",
        "config": "inference.yml",
    },
)

PP_OCR_V6_MEDIUM_REC_ONNX = ModelRepo(
    name="PP-OCRv6_medium_rec_onnx",
    stack="light",
    repos={
        "huggingface": "PaddlePaddle/PP-OCRv6_medium_rec_onnx",
        "modelscope": "PaddlePaddle/PP-OCRv6_medium_rec_onnx",
    },
    paths={
        "onnx": "inference.onnx",
        "config": "inference.yml",
    },
)

# PP-FormulaNet-Plus-M ONNX，由 RapidDoc 转出，托管在 jinzhenj 双平台镜像。
# 暂不与任何 tier 关联，仅供实验性 ONNX 后端使用。
PP_FORMULANET_PLUS_M_ONNX = ModelRepo(
    name="PP-FormulaNet_plus-M_onnx",
    stack="light",
    repos={
        "huggingface": "jinzhenj/PP-FormulaNet_plus-M_onnx",
        "modelscope": "jinzhenj/PP-FormulaNet_plus-M_onnx",
    },
    paths={
        "onnx": "inference.onnx",
        "config": "inference.yml",
    },
)

# MinerU2.5-Pro-2605-1.2B GGUF 量化版（Q8_0），用于 llama.cpp / llama-cpp-python 推理。
# 托管在 jinzhenj 双平台镜像，含主权重与多模态投影权重。
# 归入 light standard tier（llama.cpp 后端尚未集成到运行时，仅供实验性使用）。
MINERU_2_5_PRO_2605_1_2B_GGUF = ModelRepo(
    name="MinerU2.5-Pro-2605-1.2B-GGUF",
    stack="light",
    repos={
        "huggingface": "jinzhenj/MinerU2.5-Pro-2605-1.2B-GGUF",
        "modelscope": "jinzhenj/MinerU2.5-Pro-2605-1.2B-GGUF",
    },
    paths={
        "main": "MinerU2.5-Pro-2605-1.2B-Q8_0.gguf",
        "mmproj": "mmproj-MinerU2.5-Pro-2605-1.2B-Q8_0.gguf",
    },
)

MODEL_REPOS: tuple[ModelRepo, ...] = (
    PDF_EXTRACT_KIT,
    MINERU_2_5_PRO_2605_1_2B,
    PP_DOCLAYOUT_V2_ONNX,
    PP_OCR_V6_SMALL_DET_ONNX,
    PP_OCR_V6_SMALL_REC_ONNX,
    PP_OCR_V6_MEDIUM_DET_ONNX,
    PP_OCR_V6_MEDIUM_REC_ONNX,
    PP_FORMULANET_PLUS_M_ONNX,
    MINERU_2_5_PRO_2605_1_2B_GGUF,
)

MODEL_REPOS_BY_NAME: dict[str, ModelRepo] = {repo.name: repo for repo in MODEL_REPOS}


def resolve_model_stack(stack: str | None) -> Literal["light", "full"]:
    """把 ``--stack`` 参数或 config 值解析为 ``"light"`` / ``"full"``。

    ``None`` 或 ``"auto"`` 走 ``get_model_stack()``（依据 ``config.model.stack`` 与设备自动选择）。
    """
    from .runtime.device import get_model_stack

    if stack in ("light", "full"):
        return stack  # type: ignore[return-value]
    if stack is None or stack == "auto":
        return get_model_stack()  # type: ignore[return-value]
    raise ValueError(f"Unsupported stack '{stack}'. Expected one of: auto, light, full.")


_REPOS_FOR_TIER_FULL: dict[DeploymentTier, tuple[ModelRepo, ...]] = {
    "basic": (PDF_EXTRACT_KIT,),
    "standard": (PDF_EXTRACT_KIT, MINERU_2_5_PRO_2605_1_2B),
}

_REPOS_FOR_TIER_LIGHT: dict[DeploymentTier, tuple[ModelRepo, ...]] = {
    "basic": (
        PP_DOCLAYOUT_V2_ONNX,
        PP_OCR_V6_SMALL_DET_ONNX,
        PP_OCR_V6_SMALL_REC_ONNX,
        PP_FORMULANET_PLUS_M_ONNX,
    ),
    # light standard 在 basic 小模型基础上追加 GGUF VLM（llama.cpp 推理）。
    "standard": (
        PP_DOCLAYOUT_V2_ONNX,
        PP_OCR_V6_SMALL_DET_ONNX,
        PP_OCR_V6_SMALL_REC_ONNX,
        PP_FORMULANET_PLUS_M_ONNX,  # TODO: remove?
        MINERU_2_5_PRO_2605_1_2B_GGUF,
    ),
}


def get_model_repo(name: str) -> ModelRepo:
    try:
        return MODEL_REPOS_BY_NAME[name]
    except KeyError as exc:
        available = ", ".join(MODEL_REPOS_BY_NAME)
        raise ValueError(f"Unsupported model repo '{name}'. Available repos: {available}.") from exc


def validate_model_tier(tier: str) -> DeploymentTier:
    normalized = tier.strip().lower()
    if normalized in DEPLOYMENT_TIERS:
        return normalized  # type: ignore[return-value]
    supported = ", ".join(DEPLOYMENT_TIERS)
    raise ValueError(f"Unsupported model tier '{tier}'. Supported model tiers: {supported}.")


def model_repos_for_tier(
    tier: str,
    *,
    stack: str | None = None,
) -> tuple[ModelRepo, ...]:
    resolved_tier = validate_model_tier(tier)
    resolved_stack = resolve_model_stack(stack)
    mapping = _REPOS_FOR_TIER_LIGHT if resolved_stack == "light" else _REPOS_FOR_TIER_FULL
    return mapping[resolved_tier]


def model_repo_names() -> tuple[str, ...]:
    return tuple(MODEL_REPOS_BY_NAME)


__all__ = [
    "MINERU_2_5_PRO_2605_1_2B",
    "MINERU_2_5_PRO_2605_1_2B_GGUF",
    "MODEL_COMPLETE_MARKER",
    "MODEL_REPOS",
    "MODEL_REPOS_BY_NAME",
    "DownloadMode",
    "ModelPath",
    "ModelRepo",
    "PDF_EXTRACT_KIT",
    "PP_DOCLAYOUT_V2_ONNX",
    "PP_OCR_V6_SMALL_DET_ONNX",
    "PP_OCR_V6_SMALL_REC_ONNX",
    "PP_OCR_V6_MEDIUM_DET_ONNX",
    "PP_OCR_V6_MEDIUM_REC_ONNX",
    "PP_FORMULANET_PLUS_M_ONNX",
    "resolve_model_stack",
    "get_model_repo",
    "model_path_exists",
    "model_repo_names",
    "model_repos_for_tier",
    "validate_model_tier",
]
