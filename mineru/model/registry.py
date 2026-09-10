# Copyright (c) Opendatalab. All rights reserved.
"""模型仓库目录、具名资源路径与部署档位映射。"""

from __future__ import annotations

from typing import Literal

from ..types import DEPLOYMENT_TIERS, DeploymentTier
from .download import MODEL_COMPLETE_MARKER, DownloadMode, ModelPath, ModelRepo, model_path_exists


MINERU_4_MODELS_TORCH = ModelRepo(
    name="MinerU-4_models_torch",
    download_mode="required_paths",
    repos={
        "huggingface": "opendatalab/MinerU-4_models_torch",
    },
    paths={
        "pp_doclayout_v2": "Layout/PP-DocLayoutV2",
        "pp_formulanet_plus_m_weights": "MFR/pp_formulanet_plus_m/PP-FormulaNet_plus-M.pth",
        "pp_formulanet_plus_m_config": "MFR/pp_formulanet_plus_m/PP-FormulaNet_plus-M_inference.yml",
        "pytorch_paddle": "OCR/paddleocr",
        "slanet_plus": "Table/slanet-plus.onnx",
        "unet_structure": "Table/unet.onnx",
        "paddle_table_cls": "Table/PP-LCNet_x1_0_table_cls.onnx",
    },
)

MINERU_2_5_PRO_2605_1_2B = ModelRepo(
    name="MinerU2.5-Pro-2605-1.2B",
    repos={
        "huggingface": "opendatalab/MinerU2.5-Pro-2605-1.2B",
        "modelscope": "OpenDataLab/MinerU2.5-Pro-2605-1.2B",
    },
)

# Light 的全部本地模型归属同一个 ONNX 仓库，表格文件与 Torch 仓库逐字节一致。
MINERU_4_MODELS_ONNX = ModelRepo(
    name="MinerU-4_models_onnx",
    stack="light",
    download_mode="required_paths",
    repos={"huggingface": "opendatalab/MinerU-4_models_onnx"},
    paths={
        "pp_doclayout_v2": "Layout/PP-DocLayoutV2/inference.onnx",
        "pp_doclayout_v2_config": "Layout/PP-DocLayoutV2/inference.yml",
        "ocr_det": "OCR/paddleocr/ch_PP-OCRv6_tiny_det_infer.onnx",
        "ocr_det_config": "OCR/paddleocr/ch_PP-OCRv6_tiny_det_inference.yml",
        "ocr_rec": "OCR/paddleocr/ch_PP-OCRv6_small_rec_infer.onnx",
        "ocr_rec_config": "OCR/paddleocr/ch_PP-OCRv6_small_rec_inference.yml",
        "seal_det": "OCR/paddleocr/seal_PP-OCRv4_det_infer.onnx",
        "seal_det_config": "OCR/paddleocr/seal_PP-OCRv4_det_inference.yml",
        "pp_formulanet_plus_m_weights": "MFR/pp_formulanet_plus_m/PP-FormulaNet_plus-M.onnx",
        "pp_formulanet_plus_m_config": "MFR/pp_formulanet_plus_m/PP-FormulaNet_plus-M_inference.yml",
        "slanet_plus": "Table/slanet-plus.onnx",
        "unet_structure": "Table/unet.onnx",
        "paddle_table_cls": "Table/PP-LCNet_x1_0_table_cls.onnx",
    },
)

# Light standard 使用现有 GGUF 主模型与多模态投影文件。
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
    MINERU_4_MODELS_TORCH,
    MINERU_4_MODELS_ONNX,
    MINERU_2_5_PRO_2605_1_2B,
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
    "basic": (MINERU_4_MODELS_TORCH,),
    "standard": (MINERU_4_MODELS_TORCH, MINERU_2_5_PRO_2605_1_2B),
}

_REPOS_FOR_TIER_LIGHT: dict[DeploymentTier, tuple[ModelRepo, ...]] = {
    "basic": (MINERU_4_MODELS_ONNX,),
    "standard": (MINERU_4_MODELS_ONNX, MINERU_2_5_PRO_2605_1_2B_GGUF),
}


def mineru_4_models_for_stack(stack: str | None = None) -> ModelRepo:
    """按显式或当前模型栈选择本地模型仓库，避免 Light 下载 Torch 资源。"""
    return MINERU_4_MODELS_ONNX if resolve_model_stack(stack) == "light" else MINERU_4_MODELS_TORCH


def get_model_repo(name: str) -> ModelRepo:
    """按公开仓库名返回注册项，并明确提示已失效的旧名称。"""
    try:
        return MODEL_REPOS_BY_NAME[name]
    except KeyError as exc:
        available = ", ".join(MODEL_REPOS_BY_NAME)
        raise ValueError(f"Unsupported model repo '{name}'. Available repos: {available}.") from exc


def validate_model_tier(tier: str) -> DeploymentTier:
    """校验部署档位并返回规范名称。"""
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
    """返回指定档位与模型栈的完整资源集合。"""
    resolved_tier = validate_model_tier(tier)
    resolved_stack = resolve_model_stack(stack)
    mapping = _REPOS_FOR_TIER_LIGHT if resolved_stack == "light" else _REPOS_FOR_TIER_FULL
    return mapping[resolved_tier]


def model_repo_names() -> tuple[str, ...]:
    """返回 CLI 可用的仓库名称。"""
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
    "MINERU_4_MODELS_TORCH",
    "MINERU_4_MODELS_ONNX",
    "mineru_4_models_for_stack",
    "resolve_model_stack",
    "get_model_repo",
    "model_path_exists",
    "model_repo_names",
    "model_repos_for_tier",
    "validate_model_tier",
]
