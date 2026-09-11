# Copyright (c) Opendatalab. All rights reserved.
"""模型仓库目录、具名资源路径与部署档位映射。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..types import DEPLOYMENT_TIERS, DeploymentTier
from .download import MODEL_COMPLETE_MARKER, DownloadMode, ModelPath, ModelRepo, model_path_exists

if TYPE_CHECKING:
    from ..config import VlmConfig


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

# ONNX 小模型归属同一个仓库，表格文件与 Torch 仓库逐字节一致。
MINERU_4_MODELS_ONNX = ModelRepo(
    name="MinerU-4_models_onnx",
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

# llama.cpp 使用 GGUF 主模型与多模态投影文件。
MINERU_2_5_PRO_2605_1_2B_GGUF = ModelRepo(
    name="MinerU2.5-Pro-2605-1.2B-GGUF",
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


_SMALL_MODEL_REPOS: dict[str, ModelRepo] = {
    "onnx": MINERU_4_MODELS_ONNX,
    "torch": MINERU_4_MODELS_TORCH,
}
_VLM_MODEL_REPOS: dict[str, ModelRepo] = {
    "llama-cpp": MINERU_2_5_PRO_2605_1_2B_GGUF,
    "vllm": MINERU_2_5_PRO_2605_1_2B,
    "lmdeploy": MINERU_2_5_PRO_2605_1_2B,
    "mlx": MINERU_2_5_PRO_2605_1_2B,
}


def small_model_repo(backend: str | None = None) -> ModelRepo:
    """独立选择小模型仓库，不读取 VLM 引擎或其权重格式。"""
    from .runtime.device import resolve_small_model_backend

    return _SMALL_MODEL_REPOS[resolve_small_model_backend(backend)]


def vlm_model_repo(engine: str | None = None) -> ModelRepo:
    """根据实际 VLM 引擎选择 GGUF 或原始权重仓库。"""
    from .vlm.selector import resolve_vlm_engine

    return _VLM_MODEL_REPOS[resolve_vlm_engine(engine)]


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
    small_backend: str | None = None,
    vlm_engine: str | None = None,
    vlm_config: VlmConfig | None = None,
) -> tuple[ModelRepo, ...]:
    """组合档位需要的资源；远程 VLM 不要求本地权重，显式引擎可覆盖远程配置。"""
    from ..config import config

    resolved_tier = validate_model_tier(tier)
    if vlm_engine is not None:
        from .vlm.selector import resolve_vlm_engine

        # basic 不加载 VLM，但仍拒绝模型管理命令中的非法显式引擎名称。
        vlm_engine = resolve_vlm_engine(vlm_engine)
    repos = (small_model_repo(small_backend),)
    settings = vlm_config if vlm_config is not None else config.model.vlm
    if resolved_tier == "standard" and (vlm_engine is not None or not settings.server_url):
        repos += (vlm_model_repo(vlm_engine if vlm_engine is not None else settings.engine),)
    return repos


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
    "small_model_repo",
    "vlm_model_repo",
    "get_model_repo",
    "model_path_exists",
    "model_repo_names",
    "model_repos_for_tier",
    "validate_model_tier",
]
