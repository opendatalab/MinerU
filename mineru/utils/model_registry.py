from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from ..config import config
from ..types import Tier, validate_tier

if TYPE_CHECKING:
    from ..config import ModelSource

DownloadMode = Literal["full", "required_paths"]


def _join_model_path(*parts: str) -> str:
    cleaned: list[str] = []
    for raw in parts:
        value = str(raw).replace("\\", "/").strip("/")
        if not value:
            continue
        cleaned.extend(part for part in value.split("/") if part and part != ".")
    if any(part == ".." for part in cleaned):
        raise ValueError("Model paths must not contain '..' segments.")
    return "/".join(cleaned)


@dataclass(frozen=True)
class ModelPath:
    repo: "ModelRepo"
    name: str
    relative_path: str

    def path(self, relative_path: str, /, *children: str) -> "ModelPath":
        joined = _join_model_path(self.relative_path, relative_path, *children)
        name = _join_model_path(self.name, relative_path, *children)
        return ModelPath(repo=self.repo, name=name, relative_path=joined)

    def local_path(self) -> Path:
        return self.repo.local_dir() / self.relative_path

    def ensure(self, *, source: "ModelSource | None" = None) -> Path:
        from .models_download_utils import download_model_files

        download_model_files(self.repo, [self], source=source)
        return self.local_path()


@dataclass(frozen=True)
class ModelRepo:
    name: str
    local_name: str
    repos: dict[str, str]
    paths: dict[str, str]
    download_mode: DownloadMode = "full"

    def __getattr__(self, name: str) -> ModelPath:
        try:
            return self.named_path(name)
        except KeyError as exc:
            raise AttributeError(name) from exc

    def named_path(self, name: str) -> ModelPath:
        return ModelPath(repo=self, name=name, relative_path=self.paths[name])

    def path(self, relative_path: str, /, *children: str) -> ModelPath:
        joined = _join_model_path(relative_path, *children)
        return ModelPath(repo=self, name=joined, relative_path=joined)

    def required_paths(self) -> tuple[ModelPath, ...]:
        return tuple(self.named_path(name) for name in self.paths)

    def local_dir(self) -> Path:
        return Path(config.model.base_dir).expanduser() / self.local_name

    def lock_path(self) -> Path:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.local_name).strip("._") or "model"
        return Path(config.model.base_dir).expanduser() / ".locks" / f"{safe_name}.lock"

    def ensure(self, *, source: "ModelSource | None" = None) -> Path:
        from .models_download_utils import download_model_repo

        return download_model_repo(self, source=source)


PDF_EXTRACT_KIT = ModelRepo(
    name="PDF-Extract-Kit-1.0",
    local_name="PDF-Extract-Kit-1.0",
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
    local_name="MinerU2.5-Pro-2605-1.2B",
    repos={
        "huggingface": "opendatalab/MinerU2.5-Pro-2605-1.2B",
        "modelscope": "OpenDataLab/MinerU2.5-Pro-2605-1.2B",
    },
    paths={
        "config_json": "config.json",
        "model_safetensors": "model.safetensors",
        "preprocessor_config_json": "preprocessor_config.json",
        "tokenizer_config_json": "tokenizer_config.json",
        "tokenizer_json": "tokenizer.json",
    },
)

MODEL_REPOS: tuple[ModelRepo, ...] = (
    PDF_EXTRACT_KIT,
    MINERU_2_5_PRO_2605_1_2B,
)

MODEL_REPOS_BY_NAME: dict[str, ModelRepo] = {repo.name: repo for repo in MODEL_REPOS}

REPOS_FOR_TIER: dict[Tier, tuple[ModelRepo, ...]] = {
    "flash": (),
    "medium": (PDF_EXTRACT_KIT,),
    "high": (PDF_EXTRACT_KIT, MINERU_2_5_PRO_2605_1_2B),
    "xhigh": (PDF_EXTRACT_KIT, MINERU_2_5_PRO_2605_1_2B),
}


def get_model_repo(name: str) -> ModelRepo:
    try:
        return MODEL_REPOS_BY_NAME[name]
    except KeyError as exc:
        available = ", ".join(MODEL_REPOS_BY_NAME)
        raise ValueError(f"Unsupported model repo '{name}'. Available repos: {available}.") from exc


def model_repos_for_tier(tier: Tier) -> tuple[ModelRepo, ...]:
    return REPOS_FOR_TIER[validate_tier(tier)]


def model_repo_names() -> tuple[str, ...]:
    return tuple(MODEL_REPOS_BY_NAME)


def model_path_exists(path: ModelPath) -> bool:
    local_path = path.local_path()
    if local_path.is_file():
        return True
    if not local_path.is_dir():
        return False
    return any(files for _root, _dirs, files in os.walk(local_path))


__all__ = [
    "MINERU_2_5_PRO_2605_1_2B",
    "MODEL_REPOS",
    "MODEL_REPOS_BY_NAME",
    "DownloadMode",
    "ModelPath",
    "ModelRepo",
    "PDF_EXTRACT_KIT",
    "REPOS_FOR_TIER",
    "get_model_repo",
    "model_path_exists",
    "model_repo_names",
    "model_repos_for_tier",
]
