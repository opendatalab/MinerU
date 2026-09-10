# Copyright (c) Opendatalab. All rights reserved.
"""准备、导出并校验 MinerU 4 ONNX 资产；转换依赖仅用于此开发脚本。

运行：.venv1/bin/python -m scripts.prepare_onnx_models prepare --output-dir output/model-migration/models
校验：.venv1/bin/python -m scripts.prepare_onnx_models verify --repo-dir <ONNX 仓库目录>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from importlib.metadata import version
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download, snapshot_download

TORCH_REPO = "opendatalab/MinerU-4_models_torch"
TORCH_REVISION = "2b3afb86f4d23fa623f6b2f4b2279ed4ef93a89d"
SOURCES = (
    (
        "PaddlePaddle/PP-DocLayoutV2_onnx",
        "7d44592493df02e28110d99de8ca4b1fbc7309bd",
        "Layout/PP-DocLayoutV2/inference.onnx",
        "Layout/PP-DocLayoutV2/inference.yml",
    ),
    (
        "PaddlePaddle/PP-OCRv6_tiny_det_onnx",
        "2ba1506c0380b8f0b03dd142459aac66d4421f6c",
        "OCR/paddleocr/ch_PP-OCRv6_tiny_det_infer.onnx",
        "OCR/paddleocr/ch_PP-OCRv6_tiny_det_inference.yml",
    ),
    (
        "PaddlePaddle/PP-OCRv6_small_rec_onnx",
        "b8f84f0b80c529de40b4fbb3544b84fa7233a513",
        "OCR/paddleocr/ch_PP-OCRv6_small_rec_infer.onnx",
        "OCR/paddleocr/ch_PP-OCRv6_small_rec_inference.yml",
    ),
)
TABLE_FILES = ("slanet-plus.onnx", "unet.onnx", "PP-LCNet_x1_0_table_cls.onnx")


def sha256(path: Path) -> str:
    """分块计算大模型文件的校验和。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_graph(path: Path) -> dict[str, Any]:
    """检查 ONNX 图并记录可复查的输入输出及算子版本。"""
    import onnx

    onnx.checker.check_model(str(path))
    model = onnx.load(str(path), load_external_data=False)
    initializers = {item.name for item in model.graph.initializer}

    def describe_value(value: Any) -> dict[str, Any]:
        """把张量签名转成独立于 ONNX protobuf 的 JSON 数据。"""
        tensor = value.type.tensor_type
        return {
            "name": value.name,
            "dtype": onnx.TensorProto.DataType.Name(tensor.elem_type),
            "shape": [dim.dim_param or dim.dim_value for dim in tensor.shape.dim],
        }

    return {
        "ir_version": model.ir_version,
        "opsets": {item.domain or "ai.onnx": item.version for item in model.opset_import},
        "inputs": [describe_value(item) for item in model.graph.input if item.name not in initializers],
        "outputs": [describe_value(item) for item in model.graph.output],
    }


def export_seal(weight: Path, destination: Path) -> dict[str, Any]:
    """用公开 Torch API 导出印章检测概率图，并验证动态尺寸和批次。"""
    import numpy as np
    import onnxruntime as ort
    import torch

    from mineru.model._internal.pytorchocr.infer.pytorchocr_utility import get_arch_config
    from mineru.model._internal.pytorchocr.modeling.architectures.base_model import BaseModel

    torch.set_num_threads(2)
    torch.manual_seed(0)
    net = BaseModel(get_arch_config(str(weight))).eval()
    net.load_state_dict(torch.load(weight, map_location="cpu", weights_only=True), strict=True)
    for module in list(net.modules()):
        if hasattr(module, "rep"):
            module.rep()
    net.eval()
    sample = torch.randn(1, 3, 256, 320)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        net,
        (sample,),
        str(destination),
        dynamo=False,
        opset_version=17,
        input_names=["image"],
        output_names=["maps"],
        dynamic_axes={"image": {0: "batch", 2: "height", 3: "width"}, "maps": {0: "batch", 2: "height", 3: "width"}},
    )
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(destination), sess_options=options, providers=["CPUExecutionProvider"])
    comparisons = []
    for shape in ((1, 3, 256, 320), (1, 3, 384, 256), (2, 3, 256, 320)):
        sample = torch.randn(shape)
        with torch.inference_mode():
            expected = net(sample)["maps"].numpy()
        actual = session.run(None, {"image": sample.numpy()})[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
        comparisons.append({"input_shape": shape, "max_abs": float(np.abs(actual - expected).max())})
    return {"dtype": "float32", "opset": 17, "rtol": 1e-4, "atol": 1e-5, "comparisons": comparisons}


def prepare(output_dir: Path) -> Path:
    """从固定源版本准备完整仓库，全部通过检查后写出资产清单。"""
    import yaml

    from mineru.model.ocr.resources import PPOCRV6_DICT_PATH
    from mineru.model.registry import MINERU_4_MODELS_TORCH, MINERU_4_MODELS_ONNX

    full_dir = output_dir / MINERU_4_MODELS_TORCH.name
    repo_dir = output_dir / MINERU_4_MODELS_ONNX.name
    snapshot_download(TORCH_REPO, revision=TORCH_REVISION, local_dir=full_dir, max_workers=4)
    # 固定版本快照已完整下载，目录标记只用于本地验收，不上传到模型仓库。
    for relative in ("Layout/PP-DocLayoutV2", "OCR/paddleocr"):
        (full_dir / relative / ".mineru_complete").touch()
    provenance: dict[str, dict[str, Any]] = {}
    for repo, revision, model_target, config_target in SOURCES:
        for source_file, target in (("inference.onnx", model_target), ("inference.yml", config_target)):
            print(f"Preparing {target}", flush=True)
            source = Path(hf_hub_download(repo, source_file, revision=revision))
            destination = repo_dir / target
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            provenance[target] = {"repo": repo, "revision": revision, "file": source_file, "sha256": sha256(source)}
            if target.startswith("OCR/") and target.endswith(".yml"):
                configuration = yaml.safe_load(destination.read_text(encoding="utf-8"))
                # 上游训练/部署配置原样保留，额外明确 MinerU 的实际运行参数。
                configuration["MinerU"] = (
                    {
                        "limit_side_len": 960,
                        "limit_type": "max",
                        "max_side_limit": 4000,
                        "thresh": 0.3,
                        "box_thresh": 0.5,
                        "unclip_ratio": 1.5,
                        "max_candidates": 1000,
                        "use_dilation": False,
                        "box_type": "quad",
                    }
                    if "_det_" in target
                    else {
                        "image_shape": [3, 48, 320],
                        "min_width": 16,
                        "max_width": 2560,
                        "rec_batch_num": 6,
                        "use_space_char": True,
                        "dictionary": "mineru/model/ocr/data/ppocrv6_dict.txt",
                    }
                )
                destination.write_text(yaml.safe_dump(configuration, allow_unicode=True, sort_keys=False), encoding="utf-8")
                provenance[target]["adaptation"] = "append MinerU runtime profile"
    # 公式与 full 共用同一份 PTH 来源，不能再恢复成旧 Paddle 镜像图。
    from .export_formula_onnx import export_model

    formula_relative = "MFR/pp_formulanet_plus_m"
    formula_export = output_dir / "formula-export"
    export_model(full_dir / formula_relative, formula_export)
    for source, relative in (
        (formula_export / "PP-FormulaNet_plus-M.from_torch.onnx", f"{formula_relative}/PP-FormulaNet_plus-M.onnx"),
        (
            full_dir / formula_relative / "PP-FormulaNet_plus-M_inference.yml",
            f"{formula_relative}/PP-FormulaNet_plus-M_inference.yml",
        ),
    ):
        destination = repo_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_relative = f"{formula_relative}/PP-FormulaNet_plus-M.pth" if relative.endswith(".onnx") else relative
        provenance[relative] = {
            "repo": TORCH_REPO,
            "revision": TORCH_REVISION,
            "file": source_relative,
            "source_sha256": sha256(full_dir / source_relative),
            "conversion": "scripts/export_formula_onnx.py" if relative.endswith(".onnx") else "copy",
        }
    for name in TABLE_FILES:
        relative = f"Table/{name}"
        destination = repo_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(full_dir / relative, destination)
        assert sha256(destination) == sha256(full_dir / relative)
        provenance[relative] = {"repo": TORCH_REPO, "revision": TORCH_REVISION, "file": relative}

    rec_config = yaml.safe_load((repo_dir / MINERU_4_MODELS_ONNX.paths["ocr_rec_config"]).read_text())
    chars = PPOCRV6_DICT_PATH.read_text(encoding="utf-8").splitlines()
    assert rec_config["PostProcess"]["character_dict"] == chars
    seal_weight_relative = "OCR/paddleocr/seal_PP-OCRv4_det_infer.pth"
    seal_target = MINERU_4_MODELS_ONNX.paths["seal_det"]
    seal_validation = export_seal(full_dir / seal_weight_relative, repo_dir / seal_target)
    seal_config = {
        "Global": {"model_name": "PP-OCRv4_mobile_seal_det"},
        "PreProcess": {
            "limit_side_len": 736,
            "limit_type": "min",
            "max_side_limit": 4000,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "img_mode": "BGR",
        },
        "PostProcess": {
            "name": "DBPostProcess",
            "thresh": 0.2,
            "box_thresh": 0.6,
            "unclip_ratio": 0.5,
            "max_candidates": 1000,
            "box_type": "poly",
            "use_dilation": False,
        },
        "Export": seal_validation,
    }
    seal_config_target = MINERU_4_MODELS_ONNX.paths["seal_det_config"]
    (repo_dir / seal_config_target).write_text(yaml.safe_dump(seal_config, sort_keys=False), encoding="utf-8")
    for relative in (seal_target, seal_config_target):
        provenance[relative] = {
            "repo": TORCH_REPO,
            "revision": TORCH_REVISION,
            "file": seal_weight_relative,
            "source_sha256": sha256(full_dir / seal_weight_relative),
            "conversion": "scripts/prepare_onnx_models.py",
        }

    entries = []
    for relative in sorted(MINERU_4_MODELS_ONNX.paths.values()):
        path = repo_dir / relative
        entry = {"path": relative, "size": path.stat().st_size, "sha256": sha256(path), "source": provenance[relative]}
        if path.suffix == ".onnx":
            entry["graph"] = describe_graph(path)
        entries.append(entry)
    manifest = {
        "repository": "opendatalab/MinerU-4_models_onnx",
        "tools": {name: version(name) for name in ("torch", "onnx", "onnxruntime", "huggingface-hub")},
        "runtime_provider": "CPUExecutionProvider",
        "minimum_onnxruntime": "1.20.1",
        "ocr_dictionary": {
            "characters": len(chars),
            "sha256": sha256(PPOCRV6_DICT_PATH),
            "package_path": "mineru/model/ocr/data/ppocrv6_dict.txt",
        },
        "formula_export": {
            "dtype": "float32",
            "opset": 17,
            "ir_version": 8,
            "dynamic_batch": True,
            "cpu_runtime_batch_size": 8,
            "max_new_tokens": 2560,
            "forced_eos_generated_token": 1536,
        },
        "seal_export_validation": seal_validation,
        "files": entries,
    }
    (repo_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (repo_dir / "README.md").write_text(
        "# MinerU 4 ONNX models\n\n"
        "CPU ONNX Runtime models for MinerU's light stack. Requires onnxruntime >= 1.20.1 (ONNX IR 10 support).\n\n"
        "- Layout: PP-DocLayoutV2\n- OCR: PP-OCRv6 Tiny Det + Small Rec\n"
        "- Seal: PP-OCRv4 mobile seal detector + shared Small Rec\n"
        "- Formula: PP-FormulaNet plus-M (Torch export, CPU batch up to 8, forced EOS at generated token 1536)\n"
        "- Table: identical to MinerU-4_models_torch\n\n"
        "See `manifest.json` for source repositories, exact revisions, SHA-256 checksums, "
        "ONNX signatures and export validation. Original model licenses and attribution remain applicable.\n\n"
        "OCR uses the single TXT dictionary distributed with MinerU. "
        "Prepare these assets with `python -m scripts.prepare_onnx_models prepare`.\n",
        encoding="utf-8",
    )
    verify(repo_dir)
    return repo_dir


def verify(repo_dir: Path) -> None:
    """依据清单复查每个文件、ONNX 图以及当前注册表的全部必需路径。"""
    from mineru.model.registry import MINERU_4_MODELS_ONNX

    manifest = json.loads((repo_dir / "manifest.json").read_text())
    paths = set()
    for entry in manifest["files"]:
        path = repo_dir / entry["path"]
        if path.stat().st_size != entry["size"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"Asset checksum mismatch: {entry['path']}")
        if path.suffix == ".onnx":
            if describe_graph(path) != entry["graph"]:
                raise ValueError(f"ONNX signature mismatch: {entry['path']}")
        paths.add(entry["path"])
    if paths != set(MINERU_4_MODELS_ONNX.paths.values()):
        raise ValueError("Manifest and registry required paths differ")
    print(f"Verified {len(paths)} assets in {repo_dir}", flush=True)


def main() -> None:
    """解析资产准备或离线校验命令。"""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preparation = subparsers.add_parser("prepare")
    preparation.add_argument("--output-dir", type=Path, default=Path("output/model-migration/models"))
    verification = subparsers.add_parser("verify")
    verification.add_argument("--repo-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare(args.output_dir))
    else:
        verify(args.repo_dir)


if __name__ == "__main__":
    main()
