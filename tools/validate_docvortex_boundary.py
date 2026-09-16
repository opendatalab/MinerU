"""捕获两仓库的真实解析与九种输出，支持冻结源码和新 wheel 的差分验收。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def capture(args: argparse.Namespace) -> None:
    """使用显式源码或当前安装包保存完整协议、素材和可视化产物。"""
    if args.mineru_source:
        sys.path.insert(0, str(args.mineru_source.resolve()))
    if args.docvortex_source:
        sys.path.insert(0, str(args.docvortex_source.resolve()))
    import docvortex
    from docvortex.api import parse
    from docvortex.api import render as render_document

    import mineru
    from mineru.backend.analyze import doc_analyze
    from mineru.config import config
    from mineru.render import RenderFormat, render

    config.llm_aided.features.title_leveling = False
    config.llm_aided.features.cross_page_table_cell_merge = False
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"imports": {"mineru": mineru.__file__, "docvortex": docvortex.__file__}, "documents": []}
    for name in args.documents:
        source = args.samples / name
        payload = source.read_bytes()
        entry = {"name": name, "sha256": hashlib.sha256(payload).hexdigest(), "outputs": {}}
        try:
            for engine in ("mineru", "docvortex"):
                directory = args.output / source.stem / engine
                directory.mkdir(parents=True, exist_ok=True)
                if engine == "mineru":
                    middle, model = doc_analyze(payload, effort="flash", parse_mode="txt")
                    assets = None
                else:
                    result = parse(payload, file_suffix="pdf", keep_model_json=True)
                    middle, model, assets = result.middle_json, result.model_json, result.assets
                entry.setdefault("pages", {})[engine] = len(middle.pages)
                for label, value in (("model", model), ("middle", middle)):
                    (directory / f"{label}.json").write_text(value.to_json(skip_defaults=False), encoding="utf-8")
                for target in RenderFormat:
                    if engine == "docvortex" and target.value.startswith("content_list"):
                        continue
                    try:
                        if engine == "mineru":
                            content = render(middle, target)
                            data = (
                                content
                                if isinstance(content, bytes)
                                else (
                                    content
                                    if isinstance(content, str)
                                    else json.dumps(content, ensure_ascii=False, sort_keys=True)
                                ).encode()
                            )
                        else:
                            artifact = render_document(middle, target.value, assets=assets)
                            data = artifact.content
                            for asset_name, asset_bytes in artifact.assets.items():
                                asset_path = directory / asset_name
                                asset_path.parent.mkdir(parents=True, exist_ok=True)
                                asset_path.write_bytes(asset_bytes)
                        suffix = {"markdown": "md", "structured_content": "json"}.get(target.value, target.value)
                        output = directory / f"render.{suffix}"
                        output.write_bytes(data)
                        entry["outputs"][f"{engine}/{target.value}"] = hashlib.sha256(data).hexdigest()
                    except Exception as error:
                        entry["outputs"][f"{engine}/{target.value}"] = {"error": f"{type(error).__name__}: {error}"}
        except Exception as error:
            entry["error"] = f"{type(error).__name__}: {error}"
        manifest["documents"].append(entry)
        (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(entry, ensure_ascii=False), flush=True)


def main() -> None:
    """解析可重放验收所需的源码路径、样本清单和输出目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mineru-source", type=Path)
    parser.add_argument("--docvortex-source", type=Path)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("documents", nargs="+", help="样本目录内的 PDF 文件名")
    capture(parser.parse_args())


if __name__ == "__main__":
    main()
