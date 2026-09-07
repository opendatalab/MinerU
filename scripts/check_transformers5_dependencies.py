#!/usr/bin/env python3
"""从已构建 wheel 验证跨平台依赖可解性，不安装 GPU 引擎或修改目标环境。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def main() -> None:
    """在四个 Python 版本和基础/可选安装模式下解析真实发布元数据。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mineru-wheel", type=Path, required=True)
    parser.add_argument("--utils-wheel", type=Path, required=True)
    parser.add_argument("--platform", choices=("linux", "windows", "macos"), required=True)
    parser.add_argument("--transformers", choices=("5.10.1", "5.16.1"), required=True)
    parser.add_argument("--python", nargs="+", default=["3.10", "3.11", "3.12", "3.13"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    platforms = {"linux": "x86_64-manylinux_2_34", "windows": "x86_64-pc-windows-msvc", "macos": "aarch64-apple-darwin"}
    mineru = args.mineru_wheel.resolve().as_uri()
    utils = args.utils_wheel.resolve().as_uri()
    records = []
    failed = False
    for python in args.python:
        for extra in ("", "torch", "full", "all"):
            requirement = f"mineru[{extra}]" if extra else "mineru"
            source = f"{requirement} @ {mineru}\nmineru-vl-utils @ {utils}\n"
            if extra:
                source += f"transformers=={args.transformers}\n"
            process = subprocess.run(
                [
                    "uv",
                    "pip",
                    "compile",
                    "-",
                    "--python-version",
                    python,
                    "--python-platform",
                    platforms[args.platform],
                    "--no-header",
                    "--no-annotate",
                    "--no-progress",
                ],
                input=source,
                text=True,
                capture_output=True,
                check=False,
                env={**os.environ, "MACOSX_DEPLOYMENT_TARGET": "14.0"},
            )
            # 基础安装的解析结果也必须保持无 Torch/Transformers 依赖。
            heavy_in_base = not extra and any(
                line.startswith(("torch==", "transformers==")) for line in process.stdout.splitlines()
            )
            success = process.returncode == 0 and not heavy_in_base
            failed |= not success
            records.append(
                {
                    "python": python,
                    "extra": extra or "base",
                    "success": success,
                    "resolution": process.stdout,
                    "diagnostics": process.stderr,
                }
            )
            print(f"{args.platform} Python {python} {extra or 'base'}: {'PASS' if success else 'FAIL'}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"platform": args.platform, "transformers": args.transformers, "results": records}, indent=2) + "\n"
    )
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
