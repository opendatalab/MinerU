#!/usr/bin/env python3
"""从三个真实 wheel 检查 Python 声明、平台依赖和二进制覆盖；不安装 GPU 引擎。"""

from __future__ import annotations

import argparse
import email
import json
import os
import subprocess
import urllib.request
import zipfile
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.tags import compatible_tags, cpython_tags, mac_platforms
from packaging.utils import canonicalize_name, parse_wheel_filename

PLATFORMS = {
    "linux": "x86_64-manylinux_2_34",
    "windows": "x86_64-pc-windows-msvc",
    "macos": "aarch64-apple-darwin",
    "macos-intel": "x86_64-apple-darwin",
}
FORBIDDEN = {"fast-langdetect", "fasttext-predict", "robust-downloader"}


def read_wheel(path: Path) -> tuple[str, str, str]:
    """读取真实发行元数据，显式校验直接 URL 依赖也必须遵守的 Python 范围。"""
    with zipfile.ZipFile(path) as archive:
        name = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
        metadata = email.message_from_bytes(archive.read(name))
    return canonicalize_name(metadata["Name"]), metadata["Requires-Python"], path.resolve().as_uri()


def target_tags(platform: str, python: str) -> set:
    """生成普通 CPython 的目标 ABI 标签，不把 free-threaded wheel 当作可用包。"""
    version = tuple(map(int, python.split(".")))
    if platform == "linux":
        platforms = [f"manylinux_2_{minor}_x86_64" for minor in range(34, 4, -1)]
        platforms += ["manylinux2014_x86_64", "manylinux2010_x86_64", "manylinux1_x86_64", "linux_x86_64"]
    elif platform == "windows":
        platforms = ["win_amd64"]
    else:
        platforms = list(mac_platforms((14, 0), "arm64" if platform == "macos" else "x86_64"))
    interpreter = "cp" + python.replace(".", "")
    return set(cpython_tags(version, abis=[interpreter], platforms=platforms)) | set(
        compatible_tags(version, interpreter=interpreter, platforms=platforms)
    )


def verify_wheels(resolution: str, platform: str, python: str, cache: Path, *, check_wheels: bool) -> list[str]:
    """始终检查发布包的 Python 范围，按开关检查 wheel；只允许纯 Python jieba 从源码安装。"""
    tags = target_tags(platform, python)
    failures = []
    cache.mkdir(parents=True, exist_ok=True)
    for line in resolution.splitlines():
        if "==" not in line or line.startswith((" ", "#")):
            continue
        requirement = Requirement(line)
        name = canonicalize_name(requirement.name)
        version = next(iter(requirement.specifier)).version
        cached = cache / f"{name}-{version}.json"
        if not cached.exists():
            with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json", timeout=30) as response:
                temporary = cached.with_suffix(f".{os.getpid()}.tmp")
                temporary.write_bytes(response.read())
                temporary.replace(cached)
        data = json.loads(cached.read_text())
        requires_python = data["info"].get("requires_python")
        if requires_python and python not in SpecifierSet(requires_python):
            failures.append(f"{name}=={version}: Python {python} excluded by {requires_python}")
        available = not check_wheels or any(
            item["filename"].endswith(".whl") and bool(parse_wheel_filename(item["filename"])[3] & tags)
            for item in data["urls"]
        )
        if not available and name != "jieba":
            failures.append(f"{name}=={version}: no wheel for {platform} Python {python}")
    return failures


def check_backend_dependencies(selected: set[str], package: str, extra: str, platform: str) -> list[str]:
    """核验真实求解结果的平台后端组合，独立 MLX extra 不受 MinerU 默认策略影响。"""
    checks = []
    if package == "mineru":
        required = {"gradio", "lxml", "onnxruntime", "mineru-llama-cpp"}
        if platform == "macos" or extra in {"torch", "full", "all"}:
            required.update({"torch", "torchvision", "transformers", "accelerate", "safetensors"})
        if extra in {"full", "all"} and platform in {"linux", "windows"}:
            required.add("vllm" if platform == "linux" else "lmdeploy")
        checks.extend(f"missing platform dependency: {name}" for name in sorted(required - selected))
        checks.extend(f"unexpected automatic MLX dependency: {name}" for name in sorted(selected & {"mlx", "mlx-vlm"}))
    if extra == "base" and not (package == "mineru" and platform == "macos"):
        checks.extend(f"heavy base dependency: {name}" for name in sorted(selected & {"torch", "transformers"}))
    return checks


def main() -> None:
    """运行有效组合并明确记录预期冲突，结果包含完整求解输出和发行包证据。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mineru-wheel", type=Path, required=True)
    parser.add_argument("--utils-wheel", type=Path, required=True)
    parser.add_argument("--docvortex-wheel", type=Path, required=True)
    parser.add_argument("--platform", choices=tuple(PLATFORMS), required=True)
    parser.add_argument("--python", nargs="+", default=["3.10", "3.11", "3.12", "3.13", "3.14"])
    parser.add_argument("--transformers", choices=["5.10.1", "5.14.0", "5.16.1"])
    parser.add_argument("--extras", nargs="+", default=["base", "torch", "full", "all"])
    parser.add_argument("--utils-matrix", action="store_true")
    parser.add_argument("--check-wheels", action="store_true")
    parser.add_argument("--requirement", action="append", default=[])
    parser.add_argument("--expect-failure", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roots = [read_wheel(path) for path in (args.mineru_wheel, args.utils_wheel, args.docvortex_wheel)]
    by_name = {name: (requires, uri) for name, requires, uri in roots}
    cases = [("mineru", extra) for extra in args.extras]
    if args.utils_matrix:
        backend = {"linux": "vllm", "windows": "lmdeploy", "macos": "mlx", "macos-intel": "llama-cpp"}[args.platform]
        cases += [("mineru-vl-utils", extra) for extra in ("base", "transformers", "llama-cpp", backend)]
    records = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for python in args.python:
        for package, extra in cases:
            root_names = [name for name, _, _ in roots] if package == "mineru" else ["mineru-vl-utils"]
            requirements = [
                f"{name}{('[' + extra + ']') if name == package and extra != 'base' else ''} @ {by_name[name][1]}"
                for name in root_names
            ]
            if args.transformers and (extra != "base" or (package == "mineru" and args.platform == "macos")):
                requirements.append(f"transformers=={args.transformers}")
            requirements.extend(args.requirement)
            invalid_roots = [name for name in root_names if python not in SpecifierSet(by_name[name][0])]
            process = subprocess.run(
                [
                    "uv",
                    "pip",
                    "compile",
                    "-",
                    "--python-version",
                    python,
                    "--python-platform",
                    PLATFORMS[args.platform],
                    "--index-url",
                    "https://pypi.org/simple",
                    "--no-header",
                    "--no-annotate",
                    "--no-progress",
                ],
                input="\n".join(requirements) + "\n",
                text=True,
                capture_output=True,
                check=False,
                env={**os.environ, "MACOSX_DEPLOYMENT_TARGET": "14.0"},
                timeout=180,
            )
            mlx_conflict = (
                args.transformers == "5.10.1" and args.platform == "macos" and package == "mineru-vl-utils" and extra == "mlx"
            )
            expected_failure = args.expect_failure or mlx_conflict
            resolved = process.returncode == 0 and not invalid_roots
            checks = []
            if resolved:
                selected = {
                    canonicalize_name(Requirement(line).name)
                    for line in process.stdout.splitlines()
                    if line and not line.startswith((" ", "#"))
                }
                checks += [f"forbidden dependency: {name}" for name in selected & FORBIDDEN]
                if not args.requirement:
                    checks.extend(check_backend_dependencies(selected, package, extra, args.platform))
                compatibility_errors = verify_wheels(
                    process.stdout, args.platform, python, args.output.parent / "pypi-metadata", check_wheels=args.check_wheels
                )
                checks += compatibility_errors
                resolved = resolved and not compatibility_errors
            else:
                compatibility_errors = []
            rejection_confirmed = bool(invalid_roots or compatibility_errors) or "No solution found" in process.stderr
            success = not resolved and rejection_confirmed if expected_failure else resolved and not checks
            records.append(
                {
                    "package": package,
                    "python": python,
                    "extra": extra,
                    "success": success,
                    "expected_failure": expected_failure,
                    "resolved": resolved,
                    "resolver_succeeded": process.returncode == 0,
                    "invalid_roots": invalid_roots,
                    "wheel_or_dependency_errors": checks,
                    "resolution": process.stdout,
                    "diagnostics": process.stderr,
                }
            )
            print(
                f"{args.platform} Python {python} {package}[{extra}]: {'PASS' if success else 'FAIL'}"
                f"{' (expected rejection)' if expected_failure else ''}",
                flush=True,
            )
    args.output.write_text(json.dumps({"platform": args.platform, "roots": roots, "results": records}, indent=2) + "\n")
    raise SystemExit(not all(record["success"] for record in records))


if __name__ == "__main__":
    main()
