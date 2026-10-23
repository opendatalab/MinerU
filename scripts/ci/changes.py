"""根据 GitHub 事件中的固定提交判定代码和文档是否需要验证。"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any


def classify_paths(paths: list[str]) -> dict[str, bool]:
    """文档变更只构建文档，CI 公共脚本变更同时验证两个工作流。"""
    docs = any(
        path.startswith("docs/")
        or path.startswith("README")
        or path in {"mkdocs.yml", ".github/workflows/mkdocs.yml", "scripts/ci/changes.py"}
        for path in paths
    )
    code = any(not (path.startswith(("docs/", "cmds/")) or path.endswith(".md") or path == "mkdocs.yml") for path in paths)
    return {"code": code, "docs": docs}


def changed_paths(event_name: str, event: dict[str, Any], head: str) -> list[str] | None:
    """PR 对比合并提交与基分支；首次推送和手动/发布调用执行完整检查。"""
    if event_name == "pull_request":
        base = event["pull_request"]["base"]["sha"]
    elif event_name == "push" and not event.get("ref", "").startswith("refs/tags/"):
        base = event.get("before", "")
        if not base or base == "0" * 40:
            return None
    else:
        return None
    if not all(re.fullmatch(r"[0-9a-f]{40}", value) for value in (base, head)):
        raise ValueError("Expected full commit SHAs in the GitHub event")
    # 被强制推送移出的提交可能不在 checkout 中；无法取回时保守运行全部检查。
    if subprocess.run(["git", "cat-file", "-e", f"{base}^{{commit}}"], capture_output=True).returncode:
        fetched = subprocess.run(["git", "fetch", "--no-tags", "--depth=1", "origin", base], capture_output=True)
        if fetched.returncode:
            return None
    result = subprocess.run(
        ["git", "diff", "--name-only", "--no-renames", "-z", base, head], check=True, capture_output=True, text=True
    )
    return [path for path in result.stdout.split("\0") if path]


def main() -> None:
    """把变化分类写入 Actions 输出，供始终存在的汇总检查决定跳过哪些任务。"""
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8"))
    paths = changed_paths(os.environ["GITHUB_EVENT_NAME"], event, os.environ["GITHUB_SHA"])
    changes = {"code": True, "docs": True} if paths is None else classify_paths(paths)
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
        for name, changed in changes.items():
            output.write(f"{name}={str(changed).lower()}\n")
    print(json.dumps(changes))


if __name__ == "__main__":
    main()
