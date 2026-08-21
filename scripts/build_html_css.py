#!/usr/bin/env python3
"""生成并校验 MinerU HTML 渲染器使用的压缩 CSS。"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Sequence


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_PATH = _PROJECT_ROOT / "mineru" / "resources" / "html" / "mineru.css"
_OUTPUT_PATH = _PROJECT_ROOT / "mineru" / "resources" / "html" / "mineru.min.css"
_COMPACT_PUNCTUATION = frozenset("{}:;,>")


def minify_css(source: str) -> str:
    """在不改写 CSS 值的前提下删除注释和可安全省略的空白。"""
    output: list[str] = []
    pending_space = False
    quote: str | None = None
    brace_depth = 0
    index = 0

    while index < len(source):
        char = source[index]

        if quote is not None:
            output.append(char)
            if char == "\\":
                index += 1
                if index >= len(source):
                    raise ValueError("CSS 字符串以未完成的转义结尾")
                output.append(source[index])
            elif char == quote:
                quote = None
            index += 1
            continue

        if source.startswith("/*", index):
            comment_end = source.find("*/", index + 2)
            if comment_end < 0:
                raise ValueError("CSS 包含未闭合的注释")
            pending_space = bool(output)
            index = comment_end + 2
            continue

        if char in {'"', "'"}:
            if pending_space and _needs_separator(output, char):
                output.append(" ")
            pending_space = False
            quote = char
            output.append(char)
            index += 1
            continue

        if char == "\\":
            if pending_space and _needs_separator(output, char):
                output.append(" ")
            pending_space = False
            output.append(char)
            index += 1
            if index >= len(source):
                raise ValueError("CSS 以未完成的转义结尾")
            output.append(source[index])
            index += 1
            continue

        if char.isspace():
            pending_space = bool(output)
            index += 1
            continue

        if char in _COMPACT_PUNCTUATION:
            # 伪类前的空白可能是后代组合符，不能按声明冒号处理。
            if (
                char == ":"
                and pending_space
                and output
                and output[-1] not in "{,;>"
            ):
                output.append(" ")
            elif output and output[-1] == " ":
                output.pop()
            if char == "}" and output and output[-1] == ";":
                output.pop()
            if char == "{":
                brace_depth += 1
            elif char == "}":
                brace_depth -= 1
                if brace_depth < 0:
                    raise ValueError("CSS 包含多余的右花括号")
            output.append(char)
            pending_space = False
            index += 1
            continue

        if pending_space and _needs_separator(output, char):
            output.append(" ")
        pending_space = False
        output.append(char)
        index += 1

    if quote is not None:
        raise ValueError("CSS 包含未闭合的字符串")
    if brace_depth:
        raise ValueError("CSS 包含未闭合的规则块")
    return "".join(output).strip()


def _needs_separator(output: list[str], next_char: str) -> bool:
    """判断待处理空白是否用于分隔两个不能直接相邻的 CSS token。"""
    return (
        bool(output)
        and output[-1] not in _COMPACT_PUNCTUATION
        and next_char not in _COMPACT_PUNCTUATION
    )


def _write_atomically(path: Path, content: str) -> None:
    """在目标目录创建临时文件并原子替换最终 CSS 产物。"""
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as temporary:
            temporary.write(content)
            temporary_path = Path(temporary.name)
        temporary_path.chmod(0o644)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def build_html_css(*, check: bool) -> bool:
    """生成压缩 CSS，或只检查现有产物是否与源码一致。"""
    minified = minify_css(_SOURCE_PATH.read_text(encoding="utf-8"))
    existing = _OUTPUT_PATH.read_text(encoding="utf-8") if _OUTPUT_PATH.exists() else None
    if existing == minified:
        return True
    if check:
        return False
    _write_atomically(_OUTPUT_PATH, minified)
    return True


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """解析构建脚本的只校验选项。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="仅校验 mineru.min.css 是否需要重新生成")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """执行 CSS 构建并用退出码报告同步状态。"""
    args = _parse_args(argv)
    if build_html_css(check=args.check):
        action = "校验通过" if args.check else "生成完成"
        print(f"{action}: {_OUTPUT_PATH.relative_to(_PROJECT_ROOT)}")
        return 0
    print(
        f"压缩 CSS 已过期，请运行: {Path(__file__).relative_to(_PROJECT_ROOT)}",
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
