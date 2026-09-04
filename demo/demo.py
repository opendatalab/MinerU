# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import asyncio
from pathlib import Path

from mineru.filetypes import PAGE_RANGE_PARSE_EXTENSIONS, PARSEABLE_EXTENSIONS, batch_effective_parse_tier
from mineru.parser.api_client import MinerUApiParser
from mineru.parser.file_type import guess_suffix_by_path
from mineru.parser.writer import FileBasedDataWriter
from mineru.types import Tier


def collect_input_files(input_path: str | Path) -> list[Path]:
    """收集单个输入文件或目录中的当前 V1 API 支持文件。"""
    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    if path.is_file():
        file_suffix = guess_suffix_by_path(path)
        if file_suffix not in PARSEABLE_EXTENSIONS:
            raise ValueError(f"Unsupported input file type: {path.name}")
        return [path]

    if not path.is_dir():
        raise ValueError(f"Input path must be a file or directory: {path}")

    input_files = sorted(
        (
            candidate.resolve()
            for candidate in path.iterdir()
            if candidate.is_file() and guess_suffix_by_path(candidate) in PARSEABLE_EXTENSIONS
        ),
        key=lambda item: item.name,
    )
    if not input_files:
        raise ValueError(f"No supported files found in directory: {path}")
    return input_files


async def run_demo(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    api_url: str,
    api_key: str | None = None,
    tier: Tier = "standard",
    page_range: str = "",
) -> None:
    """通过公开 V1 API 异步解析文件并保存标准 ParseResult 产物。"""
    input_files = collect_input_files(input_path)
    output_root = Path(output_dir).expanduser().resolve()

    for input_file in input_files:
        file_suffix = guess_suffix_by_path(input_file)
        effective_page_range = page_range if file_suffix in PAGE_RANGE_PARSE_EXTENSIONS else ""
        effective_tier = batch_effective_parse_tier(tier, input_file)
        parser = MinerUApiParser(
            api_url=api_url,
            api_key=api_key,
            tier=effective_tier,
            include_images=True,
            include_model_output=True,
        )
        print(f"Parsing {input_file.name} with tier={effective_tier}")
        result = await parser.parse_async(input_file, page_range=effective_page_range)

        document_output = output_root / input_file.stem
        result.save(FileBasedDataWriter(str(document_output)))
        result.middle_json.export(
            document_output,
            json_name="middle_json_without_base64.json",
            overwrite=True,
        )
        print(f"Saved result to: {document_output}")


def main() -> None:
    """运行仓库内置的 V1 API 异步解析示例。"""
    demo_dir = Path(__file__).resolve().parent

    # 先运行 `mineru-kit api-server --host 127.0.0.1 --port 8000`。
    api_url = "http://127.0.0.1:8000"
    # 可改为单个文件；目录模式只读取当前层的支持格式文件。
    input_path = demo_dir / "pdfs"
    output_dir = demo_dir / "api_output"
    # V1 页码范围从 1 开始，仅对 PDF 生效；空字符串表示整份文档。
    page_range = ""

    asyncio.run(
        run_demo(
            input_path=input_path,
            output_dir=output_dir,
            api_url=api_url,
            tier="standard",
            page_range=page_range,
        )
    )


if __name__ == "__main__":
    main()
