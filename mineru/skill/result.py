# Copyright (c) Opendatalab. All rights reserved.
"""ocr-mineru skill 解析结果模型与工具"""

import base64
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union


@dataclass
class ParseResult:
    """MinerU 解析结果封装"""

    markdown: str
    content_list: list[dict[str, Any]]
    content_list_v2: list[dict[str, Any]]
    middle_json: dict[str, Any]
    model_output: Optional[dict[str, Any]]
    images: dict[str, str]
    image_paths: list[Path]
    output_dir: Path
    file_name: str

    def get_text(self, block_types: Optional[list[str]] = None) -> str:
        """提取指定类型的文本内容

        Args:
            block_types: 过滤的块类型列表，如 ["text", "title", "paragraph"]。
                        为 None 时提取所有文本块。

        Returns:
            拼接后的纯文本
        """
        if not self.content_list_v2:
            return self.markdown

        texts = []
        for item in self.content_list_v2:
            item_type = item.get("type", "")
            if block_types is None or item_type in block_types:
                text = item.get("text", "")
                if text:
                    texts.append(text)
        return "\n\n".join(texts)

    def get_tables(self) -> list[dict[str, Any]]:
        """提取所有表格内容块"""
        return [
            item
            for item in self.content_list_v2
            if item.get("type") in ("table", "simple_table", "complex_table")
        ]

    def get_images(self) -> list[tuple[str, str]]:
        """提取所有图片

        Returns:
            [(filename, base64_dataurl), ...]
        """
        return list(self.images.items())

    def save_markdown(self, path: Union[str, Path]) -> Path:
        """保存 markdown 到指定路径"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.markdown, encoding="utf-8")
        return path

    def save_content_list(self, path: Union[str, Path]) -> Path:
        """保存 content_list_v2 到指定路径"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.content_list_v2, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def save_middle_json(self, path: Union[str, Path]) -> Path:
        """保存 middle_json 到指定路径"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.middle_json, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def save_all(self, output_dir: Union[str, Path]) -> Path:
        """将所有解析产物复制到指定目录

        Args:
            output_dir: 目标目录

        Returns:
            目标目录的 Path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 复制整个结果目录
        target_dir = output_dir / self.file_name
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(self.output_dir, target_dir)

        return output_dir
