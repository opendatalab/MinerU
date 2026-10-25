"""转换结果的 Gradio 组件更新与浏览器完整回执适配。"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import quote

from .conversion import ConversionCancelled, ConversionFailure, ConversionOutcome
from .i18n import preview_placeholder


def download_updates(formats: Sequence[tuple[str, str]], *, interactive: bool, run_id: str = "") -> tuple[object, ...]:
    """同步当前素材标识，并恢复全部下载按钮的标签与交互状态。"""
    import gradio as gr

    return (run_id, *(gr.update(value=label, interactive=interactive) for _format_name, label in formats))


def conversion_updates(
    outcome: ConversionOutcome,
    *,
    status_html: str,
    download_formats: Sequence[tuple[str, str]],
) -> tuple[object, ...]:
    """将具名转换结果统一编码为既有公开 API 的组件输出顺序。"""
    import gradio as gr

    if isinstance(outcome, ConversionCancelled):
        return tuple(gr.skip() for _ in range(9 + len(download_formats)))
    if isinstance(outcome, ConversionFailure):
        return (
            status_html,
            "",
            *(gr.skip() for _ in range(4)),
            None,
            *download_updates(download_formats, interactive=False),
            "",
        )
    output = outcome.output
    preview = output.preview
    if preview.preserve_source:
        preview_updates = tuple(gr.skip() for _ in range(4))
    else:
        generic_html = preview_placeholder("result_ready") if preview.show_ready_placeholder else ""
        preview_updates = (
            gr.update(value=str(preview.pdf_path) if preview.pdf_path else None, visible=False),
            gr.update(value=str(preview.image_path) if preview.image_path else None, visible=preview.image_path is not None),
            gr.update(value="", visible=False),
            gr.update(value=generic_html, visible=bool(generic_html)),
        )
    return (
        status_html,
        output.html,
        *preview_updates,
        output.artifacts.as_state(),
        *download_updates(download_formats, interactive=True, run_id=output.artifacts.root.name),
        output.structured_json,
    )


def build_conversion_receipt(
    outcome: ConversionOutcome,
    *,
    status_html: str,
    run_id: str,
    sequence: int,
    output_root: Path,
    public_base_url: str,
    download_formats: Sequence[tuple[str, str]],
) -> str:
    """校验结果文件归属并生成既有浏览器回执，不向浏览器发送内部素材状态。"""
    from gradio.data_classes import FileData

    if isinstance(outcome, ConversionCancelled):
        return ""
    values = list(conversion_updates(outcome, status_html=status_html, download_formats=download_formats))
    # 两个文件组件使用既有输出根目录路由，输出索引仅保留在此序列化边界。
    for index in (2, 3):
        update = values[index]
        if isinstance(update, dict) and update.get("value"):
            path = Path(update["value"]).resolve()
            path.relative_to(output_root.resolve())
            file_url = f"{public_base_url}/gradio_api/file={quote(path.as_posix(), safe='/')}"
            values[index] = {
                **update,
                "value": FileData(path=str(path), url=file_url, orig_name=path.name).model_dump(mode="json"),
            }
    return json.dumps(
        {
            "run_id": run_id,
            "sequence": sequence,
            "ready_at": time.time(),
            "outputs": values[:6] + values[7:],
        },
        ensure_ascii=False,
    )


__all__ = ["build_conversion_receipt", "conversion_updates", "download_updates"]
