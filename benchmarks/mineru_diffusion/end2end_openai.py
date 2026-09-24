from __future__ import annotations

import argparse
import html
import itertools
import json
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import requests
from PIL import Image

from benchmarks.mineru_diffusion.harness import STOP_STRINGS


SYSTEM_PROMPT = "You are a helpful assistant."
LAYOUT_IMAGE_SIZE = (1036, 1036)
MIN_IMAGE_EDGE = 28
MAX_IMAGE_EDGE_RATIO = 50
PARATEXT_TYPES = {
    "header",
    "footer",
    "page_number",
    "aside_text",
    "page_footnote",
    "unknown",
}
TASK_PROMPTS = {
    "table": "\nTable Recognition:",
    "equation": "\nFormula Recognition:",
    "[default]": "\nText Recognition:",
    "[layout]": "\nLayout Detection:",
}
ANGLE_MAPPING = {
    "<|rotate_up|>": 0,
    "<|rotate_right|>": 90,
    "<|rotate_down|>": 180,
    "<|rotate_left|>": 270,
}
LAYOUT_RE = re.compile(
    r"^<\|box_start\|>(\d+)\s+(\d+)\s+(\d+)\s+(\d+)<\|box_end\|>"
    r"<\|ref_start\|>(\w+?)<\|ref_end\|>(.*)$"
)
OTSL_NL = "<nl>"
OTSL_FCEL = "<fcel>"
OTSL_ECEL = "<ecel>"
OTSL_LCEL = "<lcel>"
OTSL_UCEL = "<ucel>"
OTSL_XCEL = "<xcel>"
OTSL_TOKENS = {
    OTSL_NL,
    OTSL_FCEL,
    OTSL_ECEL,
    OTSL_LCEL,
    OTSL_UCEL,
    OTSL_XCEL,
}
OTSL_PATTERN = re.compile(
    "("
    + "|".join(
        re.escape(token)
        for token in [OTSL_NL, OTSL_FCEL, OTSL_ECEL, OTSL_LCEL, OTSL_UCEL, OTSL_XCEL]
    )
    + ")"
)


@dataclass
class ContentBlock:
    type: str
    bbox: list[float]
    angle: int | None = None
    content: str | None = None


@dataclass
class TableCell:
    text: str
    row_span: int
    col_span: int
    start_row: int
    start_col: int


@dataclass(frozen=True)
class End2EndConfig:
    image_path: Path
    output_dir: Path
    endpoint: str
    model: str = "mineru-diffusion"
    timeout: float = 600.0
    layout_max_tokens: int = 2048
    content_max_tokens: int = 1024
    table_max_tokens: int = 2048
    formula_max_tokens: int = 1024
    block_size: int = 32
    max_denoising_steps: int | None = None
    temperature: float = 1.0
    dynamic_threshold: float = 0.95
    content_concurrency: int = 1
    keep_paratext: bool = False


@dataclass
class End2EndResult:
    markdown: str
    layout_output: str
    blocks: list[ContentBlock]
    metrics: dict[str, Any]
    result_path: Path
    markdown_path: Path


@dataclass
class _BatchPageState:
    config: End2EndConfig
    page_image: Image.Image
    artifacts_dir: Path
    layout_output: str = ""
    layout_elapsed: float = 0.0
    blocks: list[ContentBlock] = field(default_factory=list)
    extracted_blocks: int = 0


class End2EndClient(Protocol):
    def infer(self, image_path: Path, prompt: str, max_tokens: int) -> str:
        ...


class End2EndOpenAIClient:
    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        timeout: float,
        block_size: int,
        dynamic_threshold: float,
        max_denoising_steps: int | None = None,
        temperature: float = 1.0,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout
        self.block_size = block_size
        self.dynamic_threshold = dynamic_threshold
        self.max_denoising_steps = max_denoising_steps
        self.temperature = temperature
        self._thread_local = threading.local()

    def _session(self) -> requests.Session:
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            session.trust_env = False
            self._thread_local.session = session
        return session

    def infer(self, image_path: Path, prompt: str, max_tokens: int) -> str:
        vllm_xargs = {
            "block_size": self.block_size,
            "dynamic_threshold": self.dynamic_threshold,
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"file://{image_path}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "stop": list(STOP_STRINGS),
            "block_size": self.block_size,
            "dynamic_threshold": self.dynamic_threshold,
            "vllm_xargs": vllm_xargs,
        }
        if self.max_denoising_steps is not None:
            payload["max_denoising_steps"] = self.max_denoising_steps
            vllm_xargs["max_denoising_steps"] = self.max_denoising_steps
        response = self._session().post(
            self.endpoint,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return trim_response(response.json()["choices"][0]["message"]["content"])


def trim_response(text: str) -> str:
    for stop in STOP_STRINGS:
        text = text.split(stop, 1)[0]
    return text.strip()


def get_rgb_image(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def resize_by_need(image: Image.Image) -> Image.Image:
    edge_ratio = max(image.size) / min(image.size)
    if edge_ratio > MAX_IMAGE_EDGE_RATIO:
        width, height = image.size
        if width > height:
            new_w, new_h = width, math.ceil(width / MAX_IMAGE_EDGE_RATIO)
        else:
            new_w, new_h = math.ceil(height / MAX_IMAGE_EDGE_RATIO), height
        new_image = Image.new(image.mode, (new_w, new_h), (255, 255, 255))
        new_image.paste(image, (int((new_w - width) / 2), int((new_h - height) / 2)))
        image = new_image
    if min(image.size) < MIN_IMAGE_EDGE:
        scale = MIN_IMAGE_EDGE / min(image.size)
        new_w, new_h = round(image.width * scale), round(image.height * scale)
        image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
    return image


def prepare_layout_image(image: Image.Image) -> Image.Image:
    return get_rgb_image(image).resize(LAYOUT_IMAGE_SIZE, Image.Resampling.BICUBIC)


def convert_bbox(raw_bbox: tuple[str, str, str, str]) -> list[float] | None:
    x1, y1, x2, y2 = map(int, raw_bbox)
    if any(coord < 0 or coord > 1000 for coord in (x1, y1, x2, y2)):
        return None
    x1, x2 = (x2, x1) if x2 < x1 else (x1, x2)
    y1, y2 = (y2, y1) if y2 < y1 else (y1, y2)
    if x1 == x2 or y1 == y2:
        return None
    return [value / 1000.0 for value in (x1, y1, x2, y2)]


def parse_angle(tail: str) -> int | None:
    for token, angle in ANGLE_MAPPING.items():
        if token in tail:
            return angle
    return None


def parse_layout_output(output: str) -> list[ContentBlock]:
    blocks: list[ContentBlock] = []
    for line in output.splitlines():
        match = LAYOUT_RE.match(line.strip())
        if not match:
            continue
        x1, y1, x2, y2, block_type, tail = match.groups()
        bbox = convert_bbox((x1, y1, x2, y2))
        if bbox is None:
            continue
        blocks.append(
            ContentBlock(
                type=block_type.lower(),
                bbox=bbox,
                angle=parse_angle(tail),
            )
        )
    return blocks


def crop_block_image(page_image: Image.Image, block: ContentBlock) -> Image.Image:
    image = get_rgb_image(page_image)
    width, height = image.size
    left = max(0, min(width - 1, math.floor(block.bbox[0] * width)))
    top = max(0, min(height - 1, math.floor(block.bbox[1] * height)))
    right = max(left + 1, min(width, math.ceil(block.bbox[2] * width)))
    bottom = max(top + 1, min(height, math.ceil(block.bbox[3] * height)))
    cropped = image.crop((left, top, right, bottom))
    if block.angle in (90, 180, 270):
        cropped = cropped.rotate(block.angle, expand=True)
    return resize_by_need(cropped)


def extract_otsl_tokens_and_text(raw_text: str) -> tuple[list[str], list[str]]:
    tokens = OTSL_PATTERN.findall(raw_text)
    text_parts = [
        part for part in OTSL_PATTERN.split(raw_text) if part and part.strip()
    ]
    return tokens, text_parts


def count_span_right(
    rows: list[list[str]],
    row_idx: int,
    col_idx: int,
    span_tokens: set[str],
) -> int:
    span = 0
    cursor = col_idx
    while cursor < len(rows[row_idx]) and rows[row_idx][cursor] in span_tokens:
        span += 1
        cursor += 1
    return span


def count_span_down(
    rows: list[list[str]],
    row_idx: int,
    col_idx: int,
    span_tokens: set[str],
) -> int:
    span = 0
    cursor = row_idx
    while (
        cursor < len(rows)
        and col_idx < len(rows[cursor])
        and rows[cursor][col_idx] in span_tokens
    ):
        span += 1
        cursor += 1
    return span


def convert_otsl_to_html(otsl_content: str) -> str:
    if otsl_content.startswith("<table") and otsl_content.endswith("</table>"):
        return otsl_content

    tokens, mixed_texts = extract_otsl_tokens_and_text(otsl_content)
    rows = [
        list(group)
        for is_nl, group in itertools.groupby(tokens, lambda item: item == OTSL_NL)
        if not is_nl
    ]
    if not rows:
        return otsl_content.strip()

    max_cols = max(len(row) for row in rows)
    for row in rows:
        while len(row) < max_cols:
            row.append(OTSL_ECEL)

    normalized_texts: list[str] = []
    text_idx = 0
    for row in rows:
        for token in row:
            normalized_texts.append(token)
            if text_idx < len(mixed_texts) and mixed_texts[text_idx] == token:
                text_idx += 1
                if (
                    text_idx < len(mixed_texts)
                    and mixed_texts[text_idx] not in OTSL_TOKENS
                ):
                    normalized_texts.append(mixed_texts[text_idx])
                    text_idx += 1
        normalized_texts.append(OTSL_NL)
        if text_idx < len(mixed_texts) and mixed_texts[text_idx] == OTSL_NL:
            text_idx += 1

    cells: list[TableCell] = []
    row_idx = 0
    col_idx = 0
    for index, part in enumerate(normalized_texts):
        if part in (OTSL_FCEL, OTSL_ECEL):
            row_span = 1
            col_span = 1
            next_offset = 1
            cell_text = ""
            if (
                index + 1 < len(normalized_texts)
                and normalized_texts[index + 1] not in OTSL_TOKENS
            ):
                cell_text = normalized_texts[index + 1].strip()
                next_offset = 2
            next_right = (
                normalized_texts[index + next_offset]
                if index + next_offset < len(normalized_texts)
                else ""
            )
            next_down = (
                rows[row_idx + 1][col_idx]
                if row_idx + 1 < len(rows) and col_idx < len(rows[row_idx + 1])
                else ""
            )
            if next_right in (OTSL_LCEL, OTSL_XCEL):
                col_span += count_span_right(
                    rows,
                    row_idx,
                    col_idx + 1,
                    {OTSL_LCEL, OTSL_XCEL},
                )
            if next_down in (OTSL_UCEL, OTSL_XCEL):
                row_span += count_span_down(
                    rows,
                    row_idx + 1,
                    col_idx,
                    {OTSL_UCEL, OTSL_XCEL},
                )
            cells.append(
                TableCell(
                    text=cell_text,
                    row_span=row_span,
                    col_span=col_span,
                    start_row=row_idx,
                    start_col=col_idx,
                )
            )
        if part in OTSL_TOKENS - {OTSL_NL}:
            col_idx += 1
        if part == OTSL_NL:
            row_idx += 1
            col_idx = 0

    html_parts = ["<table>"]
    for row in range(len(rows)):
        html_parts.append("<tr>")
        for col in range(max_cols):
            cell = next(
                (
                    item
                    for item in cells
                    if item.start_row == row and item.start_col == col
                ),
                None,
            )
            if cell is None:
                continue
            attrs = []
            if cell.row_span > 1:
                attrs.append(f' rowspan="{cell.row_span}"')
            if cell.col_span > 1:
                attrs.append(f' colspan="{cell.col_span}"')
            html_parts.append(f"<td{''.join(attrs)}>{html.escape(cell.text)}</td>")
        html_parts.append("</tr>")
    html_parts.append("</table>")
    return "".join(html_parts)


def wrap_equation(content: str) -> str:
    content = content.strip()
    if not content:
        return ""
    if not content.startswith("\\["):
        content = f"\\[\n{content}"
    if not content.endswith("\\]"):
        content = f"{content}\n\\]"
    return content


def render_block_content(block: ContentBlock) -> str:
    content = (block.content or "").strip()
    if not content:
        return ""
    if block.type == "table":
        return convert_otsl_to_html(content)
    if block.type == "equation":
        return wrap_equation(content)
    return content


def should_extract_block(block: ContentBlock) -> bool:
    return block.type not in {"image", "equation_block"}


def should_keep_block(block: ContentBlock, keep_paratext: bool) -> bool:
    if not block.content:
        return False
    if not keep_paratext and block.type in PARATEXT_TYPES:
        return False
    return True


def _write_image(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def infer_block_content(
    *,
    client: End2EndClient,
    page_image: Image.Image,
    artifacts_dir: Path,
    block: ContentBlock,
    index: int,
    config: End2EndConfig,
) -> str:
    prompt = TASK_PROMPTS.get(block.type, TASK_PROMPTS["[default]"])
    max_tokens = config.content_max_tokens
    if block.type == "table":
        max_tokens = config.table_max_tokens
    elif block.type == "equation":
        max_tokens = config.formula_max_tokens
    block_image = crop_block_image(page_image, block)
    block_image_path = artifacts_dir / f"block_{index:04d}_{block.type}.png"
    _write_image(block_image_path, block_image)
    return render_block_content(
        ContentBlock(
            type=block.type,
            bbox=list(block.bbox),
            angle=block.angle,
            content=client.infer(block_image_path, prompt, max_tokens),
        )
    )


def extract_blocks(
    *,
    client: End2EndClient,
    page_image: Image.Image,
    artifacts_dir: Path,
    blocks: list[ContentBlock],
    config: End2EndConfig,
) -> int:
    targets = [
        (index, block)
        for index, block in enumerate(blocks)
        if should_extract_block(block)
    ]
    if config.content_concurrency <= 1 or len(targets) <= 1:
        for index, block in targets:
            block.content = infer_block_content(
                client=client,
                page_image=page_image,
                artifacts_dir=artifacts_dir,
                block=block,
                index=index,
                config=config,
            )
        return len(targets)

    with ThreadPoolExecutor(max_workers=config.content_concurrency) as executor:
        futures = {
            executor.submit(
                infer_block_content,
                client=client,
                page_image=page_image,
                artifacts_dir=artifacts_dir,
                block=block,
                index=index,
                config=config,
            ): (index, block)
            for index, block in targets
        }
        for future in as_completed(futures):
            _index, block = futures[future]
            block.content = future.result()
    return len(targets)


def render_markdown(blocks: list[ContentBlock], keep_paratext: bool) -> str:
    rendered_parts = [
        block.content or ""
        for block in blocks
        if should_keep_block(block, keep_paratext)
    ]
    return "\n\n".join(part for part in rendered_parts if part)


def write_end2end_result(
    *,
    output_dir: Path,
    markdown: str,
    layout_output: str,
    blocks: list[ContentBlock],
    metrics: dict[str, Any],
) -> tuple[Path, Path]:
    timestamp = int(time.time())
    markdown_path = output_dir / f"markdown_{timestamp}.md"
    result_path = output_dir / f"result_{timestamp}.json"
    markdown_path.write_text(markdown, encoding="utf-8")
    payload = {
        "metrics": metrics,
        "layout_output": layout_output,
        "markdown": markdown,
        "blocks": [asdict(block) for block in blocks],
        "markdown_path": str(markdown_path),
    }
    result_text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    result_path.write_text(result_text, encoding="utf-8")
    (output_dir / "latest_result.json").write_text(result_text, encoding="utf-8")
    return result_path, markdown_path


def _infer_layout_for_state(
    *,
    client: End2EndClient,
    state: _BatchPageState,
) -> None:
    layout_image_path = state.artifacts_dir / "layout_input.png"
    _write_image(layout_image_path, prepare_layout_image(state.page_image))
    layout_start = time.perf_counter()
    state.layout_output = client.infer(
        layout_image_path,
        TASK_PROMPTS["[layout]"],
        state.config.layout_max_tokens,
    )
    state.layout_elapsed = time.perf_counter() - layout_start
    state.blocks = parse_layout_output(state.layout_output)


def _extract_batch_blocks(
    *,
    client: End2EndClient,
    states: list[_BatchPageState],
    content_concurrency: int,
) -> int:
    targets = [
        (state, index, block)
        for state in states
        for index, block in enumerate(state.blocks)
        if should_extract_block(block)
    ]
    if content_concurrency <= 1 or len(targets) <= 1:
        for state, index, block in targets:
            block.content = infer_block_content(
                client=client,
                page_image=state.page_image,
                artifacts_dir=state.artifacts_dir,
                block=block,
                index=index,
                config=state.config,
            )
            state.extracted_blocks += 1
        return len(targets)

    with ThreadPoolExecutor(max_workers=content_concurrency) as executor:
        futures = {
            executor.submit(
                infer_block_content,
                client=client,
                page_image=state.page_image,
                artifacts_dir=state.artifacts_dir,
                block=block,
                index=index,
                config=state.config,
            ): (state, block)
            for state, index, block in targets
        }
        for future in as_completed(futures):
            state, block = futures[future]
            block.content = future.result()
            state.extracted_blocks += 1
    return len(targets)


def run_end2end_batch(
    configs: list[End2EndConfig],
    *,
    client: End2EndClient | None = None,
    layout_concurrency: int = 1,
    content_concurrency: int | None = None,
) -> list[End2EndResult]:
    if not configs:
        return []

    first = configs[0]
    client = client or End2EndOpenAIClient(
        endpoint=first.endpoint,
        model=first.model,
        timeout=first.timeout,
        block_size=first.block_size,
        dynamic_threshold=first.dynamic_threshold,
        max_denoising_steps=first.max_denoising_steps,
        temperature=first.temperature,
    )
    content_concurrency = content_concurrency or max(
        config.content_concurrency for config in configs
    )
    states: list[_BatchPageState] = []
    for config in configs:
        output_dir = config.output_dir
        artifacts_dir = output_dir / "artifacts"
        output_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        page_image = Image.open(config.image_path.expanduser().resolve()).convert("RGB")
        states.append(
            _BatchPageState(
                config=config,
                page_image=page_image,
                artifacts_dir=artifacts_dir,
            )
        )

    started = time.perf_counter()
    layout_start = time.perf_counter()
    try:
        if layout_concurrency <= 1 or len(states) <= 1:
            for state in states:
                _infer_layout_for_state(client=client, state=state)
        else:
            with ThreadPoolExecutor(max_workers=layout_concurrency) as executor:
                futures = [
                    executor.submit(
                        _infer_layout_for_state,
                        client=client,
                        state=state,
                    )
                    for state in states
                ]
                for future in as_completed(futures):
                    future.result()
        layout_wall_elapsed = time.perf_counter() - layout_start

        extract_start = time.perf_counter()
        _extract_batch_blocks(
            client=client,
            states=states,
            content_concurrency=content_concurrency,
        )
        extract_wall_elapsed = time.perf_counter() - extract_start
        wall_elapsed = time.perf_counter() - started

        results: list[End2EndResult] = []
        for state in states:
            markdown = render_markdown(
                state.blocks,
                state.config.keep_paratext,
            )
            metrics: dict[str, Any] = {
                "layout_elapsed": state.layout_elapsed,
                "num_blocks": len(state.blocks),
                "num_extracted_blocks": state.extracted_blocks,
                "markdown_chars": len(markdown),
                "content_concurrency": content_concurrency,
                "layout_concurrency": layout_concurrency,
                "throughput_batch_size": len(states),
                "throughput_wall_elapsed_s": wall_elapsed,
                "throughput_layout_wall_elapsed_s": layout_wall_elapsed,
                "throughput_extract_wall_elapsed_s": extract_wall_elapsed,
            }
            result_path, markdown_path = write_end2end_result(
                output_dir=state.config.output_dir,
                markdown=markdown,
                layout_output=state.layout_output,
                blocks=state.blocks,
                metrics=metrics,
            )
            results.append(
                End2EndResult(
                    markdown=markdown,
                    layout_output=state.layout_output,
                    blocks=state.blocks,
                    metrics=metrics,
                    result_path=result_path,
                    markdown_path=markdown_path,
                )
            )
        return results
    finally:
        for state in states:
            state.page_image.close()


def run_end2end(
    config: End2EndConfig,
    *,
    client: End2EndClient | None = None,
) -> End2EndResult:
    output_dir = config.output_dir
    artifacts_dir = output_dir / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    image_path = config.image_path.expanduser().resolve()
    page_image = Image.open(image_path).convert("RGB")
    client = client or End2EndOpenAIClient(
        endpoint=config.endpoint,
        model=config.model,
        timeout=config.timeout,
        block_size=config.block_size,
        dynamic_threshold=config.dynamic_threshold,
        max_denoising_steps=config.max_denoising_steps,
        temperature=config.temperature,
    )

    started = time.perf_counter()
    layout_image_path = artifacts_dir / "layout_input.png"
    _write_image(layout_image_path, prepare_layout_image(page_image))
    layout_start = time.perf_counter()
    layout_output = client.infer(
        layout_image_path,
        TASK_PROMPTS["[layout]"],
        config.layout_max_tokens,
    )
    layout_elapsed = time.perf_counter() - layout_start
    blocks = parse_layout_output(layout_output)

    extract_start = time.perf_counter()
    extracted_blocks = extract_blocks(
        client=client,
        page_image=page_image,
        artifacts_dir=artifacts_dir,
        blocks=blocks,
        config=config,
    )
    extract_elapsed = time.perf_counter() - extract_start

    markdown = render_markdown(blocks, config.keep_paratext)
    metrics: dict[str, Any] = {
        "layout_elapsed": layout_elapsed,
        "extract_elapsed": extract_elapsed,
        "total_elapsed": time.perf_counter() - started,
        "num_blocks": len(blocks),
        "num_extracted_blocks": extracted_blocks,
        "markdown_chars": len(markdown),
        "content_concurrency": config.content_concurrency,
    }

    result_path, markdown_path = write_end2end_result(
        output_dir=output_dir,
        markdown=markdown,
        layout_output=layout_output,
        blocks=blocks,
        metrics=metrics,
    )
    return End2EndResult(
        markdown=markdown,
        layout_output=layout_output,
        blocks=blocks,
        metrics=metrics,
        result_path=result_path,
        markdown_path=markdown_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:18084/v1/chat/completions")
    parser.add_argument("--model", default="mineru-diffusion")
    parser.add_argument("--image-path", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results/mineru_diffusion/end2end_openai"),
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--layout-max-tokens", type=int, default=2048)
    parser.add_argument("--content-max-tokens", type=int, default=1024)
    parser.add_argument("--table-max-tokens", type=int, default=2048)
    parser.add_argument("--formula-max-tokens", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--max-denoising-steps", type=int)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--dynamic-threshold", type=float, default=0.95)
    parser.add_argument("--content-concurrency", type=int, default=1)
    parser.add_argument("--keep-paratext", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_end2end(
        End2EndConfig(
            image_path=args.image_path,
            output_dir=args.output_dir,
            endpoint=args.endpoint,
            model=args.model,
            timeout=args.timeout,
            layout_max_tokens=args.layout_max_tokens,
            content_max_tokens=args.content_max_tokens,
            table_max_tokens=args.table_max_tokens,
            formula_max_tokens=args.formula_max_tokens,
            block_size=args.block_size,
            max_denoising_steps=args.max_denoising_steps,
            temperature=args.temperature,
            dynamic_threshold=args.dynamic_threshold,
            content_concurrency=args.content_concurrency,
            keep_paratext=args.keep_paratext,
        )
    )
    print(json.dumps(result.metrics, indent=2, ensure_ascii=False))
    print(f"result: {result.result_path}")
    print(f"markdown: {result.markdown_path}")


if __name__ == "__main__":
    main()
