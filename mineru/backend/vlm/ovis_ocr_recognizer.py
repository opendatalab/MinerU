# Copyright (c) Opendatalab. All rights reserved.
"""OvisOCR2 内容识别适配器：把页级解析模型接入两阶段解耦的第二阶段。

OvisOCR2（https://huggingface.co/ATH-MaaS/OvisOCR2）是页级端到端文档解析模型，
只有一个固定prompt、不支持区域级指令。本适配器把它降维成块级识别器：
对每个layout块裁图→按angle转正→贴白底补足最小分辨率→调用其OpenAI兼容接口→
按块类型对返回的Markdown做后处理（公式剥$$留LaTeX、表格抽<table>、文本拍平）。

== 运行环境配置（重要：库兼容性） ==

OvisOCR2 要求 vllm==0.22.1，与 MinerU 锁定的 vllm>=0.10.1.1,<0.22.0 **冲突**，
两者不能装进同一个Python环境。因此本适配器为纯HTTP客户端，只依赖 MinerU
基础依赖中已有的 openai 包，MinerU 环境无需新增/升级任何库。

OvisOCR2 服务端请用独立环境或容器启动（三选一）：

  # 方式一：独立venv + vLLM（推荐）
  python -m venv ovis-env && source ovis-env/bin/activate
  pip install "vllm==0.22.1" pillow
  vllm serve "ATH-MaaS/OvisOCR2" --port 8000

  # 方式二：Docker（免环境管理）
  docker model run hf.co/ATH-MaaS/OvisOCR2

  # 方式三：SGLang（同样提供OpenAI兼容接口）

之后在 MinerU 环境中：

  from mineru.backend.vlm.ovis_ocr_recognizer import OvisOcrContentRecognizer
  recognizer = OvisOcrContentRecognizer(server_url="http://<host>:8000")
  middle_json, _ = doc_analyze(pdf_bytes, image_writer,
                               layout_detector=PipelineLayoutDetector(),
                               content_recognizer=recognizer)

注意：不要尝试在 MinerU 环境内用 transformers 直接加载 OvisOCR2 ——
其 Qwen3.5 架构对 transformers 版本另有要求，容易与 MinerU 的锁定版本冲突，
分离部署可彻底规避。
"""
import base64
import io
import re
from concurrent.futures import ThreadPoolExecutor

from loguru import logger
from PIL import Image

from .stages import ContentRecognizer

OVIS_PROMPT = (
    "Extract all readable content from the image in natural human reading order "
    "and output the result as a single Markdown document."
)

# 按纯文本处理的块类型（识别结果拍平为plain text）
TEXT_TYPES = {
    "text", "title", "header", "footer", "page_number", "page_footnote",
    "aside_text", "ref_text", "index", "phonetic", "list", "list_item",
    "image_caption", "image_footnote", "table_caption", "table_footnote",
    "code_caption", "formula_number",
}
EQUATION_TYPES = {"equation"}
TABLE_TYPES = {"table"}
CODE_TYPES = {"code", "algorithm"}
# 不需要文本识别的类型（视觉块本体等）
SKIP_TYPES = {"image", "chart", "image_block", "equation_block", "unknown"}


def _strip_math_wrappers(text: str) -> str:
    """剥掉模型返回的公式外层定界符，只留LaTeX本体。"""
    text = text.strip()
    for pattern in (r"^\$\$(.*)\$\$$", r"^\\\[(.*)\\\]$", r"^\\\((.*)\\\)$", r"^\$(.*)\$$"):
        match = re.match(pattern, text, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
    return text


def _markdown_table_to_html(text: str) -> str | None:
    """极简markdown表格转HTML，作为模型未按HTML输出时的兜底。"""
    rows = [line.strip() for line in text.splitlines() if line.strip().startswith("|")]
    rows = [row for row in rows if not re.fullmatch(r"\|[\s:\-|]+\|?", row)]
    if len(rows) < 1:
        return None
    html_rows = []
    for row in rows:
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        html_rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    return "<table>" + "".join(html_rows) + "</table>"


def _extract_table_html(text: str) -> str:
    """从模型输出中抽取<table>；无HTML时尝试markdown表格兜底。"""
    match = re.search(r"<table\b.*?</table>", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(0)
    markdown_table = _markdown_table_to_html(text)
    if markdown_table:
        return markdown_table
    return _flatten_markdown(text)


def _strip_code_fences(text: str) -> str:
    match = re.search(r"```[^\n]*\n(.*?)```", text, flags=re.DOTALL)
    if match:
        return match.group(1).rstrip()
    return text.strip()


def _flatten_markdown(text: str) -> str:
    """把Markdown拍平成接近MinerU文本块content的纯文本（保留\\(..\\)行内公式）。"""
    text = re.sub(r"<img\b[^>]*>", "", text)  # OvisOCR2的视觉区域占位
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", text)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    merged = ""
    for line in lines:
        if not merged:
            merged = line
            continue
        # CJK字符间直接连接，其余以空格连接，避免中文被塞进多余空格
        if _is_cjk(merged[-1]) and _is_cjk(line[0]):
            merged += line
        else:
            merged += " " + line
    return merged.strip()


def _is_cjk(char: str) -> bool:
    code = ord(char)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x3000 <= code <= 0x303F
        or 0xFF00 <= code <= 0xFFEF
    )


def postprocess_block_content(block_type: str, raw_output: str) -> str:
    if block_type in EQUATION_TYPES:
        return _strip_math_wrappers(_strip_code_fences(raw_output))
    if block_type in TABLE_TYPES:
        return _extract_table_html(raw_output)
    if block_type in CODE_TYPES:
        return _strip_code_fences(raw_output)
    return _flatten_markdown(raw_output)


class OvisOcrContentRecognizer(ContentRecognizer):
    """第二阶段识别器：经OpenAI兼容接口调用OvisOCR2逐块识别。"""

    name = "ovis-ocr2"

    def __init__(
        self,
        server_url: str,
        model_name: str = "ATH-MaaS/OvisOCR2",
        api_key: str = "EMPTY",
        max_concurrency: int = 8,
        http_timeout: float = 600.0,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        min_edge: int = 448,
        skip_types: set[str] | None = None,
    ):
        base_url = server_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        self.base_url = base_url
        self.model_name = model_name
        self.api_key = api_key
        self.max_concurrency = max_concurrency
        self.http_timeout = http_timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.min_edge = min_edge
        self.skip_types = SKIP_TYPES if skip_types is None else skip_types
        self._client = None  # 惰性初始化，构造时不要求服务可达

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI  # MinerU基础依赖，无需新装

            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.http_timeout,
            )
        return self._client

    # -- 图像准备 ------------------------------------------------------------

    def _prepare_block_image(self, page_image: Image.Image, block) -> Image.Image | None:
        width, height = page_image.size
        x0, y0, x1, y1 = block["bbox"]
        crop = page_image.crop((x0 * width, y0 * height, x1 * width, y1 * height))
        if crop.width < 1 or crop.height < 1:
            logger.warning(f"Skip invalid block crop size {crop.size} for {block.get('type')}")
            return None
        # 与 mineru_vl_utils 相同的转正约定：逆时针旋转 angle 度
        angle = block.get("angle")
        if angle in (90, 180, 270):
            crop = crop.rotate(angle, expand=True)
        return self._pad_to_min_edge(crop)

    def _pad_to_min_edge(self, image: Image.Image) -> Image.Image:
        """小裁块贴白底补足最小边长，避免服务端强行放大导致的劣化。"""
        target_w = max(image.width, self.min_edge)
        target_h = max(image.height, self.min_edge)
        if target_w == image.width and target_h == image.height:
            return image
        canvas = Image.new("RGB", (target_w, target_h), (255, 255, 255))
        canvas.paste(
            image.convert("RGB"),
            ((target_w - image.width) // 2, (target_h - image.height) // 2),
        )
        return canvas

    # -- 模型调用 ------------------------------------------------------------

    def _complete(self, block_image: Image.Image) -> str:
        buf = io.BytesIO()
        block_image.convert("RGB").save(buf, format="PNG")
        data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        response = self._get_client().chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": OVIS_PROMPT},
                    ],
                }
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""

    def _recognize_one(self, page_image: Image.Image, block) -> None:
        try:
            block_image = self._prepare_block_image(page_image, block)
            if block_image is None:
                return
            raw_output = self._complete(block_image)
            block["content"] = postprocess_block_content(block["type"], raw_output)
        except Exception as exc:
            logger.warning(
                f"OvisOCR2 recognize failed for {block.get('type')} block: {exc}"
            )

    # -- ContentRecognizer 接口 ------------------------------------------------

    def batch_recognize(self, images, blocks_list, image_analysis: bool = True):
        del image_analysis  # OvisOCR2不做图片语义描述，忽略该开关
        jobs = []
        for page_image, page_blocks in zip(images, blocks_list):
            for block in page_blocks:
                if block.get("type") in self.skip_types:
                    continue
                if block.get("content"):
                    continue  # 已带内容的块（如预置layout）不重复识别
                jobs.append((page_image, block))

        if jobs:
            with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
                list(executor.map(lambda job: self._recognize_one(*job), jobs))
        return [list(page_blocks) for page_blocks in blocks_list]
