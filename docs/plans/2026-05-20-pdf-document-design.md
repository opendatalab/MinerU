# PDFDocument 类设计

日期: 2026-05-20

## 动机

当前项目中对 PDF 文件的操作分散在 8 个工具模块中，通过裸 `bytes` 传递，每个调用方各自打开/关闭 pypdfium2 文档。这导致：

1. 同一份 PDF bytes 被多次打开/关闭（分析、分类、图像提取各自开一次）
2. 生命周期管理不一致（有的用 context manager，有的手动，有的靠 GC）
3. 没有面向对象接口，所有操作都是独立函数

## 目标

定义 `PDFDocument` 类，以面向对象接口收拢所有 PDF 文件操作，统一生命周期管理。

## 依赖的 PDF 库

| 库 | 用途 |
|---|---|
| pypdfium2 | 主力：打开文档、页面渲染、文本提取、页面裁切 |
| pdftext | 字符后处理：去重、字符→行→span 聚类 |
| pypdf | 可视化：PdfReader/PdfWriter 读/写 |
| reportlab | 可视化：在 PDF 上画 bbox 框和标注 |
| pdfminer.six | 分类 fallback：文本提取、图片覆盖率、CID 字体 |

## 接口定义

```python
class PDFDocument:
    """A PDF file loaded in memory, with lazy pypdfium2 access."""

    # -- 工厂 --
    @staticmethod
    def from_image(image_bytes: bytes) -> "PDFDocument":
        """将单张图片包装为单页 PDF。"""

    # -- 生命周期 --
    def __init__(self, pdf_bytes: bytes) -> None:
        """持有 PDF bytes，不立即打开 pypdfium2 文档。"""

    def close(self) -> None:
        """关闭 pypdfium2 文档，幂等。"""

    def __enter__(self) -> "PDFDocument": ...
    def __exit__(self, *args) -> None: ...
    def __del__(self) -> None: ...

    # -- 属性 --
    @property
    def page_count(self) -> int:
        """首次访问时惰性打开 pypdfium2 文档。"""

    @property
    def bytes(self) -> bytes:
        """原始 PDF bytes，零拷贝。"""

    # -- 元数据 --
    def page_size(self, page_idx: int) -> tuple[float, float]:
        """返回 (width, height)，单位 pt。"""

    # -- 渲染 --
    def render_page(self, page_idx: int, *, scale: int = 2) -> Image.Image: ...
    def render_pages(self, start: int = 0, end: int | None = None, *, scale: int = 2) -> list[Image.Image]: ...
    async def render_page_async(self, page_idx: int, *, scale: int = 2) -> Image.Image: ...
    async def render_pages_async(self, start: int = 0, end: int | None = None, *, scale: int = 2) -> list[Image.Image]: ...
    def crop_image(self, bbox: tuple[float, float, float, float], page_idx: int, *, scale: int = 2) -> bytes: ...

    # -- 文本 --
    def get_page_chars(self, page_idx: int) -> list[dict]:
        """提取单页原始字符列表（pypdfium2 输出）。"""

    def get_page_lines(self, page_idx: int) -> list[dict]:
        """字符经 pdftext 聚类为行，含 bbox + spans 树。"""

    # -- 分类 --
    def classify(self) -> str:
        """返回 'text' | 'scanned' | 'hybrid'。count_images / detect_cid_font 为内部信号不暴露。"""

    def get_text_quality(self) -> float:
        """文本质量信号 0~1，值越低越需 OCR。"""

    # -- 页面裁切 --
    def extract_page_range(self, start: int, end: int) -> "PDFDocument":
        """裁切页面范围，返回新 PDFDocument。"""

    def sample_pages(self, max_pages: int = 3) -> "PDFDocument":
        """抽取代表性样本页，返回新 PDFDocument。"""

    # -- 可视化 --
    def draw_layout_bbox(self, pages: list[PageInfo], output_path: str) -> None:
        """绘制版面检测框（pypdf + reportlab），输出标注版 PDF。"""

    def draw_span_bbox(self, pages: list[PageInfo], output_path: str) -> None:
        """绘制 span 级细粒度框，输出标注版 PDF。"""
```

## 关键设计决策

### 惰性打开 + 手动 close

构造时不打开 pypdfium2 文档。`page_count` 属性首次访问时惰性打开。用户可手动 `close()` 或依赖 context manager / `__del__` 自动关闭。

### 零拷贝持有 bytes

`bytes` 属性直接返回构造函数传入的 `bytes` 对象的引用，不复制。

### 内部信号不暴露

`count_images()` 和 `detect_cid_font()` 仅作为 `classify()` 的内部实现步骤，不暴露为公开方法。

### 与 ParseResult 的关系

`ParseResult._pdf_bytes: bytes | None` 改为 `ParseResult._pdf_doc: PDFDocument | None`。`images()` 方法内部从 `pdfium.PdfDocument(self._pdf_bytes)` 裸打开改为 `self._pdf_doc.render_page()` + `self._pdf_doc.crop_image()`。

### 放置位置

`mineru/utils/pdf_document.py`

## 影响范围

- `mineru/api/parse_result.py`: `_pdf_bytes` → `_pdf_doc`，`_extract_pdf_images()` 改用 PDFDocument 方法
- `mineru/api/pdf_parser.py`: `_prepare_input` 中图片转 PDF 改为 `PDFDocument.from_image()`，页面裁切改为 `doc.extract_page_range()`
- `mineru/backend/pipeline/pipeline_analyze.py`: `open_pdfium_document(pdf_bytes)` → 接收 PDFDocument
- `mineru/backend/vlm/vlm_analyze.py`: 同上
- `mineru/backend/hybrid/hybrid_analyze.py`: 同上
- `mineru/utils/pdf_classify.py`: `classify(pdf_bytes)` → `pdf_doc.classify()`
- `mineru/utils/draw_bbox.py`: `draw_layout_bbox(pdf_info, pdf_bytes, ...)` → `pdf_doc.draw_layout_bbox(pages, ...)`
- `mineru/utils/pdf_text_tool.py`: `get_page_chars(page, ...)` → `pdf_doc.get_page_chars(page_idx)`
- `mineru/utils/pdf_image_tools.py`: 渲染相关函数替换为 PDFDocument 方法
