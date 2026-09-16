# Output Formats and Result Contract

4.0 uses a shared document model. Distinguish what a renderer can generate from what a particular CLI/API exposes; renderer capabilities are not automatically valid command options.

## Entrypoint capabilities

| Entrypoint | Output |
| --- | --- |
| `mineru parse` | Markdown; `--json` is a command response with status, content, locators, and continuation |
| `mineru read` | Markdown or `--format image`; cached content only |
| `mineru-kit parse` | `markdown`, `middle_json`, `zip` |
| Self-hosted V1 API | `markdown`, `middle_json`, `structured_content`, `zip`; consult service capabilities |
| `ParseResult` | `markdown()`, `structured_content()`, `to_dict()`, `to_json()`, `save(writer)` |
| `mineru.render.render()` | The nine targets below |

A `mineru parse --json` response is not MiddleJson and cannot be passed directly to `ParseResult.from_dict()`. Remote output availability is defined by the remote service.

## Nine rendering targets

| `RenderFormat` | Format | Python return type |
| --- | --- | --- |
| `MARKDOWN` | Markdown | `str` |
| `HTML` | HTML | `str` |
| `LATEX` | LaTeX | `str` |
| `DOCX` | Word | `bytes` |
| `EPUB` | EPUB | `bytes` |
| `PDF` | Original block layout or semantic reflow | `bytes` |
| `STRUCTURED_CONTENT` | General structured content | `dict` |
| `CONTENT_LIST` | Content List V1 | `list[dict]` |
| `CONTENT_LIST_V2` | Content List V2 grouped by page | `list[list[dict]]` |

PDF defaults to `PdfLayout.AUTO`: PDF sources with complete geometry retain their
original pages and block positions; older results without geometry fall back to
reflow with diagnostics. `ORIGINAL` requires a PDF source, page sizes and necessary
bboxes. `REFLOW` explicitly selects semantic reflow. OFD and Office still reflow by
default. Text remains selectable; tables and charts may use region images.

```python
from mineru.render import PdfLayout, PdfRenderOptions, RenderFormat, render, render_pdf

pdf_bytes = render_pdf(result.middle_json, layout=PdfLayout.ORIGINAL)
pdf_bytes = render(result.middle_json, RenderFormat.PDF,
                   options=PdfRenderOptions(layout=PdfLayout.REFLOW))
```

Original layout keeps body text, captions, code, tables and images independently
fitted to their source boxes. `continues_prev` does not move text between original
boxes. Original pages and blank pages remain. Both PDF layouts use CJK character
wrapping for paragraphs containing Chinese, Japanese or Korean text, including
mixed-language annotations and table cells. This avoids moving long Chinese
phrases together at spaces. Pure Latin text and literal code/algorithm blocks
retain their existing rules; no layout-only characters or hyphens are inserted.

Titles reference actual fitted body fonts: following text in the same page and
column, then nearest same-column text, then the exported body's median (10.5 pt
if absent). A chapter type/level shares the median reference plus 2 pt, rounded
up to 0.1 pt, with individual increases for larger local body text. Document
titles target 2 pt above the largest chapter target, or body plus 4 pt without
chapters, retaining local hierarchy. Neither 90% coverage nor the previous style
cap forces smaller titles.

Titles that do not fit their original box may borrow whitespace above, below and
to the right in the same column, retaining their left edge and preferably their
top edge. Column width comes from the matched body; uncertain columns keep the
original width, while existing spanning titles retain their span. Other source
boxes remain occupied, with 2 pt clearance and page boundaries enforced.
Adjacent titles divide their gap at its midpoint. Insufficient space shrinks
only that title in 0.1 pt steps, then uses the existing below-6-pt fallback.
Script and formula ink extents are measured; index references and approximate
parent groups are excluded.

`pdf_title_layout_expanded` records expansion. `pdf_layout_font_exception` records
local-body increases or space-driven reductions, with reference/target/final
sizes and original/drawing bounds. Existing overlaps retain original occupancy
and report `pdf_title_geometry_conflict`; unavailable expansion clearance reports
`pdf_title_clearance_unavailable`. Only rendering context changes, with no input,
asset, protocol or public-option changes.

Requires `docvortex>=0.4.7,<1` (the current minimum dependency declared by MinerU 4.0). DocVortex clears native TXT PDF display equation
`content` once when producing model output; MinerU Flash TXT uses that output.
Flash OCR already leaves display equation content empty. MinerU does not clear
equation content again. Both Flash paths retain the bbox, orientation, image and
detected number region.
PDF, Markdown, HTML, DOCX, EPUB and LaTeX use the existing image fallback. Inline
and non-Flash equations retain their content. Old caches are not rewritten;
reparse to obtain geometry and image-only Flash display equations.

```python
from pathlib import Path
from mineru.parser import parse
from mineru.render import render, RenderFormat

result = parse("report.docx", tier="flash")
html = render(result.middle_json, RenderFormat.HTML)
Path("report.html").write_text(html, encoding="utf-8")
```

## Intermediate JSON

`ModelJson` holds analysis `pages` and `page_index_map`; `MiddleJson` holds ordered postprocessed pages and semantic blocks. JSON consumers identify the payload through the serialized `schema` and `schema_version` keys (`docvortex.model` or `docvortex.middle`, protocol `2.0`). The schema identity is named `schema_id` as a Python attribute, but it serializes as `schema`; when reading or writing JSON, always use the serialized keys. Do not identify a document type from its version number alone.

`metadata` carries the file type, producer, and document properties. `extensions["mineru"]` records the actual `tier` and final `parse_mode`. The producing version is `metadata.producer.version`, not a duplicate product extension field. Pages contain `page_idx` and `blocks`; `page_idx` is zero-based, unlike one-based CLI PDF ranges.

All PDF tiers also emit `extensions.docvortex_layout` through sync and async
analysis: `version=1` and `pages` with source `page_idx`, `width_pt`, `height_pt`
and optional `image_rotations`. Sizes share the bbox orientation. Selected and
blank pages, multiple windows and cached batch compaction retain geometry by
source page. Both ModelJson and MiddleJson serialization preserve the extension;
the main protocol remains 2.0. PDF export requires only MiddleJson and image assets.

The following example is serialized from the current public types. `4.0.0` is an illustrative stable producer version. Optional fields may be omitted; use the actual output contract:

```json
{
  "metadata": {
    "file_suffix": "html",
    "producer": {
      "name": "mineru",
      "version": "4.0.0"
    }
  },
  "extensions": {
    "mineru": {
      "tier": "flash",
      "parse_mode": "txt"
    }
  },
  "pages": [
    {
      "page_idx": 0,
      "blocks": [
        {
          "type": "text",
          "index": 0,
          "content": [
            {
              "type": "text",
              "content": "Hello MinerU"
            }
          ]
        }
      ]
    }
  ],
  "is_full_document": true,
  "schema": "docvortex.middle",
  "schema_version": "2.0"
}
```

Round-trip current results with `ParseResult.from_json(result.to_json())`. Legacy `_backend`, `pdf_info`, and `_version_name` are not part of this contract. See [migration](migration_4.md) for historical data; changing a version number alone does not migrate a payload.

## Structured Content

`structured_content()` returns a consumer-facing representation, not another name for the intermediate protocol. It retains `metadata` and `extensions` and turns natural-language spans into easier-to-consume text. It carries no schema identity: do not add `schema` or `schema_version` keys to it.

```json
{
  "pages": [
    {
      "page_idx": 0,
      "blocks": [
        {
          "type": "text",
          "content": "Hello MinerU"
        }
      ]
    }
  ],
  "metadata": {
    "file_suffix": "html",
    "producer": {
      "name": "mineru",
      "version": "4.0.0"
    }
  },
  "extensions": {
    "mineru": {
      "tier": "flash",
      "parse_mode": "txt"
    }
  },
  "is_full_document": true
}
```

## Saved files, ZIP, and assets

`ParseResult.save(writer)` materializes images on a document copy and writes `markdown.md`, `middle_json.json`, `structured_content.json`, and `images/`, plus `model_output.json` when raw model output exists. Self-hosted V1 API ZIP output and `mineru-kit parse --format zip` share this save path. All three consumer formats reference the same assets; image bytes, source page indices, block indices, and rotation metadata are preserved.

```python
from mineru.parser.writer import FileBasedDataWriter

result.save(FileBasedDataWriter("output"))
```

PDF `ParseResult.to_dict()` / `to_json()` still omit block `image_base64`, so standalone structural JSON is not a complete result package. `save(writer)` neither recrops nor rotates images and rejects unresolved asset references before writing, without reading the working directory or fetching network resources. With `include_images=True`, the API client restores direct images and images embedded in visual HTML from the ZIP. Gradio reuses those assets; PDF export needs only MiddleJson and the images, without the source PDF or ModelJson. `include_images=False` retains structural-only reading. Regenerate historical incomplete or incorrectly materialized results.

The WebUI layout PDF is a debugging artifact used to preview detections when available; otherwise the UI previews the original/cropped PDF. It serves a different purpose from `RenderFormat.PDF` above.
