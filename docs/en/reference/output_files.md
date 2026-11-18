# Output Formats and Result Contract

Find your interface in the table before choosing an output. Command responses, saved data for rendering, and files for reading serve different purposes.

## Entrypoint capabilities

| Entrypoint | Output |
| --- | --- |
| `mineru parse` | Markdown; `--json` is a command response with status, content, locators, and continuation |
| `mineru read` | Markdown or `--format image`; cached content only |
| `mineru-kit parse` | `markdown`, `middle_json`, `zip` |
| V1 Tasks / file-parse | `default`, `docx`, `html`, `latex`, including combinations; `default` is a ZIP of Markdown, Middle JSON, Structured Content, and images |
| [WebUI downloads](../usage/webui.md#downloads) | Markdown ZIP, JSON ZIP (Structured Content), HTML, DOCX, LaTeX ZIP, EPUB, and a PDF rendered from the result. |
| `ParseResult` | `markdown()`, `structured_content()`, `to_dict()`, `to_json()`, `save(writer)` |
| `mineru.render.render()` | The nine targets below |

A `mineru parse --json` response is not MiddleJson and cannot be passed directly to `ParseResult.from_dict()`. Remote output availability is defined by the remote service.

## ZIP image names

`mineru-kit parse --format zip`, V1 API ZIP, and Gradio Markdown/JSON/LaTeX downloads
share DocVortex image materialization and naming. Each entrypoint
keeps its existing archive name and document members.

Direct image/table/chart payloads use `images/page_{page_idx}_{parent_type}_{parent_index}.{ext}`;
standalone equations use `equation`. Images inside visual HTML use `image_{index}_{ordinal}`,
`table_image_{index}_{ordinal}`, or `chart_image_{index}_{ordinal}` after the page prefix.
Page indices refer to the original document and start at zero. Inline ordinals start at one
within each body, counting every `<img src=...>`, including external images whose URLs remain unchanged.

Identical bytes at the same path are reused; different bytes receive `_duplicate_n` suffixes.
Different semantic names remain separate even when their bytes match. Extensions are lowercase;
image bytes are not re-encoded. All document references are updated together with image paths.

Historical ZIPs retain their stored references, with no automatic migration or old-name aliases.
Consumers should follow actual document references instead of constructing `*_body_*` filenames.
The WebUI reads assets from the saved result bundle.

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

Original-layout export retains the source page count and blank pages. The [developer rendering reference](../../next/middle-json/rendering.md#mineru-pdf) documents fitting rules and diagnostics in Chinese.

Flash PDF parsing uses the retained formula image for display equations. Existing caches keep their previous results. Reparse to obtain updated geometry and images. The [MinerU package metadata](https://github.com/opendatalab/MinerU/blob/master/pyproject.toml) for each release specifies its DocVortex dependency range.

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

`ParseResult.save(writer)` materializes images on a document copy and writes `markdown.md`, `middle_json.json`, `structured_content.json`, and `images/`. `mineru-kit parse --format zip` uses this save path; Tasks uses a separate packaging pipeline for [default and other formats](../usage/tasks.md#outputs-and-downloads). ModelJson is internal and is not included in SDK, CLI, Tasks, or Gradio results. All three consumer formats reference the same assets; image bytes, source page indices, block indices, and rotation metadata are preserved.

```python
from mineru.parser.writer import FileBasedDataWriter

result.save(FileBasedDataWriter("output"))
```

PDF `ParseResult.to_dict()` / `to_json()` still omit block `image_base64`, so standalone structural JSON is not a complete result package. `save(writer)` neither recrops nor rotates images and rejects unresolved asset references before writing, without reading the working directory or fetching network resources. With `include_images=True`, the API client restores direct images and images embedded in visual HTML from the ZIP. Gradio reuses those assets; PDF export needs only MiddleJson and the images, without the source PDF or ModelJson. `include_images=False` retains structural-only reading. Regenerate historical incomplete or incorrectly materialized results.

When you check layout in the WebUI, region boxes appear over the source preview. The PDF from the download menu is rendered from the saved result. See [source tracing in the WebUI](../usage/webui.md#source-tracing).
