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
| `PDF` | Semantically reflowed PDF | `bytes` |
| `STRUCTURED_CONTENT` | General structured content | `dict` |
| `CONTENT_LIST` | Content List V1 | `list[dict]` |
| `CONTENT_LIST_V2` | Content List V2 grouped by page | `list[list[dict]]` |

PDF rendering produces a semantic reflow, not a pixel-identical copy of the input or a layout-debug PDF.

```python
from pathlib import Path
from mineru.parser import parse
from mineru.render import render, RenderFormat

result = parse("report.docx", tier="flash")
html = render(result.middle_json, RenderFormat.HTML)
Path("report.html").write_text(html, encoding="utf-8")
```

## Intermediate JSON

`ModelJson` holds analysis `pages` and `page_index_map`; `MiddleJson` holds ordered postprocessed pages and semantic blocks. `schema_id` distinguishes `docvortex.model` from `docvortex.middle`, while `schema_version` identifies the protocol version. Do not identify a document type from its version number alone.

`metadata` carries the file type, producer, and document properties. `extensions["mineru"]` records the actual `tier` and final `parse_mode`. The producing version is `metadata.producer.version`, not a duplicate product extension field. Pages contain `page_idx` and `blocks`; `page_idx` is zero-based, unlike one-based CLI PDF ranges.

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

`structured_content()` returns a consumer-facing representation, not another name for the intermediate protocol. It retains `metadata` and `extensions` and turns natural-language spans into easier-to-consume text. Do not invent `schema_id` or `schema_version` fields for it.

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

`ParseResult.save(writer)` writes `markdown.md`, `middle_json.json`, and `structured_content.json`, plus `model_output.json` when raw model output exists. `mineru-kit parse --format zip` packages these results.

```python
from mineru.parser.writer import FileBasedDataWriter

result.save(FileBasedDataWriter("output"))
```

Assets may be embedded or referenced by image paths, depending on the source and output entrypoint. PDF `ParseResult.to_dict()` omits block `image_base64`; saving intermediate JSON alone does not preserve every external asset. Download and retain matching assets when consuming API output references. Do not rely on a closed PDF object or the original file for later rendering.

The WebUI layout PDF is a debugging artifact used to preview detections when available; otherwise the UI previews the original/cropped PDF. It serves a different purpose from `RenderFormat.PDF` above.
