# MinerU Output Files Documentation

## Overview

After executing the `mineru` command, in addition to the main markdown file output, multiple auxiliary files are generated for debugging, quality inspection, and further processing. These files include:

The exact set of generated files depends on the backend and the input document type.

- **Visual debugging files**: Help users intuitively understand the document parsing process and results
- **Structured data files**: Contain detailed parsing data for secondary development
- In multimodal markdown output, `image` / `chart` blocks render the screenshot first; when `content` exists, a collapsed HTML `<details>` block is appended after the image, using the block `sub_type` as the summary label when available and falling back to `image content` or `chart content`

The following sections provide detailed descriptions of each file's purpose and format.

## Visual Debugging Files

### Layout Analysis File (layout.pdf)

**File naming format**: `{original_filename}_layout.pdf`

**Functionality**:

- Visualizes layout analysis results for each page
- Numbers in the top-right corner of each detection box indicate reading order
- Different background colors distinguish different types of content blocks

**Use cases**:

- Check if layout analysis is correct
- Verify if reading order is reasonable
- Debug layout-related issues

![layout page example](../images/layout_example.png)

### Text Spans File (span.pdf)

> [!NOTE]
> Only generated when span visualization is explicitly enabled.

**File naming format**: `{original_filename}_span.pdf`

**Functionality**:

- Uses different colored line boxes to annotate page content based on span type
- Used for quality inspection and issue troubleshooting

**Use cases**:

- Quickly troubleshoot text loss issues
- Check inline formula recognition
- Verify text segmentation accuracy

![span page example](../images/spans_example.png)

## Structured Data Files

> [!IMPORTANT]
> The current structured output contract uses the unified `MinerUParser` with `tier` (flash/basic/standard/advanced) and `parse_mode` (txt/ocr) parameters. Legacy middle-json files marked with `_backend: "pipeline"`/`_backend: "hybrid"`/`_backend: "office"` are no longer accepted by current readers; use schema 2.0 `MiddleJson` instead.

### Unified Model Output Results

#### Model Inference Results (model.json)

**File naming format**: `{original_filename}_model.json`

##### Sample Data

```json
[
    {
        "cls_id": 12,
        "label": "header",
        "score": 0.93,
        "bbox": [
            1217,
            104,
            1516,
            134
        ],
        "index": 2
    },
    {
        "cls_id": 6,
        "label": "doc_title",
        "score": 0.9751,
        "bbox": [
            275,
            181,
            1512,
            292
        ],
        "index": 3
    },
    {
        "cls_id": 22,
        "label": "text",
        "score": 0.9217,
        "bbox": [
            275,
            330,
            524,
            370
        ],
        "index": 4
    }
]
```

#### Intermediate Processing Results (middle.json)

**File naming format**: `{original_filename}_middle.json`

##### Top-level Structure (schema 2.0)

| Field Name | Type | Description |
|------------|------|-------------|
| `pages` | `list[PageInfo]` | Array of parsing results for each page, strictly increasing by `page_idx` |
| `is_full_document` | `bool` | Whether the input is a full document (empty `page_index_map`) |
| `file_suffix` | `string` | Input file type: `pdf`, `doc`, `docx`, `ppt`, `pptx`, `xls`, `xlsx`, or `csv` |
| `effort` | `string` | Analysis effort: `flash`, `medium`, `high`, or `xhigh` |
| `parse_mode` | `string` | Parse mode: `txt` or `ocr` |
| `mineru_version` | `string` | MinerU version number |

Legacy fields `pdf_info`/`_backend`/`_version_name`/`_ocr_enable`/`_vlm_ocr_enable` are removed in schema 2.0.

##### Page Information Structure (pages)

| Field Name | Description |
|------------|-------------|
| `page_idx` | Page number, starting from 0 |
| `blocks` | List of top-level page blocks (the only content field) |

Legacy fields `preproc_blocks`/`para_blocks`/`page_size`/`images`/`tables`/`interline_equations`/`discarded_blocks`/`_layout_tree`/`layout_bboxes` are removed in schema 2.0 — all content is expressed in the `blocks` tree.

##### Block Structure Hierarchy

```
Top-level page blocks (text | title | equation | image | table | chart | code | list | index | ...)
└── Visual parent blocks (image | table | chart | code) contain child blocks
    └── body block + optional caption/footnote blocks
```

Leaf blocks (text, title, equation, etc.) carry `content: str` directly. Visual parent blocks (image, table, chart, code) carry `content: list[child blocks]`, where children include exactly one body plus optional caption/footnote. There are no independent `Line`/`Span` types in schema 2.0.

##### Common Block Fields

| Field Name | Description |
|------------|-------------|
| `type` | Block type (see table below) |
| `bbox` | Rectangular box coordinates of the block `[x0, y0, x1, y1]`, normalized to `[0, 1]` |
| `index` | Block index for reading order (required for top-level blocks) |
| `content` | For leaf blocks: a string; for visual parent blocks: a list of child blocks |

##### Block Types

| Type | Description |
|------|-------------|
| `text` | Text block (leaf, `content: str`) |
| `doc_title` | Document title (level=1) |
| `paragraph_title` | Paragraph title (level 2-6) |
| `equation` | Display (interline) formula block (renamed from `interline_equation`) |
| `image` | Image container; `content` includes `image_body` + optional `image_caption`/`image_footnote` |
| `table` | Table container; `content` includes `table_body` + optional `table_caption`/`table_footnote` |
| `chart` | Chart container; `content` includes `chart_body` + optional `chart_caption`/`chart_footnote` |
| `code` | Code container; `content` includes `code_body` + optional `code_caption`/`code_footnote`; `sub_type` is `code` or `algorithm` |
| `list` | List container; `content` is a list of `text`/`ref_text`/nested `list` blocks; `sub_type` is `text` or `ref_text` |
| `index` | Index (table of contents) container; `content` is a list of `text`/`doc_title`/`paragraph_title`/nested `index` blocks |
| `ref_text` | Reference / citation text block |
| `header` / `footer` / `page_number` / `aside_text` / `page_footnote` | Page auxiliary blocks (leaf, `content: str`) |

##### Sample Data (schema 2.0)

```json
{
    "pages": [
        {
            "page_idx": 0,
            "blocks": [
                {
                    "type": "doc_title",
                    "index": 0,
                    "bbox": [0.45, 0.23, 0.55, 0.28],
                    "content": "1 Introduction",
                    "level": 1
                },
                {
                    "type": "text",
                    "index": 1,
                    "bbox": [0.08, 0.30, 0.46, 0.40],
                    "content": "dependent on the service headway and the reliability of the departure"
                },
                {
                    "type": "image",
                    "index": 2,
                    "bbox": [0.52, 0.30, 0.95, 0.55],
                    "content": [
                        {
                            "type": "image_body",
                            "index": 2,
                            "bbox": [0.52, 0.30, 0.95, 0.55],
                            "content": "",
                            "image_path": "images/page_0_image_body_2.png"
                        },
                        {
                            "type": "image_caption",
                            "index": 3,
                            "bbox": [0.52, 0.56, 0.95, 0.58],
                            "content": "Figure 1: Example figure"
                        }
                    ],
                    "sub_type": null
                }
            ]
        }
    ],
    "file_suffix": "pdf",
    "effort": "high",
    "parse_mode": "ocr",
    "mineru_version": "1.x.x"
}
```

#### Content List (content_list.json)

> [!NOTE]
> `content_list.json` is deprecated and no longer generated. Use `structured_content.json` for structured content output. See the [Common Structured Content](#common-structured-content-structured_contentjsondevelopment-version-subject-to-change) section below.

### Common Structured Content (structured_content.json)(development version, subject to change)

**File naming format**: `{original_filename}_structured_content.json`

##### Functionality

`structured_content.json` is the new structured output added in 3.0:

- The top level is grouped by page for page-oriented consumption
- Each item uses a unified `type + content` structure for easier programmatic processing
- The exact supported `type` values depend on the backend and input type

##### Common Fields

| Field | Type | Description |
|------|------|-------------|
| `type` | `string` | Content type |
| `content` | `dict` | Structured payload for the given `type` |
| `bbox` | `list[int]` | Optional bounding box mapped into the 0-1000 coordinate range |
| `anchor` | `string` | Optional anchor; some `DOCX` titles or index items may include it |

`image` / `chart` items may also include an optional top-level `sub_type` field for visual subtype propagation.

##### Common Types

| Type | Description |
|------|-------------|
| `title` | Title block with `title_content` and `level` |
| `paragraph` | Paragraph block with `paragraph_content` |
| `equation_interline` | Interline formula with `math_content` and `math_type` |
| `image` / `table` / `chart` | Visual blocks with image paths, captions, and related structured fields. Seal content uses `image` with `sub_type: "seal"` |
| `code` | Code block with `code_content`, `code_caption`, `code_footnote`, and `code_language` |
| `algorithm` | Algorithm block with `algorithm_content`, `algorithm_caption`, and `algorithm_footnote` |
| `list` / `index` | List and index blocks with `list_items` |
| `page_header` / `page_footer` / `page_number` / `page_aside_text` / `page_footnote` | Page auxiliary blocks |

Inline fields such as `title_content`, `paragraph_content`, and captions are
usually span lists. A `hyperlink` span contains `content` and `url`; when one
link contains text fragments with different styles, it may also contain
`children`. In that case, `content` is the concatenated child text, and the
exact styles are represented by the child `text` spans.

##### Sample Data

```json
[
  [
    {
      "type": "title",
      "content": {
        "title_content": [
          {
            "type": "text",
            "content": "1 Introduction"
          }
        ],
        "level": 1
      },
      "bbox": [83, 121, 917, 156]
    },
    {
      "type": "page_footnote",
      "content": {
        "page_footnote_content": [
          {
            "type": "text",
            "content": "* Corresponding author"
          }
        ]
      },
      "bbox": [71, 815, 915, 841]
    }
  ]
]
```

### Multimodal Model Output Results

#### Model Inference Results (model.json)

**File naming format**: `{original_filename}_model.json`

##### File format description
- Two-level nested list: outer list = pages; inner list = content blocks of that page
- Each block is a dict with at least: `type`, `bbox`, `angle`, `content` (some types add extra fields like `score`, `block_tags`, `content_tags`, `format`)
- Designed for direct, raw model inspection

##### Supported content types (type field values)
```json
{
  "text": "Plain text",
  "title": "Title",
  "equation": "Display (interline) formula",
  "image": "Image",
  "image_caption": "Image caption",
  "image_footnote": "Image footnote",
  "table": "Table",
  "table_caption": "Table caption",
  "table_footnote": "Table footnote",
  "phonetic": "Phonetic annotation",
  "code": "Code block",
  "code_caption": "Code caption",
  "ref_text": "Reference / citation entry",
  "algorithm": "Algorithm block (treated as code subtype)",
  "list": "List container",
  "header": "Page header",
  "footer": "Page footer",
  "page_number": "Page number",
  "aside_text": "Side / margin note",
  "page_footnote": "Page footnote"
}
```

##### Coordinate system
- `bbox` = `[x0, y0, x1, y1]` (top-left, bottom-right)
- Origin at top-left of the page
- All coordinates are normalized percentages in `[0,1]`

##### Sample data
```json
[
  [
    {
      "type": "header",
      "bbox": [0.077, 0.095, 0.18, 0.181],
      "angle": 0,
      "score": null,
      "block_tags": null,
      "content": "ELSEVIER",
      "format": null,
      "content_tags": null
    },
    {
      "type": "title",
      "bbox": [0.157, 0.228, 0.833, 0.253],
      "angle": 0,
      "score": null,
      "block_tags": null,
      "content": "The response of flow duration curves to afforestation",
      "format": null,
      "content_tags": null
    }
  ]
]
```

#### Intermediate Processing Results (middle.json)

**File naming format**: `{original_filename}_middle.json`

In schema 2.0, the multimodal tier produces the same unified `MiddleJson` structure as other tiers. The block types below are part of the standard `blocks` tree (not extensions):

- `list` is a container block; `content` holds child `text`/`ref_text`/nested `list` blocks; `sub_type` distinguishes list categories:
    * `text`: ordinary list
    * `ref_text`: reference / bibliography style list
- `code` is a container block; `content` holds a `code_body` plus optional `code_caption`/`code_footnote`; `sub_type` is:
    * `code`
    * `algorithm`
- Page auxiliary blocks (`header`, `footer`, `page_number`, `aside_text`, `page_footnote`) appear as top-level leaf blocks in `blocks` with `content: str` — there is no separate `discarded_blocks` field in schema 2.0.
- All blocks may include an `angle` field indicating rotation (one of `0, 90, 180, 270`).

##### Examples
- Example: list block
    ```json
    {
      "type": "list",
      "bbox": [0.068, 0.121, 0.319, 0.260],
      "index": 11,
      "content": [
        {
          "type": "text",
          "bbox": [0.068, 0.123, 0.122, 0.137],
          "index": 3,
          "content": "H.1 Introduction"
        },
        {
          "type": "text",
          "bbox": [0.068, 0.142, 0.181, 0.179],
          "index": 4,
          "content": "H.2 Example: Divide by Zero without Exception Handling"
        }
      ],
      "sub_type": "text"
    }
    ```

- Example: code block with optional caption:
    ```json
    {
      "type": "code",
      "bbox": [0.045, 0.610, 0.346, 0.964],
      "index": 17,
      "content": [
        {
          "type": "code_body",
          "bbox": [0.045, 0.610, 0.346, 0.964],
          "index": 17,
          "content": "1 // Fig. H.1: DivideByZeroNoExceptionHandling.java  \n2 // Integer division without exception handling.  \n3 import java.util.Scanner;  \n4  \n5 public class DivideByZeroNoExceptionHandling  \n6 {  \n7 // demonstrates throwing an exception when a divide-by-zero occurs  \n8 public static int quotient( int numerator, int denominator )  \n9 {  \n10 return numerator / denominator; // possible division by zero  \n11 } // end method quotient  \n12  \n13 public static void main(String[] args)  \n14 {  \n15 Scanner scanner = new Scanner(System.in); // scanner for input  \n16  \n17 System.out.print(\"Please enter an integer numerator: \");  \n18 int numerator = scanner.nextInt();  \n19 System.out.print(\"Please enter an integer denominator: \");  \n20 int denominator = scanner.nextInt();  \n21"
        },
        {
          "type": "code_caption",
          "bbox": [0.339, 0.125, 0.500, 0.148],
          "index": 19,
          "content": "Algorithm 1 Modules for MCTSteg"
        }
      ],
      "sub_type": "code",
      "guess_lang": "java"
    }
    ```

#### Content List (content_list.json)

> [!NOTE]
> `content_list.json` is deprecated and no longer generated. Use `structured_content.json` for structured content output. See the [Common Structured Content](#common-structured-content-structured_contentjsondevelopment-version-subject-to-change) section above for the shared structure.

## Summary

The above files constitute MinerU's complete output results. Users can choose appropriate files for subsequent processing based on their needs:

- **Model outputs** (Use raw outputs):  
    * model.json
  
- **Debugging and verification** (Use visualization files):
    * layout.pdf
    * span.pdf 
  
- **Content extraction**: (Use simplified files):
    * *.md
    * structured_content.json
  
- **Secondary development**: (Use structured files):
    * middle.json
