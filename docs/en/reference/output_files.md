# MinerU Output Files Documentation

## Overview

After executing the `mineru` command, in addition to the main markdown file output, multiple auxiliary files are generated for debugging, quality inspection, and further processing. These files include:

- **Visual debugging files**: Help users intuitively understand the document parsing process and results
- **Structured data files**: Contain detailed parsing data for secondary development

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

### Text Spans File (spans.pdf)

> [!NOTE]
> Only applicable to pipeline backend

**File naming format**: `{original_filename}_spans.pdf`

**Functionality**:

- Uses different colored line boxes to annotate page content based on span type
- Used for quality inspection and issue troubleshooting

**Use cases**:

- Quickly troubleshoot text loss issues
- Check inline formula recognition
- Verify text segmentation accuracy

![span page example](../images/spans_example.png)

## Structured Data Files

### Model Inference Results (model.json)

> [!NOTE]
> Only applicable to pipeline backend

**File naming format**: `{original_filename}_model.json`

#### Data Structure Definition

```python
from pydantic import BaseModel, Field
from enum import IntEnum

class CategoryType(IntEnum):
    """Content category enumeration"""
    title = 0               # Title
    plain_text = 1          # Text
    abandon = 2             # Including headers, footers, page numbers, and page annotations
    figure = 3              # Image
    figure_caption = 4      # Image caption
    table = 5               # Table
    table_caption = 6       # Table caption
    table_footnote = 7      # Table footnote
    isolate_formula = 8     # Interline formula
    formula_caption = 9     # Interline formula number
    embedding = 13          # Inline formula
    isolated = 14           # Interline formula
    text = 15               # OCR recognition result

class PageInfo(BaseModel):
    """Page information"""
    page_no: int = Field(description="Page number, first page is 0", ge=0)
    height: int = Field(description="Page height", gt=0)
    width: int = Field(description="Page width", ge=0)

class ObjectInferenceResult(BaseModel):
    """Object recognition result"""
    category_id: CategoryType = Field(description="Category", ge=0)
    poly: list[float] = Field(description="Quadrilateral coordinates, format: [x0,y0,x1,y1,x2,y2,x3,y3]")
    score: float = Field(description="Confidence score of inference result")
    latex: str | None = Field(description="LaTeX parsing result", default=None)
    html: str | None = Field(description="HTML parsing result", default=None)

class PageInferenceResults(BaseModel):
    """Page inference results"""
    layout_dets: list[ObjectInferenceResult] = Field(description="Page recognition results")
    page_info: PageInfo = Field(description="Page metadata")

# Complete inference results
inference_result: list[PageInferenceResults] = []
```

#### Coordinate System Description

`poly` coordinate format: `[x0, y0, x1, y1, x2, y2, x3, y3]`

- Represents coordinates of top-left, top-right, bottom-right, bottom-left points respectively
- Coordinate origin is at the top-left corner of the page

![poly coordinate diagram](../images/poly.png)

#### Sample Data

```json
[
    {
        "layout_dets": [
            {
                "category_id": 2,
                "poly": [
                    99.1906967163086,
                    100.3119125366211,
                    730.3707885742188,
                    100.3119125366211,
                    730.3707885742188,
                    245.81326293945312,
                    99.1906967163086,
                    245.81326293945312
                ],
                "score": 0.9999997615814209
            }
        ],
        "page_info": {
            "page_no": 0,
            "height": 2339,
            "width": 1654
        }
    },
    {
        "layout_dets": [
            {
                "category_id": 5,
                "poly": [
                    99.13092803955078,
                    2210.680419921875,
                    497.3183898925781,
                    2210.680419921875,
                    497.3183898925781,
                    2264.78076171875,
                    99.13092803955078,
                    2264.78076171875
                ],
                "score": 0.9999997019767761
            }
        ],
        "page_info": {
            "page_no": 1,
            "height": 2339,
            "width": 1654
        }
    }
]
```

### VLM Output Results (model_output.txt)

> [!NOTE]
> Only applicable to VLM backend

**File naming format**: `{original_filename}_model_output.txt`

#### File Format Description

- Uses `----` to separate output results for each page
- Each page contains multiple text blocks starting with `<|box_start|>` and ending with `<|md_end|>`

#### Field Meanings

| Tag | Format | Description |
|-----|--------|-------------|
| Bounding box | `<\|box_start\|>x0 y0 x1 y1<\|box_end\|>` | Quadrilateral coordinates (top-left, bottom-right points), coordinate values after scaling page to 1000×1000 |
| Type tag | `<\|ref_start\|>type<\|ref_end\|>` | Content block type identifier |
| Content | `<\|md_start\|>markdown content<\|md_end\|>` | Markdown content of the block |

#### Supported Content Types

```json
{
    "text": "Text",
    "title": "Title", 
    "image": "Image",
    "image_caption": "Image caption",
    "image_footnote": "Image footnote",
    "table": "Table",
    "table_caption": "Table caption", 
    "table_footnote": "Table footnote",
    "equation": "Interline formula"
}
```

#### Special Tags

- `<|txt_contd|>`: Appears at the end of text, indicating that this text block can be connected with subsequent text blocks
- Table content uses `otsl` format and needs to be converted to HTML for rendering in Markdown

### Intermediate Processing Results (middle.json)

**File naming format**: `{original_filename}_middle.json`

#### Top-level Structure

| Field Name | Type | Description |
|------------|------|-------------|
| `pdf_info` | `list[dict]` | Array of parsing results for each page |
| `_backend` | `string` | Parsing mode: `pipeline` or `vlm` |
| `_version_name` | `string` | MinerU version number |

#### Page Information Structure (pdf_info)

| Field Name | Description |
|------------|-------------|
| `preproc_blocks` | Unsegmented intermediate results after PDF preprocessing |
| `layout_bboxes` | Layout segmentation results, including layout direction and bounding boxes, sorted by reading order |
| `page_idx` | Page number, starting from 0 |
| `page_size` | Page width and height `[width, height]` |
| `_layout_tree` | Layout tree structure |
| `images` | Image block information list |
| `tables` | Table block information list |
| `interline_equations` | Interline formula block information list |
| `discarded_blocks` | Block information to be discarded |
| `para_blocks` | Content block results after segmentation |

#### Block Structure Hierarchy

```
Level 1 blocks (table | image)
└── Level 2 blocks
    └── Lines
        └── Spans
```

#### Level 1 Block Fields

| Field Name | Description |
|------------|-------------|
| `type` | Block type: `table` or `image` |
| `bbox` | Rectangular box coordinates of the block `[x0, y0, x1, y1]` |
| `blocks` | List of contained level 2 blocks |

#### Level 2 Block Fields

| Field Name | Description |
|------------|-------------|
| `type` | Block type (see table below) |
| `bbox` | Rectangular box coordinates of the block |
| `lines` | List of contained line information |

#### Level 2 Block Types

| Type | Description |
|------|-------------|
| `image_body` | Image body |
| `image_caption` | Image caption text |
| `image_footnote` | Image footnote |
| `table_body` | Table body |
| `table_caption` | Table caption text |
| `table_footnote` | Table footnote |
| `text` | Text block |
| `title` | Title block |
| `index` | Index block |
| `list` | List block |
| `interline_equation` | Interline formula block |

#### Line and Span Structure

**Line fields**:
- `bbox`: Rectangular box coordinates of the line
- `spans`: List of contained spans

**Span fields**:
- `bbox`: Rectangular box coordinates of the span
- `type`: Span type (`image`, `table`, `text`, `inline_equation`, `interline_equation`)
- `content` | `img_path`: Text content or image path

#### Sample Data

```json
{
    "pdf_info": [
        {
            "preproc_blocks": [
                {
                    "type": "text",
                    "bbox": [
                        52,
                        61.956024169921875,
                        294,
                        82.99800872802734
                    ],
                    "lines": [
                        {
                            "bbox": [
                                52,
                                61.956024169921875,
                                294,
                                72.0000228881836
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        54.0,
                                        61.956024169921875,
                                        296.2261657714844,
                                        72.0000228881836
                                    ],
                                    "content": "dependent on the service headway and the reliability of the departure ",
                                    "type": "text",
                                    "score": 1.0
                                }
                            ]
                        }
                    ]
                }
            ],
            "layout_bboxes": [
                {
                    "layout_bbox": [
                        52,
                        61,
                        294,
                        731
                    ],
                    "layout_label": "V",
                    "sub_layout": []
                }
            ],
            "page_idx": 0,
            "page_size": [
                612.0,
                792.0
            ],
            "_layout_tree": [],
            "images": [],
            "tables": [],
            "interline_equations": [],
            "discarded_blocks": [],
            "para_blocks": [
                {
                    "type": "text",
                    "bbox": [
                        52,
                        61.956024169921875,
                        294,
                        82.99800872802734
                    ],
                    "lines": [
                        {
                            "bbox": [
                                52,
                                61.956024169921875,
                                294,
                                72.0000228881836
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        54.0,
                                        61.956024169921875,
                                        296.2261657714844,
                                        72.0000228881836
                                    ],
                                    "content": "dependent on the service headway and the reliability of the departure ",
                                    "type": "text",
                                    "score": 1.0
                                }
                            ]
                        }
                    ]
                }
            ]
        }
    ],
    "_backend": "pipeline",
    "_version_name": "0.6.1"
}
```

### Content List (content_list.json)

**File naming format**: `{original_filename}_content_list.json`

#### Functionality

This is a simplified version of `middle.json` that stores all readable content blocks in reading order as a flat structure, removing complex layout information for easier subsequent processing.

#### Content Types

| Type | Description |
|------|-------------|
| `image` | Image |
| `table` | Table |
| `text` | Text/Title |
| `equation` | Interline formula |

#### Text Level Identification

Text levels are distinguished through the `text_level` field:

- No `text_level` or `text_level: 0`: Body text
- `text_level: 1`: Level 1 heading
- `text_level: 2`: Level 2 heading
- And so on...

#### Common Fields

All content blocks include a `page_idx` field indicating the page number (starting from 0).

#### Sample Data

```json
[
        {
        "type": "text",
        "text": "The response of flow duration curves to afforestation ",
        "text_level": 1,
        "page_idx": 0
    },
    {
        "type": "text",
        "text": "Received 1 October 2003; revised 22 December 2004; accepted 3 January 2005 ",
        "page_idx": 0
    },
    {
        "type": "text",
        "text": "Abstract ",
        "text_level": 2,
        "page_idx": 0
    },
    {
        "type": "text",
        "text": "The hydrologic effect of replacing pasture or other short crops with trees is reasonably well understood on a mean annual basis. The impact on flow regime, as described by the annual flow duration curve (FDC) is less certain. A method to assess the impact of plantation establishment on FDCs was developed. The starting point for the analyses was the assumption that rainfall and vegetation age are the principal drivers of evapotranspiration. A key objective was to remove the variability in the rainfall signal, leaving changes in streamflow solely attributable to the evapotranspiration of the plantation. A method was developed to (1) fit a model to the observed annual time series of FDC percentiles; i.e. 10th percentile for each year of record with annual rainfall and plantation age as parameters, (2) replace the annual rainfall variation with the long term mean to obtain climate adjusted FDCs, and (3) quantify changes in FDC percentiles as plantations age. Data from 10 catchments from Australia, South Africa and New Zealand were used. The model was able to represent flow variation for the majority of percentiles at eight of the 10 catchments, particularly for the 10–50th percentiles. The adjusted FDCs revealed variable patterns in flow reductions with two types of responses (groups) being identified. Group 1 catchments show a substantial increase in the number of zero flow days, with low flows being more affected than high flows. Group 2 catchments show a more uniform reduction in flows across all percentiles. The differences may be partly explained by storage characteristics. The modelled flow reductions were in accord with published results of paired catchment experiments. An additional analysis was performed to characterise the impact of afforestation on the number of zero flow days $( N _ { \\mathrm { z e r o } } )$ for the catchments in group 1. This model performed particularly well, and when adjusted for climate, indicated a significant increase in $N _ { \\mathrm { z e r o } }$ . The zero flow day method could be used to determine change in the occurrence of any given flow in response to afforestation. The methods used in this study proved satisfactory in removing the rainfall variability, and have added useful insight into the hydrologic impacts of plantation establishment. This approach provides a methodology for understanding catchment response to afforestation, where paired catchment data is not available. ",
        "page_idx": 0
    },
    {
        "type": "text",
        "text": "1. Introduction ",
        "text_level": 2,
        "page_idx": 1
    },
    {
        "type": "image",
        "img_path": "images/a8ecda1c69b27e4f79fce1589175a9d721cbdc1cf78b4cc06a015f3746f6b9d8.jpg",
        "img_caption": [
            "Fig. 1. Annual flow duration curves of daily flows from Pine Creek, Australia, 1989–2000. "
        ],
        "img_footnote": [],
        "page_idx": 1
    },
    {
        "type": "equation",
        "img_path": "images/181ea56ef185060d04bf4e274685f3e072e922e7b839f093d482c29bf89b71e8.jpg",
        "text": "$$\nQ _ { \\% } = f ( P ) + g ( T )\n$$",
        "text_format": "latex",
        "page_idx": 2
    },
    {
        "type": "table",
        "img_path": "images/e3cb413394a475e555807ffdad913435940ec637873d673ee1b039e3bc3496d0.jpg",
        "table_caption": [
            "Table 2 Significance of the rainfall and time terms "
        ],
        "table_footnote": [
            "indicates that the rainfall term was significant at the $5 \\%$ level, $T$ indicates that the time term was significant at the $5 \\%$ level, \\* represents significance at the $10 \\%$ level, and na denotes too few data points for meaningful analysis. "
        ],
        "table_body": "<html><body><table><tr><td rowspan=\"2\">Site</td><td colspan=\"10\">Percentile</td></tr><tr><td>10</td><td>20</td><td>30</td><td>40</td><td>50</td><td>60</td><td>70</td><td>80</td><td>90</td><td>100</td></tr><tr><td>Traralgon Ck</td><td>P</td><td>P,*</td><td>P</td><td>P</td><td>P,</td><td>P,</td><td>P,</td><td>P,</td><td>P</td><td>P</td></tr><tr><td>Redhill</td><td>P,T</td><td>P,T</td><td>，*</td><td>**</td><td>P.T</td><td>P,*</td><td>P*</td><td>P*</td><td>*</td><td>，*</td></tr><tr><td>Pine Ck</td><td></td><td>P,T</td><td>P,T</td><td>P,T</td><td>P,T</td><td>T</td><td>T</td><td>T</td><td>na</td><td>na</td></tr><tr><td>Stewarts Ck 5</td><td>P,T</td><td>P,T</td><td>P,T</td><td>P,T</td><td>P.T</td><td>P.T</td><td>P,T</td><td>na</td><td>na</td><td>na</td></tr><tr><td>Glendhu 2</td><td>P</td><td>P,T</td><td>P,*</td><td>P,T</td><td>P.T</td><td>P,ns</td><td>P,T</td><td>P,T</td><td>P,T</td><td>P,T</td></tr><tr><td>Cathedral Peak 2</td><td>P,T</td><td>P,T</td><td>P,T</td><td>P,T</td><td>P,T</td><td>*,T</td><td>P,T</td><td>P,T</td><td>P,T</td><td>T</td></tr><tr><td>Cathedral Peak 3</td><td>P.T</td><td>P.T</td><td>P,T</td><td>P,T</td><td>P,T</td><td>T</td><td>P,T</td><td>P,T</td><td>P,T</td><td>T</td></tr><tr><td>Lambrechtsbos A</td><td>P,T</td><td>P</td><td>P</td><td>P,T</td><td>*,T</td><td>*,T</td><td>*,T</td><td>*,T</td><td>*,T</td><td>T</td></tr><tr><td>Lambrechtsbos B</td><td>P,T</td><td>P,T</td><td>P,T</td><td>P,T</td><td>P,T</td><td>P,T</td><td>P,T</td><td>P,T</td><td>T</td><td>T</td></tr><tr><td>Biesievlei</td><td>P,T</td><td>P.T</td><td>P,T</td><td>P,T</td><td>*,T</td><td>*,T</td><td>T</td><td>T</td><td>P,T</td><td>P,T</td></tr></table></body></html>",
        "page_idx": 5
    }
]
```

## Summary

The above files constitute MinerU's complete output results. Users can choose appropriate files for subsequent processing based on their needs:

- **Model outputs**: Use raw outputs (model.json, model_output.txt)
- **Debugging and verification**: Use visualization files (layout.pdf, spans.pdf) 
- **Content extraction**: Use simplified files (*.md, content_list.json)
- **Secondary development**: Use structured files (middle.json)
