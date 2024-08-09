

## Overview
After executing the `magic-pdf` command, in addition to outputting files related to markdown, several other files unrelated to markdown will also be generated. These files will be introduced one by one.


### layout.pdf
Each page layout consists of one or more boxes. The number at the top left of each box indicates its sequence number. Additionally, in `layout.pdf`, different content blocks are highlighted with different background colors.

![layout example](images/layout_example.png)


### spans.pdf
All spans on the page are drawn with different colored line frames according to the span type. This file can be used for quality control, allowing for quick identification of issues such as missing text or unrecognized inline formulas.

![spans example](images/spans_example.png)


### model.json

#### Structure Definition
```python
from pydantic import BaseModel, Field
from enum import IntEnum

class CategoryType(IntEnum):
     title = 0               # Title
     plain_text = 1          # Text
     abandon = 2             # Includes headers, footers, page numbers, and page annotations
     figure = 3              # Image
     figure_caption = 4      # Image description
     table = 5               # Table
     table_caption = 6       # Table description
     table_footnote = 7      # Table footnote
     isolate_formula = 8     # Block formula
     formula_caption = 9     # Formula label
     
     embedding = 13          # Inline formula
     isolated = 14           # Block formula
     text = 15               # OCR recognition result
   
     
class PageInfo(BaseModel):
    page_no: int = Field(description="Page number, the first page is 0", ge=0)
    height: int = Field(description="Page height", gt=0)
    width: int = Field(description="Page width", ge=0)

class ObjectInferenceResult(BaseModel):
    category_id: CategoryType = Field(description="Category", ge=0)
    poly: list[float] = Field(description="Quadrilateral coordinates, representing the coordinates of the top-left, top-right, bottom-right, and bottom-left points respectively")
    score: float = Field(description="Confidence of the inference result")
    latex: str | None = Field(description="LaTeX parsing result", default=None)
    html: str | None = Field(description="HTML parsing result", default=None)
  
class PageInferenceResults(BaseModel):
     layout_dets: list[ObjectInferenceResult] = Field(description="Page recognition results", ge=0)
     page_info: PageInfo = Field(description="Page metadata")
    
    
# The inference results of all pages, ordered by page number, are stored in a list as the inference results of MinerU
inference_result: list[PageInferenceResults] = []

```

The format of the poly coordinates is [x0, y0, x1, y1, x2, y2, x3, y3], representing the coordinates of the top-left, top-right, bottom-right, and bottom-left points respectively.
![Poly Coordinate Diagram](images/poly.png)



#### example

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


### middle.json

| Field Name | Description |
| :-----|:------------------------------------------|
|pdf_info | list, each element is a dict representing the parsing result of each PDF page, see the table below for details |
|_parse_type | ocr \| txt, used to indicate the mode used in this intermediate parsing state |
|_version_name | string, indicates the version of magic-pdf used in this parsing |


<br>

**pdf_info**

Field structure description

| Field Name | Description | 
| :-----| :---- |
| preproc_blocks | Intermediate result after PDF preprocessing, not yet segmented |
| layout_bboxes | Layout segmentation results, containing layout direction (vertical, horizontal), and bbox, sorted by reading order |
| page_idx | Page number, starting from 0 |
| page_size | Page width and height | 
| _layout_tree | Layout tree structure |
| images | list, each element is a dict representing an img_block |
| tables | list, each element is a dict representing a table_block |
| interline_equations | list, each element is a dict representing an interline_equation_block |
| discarded_blocks | List, block information returned by the model that needs to be dropped |
| para_blocks | Result after segmenting preproc_blocks |

In the above table, `para_blocks` is an array of dicts, each dict representing a block structure. A block can support up to one level of nesting.

<br>

**block**

The outer block is referred to as a first-level block, and the fields in the first-level block include:

| Field Name | Description |
| :-----| :---- |
| type | Block type (table\|image)|
|bbox | Block bounding box coordinates |
|blocks |list, each element is a dict representing a second-level block |

<br>
There are only two types of first-level blocks: "table" and "image". All other blocks are second-level blocks.

The fields in a second-level block include:

| Field Name | Description |
| :-----| :---- |
| type | Block type |
| bbox | Block bounding box coordinates |
| lines | list, each element is a dict representing a line, used to describe the composition of a line of information| 

Detailed explanation of second-level block types

| type               | Description | 
|:-------------------| :---- |
| image_body         | Main body of the image |
| image_caption      | Image description text |
| table_body         | Main body of the table |
| table_caption      | Table description text |
| table_footnote     | Table footnote |
| text               | Text block |
| title              | Title block |
| interline_equation | Block formula| 

<br>

**line**

The field format of a line is as follows:

| Field Name | Description | 
| :-----| :---- |
| bbox | Bounding box coordinates of the line |
| spans | list, each element is a dict representing a span, used to describe the composition of the smallest unit |


<br>

**span**

| Field Name | Description | 
| :-----| :---- |
| bbox | Bounding box coordinates of the span |
| type | Type of the span |
| content \| img_path | Text spans use content, chart spans use img_path to store the actual text or screenshot path information |

The types of spans are as follows:

| type | Description | 
| :-----| :---- |
| image | Image | 
| table | Table |
| text | Text |
| inline_equation | Inline formula |
| interline_equation | Block formula |

**Summary**

A span is the smallest storage unit for all elements.

The elements stored within para_blocks are block information.

The block structure is as follows:

First-level block (if any) -> Second-level block -> Line -> Span


#### example

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
    "_parse_type": "txt",
    "_version_name": "0.6.1"
}
```
