# MinerU 输出文件说明

## 概览

`mineru` 命令执行后，除了输出主要的 markdown 文件外，还会生成多个辅助文件用于调试、质检和进一步处理。这些文件包括：

具体会生成哪些文件，取决于后端类型和输入文档类型。

- **可视化调试文件**：帮助用户直观了解文档解析过程和结果
- **结构化数据文件**：包含详细的解析数据，可用于二次开发
- 多模态 markdown 输出中，`image` / `chart` 默认以截图为主；若块内存在 `content`，会在图片后追加一个默认折叠的 HTML `<details>` 内容块，其中折叠标题优先使用块的 `sub_type`，否则回退为 `image content` 或 `chart content`

下面将详细介绍每个文件的作用和格式。

## 可视化调试文件

### 布局分析文件 (layout.pdf)

**文件命名格式**：`{原文件名}_layout.pdf`

**功能说明**：

- 可视化展示每一页的布局分析结果
- 每个检测框右上角的数字表示阅读顺序
- 使用不同背景色块区分不同类型的内容块

**使用场景**：

- 检查布局分析是否正确
- 确认阅读顺序是否合理
- 调试布局相关问题

![layout 页面示例](../images/layout_example.png)

### 文本片段文件 (span.pdf)

> [!NOTE]
> 仅在显式启用 span 可视化时生成。

**文件命名格式**：`{原文件名}_span.pdf`

**功能说明**：

- 根据 span 类型使用不同颜色线框标注页面内容
- 用于质量检查和问题排查

**使用场景**：

- 快速排查文本丢失问题
- 检查行内公式识别情况
- 验证文本分割准确性

![span 页面示例](../images/spans_example.png)

## 结构化数据文件

> [!IMPORTANT]
> 当前结构化输出合约使用统一的 `MinerUParser`，通过 `tier`（flash/basic/standard/advanced）和 `parse_mode`（txt/ocr）参数控制。旧 middle-json 中标记 `_backend: "pipeline"`/`_backend: "hybrid"`/`_backend: "office"` 的产物不再被当前读取逻辑兼容，请使用 schema 2.0 的 `MiddleJson`。

### 统一模型输出结果

#### 模型推理结果 (model.json)

**文件命名格式**：`{原文件名}_model.json`

##### 示例数据

```json
[
    {
        “cls_id”: 12,
        “label”: “header”,
        “score”: 0.93,
        “bbox”: [
            1217,
            104,
            1516,
            134
        ],
        “index”: 2
    },
    {
        “cls_id”: 6,
        “label”: “doc_title”,
        “score”: 0.9751,
        “bbox”: [
            275,
            181,
            1512,
            292
        ],
        “index”: 3
    },
    {
        “cls_id”: 22,
        “label”: “text”,
        “score”: 0.9217,
        “bbox”: [
            275,
            330,
            524,
            370
        ],
        “index”: 4
    }
]
```

#### 中间处理结果 (middle.json)

**文件命名格式**：`{原文件名}_middle.json`

##### 顶层结构（schema 2.0）

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `pages` | `list[PageInfo]` | 每一页的解析结果数组，按 `page_idx` 严格升序 |
| `is_full_document` | `bool` | 是否整本文档解析（`page_index_map` 为空时为 `True`） |
| `file_suffix` | `string` | 输入文件类型：`pdf`、`doc`、`docx`、`ppt`、`pptx`、`xls`、`xlsx` 或 `csv` |
| `effort` | `string` | 分析强度：`flash`、`medium`、`high` 或 `xhigh` |
| `parse_mode` | `string` | 解析模式：`txt` 或 `ocr` |
| `mineru_version` | `string` | MinerU 版本号 |

schema 2.0 已移除旧字段 `pdf_info`/`_backend`/`_version_name`/`_ocr_enable`/`_vlm_ocr_enable`。

##### 页面信息结构 (pages)

| 字段名 | 说明 |
|--------|------|
| `page_idx` | 页码，从 0 开始 |
| `blocks` | 顶层页面块列表（唯一内容字段） |

schema 2.0 已移除旧字段 `preproc_blocks`/`para_blocks`/`page_size`/`images`/`tables`/`interline_equations`/`discarded_blocks`/`_layout_tree`/`layout_bboxes`——所有内容统一在 `blocks` 树里表达。

##### 块结构层次

```
顶层页面块 (text | title | equation | image | table | chart | code | list | index | ...)
└── 视觉父块 (image | table | chart | code) 包含子块
    └── body 块 + 可选 caption/footnote 块
```

叶子块（text、title、equation 等）直接持有 `content: str`。视觉父块（image、table、chart、code）持有 `content: list[子块]`，子块包含唯一 body 加可选 caption/footnote。schema 2.0 不再有独立的 `Line`/`Span` 类型。

##### 通用块字段

| 字段名 | 说明 |
|--------|------|
| `type` | 块类型（详见下表） |
| `bbox` | 块的矩形框坐标 `[x0, y0, x1, y1]`，归一化到 `[0, 1]` |
| `index` | 块的阅读序号（顶层块必填） |
| `content` | 叶子块为字符串；视觉父块为子块列表 |

##### 块类型

| 类型 | 说明 |
|------|------|
| `text` | 文本块（叶子块，`content: str`） |
| `doc_title` | 文档标题（level=1） |
| `paragraph_title` | 段落标题（level 2-6） |
| `equation` | 行间公式块（由 `interline_equation` 重命名） |
| `image` | 图片容器；`content` 包含 `image_body` + 可选 `image_caption`/`image_footnote` |
| `table` | 表格容器；`content` 包含 `table_body` + 可选 `table_caption`/`table_footnote` |
| `chart` | 图表容器；`content` 包含 `chart_body` + 可选 `chart_caption`/`chart_footnote` |
| `code` | 代码容器；`content` 包含 `code_body` + 可选 `code_caption`/`code_footnote`；`sub_type` 为 `code` 或 `algorithm` |
| `list` | 列表容器；`content` 为 `text`/`ref_text`/嵌套 `list` 块列表；`sub_type` 为 `text` 或 `ref_text` |
| `index` | 目录容器；`content` 为 `text`/`doc_title`/`paragraph_title`/嵌套 `index` 块列表 |
| `ref_text` | 参考文献/引用文本块 |
| `header` / `footer` / `page_number` / `aside_text` / `page_footnote` | 页面辅助块（叶子块，`content: str`） |

##### 示例数据（schema 2.0）

```json
{
    “pages”: [
        {
            “page_idx”: 0,
            “blocks”: [
                {
                    “type”: “doc_title”,
                    “index”: 0,
                    “bbox”: [0.45, 0.23, 0.55, 0.28],
                    “content”: “1 Introduction”,
                    “level”: 1
                },
                {
                    “type”: “text”,
                    “index”: 1,
                    “bbox”: [0.08, 0.30, 0.46, 0.40],
                    “content”: “dependent on the service headway and the reliability of the departure”
                },
                {
                    “type”: “image”,
                    “index”: 2,
                    “bbox”: [0.52, 0.30, 0.95, 0.55],
                    “content”: [
                        {
                            “type”: “image_body”,
                            “index”: 2,
                            “bbox”: [0.52, 0.30, 0.95, 0.55],
                            “content”: “”,
                            “image_path”: “images/page_0_image_body_2.png”
                        },
                        {
                            “type”: “image_caption”,
                            “index”: 3,
                            “bbox”: [0.52, 0.56, 0.95, 0.58],
                            “content”: “Figure 1: Example figure”
                        }
                    ],
                    “sub_type”: null
                }
            ]
        }
    ],
    “file_suffix”: “pdf”,
    “effort”: “high”,
    “parse_mode”: “ocr”,
    “mineru_version”: “1.x.x”
}
```

#### 内容列表 (content_list.json)

> [!NOTE]
> `content_list.json` 已废弃，不再生成。结构化内容输出请使用 `structured_content.json`，详见下文”通用结构化内容”章节。

### 通用结构化内容 (structured_content.json)(开发中，格式可能调整)

**文件命名格式**：`{原文件名}_structured_content.json`

##### 功能说明

`structured_content.json` 是 3.0 起新增的结构化输出文件，所有后端都会输出该文件：

- 顶层是按页分组的列表，便于按页消费结果
- 每个内容块使用统一的 `type + content` 结构，适合程序化处理
- 不同后端和输入类型支持的 `type` 会有所不同

##### 通用字段

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `type` | `string` | 内容类型 |
| `content` | `dict` | 与 `type` 对应的结构化内容 |
| `bbox` | `list[int]` | 可选，0-1000 范围的边界框 |
| `anchor` | `string` | 可选，部分 `DOCX` 标题或索引项会携带锚点 |

其中 `image` / `chart` 类型还可能包含可选顶层字段 `sub_type`，用于表示视觉子类型。

##### 常见类型

| 类型 | 说明 |
|------|------|
| `title` | 标题块，包含 `title_content` 与 `level` |
| `paragraph` | 段落块，包含 `paragraph_content` |
| `equation_interline` | 行间公式，包含 `math_content`、`math_type` |
| `image` / `table` / `chart` | 视觉类块，包含图片路径、说明文字等结构化字段；印章使用 `sub_type: "seal"` 的 `image` 表示 |
| `code` | 代码块，包含 `code_content`、`code_caption`、`code_footnote`、`code_language` |
| `algorithm` | 算法块，包含 `algorithm_content`、`algorithm_caption`、`algorithm_footnote` |
| `list` / `index` | 列表与索引，包含 `list_items` |
| `page_header` / `page_footer` / `page_number` / `page_aside_text` / `page_footnote` | 页面辅助块 |

`title_content`、`paragraph_content`、说明文字等行内内容通常由 span 列表组成。
`hyperlink` span 包含 `content`、`url`，当同一个链接内存在多段不同样式文本时，
还会包含 `children`；此时 `content` 是 children 文本的拼接，精确样式以
`children` 中的 `text` span 为准。

##### 示例数据

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
            "bbox": [
                83,
                121,
                917,
                156
            ]
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
            "bbox": [
                71,
                815,
                915,
                841
            ]
        }
    ]
]
```

### 多模态模型输出结果

#### 模型推理结果 (model.json)

**文件命名格式**：`{原文件名}_model.json`

##### 文件格式说明

- 该文件为多模态模型的原始输出结果，包含两层嵌套list，外层表示页面，内层表示该页的内容块
- 每个内容块都是一个dict，包含 `type`、`bbox`、`angle`、`content` 字段


##### 支持的内容类型

```json
{
    “text”: “文本”,
    “title”: “标题”,
    “equation”: “行间公式”,
    “image”: “图片”,
    “image_caption”: “图片描述”,
    “image_footnote”: “图片脚注”,
    “table”: “表格”,
    “table_caption”: “表格描述”,
    “table_footnote”: “表格脚注”,
    “phonetic”: “拼音”,
    “code”: “代码块”,
    “code_caption”: “代码描述”,
    “ref_text”: “参考文献”,
    “algorithm”: “算法块”,
    “list”: “列表”,
    “header”: “页眉”,
    “footer”: “页脚”,
    “page_number”: “页码”,
    “aside_text”: “装订线旁注”,
    “page_footnote”: “页面脚注”
}
```

##### 坐标系统说明

`bbox` 坐标格式：`[x0, y0, x1, y1]`

- 分别表示左上、右下两点的坐标
- 坐标原点在页面左上角
- 坐标为相对于原始页面尺寸的百分比，范围在0-1之间

##### 示例数据

```json
[
    [
        {
            “type”: “header”,
            “bbox”: [
                0.077,
                0.095,
                0.18,
                0.181
            ],
            “angle”: 0,
            “score”: null,
            “block_tags”: null,
            “content”: “ELSEVIER”,
            “format”: null,
            “content_tags”: null
        },
        {
            “type”: “title”,
            “bbox”: [
                0.157,
                0.228,
                0.833,
                0.253
            ],
            “angle”: 0,
            “score”: null,
            “block_tags”: null,
            “content”: “The response of flow duration curves to afforestation”,
            “format”: null,
            “content_tags”: null
        }
    ]
]
```

#### 中间处理结果 (middle.json)

**文件命名格式**：`{原文件名}_middle.json`

##### 文件格式说明

在 schema 2.0 中，多模态 tier 与其他 tier 产出相同的统一 `MiddleJson` 结构。以下块类型是标准 `blocks` 树的一部分（不再是扩展）：

- list 是容器块，`content` 持有子 `text`/`ref_text`/嵌套 `list` 块，`sub_type` 区分 list 类型:
    * `text`（文本类型）
    * `ref_text`（引用类型）

- code 是容器块，`content` 持有 `code_body` 加可选 `code_caption`/`code_footnote`，`sub_type` 为:
    * `code`
    * `algorithm`

- 页面辅助块（`header`、`footer`、`page_number`、`aside_text`、`page_footnote`）作为顶层叶子块出现在 `blocks` 中，`content: str`——schema 2.0 不再有独立的 `discarded_blocks` 字段。
- 所有 block 可能包含 `angle` 字段，用来表示旋转角度，0，90，180，270


##### 示例数据
- list block 示例
    ```json
    {
        “type”: “list”,
        “bbox”: [0.068, 0.121, 0.319, 0.260],
        “index”: 11,
        “content”: [
            {
                “type”: “text”,
                “bbox”: [0.068, 0.123, 0.122, 0.137],
                “index”: 3,
                “content”: “H.1 Introduction”
            },
            {
                “type”: “text”,
                “bbox”: [0.068, 0.142, 0.181, 0.179],
                “index”: 4,
                “content”: “H.2 Example: Divide by Zero without Exception Handling”
            }
        ],
        “sub_type”: “text”
    }
    ```
- code block 示例
    ```json
    {
        “type”: “code”,
        “bbox”: [0.045, 0.610, 0.346, 0.964],
        “index”: 17,
        “content”: [
            {
                “type”: “code_body”,
                “bbox”: [0.045, 0.610, 0.346, 0.964],
                “index”: 17,
                “content”: “1 // Fig. H.1: DivideByZeroNoExceptionHandling.java  \n2 // Integer division without exception handling.  \n3 import java.util.Scanner;  \n4  \n5 public class DivideByZeroNoExceptionHandling  \n6 {  \n7 // demonstrates throwing an exception when a divide-by-zero occurs  \n8 public static int quotient( int numerator, int denominator )  \n9 {  \n10 return numerator / denominator; // possible division by zero  \n11 } // end method quotient  \n12  \n13 public static void main(String[] args)  \n14 {  \n15 Scanner scanner = new Scanner(System.in); // scanner for input  \n16  \n17 System.out.print(\”Please enter an integer numerator: \”);  \n18 int numerator = scanner.nextInt();  \n19 System.out.print(\”Please enter an integer denominator: \”);  \n20 int denominator = scanner.nextInt();  \n21”
            },
            {
                “type”: “code_caption”,
                “bbox”: [0.339, 0.125, 0.500, 0.148],
                “index”: 19,
                “content”: “Algorithm 1 Modules for MCTSteg”
            }
        ],
        “sub_type”: “code”,
        “guess_lang”: “java”
    }
    ```

#### 内容列表 (content_list.json)

> [!NOTE]
> `content_list.json` 已废弃，不再生成。结构化内容输出请使用 `structured_content.json`，其通用结构见上文”通用结构化内容”章节。

## 总结

以上文件为 MinerU 的完整输出结果，用户可根据需要选择合适的文件进行后续处理：

- **模型输出**(使用原始输出):
    * model.json

- **调试和验证**(使用可视化文件):
    * layout.pdf
    * span.pdf 
  
- **内容提取**(使用简化文件):
    * *.md
    * structured_content.json
  
- **二次开发**(使用结构化文件):
    * middle.json
