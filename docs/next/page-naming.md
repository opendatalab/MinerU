# Page Naming Convention

状态: Draft
读者: 核心开发者、编程 Agent、SDK/API/CLI 设计参与者、文档作者
范围: MinerU Next 中页码、页码范围、页面数组和页数统计的命名约定
非目标: 定义 page range 语法；定义 Middle JSON 页面结构

## 1. 目标

本文件统一项目中与页面相关的字段、变量和参数命名，避免 `pages` 同时表示“页码范围字符串”“页面数组”“页数统计”等多种语义。

核心原则:

- `page_range` 表示页码范围字符串。
- `pages` 只表示页面对象列表。
- `page_idx` / `page_indices` 表示 0-based index。
- `page_no` / `page_numbers` 表示 1-based page number。
- `page_count` 表示页数统计。

## 2. 推荐命名

| 名称 | 含义 | 类型示例 | 说明 |
|------|------|----------|------|
| `page_range` | 页码范围字符串 | `str \| None` | 例如 `"1~10"`、`"1,3,5~7"`、`"all"`。是否已经规范化由所在层契约定义。 |
| `pages` | 页面对象列表 | `list[PageInfo]` / `list[dict]` | 只用于真实页面结构，例如 `ParseResult.pages`、Middle JSON 顶层 `{"pages": [...]}`。 |
| `<state>_pages` | 某状态下的页面对象列表 | `done_pages: list[PageInfo]` | 仅当值仍然是 `PageInfo` / page dict 列表时使用。 |
| `page_idx` | 0-based 页面索引 | `int` | 第一页为 `0`。对应 `PageInfo.page_idx`。 |
| `page_indices` | 0-based 页面索引集合 | `list[int]` / `set[int]` | 用于内部索引计算。 |
| `page_no` | 1-based 页码 | `int` | 第一页为 `1`。适合用户可见 locator、cursor 和提示。 |
| `page_numbers` | 1-based 页码集合 | `list[int]` / `set[int]` | 表示一组 1-based 页码；`page_range` 展开后的结果也是 `page_numbers`。 |
| `doc.page_count` | 文档总页数 | `int \| None` | 文档级属性，表示这个 doc 本身有多少页。 |
| `total_page_count` | 当前上下文总页数 | `int` | 用于局部计算上下文，不一定等于某个 doc 的持久字段。 |

## 3. 禁止和避免

避免使用:

- `num_pages`
- `total_pages`
- 用 `pages` 表示页码范围字符串
- 用 `pages` 表示页数统计

替代规则:

| 避免写法 | 推荐写法 |
|----------|----------|
| `pages: str` | `page_range: str` |
| `result.pages == "1~10"` | `result.page_range == "1~10"` |
| `num_pages` | `page_count` 或 `total_page_count` |
| `total_pages` | `page_count` 或 `total_page_count` |
| `done_pages: str` | `done_page_range: str` |
| `active_pages: str` | `active_page_range: str` |
| `missing_pages: str` | `missing_page_range: str` |

## 4. 边界规则

### 4.1 `page_range`

`page_range` 是范围表达，不是页面对象，也不是展开后的页码集合。

适用位置:

- API / SDK 请求字段。
- doclib parse record 字段。
- parsing-rule 字段。
- content response 中的 request scope、content range 和 next request。
- parse batch JSON 文件名的一部分。

示例:

```python
ParseRequest(page_range="1~10")
ParseInfo(page_range="1~5")
ContentNextRequest(page_range="11~20")
```

### 4.2 `pages`

`pages` 只表示页面对象列表。

适用位置:

- `ParseResult.pages`
- `list[PageInfo]`
- Middle JSON 顶层 `{"pages": [...]}`
- 从 JSON 文件中读取出来的 page dict 列表

示例:

```python
pages: list[PageInfo] = result.pages
payload = {"pages": [page.to_dict() for page in pages]}
```

### 4.3 `<state>_pages`

`<state>_pages` 只有在值是页面对象列表时才允许使用。

允许:

```python
done_pages: list[PageInfo]
rendered_pages: list[dict]
```

不允许:

```python
done_pages: str          # should be done_page_range
missing_pages: set[int]  # should be missing_page_numbers
```

如果值是页码范围字符串，使用 `<state>_page_range`。

如果值是 1-based 页码集合，使用 `<state>_page_numbers`。

如果值是 0-based 索引集合，使用 `<state>_page_indices`。

### 4.4 `page_idx` 与 `page_no`

`page_idx` 是 0-based index，用于内部页面结构和数组索引。

`page_no` 是 1-based number，用于用户可见语义、locator 和 cursor。

示例:

```python
page_idx = page.page_idx      # 0 for first page
page_no = page_idx + 1        # 1 for first page
```

### 4.5 `page_numbers`

`page_numbers` 表示 1-based 页码集合。它不要求来源一定是 `page_range`；只要集合中的值是面向用户语义的 1-based 页码，就应使用这个名字。

示例:

```python
page_numbers = sorted(parse_page_range_set(page_range))
```

也可以来自搜索命中、用户选择、渲染范围或其它计算过程。

如果集合中的值是 0-based index，必须使用 `page_indices`。

### 4.6 `page_count`

`page_count` 表示“有多少页”。

使用规则:

- 文档实体字段使用 `page_count`，例如 `DocInfo.page_count`。
- 局部上下文中如果需要表达当前计算范围的总页数，使用 `total_page_count`。
- 不使用 `num_pages` 或 `total_pages`。

## 5. 快速判断

命名时先问这几个问题:

1. 值是不是 `"1~10"` 这样的范围字符串？
   - 是: 用 `page_range`。
2. 值是不是 `PageInfo` 或 page dict 的列表？
   - 是: 用 `pages` 或 `<state>_pages`。
3. 值是不是 0-based 页面索引？
   - 是: 用 `page_idx` / `page_indices`。
4. 值是不是 1-based 页码？
   - 是: 用 `page_no` / `page_numbers`。
5. 值是不是页数统计？
   - 是: 用 `page_count` 或 `total_page_count`。

## 6. 允许保留 `--pages`

CLI 用户参数可以继续使用 `--pages`。

原因:

- `--pages` 对用户更短、更自然。
- CLI 层负责把 `--pages` 映射到内部请求字段 `page_range`。
- JSON 输出、SDK 类型和 HTTP API 字段应优先使用 `page_range`。

示例:

```python
pages: str | None = typer.Option(None, "--pages")
request = ParseRequest(page_range=pages)
```
