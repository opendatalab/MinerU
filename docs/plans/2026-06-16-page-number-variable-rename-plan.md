# Page Number Variable Rename Plan

状态: Draft
日期: 2026-06-16
范围: doclib / parser API server / tests 中与页码集合相关的局部变量和 helper 命名
非目标: 修改 CLI 用户参数 `--pages`；修改 Middle JSON `pages`；修改 `ParseResult.pages`

## 1. 背景

项目已约定:

- `page_range`: 页码范围字符串。
- `pages`: `list[PageInfo]` 或 page dict list。
- `page_idx` / `page_indices`: 0-based page index。
- `page_no` / `page_numbers`: 1-based page number。
- `page_count` / `total_page_count`: 页数统计。

当前代码中仍有一些 `pages` / `xxx_pages` 变量实际表示 1-based page numbers，容易与 `pages: list[PageInfo]` 混淆。

本计划只处理这类命名问题。

## 2. 总原则

保留:

- `ParseResult.pages`
- Middle JSON 顶层 `{"pages": [...]}`
- `pages: list[PageInfo]`
- `loaded_pages`
- `new_pages`
- `json_pages`
- `rendered_pages`
- backend / render 中表示页面对象列表的 `pages`
- Office converter 中表示页面结构列表的 `self.pages`
- CLI 用户参数局部变量 `pages = typer.Option(... "--pages")`

重命名:

- 任何表示 1-based 页码集合的 `pages` / `xxx_pages`。
- 任何表示页数统计的 `pages`。
- 任何 helper 名称中 `page_set` 但实际含义是 1-based page numbers 的命名。

## 3. 代码重命名清单

### 3.1 `mineru/doclib/services/parse_svc.py`

#### Helper

| 当前 | 建议 |
|------|------|
| `parse_range_set()` | `parse_page_range_set()` |
| `_page_set_to_range_str()` | `_page_numbers_to_range_str()` |

说明:

- `parse_page_range_set()` 输入是 `page_range`，输出是 `set[int]` 类型的 1-based page numbers。
- `_page_numbers_to_range_str()` 输入是 1-based page numbers，输出是 compact page range string。

#### 局部变量

| 当前 | 实际含义 | 建议 |
|------|----------|------|
| `pages` in `parse_range_set()` | 1-based page numbers | `page_numbers` |
| `requested` in `filter_pages_by_user_range()` | 1-based page numbers | `requested_page_numbers` |
| `needed` | 仍需处理的 1-based page numbers | `needed_page_numbers` |
| `covered` | 已覆盖的 1-based page numbers | `covered_page_numbers` |
| `active_covered` | active batch 覆盖的 1-based page numbers | `active_covered_page_numbers` |
| `done_pages` | done 状态的 1-based page numbers | `done_page_numbers` |
| `active_pages` | active 状态的 1-based page numbers | `active_page_numbers` |
| `missing_pages` | missing 状态的 1-based page numbers | `missing_page_numbers` |
| `row_pages` | 单个 parse row 的 1-based page numbers | `row_page_numbers` |

#### 注释

把类似:

```text
remove pages covered by ...
```

改成:

```text
remove page numbers covered by ...
```

或:

```text
remove covered page_range numbers
```

### 3.2 `mineru/doclib/server.py`

| 当前 | 实际含义 | 建议 |
|------|----------|------|
| `current_pages` | 当前 content range 的 1-based page numbers | `current_page_numbers` |
| `item_pages` | 下一个 content range 的 1-based page numbers | `item_page_numbers` |
| `page_set` in `_last_requested_page()` | 1-based page numbers | `page_numbers` |
| `done_pages` | done 状态的 1-based page numbers | `done_page_numbers` |
| `active_pages` | active 状态的 1-based page numbers | `active_page_numbers` |
| `missing_pages` | missing 状态的 1-based page numbers | `missing_page_numbers` |
| `row_pages` | 单个 parse row 的 1-based page numbers | `row_page_numbers` |
| `total_pages` in `_next_content_request()` | 当前上下文总页数 | `total_page_count` |

同时更新 import:

```python
parse_range_set -> parse_page_range_set
_page_set_to_range_str -> _page_numbers_to_range_str
```

如果 `_page_numbers_to_range_str()` 仍是 parse_svc 内部 helper，需确认 `server.py` 的导入边界是否合适。若当前已有导入，可继续保持；不在本任务中抽公共模块。

### 3.3 `mineru/doclib/background/compaction.py`

| 当前 | 实际含义 | 建议 |
|------|----------|------|
| `all_pages` | 所有 done batch 的 1-based page numbers | `all_page_numbers` |
| `sorted_pages` | 排序后的 1-based page numbers | `sorted_page_numbers` |
| `page_set` | 当前 compacted range 的 1-based page numbers | `page_numbers` |

保留:

- `json_pages`: page dict list。
- `pages_by_idx`: dict key 是 0-based `page_idx`，value 是 page dict；命名可接受，但如要更精确可改为 `pages_by_page_idx`。本轮可选，不强制。

### 3.4 `mineru/parser/api_server.py`

| 当前 | 实际含义 | 建议 |
|------|----------|------|
| `pages` in `usage()` | 已处理页数统计 | `processed_page_count` |

保留:

- `_compact_page_numbers(page_numbers)`。
- `_page_range_from_result_pages(pages)`，其中 `pages` 是 result page list。
- `pages_processed` 响应字段，属于 API schema，是否改名不在本计划范围内。

## 4. 测试重命名清单

### 4.1 `tests/unittest/test_doclib_cache_semantics.py`

| 当前 | 实际含义 | 建议 |
|------|----------|------|
| `_write_batch(..., pages: str, ...)` | page range string | `_write_batch(..., page_range: str, ...)` |
| fake parser 参数 `pages: str` | page range string | `page_range: str` |

保留:

- `pages = load_pages_from_done_batches(...)`
- `ParseResult(pages=[...])`
- `json.dumps({"pages": json_pages})`
- `compacted["pages"]`

### 4.2 其它测试

只改引用被重命名 helper 的测试:

- `parse_range_set` -> `parse_page_range_set`
- `_page_set_to_range_str` -> `_page_numbers_to_range_str`

真实页面对象列表变量保持 `pages`。

## 5. 文档处理

本任务不要求大规模清理历史底稿。

只在必要时更新:

- `docs/next/page-naming.md`: 如果发现新边界需要补充。
- 与被改 helper 直接相关的实现计划或 ADR。

不处理:

- CLI 文档中的 `--pages`。
- Middle JSON 文档中的 `pages`。
- 历史底稿中的 `parsed_pages` 旧设计残留。
- Telemetry 页数统计指标应使用 `parse.processed_page_count` / `parse_server.processed_page_count`。

## 6. 执行顺序

1. 重命名 `parse_svc.py` 中的 helper。
2. 更新 `parse_svc.py` 内部调用和局部变量。
3. 更新 `server.py` import、调用和局部变量。
4. 更新 `compaction.py` import、调用和局部变量。
5. 更新 `api_server.py` 的 usage 统计变量。
6. 更新测试中的 helper 引用和 fake 参数名。
7. 运行静态搜索，确认剩余 `xxx_pages: set[int]` 或 `pages: set[int]` 不再表示 page numbers。
8. 运行 lint 和测试。

## 7. 验证命令

```bash
.venv/bin/python -m ruff check mineru/doclib mineru/parser/api_server.py tests/unittest
```

```bash
.venv/bin/python -m pytest -o addopts='' \
  tests/unittest/test_doclib_cache_semantics.py \
  tests/unittest/test_doclib_progressive_reading.py \
  tests/unittest/test_doclib_interface_contract.py \
  tests/unittest/test_parser_api_contract.py \
  tests/unittest/test_cli_next_command_design.py \
  -q
```

建议最终审计:

```bash
rg -n "\b[a-zA-Z_]*pages: set\[int\]|parse_range_set|_page_set_to_range_str|total_pages|pages: str" mineru/doclib mineru/parser/api_server.py tests/unittest -g '*.py'
```

逐项确认剩余命中只属于:

- `list[PageInfo]` / page dict list。
- Middle JSON `{"pages": [...]}`。
- CLI `--pages`。
- human-readable output text。
- API schema 中暂不改名的历史字段。

## 8. 风险

- helper 重命名会影响 `server.py`、`compaction.py`、测试引用，必须一次性改完。
- `parse_page_range_set()` 返回 1-based page numbers；调用方如果误当 0-based index 使用，会产生 off-by-one bug。重命名后应优先检查所有 `+ 1` / `- 1` 位置。
- `pages_processed` 等外部 API 字段暂不改，避免把内部命名清理扩大成 API breaking change。
