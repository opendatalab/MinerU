# Page Range Rename Plan

Date: 2026-06-16

## Goal

Rename doclib fields and parameters that use `pages` to mean a page range string to `page_range`.

This is a naming cleanup only. It does not change the CLI user-facing `--pages` option and does not rename Middle JSON `pages`.

## Naming Rules

- CLI flag remains `--pages`.
- Code, interface, and DB fields that represent a page range string should use `page_range`.
- `pages` should be reserved for `list[PageInfo]` and Middle JSON arrays:

```json
{
  "pages": []
}
```

- Do not introduce `raw_page_range`.
- Whether a `page_range` is raw input or canonical storage is defined by the layer contract, not by the field name.

## Scope

### DB Schema

Rename:

- `parses.pages` -> `parses.page_range`
- `parsing_rules.pages` -> `parsing_rules.page_range`

Because NEXT is still in development and there is no compatibility burden, update `001_init.sql` directly. Do not add a migration for this rename.

### Doclib Types And Interface

Rename range-string fields:

- `ParseRequest.pages` -> `ParseRequest.page_range`
- `ParseResponse.pages` -> `ParseResponse.page_range`
- `ParseInfo.pages` -> `ParseInfo.page_range`
- `TierParseInfo.pages` -> `TierParseInfo.page_range`
- `ParsingRuleRequest.pages` -> `ParsingRuleRequest.page_range`
- `ParsingRuleInfo.pages` -> `ParsingRuleInfo.page_range`
- `DocContentRequest.pages` -> `DocContentRequest.page_range`
- `DocContentExportRequest.pages` -> `DocContentExportRequest.page_range`
- `ContentRequestScope.pages` -> `ContentRequestScope.page_range`
- `ContentRange.pages` -> `ContentRange.page_range`
- `ContentNextRequest.pages` -> `ContentNextRequest.page_range`

### Doclib Service, Server, Client

Rename function parameters and local variables where they represent a page range string:

- `request_parse(..., pages=...)` -> `request_parse(..., page_range=...)`
- `request_pages_str` -> `request_page_range`
- `default_pages` -> `default_page_range`
- `initial_pages` -> `initial_page_range`
- `rule_pages` -> `rule_page_range`
- `uncovered_str` -> `uncovered_page_range`

Rename helpers:

- `expand_pages()` -> `expand_page_range()`
- `_pages_set_to_str()` -> `_page_numbers_to_range_str()`
- `pages_covered()` -> `page_range_covered()`
- `pages_uncovered()` -> `page_range_uncovered()`

`parse_range_set()` should become `parse_page_range_set()` when it parses a page range string into 1-based page numbers.

### CLI

Keep the public option:

```python
pages: str | None = typer.Option(None, "-p", "--pages", ...)
```

When constructing doclib requests, map it to `page_range`:

```python
ParseRequest(page_range=pages)
```

Human-readable CLI text may continue to print `pages=...` if that is clearer for users. JSON output should prefer `page_range`.

### Parser SDK And V1 API

No major rename is needed:

- `DocumentParser.parse(page_range=...)` is already correct.
- `MinerUApiParser` already uses `page_range`.
- v1 API already uses `files[].page_range`.
- `api_server.py` should continue using `page_range`.

Do not rename Middle JSON:

- `ParseResult.pages`
- `PageInfo` lists
- output JSON `{ "pages": [...] }`

### Tests

Update tests that refer to page range fields:

- `ParseRequest(..., pages=...)` -> `ParseRequest(..., page_range=...)`
- `result.pages == "1~10"` -> `result.page_range == "1~10"`
- DB inserts/selects for `parses.pages` -> `parses.page_range`
- DB inserts/selects for `parsing_rules.pages` -> `parsing_rules.page_range`

Do not change tests where `pages` is a `list[PageInfo]` or Middle JSON key.

### Documentation

Update docs under `docs/next` that describe doclib/interface fields:

- replace range-string `pages` fields with `page_range`
- keep CLI examples using `--pages`
- explain that CLI `--pages` maps to the request field `page_range`

Unified API docs under `docs/next/api/` already use `page_range` for v1 parse jobs. Only adjust them if doclib-specific `pages` range fields appear there.

## Execution Order

1. Update DB schema and row typed dicts.
2. Update `doclib/types.py` request/response models.
3. Update `doclib/base.py`, `doclib/client.py`, and `doclib/server.py` signatures and route calls.
4. Update `doclib/services/parse_svc.py`, `config_svc.py`, compaction, and related SQL.
5. Update CLI request construction and JSON output.
6. Update tests.
7. Run global search for `pages` and classify each remaining hit.
8. Update docs.
9. Run focused tests and ruff.

## Acceptance Criteria

After the rename, remaining `pages` usages should primarily be:

- CLI flag `--pages`
- `ParseResult.pages`
- `PageInfo` list variables
- Middle JSON `{ "pages": [...] }`
- ordinary prose where "pages" means document pages

The following should no longer use `pages` to mean a page range string:

- DB `parses`
- DB `parsing_rules`
- doclib interface request/response models
- parse batch records
- coverage fields
- content request scope
- content ranges
- content continuation `next_request`

## Verification

Run at minimum:

```bash
.venv/bin/python -m pytest -o addopts='' tests/unittest/test_doclib_interface_contract.py tests/unittest/test_doclib_cache_semantics.py tests/unittest/test_cli_next_command_design.py -q
.venv/bin/python -m ruff check mineru/doclib mineru/cli_next tests/unittest/test_doclib_interface_contract.py tests/unittest/test_doclib_cache_semantics.py tests/unittest/test_cli_next_command_design.py
```

Then run a targeted global check:

```bash
rg -n "\\bpages\\b" mineru/doclib mineru/cli_next tests docs/next
```

Every remaining range-string usage of `pages` should either be intentionally user-facing CLI wording or be renamed.
