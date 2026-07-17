# Text Inputs Do Not Require Parsing

## Context

`mineru parse` currently returns a successful synthetic `flash` response for `txt`, `md`, `markdown`, `csv`, `rst`, and
`tex`, then fails with `not_cached` when the CLI tries to read parsed pages. These inputs are plain text and do not produce
Middle JSON under the parsed cache.

## Decision

Text inputs remain ingestible, discoverable, and searchable, but are not parseable. An explicit parse request returns:

- code: `parse_not_required`
- type: `invalid_request_error`
- param: `path`
- message: text files do not require MinerU parsing and should be read directly

The same error is returned regardless of `--force`. Office and HTML inputs retain their existing `flash` parse behavior.

## Implementation

- Register `parse_not_required` in the shared error catalog.
- Replace the synthetic text parse response in `ParseService.request_parse()` with an `InvalidRequestError`.
- Remove the unused `_text_response()` helper.
- Update file-type, tier, and CLI documentation so text indexing is distinct from parsing.
- Update the MinerU Skill to tell agents to read these text formats directly.

## Verification

- Cover all six text extension families, including the `markdown` alias.
- Verify JSON and human-readable CLI errors.
- Verify `--force` does not bypass the error.
- Preserve text ingest and FTS coverage.
