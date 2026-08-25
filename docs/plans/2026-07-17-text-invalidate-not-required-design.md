# Text Invalidation Is Not Required

## Decision

Invalidation returns `parse_not_required` when its resolved target is a `.txt`, `.md`, `.markdown`, `.csv`, `.rst`, or `.tex`
document. This applies whether the request identifies the target by `path` or `doc_ref`. The error uses type
`invalid_request_error`, the identifier field as its param, and the same direct-read guidance as `mineru parse`.

Text ingestion and full-text indexing remain unchanged. The request does not supersede parse rows because text files never create
them. Other file types retain the existing invalidation behavior.

## Verification

- Cover all text extensions.
- Verify an untracked text file is still ingested and indexed before the error is returned.
- Verify path and doc-ref requests return `path` and `doc_ref` as their respective error params.
- Verify no parse rows are created or changed.
- Verify human-readable CLI output and the shared HTTP error envelope preserve `parse_not_required`.

Refs #5276
