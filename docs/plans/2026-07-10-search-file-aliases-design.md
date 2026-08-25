# Search File Aliases Design

Issue: #101

## Problem

Content search is document-level and deduplicates results by SHA-256, while `filename` and `ext` belong to individual file paths. The current FTS content row stores one filename from the path that last built the index, so a search result can combine that stale alias with unrelated current paths.

## FTS Contract

`fts_contents` stores document-level searchable data only: SHA-256, tier, text, title, and author. Remove `filename` from the virtual table and from `FTSManager.replace()`. Filename search remains the responsibility of `fts_filenames` and `mineru find`.

The product has not been released and has no user databases to migrate. Update `001_init.sql` directly without adding a migration or changing the schema version.

## Search Response

Remove the path-level `filename`, `ext`, and `paths` fields from `SearchResult`. Add a structured `files` list:

```python
class SearchFile(DoclibModel):
    path: str
    filename: str
    ext: str
    status: FileStatus
```

JSON responses always include every file row associated with the matching document. Files are ordered by `files.id DESC`, which represents reverse insertion order and is unambiguous because the primary key is unique. Document-level `size_bytes` comes from `docs`, not an arbitrary file row.

An indexed document with no file rows remains a search result with `files=[]`.

## CLI Rendering

Filtering is presentation-only:

- If at least one file is active, display all active files and hide inactive aliases.
- If no file is active, display all inactive files.
- Append `(deleted)` or `(unreachable)` to the corresponding path.
- If `files` is empty, display `File no longer exists.`

The result heading uses the document title when available and otherwise `Document <short_id>`. JSON mode does not apply the CLI filtering.

## Tests

Cover different filenames for one SHA, deterministic `id DESC` ordering, JSON preservation of all statuses, CLI active-only presentation, inactive fallback labels, orphan document results, and the separation between `search` content matching and `find` filename matching.
