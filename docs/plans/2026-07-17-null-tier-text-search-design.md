# Null Tier for Text Search Results

## Decision

Text content indexed directly from `.txt`, `.md`, `.markdown`, `.csv`, `.rst`, and `.tex` files stores `NULL` in
`fts_contents.tier`. Parsed content continues to store its actual MinerU tier.

`SearchResult.tier` and the corresponding internal row types become `Tier | None`. No new source field and no historical-data
migration are added.

Search filtering follows these rules:

- Without `tier` or `min_tier`, results with both concrete tiers and `NULL` tiers are returned.
- With either filter, `NULL`-tier results are excluded.
- Concrete tiers retain their existing exact-tier and minimum-tier ordering behavior.

The non-JSON CLI already omits the tier label when `tier` is null. JSON returns `"tier": null`.

## Verification

- Verify text ingestion stores `NULL` in `fts_contents.tier`.
- Verify unfiltered search returns text results with `tier=null`.
- Verify `tier` and `min_tier` filters exclude text results.
- Preserve parsed-result filtering and CLI rendering.

Refs #5276
