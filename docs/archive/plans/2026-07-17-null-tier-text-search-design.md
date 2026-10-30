# Null Tier for Text Search Results

> 历史归档：保留原设计或验收时的记录，不作为当前版本的安装、接口或实现指引。
> 当前入口见 [开发文档](https://opendatalab.github.io/MinerU/next/)；当时的版本、测试结论和未完成事项不追溯改写。

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
