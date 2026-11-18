# Document library and agent reading {#using-mineru}

Use the library when you need to search, cite, or revisit the same documents. For a one-time conversion of a file or folder to Markdown or ZIP, use `mineru-kit parse` in the [quick start](../quick_start/index.md).

## Parse a file into the library {#quick-usage-via-command-line}

Start the local library service, then try the `report.html` from the [quick start](../quick_start/index.md). The parsed result is kept in the library, and the command's JSON response is written to `parsed.json`. Native HTML is parsed in full using Flash.

```bash
mineru server start
mineru parse report.html --wait 600 --json > parsed.json
```

On completion, `content` contains the returned text and its `content_ranges`. Each range has a `start` and an `end` locator with the actual document ID, tier, and content position. To write a complete Markdown file, use `-o`. [Parsing, exporting, and waiting](cli_tools.md)

## Search and read cached content {#library-search-and-continuation}

```bash
mineru search "MinerU" --json
```

Search results provide a document `short_id`, snippet, and available paths. To read a specific position from the file you just parsed, copy `content.content_ranges[0].start` from `parsed.json` and pass it as the first argument to `mineru read`. With Bash and `jq`, the following commands pass that locator directly to the CLI.

```bash
locator=$(jq -er '.content.content_ranges[0].start' parsed.json) &&
  mineru read "$locator" --json
```

This reads the start of the first returned range from the cache. If a read response includes a `next_request`, pass its `locator` to the next read and supply `--after` when `after` has a value. In text mode, the `<!-- Next: ... -->` marker gives the next command.

## Read additional PDF pages {#pdf-page-selection}

`mineru parse document.pdf` requests the first 10 pages by default. To read page 11, request that page before using the locator in the response. To export the entire PDF, request all pages explicitly.

```bash
mineru parse document.pdf --pages 11 --wait 600 --json
mineru parse document.pdf --pages all -o document.md --wait 600
```

PDF pages start at 1. `1-5,8,r1` selects the first five pages, page eight, and the last page. `--pages` sets the requested pages, `--limit` bounds one terminal response, and `-o` writes the complete requested range. After a wait timeout, repeat the same parse command to keep waiting. [PDF page ranges](../../next/page-ranges.md)

## Services and configuration

- [Prepare models for offline use](model_source.md){#quick-model-source-configuration} or look up [model paths and backends](model_source.md){#extending-mineru-functionality-with-configuration-files}.
- [Deploy a parsing service](../deployment/index.md){#advanced-usage-via-api-webui-and-services} or [connect the WebUI to an existing service](webui.md).
- [Configure title leveling and cross-page table continuation](model_source.md#llm-post-processing){#configuring-llm-aided-post-processing-with-configyaml}.
- Library usage and diagnostic telemetry is enabled initially. Inspect it with `mineru telemetry preview`, disable collection and sending with `mineru telemetry disable`, or set `MINERU_TELEMETRY_DISABLED=1` before starting the service. [Telemetry scope and controls](../../next/telemetry-minimum.md)
