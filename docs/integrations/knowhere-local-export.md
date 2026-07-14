# Knowhere Local Export Integration

## Status and baseline

This document defines the versioned artifact boundary used by Knowhere's
standalone Codex review-package exporter. The implementation baseline is
MinerU `79d6d8d79fb8f3ddba5cc34c07a16f0ec36f56c7` (version 3.4.4) on branch
`feat/kiwi-shane/knowhere-local-export-adapter`.

The adapter runs inside the MinerU environment and calls
`mineru.cli.common.do_parse()` directly. Knowhere invokes the adapter as a
separate process and consumes files from the output directory. Neither
repository imports the other's Python modules.

## Process and artifact contract

The `mineru-knowhere-export` command accepts one local PDF or DOCX and writes a
`knowhere-mineru-artifacts/1.0` manifest only after all required artifacts have
been validated. Required files are Markdown, `middle.json`,
`content_list.json`, and `content_list_v2.json`; an images directory is also
declared. Manifest paths are relative to the output root, hashes are SHA-256,
and paths that are absolute or escape the root are invalid.

The adapter does not start `mineru-api`, a FastAPI server, or any other HTTP
service. It does not require `MINERU_API_KEYS`. Model output and normalized
source-file dumps are disabled for this integration.

## Local and offline behavior

The default backend is the local `pipeline` backend. Offline mode rejects HTTP
client backends and remote server URLs, sets the supported Hugging Face and
Transformers offline environment flags, and requires models to have been
downloaded before the run. These application flags prevent intended download
paths; they do not prove that the host had no network access. Therefore the
manifest keeps `offline_verified` false unless an external network-denial
control independently verifies the run.

## DOCX handling

MinerU parses DOCX through its Office backend and writes artifacts below the
effective `office` parse directory. Office logical pages are parser-derived
navigation locators. They must not be represented as stable physical DOCX
pages or silently mapped to pages produced later by LibreOffice.

## Non-goals

This integration does not replace MinerU's public CLI or API, change Knowhere's
production ingestion path, add semantic page selection or summaries, or draw
regulatory, quality, approval, compliance, equivalence, or Pass/Fail
conclusions.

## Licensing and attribution

No MinerU source is copied into Knowhere. MinerU remains governed by
`LICENSE.md`, including its additional terms. Deployments that expose an online
service based on MinerU must evaluate the applicable attribution obligation and
other deployment-specific licensing requirements; this document is not legal
advice.

## Implementation notes and deviations

- The observed baseline matches the planned snapshot.
- `content_list_v2.json` is a page-indexed outer array for pipeline and Office
  output, including empty page arrays.
- On the Windows implementation host, the existing model E2E test is not a
  baseline unit test because it loads local models. Baseline verification used
  compilation and imports of `do_parse`, `read_fn`, and `resolve_parse_dir`.

