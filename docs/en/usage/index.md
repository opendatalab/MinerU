# Usage Guide

## Choose a workflow {#table-of-contents}

| What do you need? | Workflow |
| --- | --- |
| Markdown, Middle JSON, or a ZIP from one file or a folder | Follow the [first conversion](../quick_start/index.md) with `mineru-kit parse`. It processes all PDF pages by default and writes to the path supplied with `-o`. |
| A browser view to check recognition against the source | Open the [WebUI](webui.md), select a file and page range, and click either the source or the result to see its counterpart. |
| A searchable library with cached results | Add a file with `mineru parse`, then search and read it with the [document library](quick_usage.md). |
| Parsing inside an application | Use the [local or remote Python SDK](sdk_api.md), [asynchronous Tasks](tasks.md), or the [single-file synchronous API](file_parse.md). |
| An existing application or workflow | See [third-party integrations](integrations.md) for the n8n download template and the scope of historical tutorials. |

## Options and troubleshooting {#getting-started}

Look up export and wait options in the [CLI guide](cli_tools.md) and available files in [output formats](../reference/output_files.md). For models, API servers, and multi-GPU workers, start with [deployment](../deployment/index.md). [FAQ](../faq/index.md)
