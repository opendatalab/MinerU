# Reference Documentation

## SDK and HTTP interfaces {#table-of-contents}

| Interface | When to use it | Reference |
| --- | --- | --- |
| Python SDK | Parse locally or connect to a service from Python and receive a `ParseResult` | [Examples and parameters](../usage/sdk_api.md) |
| Tasks API | Submit one or more files, track their status, cancel work, and download per-file results | [First HTTP request](../usage/http_api.md) · [Fields, states, and errors](../usage/tasks.md) |
| file-parse | Upload one file to the separate service, wait for an attachment, or resume with a request ID | [Form fields and recovery](../usage/file_parse.md) |

## Outputs and developer reference {#documentation-overview}

[Output formats](output_files.md){#output-files-documentation} lists downloads by interface, ZIP members, and how to save assets. The [developer reference](development.md) links to the Middle JSON specification, detailed CLI and SDK contracts, architecture, and maintenance decisions.

## Releases and migration {#changelog}

See the [changelog](changelog.md), [3.x → 4.0](migration_4.md), [4.0.x → 4.1](migration_4_1.md), and [legacy platform guides](../usage/compatibility.md).
