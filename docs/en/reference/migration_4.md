# Migrating from 3.x to 4.0

4.0 changes command entrypoints, dependency groups, configuration, and APIs. Validate in an isolated environment before migrating production callers. AMD and vendor accelerators stay on `mineru<4` using the [legacy guides](../usage/compatibility.md).

## Entrypoints and installation

| Previous usage | 4.0 usage |
| --- | --- |
| `mineru -p input.pdf -o output` | Library: `mineru parse input.pdf --pages all -o output.md`; one-off conversion: `mineru-kit parse input.pdf -o output.md` |
| Directory batch conversion | `mineru-kit parse ./documents -o ./output --format zip` |
| `mineru-gradio` / development-era `mineru-kit gradio` | `mineru-webui` / `mineru-kit webui` |
| Model download option `-m all` | `mineru-kit models download --tier standard` |
| Old HTTP parsing calls | V1 uploads → parse jobs → output files |
| Extras such as `mineru[core]` and `mineru[pipeline]` | Base package, `mineru[torch]`, or `mineru[full]` |

Install stable with `"mineru>=4.0,<5"`, inserting extras after the package name when needed. The package requires Python `>=3.10,<3.15`; optional engines also require suitable wheels and drivers. Stop services in the original environment before upgrading and check `mineru version --json` afterwards. Do not replace legacy platform images or installation commands with mainline 4.0 instructions.

## Tiers, pages, and outputs

- Select parsing quality with `flash/basic/standard/advanced`. Basic uses small models; Standard / Advanced use a VLM. Do not infer new quality defaults from old backend names.
- `mineru parse` uses the library, cache, and continuation protocol, defaulting to the first 10 PDF pages. `mineru-kit parse` and the Python SDK default to all pages.
- PDF ranges use `1-5,8,r3-r1`; `r1` is the last page and `all` selects the whole document. Non-PDF native documents are parsed as a whole.
- `-o` file/directory semantics depend on the entrypoint. `mineru-kit parse` defaults to Markdown for one file; directory or multi-file inputs write to a directory.
- Fields such as `_backend` and `pdf_info` in legacy results are not templates for 4.0. Read actual JSON using the [current output contract](output_files.md). Legacy artifacts are accepted only by explicitly supported compatibility readers; not every old JSON document can be passed to `ParseResult.from_dict()`.

## Configuration and models

Configuration lives in `$MINERU_HOME/config.yaml`, optionally overridden by `MINERU_CONFIG`. Old `mineru.json` configuration is not read. `model.stack`, `MINERU_MODEL_STACK`, and `--stack` have been removed; configure the two components independently:

```yaml
model:
  source: auto
  base_dir: ~/.mineru/models
  small_backend: auto
  vlm:
    engine: auto
```

Automatic selection follows the platform. Explicit `onnx + llama-cpp` can serve lighter environments. Torch and VLM engines can also be combined when their dependencies are available. Apple Silicon defaults to llama.cpp for the VLM; MLX requires manual installation and selection.

Restart relevant services after configuration changes, and download and verify models with matching backend choices. 4.0 has separate small-model bundles: the presence of an old model directory does not prove readiness. See [Model Source](../usage/model_source.md).

## API and WebUI

V1 uses `/v1/health`, `/v1/tiers`, `/v1/uploads`, `/v1/parse/jobs`, and `/v1/files`. Legacy `/file_parse` and `/tasks` are not provided. SDK clients and the WebUI must connect to a V1 service. The parsing API and OpenAI-compatible VLM server are different interfaces; their addresses are not interchangeable.

Existing Gradio clients also need to follow the new event interface; renaming the command alone is insufficient. Retained third-party integration guides describe their original adaptations and do not imply V1 compatibility.

Validate migration with a native document and a text-layer PDF. Check pages, Markdown, structured JSON, and assets, then validate the required quality tiers and API calls on the actual target hardware. See [SDK and API](../usage/sdk_api.md).
