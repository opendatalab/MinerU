# MinerU Kit Command Registration Design

Date: 2026-07-21

## Goal

Remove duplicate CLI parameter declarations for `mineru-kit parse` and `mineru-kit api-server`. Each command implementation becomes the single source of truth for its Typer arguments, options, defaults, types, help text, validation, and execution.

The user-facing command names and behavior remain unchanged.

## Current Problem

`mineru/kit/main.py` currently defines full Typer callbacks for `parse` and `api-server`, then forwards every value to functions in `mineru/kit/commands/`. The implementation functions define another set of signatures and defaults. Changes therefore need to be applied twice, and the two layers already require `type: ignore` annotations where the entrypoint uses broad strings but the implementation uses typed literals.

The main `mineru` CLI avoids this duplication. Command modules define their Typer parameters directly, and `mineru/cli/main.py` only registers those functions with `app.command()`.

## Design

Move the Typer `Argument` and `Option` metadata for `parse` into `mineru/kit/commands/parse.py`, on `parse_cmd()` itself. Move the corresponding metadata for `api-server` into `mineru/kit/commands/api_server.py`, on `api_server_cmd()` itself.

Register the implementation functions directly in `mineru/kit/main.py`:

```python
app.command("parse")(parse.parse_cmd)
app.command("api-server")(api_server.api_server_cmd)
```

This matches the registration pattern used by the main `mineru` CLI.

The command modules own:

- option names and aliases;
- defaults;
- help text;
- public type annotations;
- validation and error translation;
- command execution.

`mineru/kit/main.py` continues to own the root Typer application, top-level command ordering, and command registration only.

## Scope Boundaries

The `models` command group already owns its module-local Typer app and remains unchanged.

The `router` and `vlm-server` wrappers remain in `mineru/kit/main.py`. They require command-specific `context_settings`, access to `typer.Context`, unknown-option forwarding, and argument reshaping, so direct replacement is not part of this refactor.

No parser behavior, API-server behavior, request schema, or command-line option is added or removed.

## Error Handling

Existing command validation remains in the implementation modules. The API-server wrapper continues translating Click exceptions from the underlying parser API server into MinerU Kit errors. Direct Typer registration must preserve the current exit codes and messages for invalid tiers, conflicting Flash options, invalid languages, and parse failures.

## Verification

Tests verify:

1. `mineru-kit parse --help` and `mineru-kit api-server --help` retain their arguments, options, defaults, and help text.
2. Parse command values reach the existing implementation behavior.
3. API-server options, including the single startup tier and `--no-flash`, reach the underlying Click command unchanged.
4. Invalid API-server startup tiers and `--tier flash --no-flash` retain structured CLI failures.
5. Root command order remains unchanged.
6. The tracked unit-test suite remains green.
