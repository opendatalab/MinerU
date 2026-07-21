# MinerU Kit Command Registration Design

Date: 2026-07-21

## Goal

Remove duplicate CLI parameter declarations for the `mineru-kit parse`, `api-server`, `vlm-server`, and `router` commands. Each command implementation becomes the single source of truth for its Typer arguments, options, defaults, types, help text, validation, and execution.

The user-facing command names and behavior remain unchanged.

## Current Problem

`mineru/kit/main.py` currently defines full Typer callbacks for the four commands, then forwards every value to functions in `mineru/kit/commands/`. The implementation functions define another set of signatures and defaults. Changes therefore need to be applied twice, and the two layers already require `type: ignore` annotations where the entrypoint uses broad strings but the implementation uses typed literals.

The `vlm-server` and `router` wrappers additionally configure unknown-option forwarding through Click context settings. The router wrapper also renames its repeatable `upstream_url` CLI value to the implementation's `upstream_urls` parameter. These responsibilities can remain intact without retaining duplicate callbacks.

The main `mineru` CLI avoids this duplication. Command modules define their Typer parameters directly, and `mineru/cli/main.py` only registers those functions with `app.command()`.

## Design

Move the Typer `Argument` and `Option` metadata for each command into its corresponding function under `mineru/kit/commands/`.

The `vlm-server` and `router` implementation functions accept `typer.Context` directly and continue reading `ctx.args` for unknown arguments. Their `context_settings` stay on the registration calls in `mineru/kit/main.py`, because those settings configure Click command construction rather than command behavior. The router implementation adopts the public CLI name `upstream_url`, eliminating the wrapper-only singular-to-plural translation.

Register the implementation functions directly in `mineru/kit/main.py`:

```python
app.command("parse")(parse.parse_cmd)
app.command("api-server")(api_server.api_server_cmd)
app.command("vlm-server", context_settings=FORWARD_CONTEXT_SETTINGS)(vlm_server.vlm_server_cmd)
app.command("router", context_settings=FORWARD_CONTEXT_SETTINGS)(router.router_cmd)
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

The `models` command group already owns its module-local Typer app and remains unchanged. The root application, ordered command group, command ordering, and forwarding context settings remain in `mineru/kit/main.py`.

No parser behavior, API-server behavior, request schema, or command-line option is added or removed.

## Error Handling

Existing command validation remains in the implementation modules. The API-server implementation continues translating Click exceptions from the underlying parser API server into MinerU Kit errors. Direct Typer registration must preserve the current exit codes and messages for invalid tiers, conflicting Flash options, invalid languages, and parse failures.

## Verification

Tests verify:

1. All four directly registered commands retain their arguments, options, defaults, and help text.
2. Parse command values reach the existing implementation behavior.
3. API-server options, including the single startup tier and `--no-flash`, reach the underlying Click command unchanged.
4. Invalid API-server startup tiers and `--tier flash --no-flash` retain structured CLI failures.
5. `vlm-server` and `router` continue receiving unknown arguments through `ctx.args`.
6. Router options, including repeated `--upstream-url`, reach the legacy router unchanged.
7. Root command order remains unchanged.
8. The tracked unit-test suite remains green.
