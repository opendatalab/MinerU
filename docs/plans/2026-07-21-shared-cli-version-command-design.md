# Shared CLI Version Command Design

Date: 2026-07-21

## Goal

Add `mineru-kit version` and `mineru-kit --version` with exactly the same text output, JSON output, and exit behavior as the existing `mineru` entry points.

## Design

Extract the existing version command implementation from `mineru/cli/main.py` into a lightweight shared module, `mineru/cli/version_command.py`. The module owns:

- the `VersionInfo` result type;
- MinerU and Python version collection;
- plain-text rendering;
- the `version --json` Typer command callback;
- the eager root `--version` option callback.

Both root applications import and register the same `version_cmd` callback. Both root callbacks use the same eager `show_version` option callback. The main `mineru` callback continues preparing telemetry for normal commands; eager version output exits before telemetry preparation, as it does today. The MinerU Kit callback has no additional side effects.

The shared module uses the existing `CliContext` and `run_cli` path so that `version --json` retains its current serialization behavior. It imports only lightweight CLI runtime code and does not import either root command tree.

## Command Registration

The main CLI continues registering `version` at the end of its ordered command list. MinerU Kit adds `version` after `models` in its ordered command list and registers the shared callback directly:

```python
app.command("version")(version_cmd)
```

Both root applications expose an eager `--version` option with the existing help text, `Show the version and exit.`

## Output Contract

Plain output remains:

```text
MinerU version: <version>
Python version: <version>
```

JSON output remains a single object with `mineru_version` and `python_version` fields. The product label remains `MinerU version` for both executables.

## Error and Import Boundaries

Version collection reads the local package version and current Python version only. It performs no network access and loads no parser, model, legacy router, or server implementation. Existing `mineru-kit` lazy-import boundaries remain intact.

## Verification

Tests verify:

1. Existing `mineru version`, `mineru version --json`, and `mineru --version` behavior is unchanged.
2. MinerU Kit exposes `version` in the final top-level command position and `--version` in root help.
3. `mineru-kit version` matches `mineru version` exactly.
4. `mineru-kit --version` matches its `version` subcommand.
5. Both JSON subcommands emit the same single JSON object.
6. Importing `mineru.kit.main` still avoids legacy router and server implementations.
7. The tracked unit-test suite remains green.
