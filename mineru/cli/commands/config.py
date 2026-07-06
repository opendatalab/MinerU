"""mineru config — configuration and rule management."""

from __future__ import annotations

import typer
from rich.table import Table

from ...doclib.client import DoclibClient
from ...doclib.types import (
    ConfigResponse,
    ConfigSetRequest,
    ConfigSetResponse,
    ConfigUnsetResponse,
    ConfigValueResponse,
    ExcludeRuleInfo,
    ExcludeRuleListResponse,
    ExcludeRuleRequest,
    ParsingRuleInfo,
    ParsingRuleListResponse,
    ParsingRuleRequest,
    RemoveExcludeRuleResponse,
    RemoveParsingRuleResponse,
)
from ...types import Tier
from ..contracts import CliContext
from ..runtime import run_cli

app = typer.Typer(help="Configuration management", no_args_is_help=True)

exclude_rules_app = typer.Typer(help="Exclude rule management", no_args_is_help=True)
parsing_rules_app = typer.Typer(help="Parsing rule management", no_args_is_help=True)

app.add_typer(exclude_rules_app, name="exclude-rules")
app.add_typer(parsing_rules_app, name="parsing-rules")


@app.command("show")
def config_show(json_mode: bool = typer.Option(False, "--json", help="JSON output")) -> None:
    """Show effective configuration values."""
    ctx = CliContext(json_mode=json_mode)
    run_cli(ctx, lambda: _client().get_config(), render=_render_config)


@app.command("get")
def config_get(
    key: str = typer.Argument(..., help="Configuration key"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Show one effective configuration value."""
    ctx = CliContext(json_mode=json_mode)
    run_cli(ctx, lambda: _client().get_config_key(key), render=_render_config_value)


@app.command("set")
def config_set(
    key: str = typer.Argument(..., help="Configuration key"),
    value: str = typer.Argument(..., help="Configuration value"),
) -> None:
    """Set a configuration override."""
    ctx = CliContext(json_mode=False)
    run_cli(ctx, lambda: _client().set_config(key, ConfigSetRequest(value=value)), render=_render_config_set)


@app.command("unset")
def config_unset(key: str = typer.Argument(..., help="Configuration key")) -> None:
    """Remove a configuration override and fall back to the default."""
    ctx = CliContext(json_mode=False)
    run_cli(ctx, lambda: _client().unset_config(key), render=_render_config_unset)


@exclude_rules_app.command("add")
def exclude_rules_add(
    pattern: str = typer.Argument(..., help="Glob pattern to exclude"),
    priority: int = typer.Option(0, "--priority", help="Rule priority"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Add an exclusion rule."""
    ctx = CliContext(json_mode=json_mode)
    run_cli(
        ctx,
        lambda: _client().add_exclude_rule(ExcludeRuleRequest(pattern=pattern, priority=priority)),
        render=_render_exclude_rule_added,
    )


@exclude_rules_app.command("list")
def exclude_rules_list(json_mode: bool = typer.Option(False, "--json", help="JSON output")) -> None:
    """List exclusion rules."""
    ctx = CliContext(json_mode=json_mode)
    run_cli(ctx, lambda: _client().list_exclude_rules(), render=_render_exclude_rules)


@exclude_rules_app.command("remove")
def exclude_rules_remove(rule_id: int = typer.Argument(..., help="Rule id to remove")) -> None:
    """Remove an exclusion rule."""
    ctx = CliContext(json_mode=False)
    run_cli(ctx, lambda: _client().remove_exclude_rule(rule_id), render=_render_exclude_rule_removed)


@parsing_rules_app.command("add")
def parsing_rules_add(
    pattern: str = typer.Argument(..., help="Glob pattern to match"),
    tier: Tier | None = typer.Option(None, "--tier", help="Parse tier: flash, medium, high, extra_high"),
    pages: str | None = typer.Option(None, "--pages", help="Page range, e.g. all or 1~10"),
    remote: bool = typer.Option(False, "--remote", help="Allow remote parsing"),
    name: str | None = typer.Option(None, "--name", help="Rule name"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Add a parsing rule."""
    ctx = CliContext(json_mode=json_mode)
    run_cli(
        ctx,
        lambda: _client().add_parsing_rule(
            ParsingRuleRequest(pattern=pattern, tier=tier, page_range=pages, remote=remote, name=name)
        ),
        render=_render_parsing_rule_added,
    )


@parsing_rules_app.command("list")
def parsing_rules_list(json_mode: bool = typer.Option(False, "--json", help="JSON output")) -> None:
    """List parsing rules."""
    ctx = CliContext(json_mode=json_mode)
    run_cli(ctx, lambda: _client().list_parsing_rules(), render=_render_parsing_rules)


@parsing_rules_app.command("remove")
def parsing_rules_remove(rule_id: int = typer.Argument(..., help="Rule id to remove")) -> None:
    """Remove a parsing rule."""
    ctx = CliContext(json_mode=False)
    run_cli(ctx, lambda: _client().remove_parsing_rule(rule_id), render=_render_parsing_rule_removed)


def _client() -> DoclibClient:
    return DoclibClient(timeout=30)


def _render_config(data: ConfigResponse) -> Table:
    table = Table(title="Config")
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Source")
    for key in sorted(data.config):
        value = data.config[key]
        source = data.sources.get(key, "default")
        table.add_row(key, value, source)
    return table


def _render_config_value(data: ConfigValueResponse) -> str:
    return f"{data.key} = {data.value}  [{data.source}]"


def _render_config_set(data: ConfigSetResponse) -> str:
    return f"{data.key} = {data.value}  [{data.source}]"


def _render_config_unset(data: ConfigUnsetResponse) -> str:
    action = "removed" if data.removed else "unchanged"
    return f"{data.key} = {data.value}  [{data.source}] ({action})"


def _render_exclude_rule_added(data: ExcludeRuleInfo) -> str:
    return f"Exclude rule added: id={data.id}"


def _render_exclude_rules(data: ExcludeRuleListResponse) -> Table | str:
    if not data.rules:
        return "No exclude rules configured."
    table = Table(title="Exclude Rules")
    table.add_column("ID", justify="right")
    table.add_column("Pattern", style="cyan")
    table.add_column("Priority", justify="right")
    for rule in data.rules:
        table.add_row(str(rule.id), rule.pattern, str(rule.priority))
    return table


def _render_exclude_rule_removed(data: RemoveExcludeRuleResponse) -> str:
    action = "removed" if data.removed else "unchanged"
    return f"Exclude rule {data.rule_id} {action}."


def _render_parsing_rule_added(data: ParsingRuleInfo) -> str:
    return f"Parsing rule added: id={data.id}"


def _render_parsing_rules(data: ParsingRuleListResponse) -> Table | str:
    if not data.rules:
        return "No parsing rules configured."
    table = Table(title="Parsing Rules")
    table.add_column("ID", justify="right")
    table.add_column("Pattern", style="cyan")
    table.add_column("Tier")
    table.add_column("Pages")
    table.add_column("Remote")
    table.add_column("Name")
    for rule in data.rules:
        table.add_row(
            str(rule.id),
            rule.pattern,
            rule.tier or "-",
            rule.page_range or "-",
            "yes" if rule.remote else "no",
            rule.name or "-",
        )
    return table


def _render_parsing_rule_removed(data: RemoveParsingRuleResponse) -> str:
    action = "removed" if data.removed else "unchanged"
    return f"Parsing rule {data.rule_id} {action}."
