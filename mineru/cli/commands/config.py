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
from ...parser.page_range import normalize_page_range_input
from ...types import Tier
from ...utils.i18n import t
from ..contracts import CliContext
from ..runtime import run_cli

app = typer.Typer(help=t("Configuration management"), no_args_is_help=True)

exclude_rules_app = typer.Typer(help=t("Exclude rule management"), no_args_is_help=True)
parsing_rules_app = typer.Typer(help=t("Parsing rule management"), no_args_is_help=True)

app.add_typer(exclude_rules_app, name="exclude-rules")
app.add_typer(parsing_rules_app, name="parsing-rules")


@app.command("show", help=t("Show effective configuration values."))
def config_show(json_mode: bool = typer.Option(False, "--json", help=t("JSON output"))) -> None:
    """Show effective configuration values."""
    ctx = CliContext(json_mode=json_mode)
    run_cli(ctx, lambda: _client().get_config(), render=_render_config)


@app.command("get", help=t("Show one effective configuration value."))
def config_get(
    key: str = typer.Argument(..., help=t("Configuration key")),
    json_mode: bool = typer.Option(False, "--json", help=t("JSON output")),
) -> None:
    """Show one effective configuration value."""
    ctx = CliContext(json_mode=json_mode)
    run_cli(ctx, lambda: _client().get_config_key(key), render=_render_config_value)


@app.command("set", help=t("Set a configuration override."))
def config_set(
    key: str = typer.Argument(..., help=t("Configuration key")),
    value: str = typer.Argument(..., help=t("Configuration value")),
) -> None:
    """Set a configuration override."""
    ctx = CliContext(json_mode=False)
    run_cli(ctx, lambda: _client().set_config(key, ConfigSetRequest(value=value)), render=_render_config_set)


@app.command("unset", help=t("Remove a configuration override and fall back to the default."))
def config_unset(key: str = typer.Argument(..., help=t("Configuration key"))) -> None:
    """Remove a configuration override and fall back to the default."""
    ctx = CliContext(json_mode=False)
    run_cli(ctx, lambda: _client().unset_config(key), render=_render_config_unset)


@exclude_rules_app.command("add", help=t("Add an exclusion rule."))
def exclude_rules_add(
    pattern: str = typer.Argument(..., help=t("Glob pattern to exclude")),
    priority: int = typer.Option(0, "--priority", help=t("Rule priority")),
    json_mode: bool = typer.Option(False, "--json", help=t("JSON output")),
) -> None:
    """Add an exclusion rule."""
    ctx = CliContext(json_mode=json_mode)
    run_cli(
        ctx,
        lambda: _client().add_exclude_rule(ExcludeRuleRequest(pattern=pattern, priority=priority)),
        render=_render_exclude_rule_added,
    )


@exclude_rules_app.command("list", help=t("List exclusion rules."))
def exclude_rules_list(json_mode: bool = typer.Option(False, "--json", help=t("JSON output"))) -> None:
    """List exclusion rules."""
    ctx = CliContext(json_mode=json_mode)
    run_cli(ctx, lambda: _client().list_exclude_rules(), render=_render_exclude_rules)


@exclude_rules_app.command("remove", help=t("Remove an exclusion rule."))
def exclude_rules_remove(rule_id: int = typer.Argument(..., help=t("Rule id to remove"))) -> None:
    """Remove an exclusion rule."""
    ctx = CliContext(json_mode=False)
    run_cli(ctx, lambda: _client().remove_exclude_rule(rule_id), render=_render_exclude_rule_removed)


@parsing_rules_app.command("add", help=t("Add a parsing rule."))
def parsing_rules_add(
    pattern: str = typer.Argument(..., help=t("Glob pattern to match")),
    tier: Tier | None = typer.Option(None, "--tier", help=t("Parse tier: flash, basic, standard, advanced")),
    pages: str | None = typer.Option(None, "--pages", help=t("PDF pages, e.g. all, 1-10 or r3-r1")),
    remote: bool = typer.Option(False, "--remote", help=t("Allow remote parsing")),
    name: str | None = typer.Option(None, "--name", help=t("Rule name")),
    json_mode: bool = typer.Option(False, "--json", help=t("JSON output")),
) -> None:
    """Add a parsing rule."""
    ctx = CliContext(json_mode=json_mode)
    run_cli(
        ctx,
        lambda: _add_parsing_rule(pattern, tier=tier, page_range=pages, remote=remote, name=name),
        render=_render_parsing_rule_added,
    )


def _add_parsing_rule(
    pattern: str, *, tier: Tier | None, page_range: str | None, remote: bool, name: str | None
) -> ParsingRuleInfo:
    """在访问 Doclib 前校验规则中的页码表达式。"""
    page_range = normalize_page_range_input(page_range) or None
    return _client().add_parsing_rule(
        ParsingRuleRequest(pattern=pattern, tier=tier, page_range=page_range, remote=remote, name=name)
    )


@parsing_rules_app.command("list", help=t("List parsing rules."))
def parsing_rules_list(json_mode: bool = typer.Option(False, "--json", help=t("JSON output"))) -> None:
    """List parsing rules."""
    ctx = CliContext(json_mode=json_mode)
    run_cli(ctx, lambda: _client().list_parsing_rules(), render=_render_parsing_rules)


@parsing_rules_app.command("remove", help=t("Remove a parsing rule."))
def parsing_rules_remove(rule_id: int = typer.Argument(..., help=t("Rule id to remove"))) -> None:
    """Remove a parsing rule."""
    ctx = CliContext(json_mode=False)
    run_cli(ctx, lambda: _client().remove_parsing_rule(rule_id), render=_render_parsing_rule_removed)


def _client() -> DoclibClient:
    return DoclibClient(timeout=30)


def _render_config(data: ConfigResponse) -> Table:
    table = Table(title=t("Config"))
    table.add_column(t("Key"), style="cyan")
    table.add_column(t("Value"), style="green")
    table.add_column(t("Source"))
    for key in sorted(data.config):
        value = data.config[key]
        source = data.sources.get(key, t("default"))
        table.add_row(key, value, source)
    return table


def _render_config_value(data: ConfigValueResponse) -> str:
    return t("{key} = {value}  [{source}]", key=data.key, value=data.value, source=data.source)


def _render_config_set(data: ConfigSetResponse) -> str:
    return t("{key} = {value}  [{source}]", key=data.key, value=data.value, source=data.source)


def _render_config_unset(data: ConfigUnsetResponse) -> str:
    action = t("removed") if data.removed else t("unchanged")
    return t("{key} = {value}  [{source}] ({action})", key=data.key, value=data.value, source=data.source, action=action)


def _render_exclude_rule_added(data: ExcludeRuleInfo) -> str:
    return t("Exclude rule added: id={id}", id=data.id)


def _render_exclude_rules(data: ExcludeRuleListResponse) -> Table | str:
    if not data.rules:
        return t("No exclude rules configured.")
    table = Table(title=t("Exclude Rules"))
    table.add_column(t("ID"), justify="right")
    table.add_column(t("Pattern"), style="cyan")
    table.add_column(t("Priority"), justify="right")
    for rule in data.rules:
        table.add_row(str(rule.id), rule.pattern, str(rule.priority))
    return table


def _render_exclude_rule_removed(data: RemoveExcludeRuleResponse) -> str:
    if data.removed:
        return t("Exclude rule {rule_id} removed.", rule_id=data.rule_id)
    return t("Exclude rule {rule_id} unchanged.", rule_id=data.rule_id)


def _render_parsing_rule_added(data: ParsingRuleInfo) -> str:
    return t("Parsing rule added: id={id}", id=data.id)


def _render_parsing_rules(data: ParsingRuleListResponse) -> Table | str:
    if not data.rules:
        return t("No parsing rules configured.")
    table = Table(title=t("Parsing Rules"))
    table.add_column(t("ID"), justify="right")
    table.add_column(t("Pattern"), style="cyan")
    table.add_column(t("Tier"))
    table.add_column(t("Pages"))
    table.add_column(t("Remote"))
    table.add_column(t("Name"))
    for rule in data.rules:
        table.add_row(
            str(rule.id),
            rule.pattern,
            rule.tier or "-",
            rule.page_range or "-",
            t("yes") if rule.remote else t("no"),
            rule.name or "-",
        )
    return table


def _render_parsing_rule_removed(data: RemoveParsingRuleResponse) -> str:
    if data.removed:
        return t("Parsing rule {rule_id} removed.", rule_id=data.rule_id)
    return t("Parsing rule {rule_id} unchanged.", rule_id=data.rule_id)
