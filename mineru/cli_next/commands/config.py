"""mineru config — configuration and rule management."""

from __future__ import annotations

import typer

from ...doclib.client import DoclibClient
from ...doclib.types import ConfigResponse, ConfigSetRequest, ExcludeRuleRequest, ParsingRuleRequest
from ...types import Tier
from ..output import print_error, print_info, print_json, print_success

app = typer.Typer(help="Configuration management", no_args_is_help=True)

exclude_rules_app = typer.Typer(help="Exclude rule management", no_args_is_help=True)
parsing_rules_app = typer.Typer(help="Parsing rule management", no_args_is_help=True)

app.add_typer(exclude_rules_app, name="exclude-rules")
app.add_typer(parsing_rules_app, name="parsing-rules")


@app.command("show")
def config_show(json_mode: bool = typer.Option(False, "--json", help="JSON output")) -> None:
    """Show effective configuration values."""
    try:
        data = _client().get_config()
    except Exception as exc:
        print_error(str(exc))
        raise typer.Exit(1) from None

    if json_mode:
        print_json(data)
        return
    _print_config(data)


@app.command("get")
def config_get(
    key: str = typer.Argument(..., help="Configuration key"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Show one effective configuration value."""
    try:
        data = _client().get_config_key(key)
    except Exception as exc:
        print_error(str(exc))
        raise typer.Exit(1) from None

    if json_mode:
        print_json(data)
        return
    print(f"{data.key} = {data.value}  [{data.source}]")


@app.command("set")
def config_set(
    key: str = typer.Argument(..., help="Configuration key"),
    value: str = typer.Argument(..., help="Configuration value"),
) -> None:
    """Set a configuration override."""
    try:
        data = _client().set_config(key, ConfigSetRequest(value=value))
    except Exception as exc:
        print_error(str(exc))
        raise typer.Exit(1) from None

    print_success(f"{data.key} = {data.value}  [{data.source}]")


@app.command("unset")
def config_unset(key: str = typer.Argument(..., help="Configuration key")) -> None:
    """Remove a configuration override and fall back to the default."""
    try:
        data = _client().unset_config(key)
    except Exception as exc:
        print_error(str(exc))
        raise typer.Exit(1) from None

    action = "removed" if data.removed else "unchanged"
    print_success(f"{data.key} = {data.value}  [{data.source}] ({action})")


@exclude_rules_app.command("add")
def exclude_rules_add(
    pattern: str = typer.Argument(..., help="Glob pattern to exclude"),
    priority: int = typer.Option(0, "--priority", help="Rule priority"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Add an exclusion rule."""
    try:
        data = _client().add_exclude_rule(ExcludeRuleRequest(pattern=pattern, priority=priority))
    except Exception as exc:
        print_error(str(exc))
        raise typer.Exit(1) from None

    if json_mode:
        print_json(data)
        return
    print_success(f"Exclude rule added: id={data.id}")


@exclude_rules_app.command("list")
def exclude_rules_list(json_mode: bool = typer.Option(False, "--json", help="JSON output")) -> None:
    """List exclusion rules."""
    try:
        data = _client().list_exclude_rules()
    except Exception as exc:
        print_error(str(exc))
        raise typer.Exit(1) from None

    if json_mode:
        print_json(data)
        return
    if not data.rules:
        print_info("No exclude rules configured.")
        return
    for rule in data.rules:
        print(f"  [{rule.id}] {rule.pattern}  priority={rule.priority}")


@exclude_rules_app.command("remove")
def exclude_rules_remove(rule_id: int = typer.Argument(..., help="Rule id to remove")) -> None:
    """Remove an exclusion rule."""
    try:
        _client().remove_exclude_rule(rule_id)
    except Exception as exc:
        print_error(str(exc))
        raise typer.Exit(1) from None

    print_success(f"Exclude rule {rule_id} removed.")


@parsing_rules_app.command("add")
def parsing_rules_add(
    pattern: str = typer.Argument(..., help="Glob pattern to match"),
    tier: Tier | None = typer.Option(None, "--tier", help="Parse tier: flash, standard, pro"),
    pages: str | None = typer.Option(None, "--pages", help="Page range, e.g. all or 1~10"),
    remote: bool = typer.Option(False, "--remote", help="Allow remote parsing"),
    name: str | None = typer.Option(None, "--name", help="Rule name"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Add a parsing rule."""
    try:
        data = _client().add_parsing_rule(
            ParsingRuleRequest(pattern=pattern, tier=tier, page_range=pages, remote=remote, name=name)
        )
    except Exception as exc:
        print_error(str(exc))
        raise typer.Exit(1) from None

    if json_mode:
        print_json(data)
        return
    print_success(f"Parsing rule added: id={data.id}")


@parsing_rules_app.command("list")
def parsing_rules_list(json_mode: bool = typer.Option(False, "--json", help="JSON output")) -> None:
    """List parsing rules."""
    try:
        data = _client().list_parsing_rules()
    except Exception as exc:
        print_error(str(exc))
        raise typer.Exit(1) from None

    if json_mode:
        print_json(data)
        return
    if not data.rules:
        print_info("No parsing rules configured.")
        return
    for rule in data.rules:
        flags = []
        if rule.tier:
            flags.append(f"tier={rule.tier}")
        if rule.page_range:
            flags.append(f"pages={rule.page_range}")
        if rule.remote:
            flags.append("remote")
        suffix = f"  {', '.join(flags)}" if flags else ""
        print(f"  [{rule.id}] {rule.pattern}{suffix}")


@parsing_rules_app.command("remove")
def parsing_rules_remove(rule_id: int = typer.Argument(..., help="Rule id to remove")) -> None:
    """Remove a parsing rule."""
    try:
        _client().remove_parsing_rule(rule_id)
    except Exception as exc:
        print_error(str(exc))
        raise typer.Exit(1) from None

    print_success(f"Parsing rule {rule_id} removed.")


def _client() -> DoclibClient:
    return DoclibClient(timeout=30)


def _print_config(data: ConfigResponse) -> None:
    print("\n[Config]")
    for key in sorted(data.config):
        value = data.config[key]
        source = data.sources.get(key, "default")
        print(f"  {key} = {value}  [{source}]")
