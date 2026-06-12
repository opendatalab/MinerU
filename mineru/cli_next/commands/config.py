"""mineru config — configuration management."""

from __future__ import annotations

import typer

from ...doclib.client import DoclibClient
from ...doclib.types import ConfigResponse, ConfigSetRequest, ExcludeRuleRequest, ParsingRuleRequest, WatchRequest
from ...types import Tier
from ..output import print_error, print_success, print_info, print_json

app = typer.Typer(help="Configuration management", no_args_is_help=True)

# sub-command groups
watch_app = typer.Typer(help="Watch directory management", no_args_is_help=True)
exclude_app = typer.Typer(help="Exclusion rule management", no_args_is_help=True)
parsing_rules_app = typer.Typer(help="Parsing rule management", no_args_is_help=True)

app.add_typer(watch_app, name="watch")
app.add_typer(exclude_app, name="exclude")
app.add_typer(parsing_rules_app, name="parsing-rules")


# ── config show ─────────────────────────────────────────────────


@app.command()
def show(json_mode: bool = typer.Option(False, "--json", help="JSON output")) -> None:
    """Show all configuration."""
    try:
        client = DoclibClient(timeout=10)
        data = client.get_config()
        if json_mode:
            print_json(data)
        else:
            _print_config(data)
    except Exception as exc:
        print_error(str(exc))
        raise typer.Exit(1) from None


# ── watch ────────────────────────────────────────────────────────


@watch_app.command("add")
def watch_add(
    path: str = typer.Argument(..., help="Directory path to watch"),
    removable: bool = typer.Option(False, "--removable", help="Removable device"),
    label: str = typer.Option(None, "--label", help="Label for this watch"),
) -> None:
    """Add a directory to watch."""
    try:
        client = DoclibClient(timeout=10)
        data = client.add_watch(WatchRequest(path=path, removable=removable, label=label))
        print_success(f"Watch added: {data.path} (id={data.id})")
    except Exception as exc:
        print_error(str(exc))
        raise typer.Exit(1) from None


@watch_app.command("list")
def watch_list(json_mode: bool = typer.Option(False, "--json", help="JSON output")) -> None:
    """List all watched directories."""
    try:
        client = DoclibClient(timeout=10)
        data = client.list_watches()
        if json_mode:
            print_json(data)
        else:
            watches = data.watches
            if not watches:
                print_info("No watches configured.")
            for w in watches:
                status = w.watch_status
                icon = "✓" if status == "active" else "✗"
                extra = " [removable]" if w.removable else ""
                print(f"  {icon} {w.path}{extra}  [{status}]")
    except Exception as exc:
        print_error(str(exc))
        raise typer.Exit(1) from None


@watch_app.command("rm")
def watch_rm(path: str = typer.Argument(..., help="Watch directory path to remove")) -> None:
    """Remove a watched directory."""
    try:
        client = DoclibClient(timeout=10)
        client.remove_watch(path)
        print_success(f"Watch removed: {path}")
    except Exception as exc:
        print_error(str(exc))
        raise typer.Exit(1) from None


# ── exclude ──────────────────────────────────────────────────────


@exclude_app.command("add")
def exclude_add(
    pattern: str = typer.Argument(..., help="Glob pattern to exclude"),
    priority: int = typer.Option(0, "--priority", help="Rule priority"),
) -> None:
    """Add an exclusion pattern."""
    try:
        client = DoclibClient(timeout=10)
        data = client.add_exclude_rule(ExcludeRuleRequest(pattern=pattern, priority=priority))
        print_success(f"Exclude rule added: id={data.id}")
    except Exception as exc:
        print_error(str(exc))
        raise typer.Exit(1) from None


@exclude_app.command("list")
def exclude_list(
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """List all exclusion rules."""
    try:
        client = DoclibClient(timeout=10)
        data = client.list_exclude_rules()
        if json_mode:
            print_json(data)
        else:
            rules = data.rules
            if not rules:
                print_info("No exclude rules configured.")
            for r in rules:
                print(f"  [{r.id}] {r.pattern}  priority={r.priority}")
    except Exception as exc:
        print_error(str(exc))
        raise typer.Exit(1) from None


@exclude_app.command("rm")
def exclude_rm(rule_id: int = typer.Argument(..., help="Rule ID to remove")) -> None:
    """Remove an exclusion rule."""
    try:
        client = DoclibClient(timeout=10)
        client.remove_exclude_rule(rule_id)
        print_success(f"Exclude rule {rule_id} removed.")
    except Exception as exc:
        print_error(str(exc))
        raise typer.Exit(1) from None


# ── parsing-rules ────────────────────────────────────────────────


@parsing_rules_app.command("add")
def parsing_rules_add(
    pattern: str = typer.Argument(..., help="Glob pattern to match"),
    tier: Tier | None = typer.Option(None, "--tier", help="Parse tier: flash, standard, pro"),
    pages: str = typer.Option(None, "--pages", help="Page range, e.g. 'all' or '1~10'"),
    remote: bool = typer.Option(False, "--remote", help="Allow remote parsing"),
    name: str = typer.Option(None, "--name", help="Rule name"),
) -> None:
    """Add a parsing rule."""
    try:
        client = DoclibClient(timeout=10)
        data = client.add_parsing_rule(
            ParsingRuleRequest(pattern=pattern, tier=tier, pages=pages, remote=remote, name=name)
        )
        print_success(f"Parsing rule added: id={data.id}")
    except Exception as exc:
        print_error(str(exc))
        raise typer.Exit(1) from None


@parsing_rules_app.command("list")
def parsing_rules_list(
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """List all parsing rules."""
    try:
        client = DoclibClient(timeout=10)
        data = client.list_parsing_rules()
        if json_mode:
            print_json(data)
        else:
            rules = data.rules
            if not rules:
                print_info("No parsing rules configured.")
            for r in rules:
                flags = []
                if r.tier:
                    flags.append(f"tier={r.tier}")
                if r.pages:
                    flags.append(f"pages={r.pages}")
                if r.remote:
                    flags.append("remote")
                print(f"  [{r.id}] {r.pattern}  {', '.join(flags)}")
    except Exception as exc:
        print_error(str(exc))
        raise typer.Exit(1) from None


@parsing_rules_app.command("rm")
def parsing_rules_rm(rule_id: int = typer.Argument(..., help="Rule ID to remove")) -> None:
    """Remove a parsing rule."""
    try:
        client = DoclibClient(timeout=10)
        client.remove_parsing_rule(rule_id)
        print_success(f"Parsing rule {rule_id} removed.")
    except Exception as exc:
        print_error(str(exc))
        raise typer.Exit(1) from None


parse_server_app = typer.Typer(help="Parse-server configuration", no_args_is_help=True)
app.add_typer(parse_server_app, name="parse-server")


# ── parse-server config ──────────────────────────────────────────


@parse_server_app.command("local.mode")
def parse_server_local_mode(
    mode: str = typer.Argument(..., help="Mode: disabled, managed, self_hosted"),
) -> None:
    """Set local parse-server mode."""
    valid = {"disabled", "managed", "self_hosted"}
    if mode not in valid:
        print_error(f"Invalid mode '{mode}'. Must be one of: {', '.join(sorted(valid))}")
        raise typer.Exit(1)
    try:
        client = DoclibClient(timeout=10)
        client.set_config(ConfigSetRequest(key="parse_server.local.mode", value=mode))
        print_success(f"Local parse-server mode set to '{mode}'.")
    except Exception as exc:
        print_error(str(exc))
        raise typer.Exit(1) from None


@parse_server_app.command("local.managed-tier")
def parse_server_managed_tier(
    tier: Tier = typer.Argument(..., help="Tier: standard, pro"),
) -> None:
    """Set tier for managed local parse-server."""
    if tier not in ("standard", "pro"):
        print_error("Tier must be 'standard' or 'pro'.")
        raise typer.Exit(1)
    try:
        client = DoclibClient(timeout=10)
        client.set_config(ConfigSetRequest(key="parse_server.local.managed_tier", value=tier))
        print_success(f"Managed tier set to '{tier}'.")
    except Exception as exc:
        print_error(str(exc))
        raise typer.Exit(1) from None


@parse_server_app.command("local.self-hosted-url")
def parse_server_self_hosted_url(
    url: str = typer.Argument(..., help="HTTP URL of self-hosted parse-server"),
) -> None:
    """Set URL for self-hosted parse-server."""
    try:
        client = DoclibClient(timeout=10)
        client.set_config(ConfigSetRequest(key="parse_server.local.self_hosted_url", value=url))
        print_success(f"Self-hosted URL set to '{url}'.")
    except Exception as exc:
        print_error(str(exc))
        raise typer.Exit(1) from None


@parse_server_app.command("local.self-hosted-api-key")
def parse_server_self_hosted_api_key(
    api_key: str = typer.Argument(..., help="API key for self-hosted parse-server"),
) -> None:
    """Set API key for self-hosted parse-server."""
    try:
        client = DoclibClient(timeout=10)
        client.set_config(ConfigSetRequest(key="parse_server.local.self_hosted_api_key", value=api_key))
        print_success("Self-hosted API key set.")
    except Exception as exc:
        print_error(str(exc))
        raise typer.Exit(1) from None


@parse_server_app.command("remote.url")
def parse_server_remote_url(
    url: str = typer.Argument(..., help="Remote parse-server URL (default: https://mineru.net/api)"),
) -> None:
    """Set remote parse-server URL."""
    try:
        client = DoclibClient(timeout=10)
        client.set_config(ConfigSetRequest(key="parse_server.remote.url", value=url))
        print_success(f"Remote URL set to '{url}'.")
    except Exception as exc:
        print_error(str(exc))
        raise typer.Exit(1) from None


@parse_server_app.command("remote.api-key")
def parse_server_remote_api_key(
    api_key: str = typer.Argument(..., help="API key for remote parse-server"),
) -> None:
    """Set API key for remote parse-server."""
    try:
        client = DoclibClient(timeout=10)
        client.set_config(ConfigSetRequest(key="parse_server.remote.api_key", value=api_key))
        print_success("Remote API key set.")
    except Exception as exc:
        print_error(str(exc))
        raise typer.Exit(1) from None


# ── helpers ──────────────────────────────────────────────────────


def _print_config(data: ConfigResponse) -> None:
    cfg = data.config

    print("\n[Global Config]")
    for k, v in cfg.items():
        print(f"  {k} = {v}")
