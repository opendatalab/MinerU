"""mineru config — configuration management."""

from __future__ import annotations

import typer

from mineru.doclib.client import MineruClient
from mineru.cli_next.output import print_error, print_success, print_info, print_json

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
        client = MineruClient(timeout=10)
        data = client.config_show()
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
        client = MineruClient(timeout=10)
        data = client.config_watch_add(path, removable=removable, label=label)
        print_success(f"Watch added: {data.get('path')} (id={data.get('id')})")
    except Exception as exc:
        print_error(str(exc))
        raise typer.Exit(1) from None


@watch_app.command("list")
def watch_list(json_mode: bool = typer.Option(False, "--json", help="JSON output")) -> None:
    """List all watched directories."""
    try:
        client = MineruClient(timeout=10)
        data = client.config_watch_list()
        if json_mode:
            print_json(data)
        else:
            watches = data.get("watches", [])
            if not watches:
                print_info("No watches configured.")
            for w in watches:
                status = w.get("watch_status", "?")
                icon = "✓" if status == "active" else "✗"
                extra = " [removable]" if w.get("removable") else ""
                print(f"  {icon} {w['path']}{extra}  [{status}]")
    except Exception as exc:
        print_error(str(exc))
        raise typer.Exit(1) from None


@watch_app.command("rm")
def watch_rm(path: str = typer.Argument(..., help="Watch directory path to remove")) -> None:
    """Remove a watched directory."""
    try:
        client = MineruClient(timeout=10)
        client.config_watch_rm(path)
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
        client = MineruClient(timeout=10)
        data = client.config_exclude_add(pattern, priority=priority)
        print_success(f"Exclude rule added: id={data.get('id')}")
    except Exception as exc:
        print_error(str(exc))
        raise typer.Exit(1) from None


@exclude_app.command("list")
def exclude_list(
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """List all exclusion rules."""
    try:
        client = MineruClient(timeout=10)
        data = client.config_exclude_list()
        if json_mode:
            print_json(data)
        else:
            rules = data.get("rules", [])
            if not rules:
                print_info("No exclude rules configured.")
            for r in rules:
                print(f"  [{r['id']}] {r['pattern']}  priority={r.get('priority', 0)}")
    except Exception as exc:
        print_error(str(exc))
        raise typer.Exit(1) from None


@exclude_app.command("rm")
def exclude_rm(rule_id: int = typer.Argument(..., help="Rule ID to remove")) -> None:
    """Remove an exclusion rule."""
    try:
        client = MineruClient(timeout=10)
        client.config_exclude_rm(rule_id)
        print_success(f"Exclude rule {rule_id} removed.")
    except Exception as exc:
        print_error(str(exc))
        raise typer.Exit(1) from None


# ── parsing-rules ────────────────────────────────────────────────


@parsing_rules_app.command("add")
def parsing_rules_add(
    pattern: str = typer.Argument(..., help="Glob pattern to match"),
    tier: str = typer.Option(None, "--tier", help="Parse tier: flash, standard, pro"),
    pages: str = typer.Option(None, "--pages", help="Page range, e.g. 'all' or '1~10'"),
    remote: bool = typer.Option(False, "--remote", help="Allow remote parsing"),
    name: str = typer.Option(None, "--name", help="Rule name"),
) -> None:
    """Add a parsing rule."""
    try:
        client = MineruClient(timeout=10)
        data = client.config_parsing_rules_add(
            pattern, tier=tier, pages=pages, remote=remote, name=name,
        )
        print_success(f"Parsing rule added: id={data.get('id')}")
    except Exception as exc:
        print_error(str(exc))
        raise typer.Exit(1) from None


@parsing_rules_app.command("list")
def parsing_rules_list(
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """List all parsing rules."""
    try:
        client = MineruClient(timeout=10)
        data = client.config_parsing_rules_list()
        if json_mode:
            print_json(data)
        else:
            rules = data.get("rules", [])
            if not rules:
                print_info("No parsing rules configured.")
            for r in rules:
                flags = []
                if r.get("tier"):
                    flags.append(f"tier={r['tier']}")
                if r.get("pages"):
                    flags.append(f"pages={r['pages']}")
                if r.get("remote"):
                    flags.append("remote")
                print(f"  [{r['id']}] {r['pattern']}  {', '.join(flags)}")
    except Exception as exc:
        print_error(str(exc))
        raise typer.Exit(1) from None


@parsing_rules_app.command("rm")
def parsing_rules_rm(rule_id: int = typer.Argument(..., help="Rule ID to remove")) -> None:
    """Remove a parsing rule."""
    try:
        client = MineruClient(timeout=10)
        client.config_parsing_rules_rm(rule_id)
        print_success(f"Parsing rule {rule_id} removed.")
    except Exception as exc:
        print_error(str(exc))
        raise typer.Exit(1) from None


# ── helpers ──────────────────────────────────────────────────────


def _print_config(data: dict) -> None:
    cfg = data.get("config", {})
    watches = data.get("watches", [])
    rules = data.get("rules", [])

    print("\n[Global Config]")
    for k, v in cfg.items():
        print(f"  {k} = {v}")

    print(f"\n[Watches] ({len(watches)})")
    for w in watches:
        print(f"  {w['path']} [{w.get('watch_status','?')}]")

    print(f"\n[Rules] ({len(rules)})")
    for r in rules:
        print(f"  [{r['rule_type']}] {r['pattern']}")
