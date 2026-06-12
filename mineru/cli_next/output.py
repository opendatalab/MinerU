"""Rich-based output formatting for mineru CLI."""

from __future__ import annotations

import json
import sys
from typing import Any

from pydantic import BaseModel
from pydantic_core import to_json

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    console = Console()
except ImportError:
    console = None


def print_error(msg: str) -> None:
    if console:
        console.print(f"[red]Error:[/red] {msg}")
    else:
        print(f"Error: {msg}", file=sys.stderr)


def print_success(msg: str) -> None:
    if console:
        console.print(f"[green]{msg}[/green]")
    else:
        print(msg)


def print_info(msg: str) -> None:
    if console:
        console.print(f"[dim]{msg}[/dim]")
    else:
        print(msg)


def print_json(data: Any) -> None:
    if isinstance(data, BaseModel):
        print(to_json(data, indent=2).decode("utf-8"))
        return
    print(json.dumps(data, ensure_ascii=False, indent=2))


def _get(data: Any, key: str, default: Any = None) -> Any:
    if isinstance(data, dict):
        return data.get(key, default)
    return getattr(data, key, default)


# ── parse ──────────────────────────────────────────────────────


def format_parse_result(data: Any, json_mode: bool = False) -> None:
    if json_mode:
        print_json(data)
        return

    status = _get(data, "status", "?")
    tip = _get(data, "tip", "")

    if status == "pending" or status == "parsing":
        print_info(f"Parse {status}... {tip}")
    elif status == "failed":
        print_error(f"Parse failed: {_get(data, 'error_code', '?')} — {_get(data, 'error_msg', '')}")
    else:
        print_success(f"Parse complete (tier={_get(data, 'tier', '?')}) {tip}")


# ── search ─────────────────────────────────────────────────────


def format_search_results(data: Any, json_mode: bool = False) -> None:
    results = _get(data, "results", [])
    total = _get(data, "total", 0)

    if json_mode:
        print_json(data)
        return

    if not results:
        print_info("No results found.")
        return

    table = Table(title=f"Search results ({total} total)")
    table.add_column("File", style="cyan", no_wrap=True)
    table.add_column("Tier", style="green", width=10)
    table.add_column("Snippet", style="dim")
    table.add_column("Paths", style="dim")

    for r in results:
        table.add_row(
            _get(r, "title") or _get(r, "filename", "?"),
            _get(r, "tier", ""),
            (_get(r, "snippet", "") or "")[:80],
            ", ".join(_get(r, "paths", [])[:3]),
        )

    if console:
        console.print(table)
    else:
        for r in results:
            print(f"{_get(r, 'title')} [{_get(r, 'tier')}] — {', '.join(_get(r, 'paths', []))}")


# ── server status ──────────────────────────────────────────────


def format_server_status(data: Any, json_mode: bool = False) -> None:
    if json_mode:
        print_json(data)
        return

    if not _get(data, "running"):
        print_info("Server is not running.")
        return

    if console:
        table = Table(title="MinerU Server")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("PID", str(_get(data, "pid", "?")))
        table.add_row("Uptime", f"{_get(data, 'uptime_seconds', 0):.0f}s")
        table.add_row("Socket", _get(data, "socket_path", ""))
        table.add_row("Data dir", _get(data, "data_dir", ""))
        table.add_row("Files tracked", str(_get(data, "files_total", 0)))
        table.add_row("Docs indexed", str(_get(data, "docs_total", 0)))
        table.add_row("Parse queue", str(_get(data, "parse_queue_length", 0)))
        table.add_row("Ingest queue", str(_get(data, "ingest_queue_length", 0)))
        table.add_row("Watches", str(_get(data, "watch_count", 0)))
        console.print(table)

        # parse-server status
        ps_data = _get(data, "parse_server")
        if ps_data:
            ps_table = Table(title="Parse Server")
            ps_table.add_column("Target", style="cyan")
            ps_table.add_column("Healthy", style="green")
            ps_table.add_column("Tiers", style="green")
            for label, key in [("Local", "local"), ("Remote", "remote")]:
                ps = _get(ps_data, key, {})
                if _get(ps, "starting"):
                    healthy_str = "starting"
                elif _get(ps, "healthy"):
                    healthy_str = "yes"
                else:
                    healthy_str = "no"
                tiers_str = ", ".join(_get(ps, "supported_tiers", [])) or "-"
                mode = _get(ps, "mode", "")
                label_str = f"{label} ({mode})" if mode else label
                ps_table.add_row(label_str, healthy_str, tiers_str)
            console.print(ps_table)

        # recent logs
        logs = _get(data, "recent_logs", [])
        if logs:
            log_text = "".join(logs)  # last 100 lines from server
            panel = Panel(log_text.strip() or "(empty)", title="Recent Logs", border_style="dim")
            console.print(panel)
    else:
        print(f"Server running (PID {_get(data, 'pid')}), {_get(data, 'files_total')} files")


# ── info ───────────────────────────────────────────────────────


def format_info(data: Any, json_mode: bool = False) -> None:
    if json_mode:
        print_json(data)
        return

    file_data = _get(data, "file")
    if not file_data:
        print_info("File not found in database.")
        return
    doc_data = _get(data, "doc")

    if console:
        table = Table(title=f"File Info: {_get(file_data, 'filename', '?')}")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Path", _get(file_data, "path", "?"))
        table.add_row("Type", _get(file_data, "ext", "?"))
        table.add_row("Size", _format_bytes(_get(file_data, "size_bytes")))
        table.add_row("SHA-256", (_get(file_data, "sha256") or "")[:16] + "...")
        table.add_row("Page count", str(_get(doc_data, "page_count", "?") if doc_data else "?"))
        table.add_row("Title", (_get(doc_data, "title") if doc_data else None) or "—")
        table.add_row("Author", (_get(doc_data, "author") if doc_data else None) or "—")

        tiers = _get(data, "parsed_tiers", [])
        if tiers:
            tier_str = ", ".join(f"{_get(t, 'tier')}={_get(t, 'status')}" for t in tiers)
            table.add_row("Tiers", tier_str)

        console.print(table)
    else:
        print(f"{_get(file_data, 'filename')} — {(_get(doc_data, 'title') if doc_data else None) or 'No title'}")


# ── helpers ────────────────────────────────────────────────────


def _format_bytes(n: int | None) -> str:
    if n is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n} {unit}"
        n //= 1024
    return f"{n} TB"
