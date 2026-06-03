"""Rich-based output formatting for mineru CLI."""

from __future__ import annotations

import json
import sys

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


def print_json(data: dict) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


# ── parse ──────────────────────────────────────────────────────


def format_parse_result(data: dict, json_mode: bool = False) -> None:
    if json_mode:
        print_json(data)
        return

    status = data.get("status", "?")
    tip = data.get("tip", "")

    if status == "pending" or status == "parsing":
        print_info(f"Parse {status}... {tip}")
    elif status == "failed":
        err = data.get("error", {})
        print_error(f"Parse failed: {err.get('code','?')} — {err.get('message','')}")
    else:
        print_success(f"Parse complete (tier={data.get('tier','?')}) {tip}")


# ── search ─────────────────────────────────────────────────────


def format_search_results(data: dict, json_mode: bool = False) -> None:
    results = data.get("results", [])
    total = data.get("total", 0)

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
            r.get("title") or r.get("filename", "?"),
            r.get("tier", ""),
            (r.get("snippet") or "")[:80],
            ", ".join(r.get("paths", [])[:3]),
        )

    if console:
        console.print(table)
    else:
        for r in results:
            print(f"{r.get('title')} [{r.get('tier')}] — {', '.join(r.get('paths', []))}")


# ── server status ──────────────────────────────────────────────


def format_server_status(data: dict, json_mode: bool = False) -> None:
    if json_mode:
        print_json(data)
        return

    if not data.get("running"):
        print_info("Server is not running.")
        return

    if console:
        table = Table(title="MinerU Server")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("PID", str(data.get("pid", "?")))
        table.add_row("Uptime", f"{data.get('uptime_seconds', 0):.0f}s")
        table.add_row("Socket", data.get("socket_path", ""))
        table.add_row("Data dir", data.get("data_dir", ""))
        table.add_row("Files tracked", str(data.get("files_total", 0)))
        table.add_row("Docs indexed", str(data.get("docs_total", 0)))
        table.add_row("Parse queue", str(data.get("parse_queue_length", 0)))
        table.add_row("Reg queue", str(data.get("reg_queue_length", 0)))
        table.add_row("Watches", str(data.get("watch_count", 0)))
        console.print(table)
    else:
        print(f"Server running (PID {data.get('pid')}), {data.get('files_total')} files")


# ── info ───────────────────────────────────────────────────────


def format_info(data: dict, json_mode: bool = False) -> None:
    if json_mode:
        print_json(data)
        return

    if not data.get("found"):
        print_info("File not found in database.")
        return

    if console:
        table = Table(title=f"File Info: {data.get('filename','?')}")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Path", data.get("path", "?"))
        table.add_row("Type", data.get("ext", "?"))
        table.add_row("Size", _format_bytes(data.get("size_bytes")))
        table.add_row("SHA-256", (data.get("sha256") or "")[:16] + "...")
        table.add_row("Page count", str(data.get("page_count", "?")))
        table.add_row("Title", data.get("title") or "—")
        table.add_row("Author", data.get("author") or "—")

        tiers = data.get("tiers", [])
        if tiers:
            tier_str = ", ".join(f"{t['tier']}={t['status']}" for t in tiers)
            table.add_row("Tiers", tier_str)

        console.print(table)
    else:
        print(f"{data.get('filename')} — {data.get('title') or 'No title'}")


# ── helpers ────────────────────────────────────────────────────


def _format_bytes(n: int | None) -> str:
    if n is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n} {unit}"
        n //= 1024
    return f"{n} TB"
