# Disable Implicit CLI Markup Design

## Problem

Rich parses plain string renderables as markup by default. CLI tables, dynamic titles, notices, warnings, and shared output helpers can receive paths, metadata, rules, or service messages containing bracketed text. A malformed closing tag raises `MarkupError`; a valid tag changes presentation unexpectedly.

## Decision

Construct the shared stdout and stderr Rich consoles with `markup=False`. Plain strings are always treated as literal text, while explicit Rich renderables such as `Text`, `Table`, and `Panel` retain their assigned styles.

Keep the server log panel's explicit `Text` wrapper so that its renderer remains safe when consumed by a different Rich console.

Commands whose renderers return ordinary strings, including `search` and `find`, continue to use Python `print()` and are unaffected.

## Tests

Verify markup-like content renders literally through both shared consoles, through table cells and titles, and through notice output. Verify an explicitly styled `Text` renderable remains supported.
