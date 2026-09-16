# Text Read Does Not Use Parsed Content

## Decision

`mineru read <locator>` returns `parse_not_required` when the locator resolves to a text document (`.txt`, `.md`, `.markdown`,
`.csv`, `.rst`, or `.tex`). The error uses type `invalid_request_error`, param `locator`, and tells the caller to read the source
file directly.

The check runs after resolving the document and before selecting a cached tier or reading parse artifacts. Both a root locator and
an explicit `tier:flash` locator therefore return the same error instead of `tier_not_cached` or `not_cached`.

This change does not make `mineru read` read source files and does not change content export endpoints.

## Verification

- Cover all text extensions.
- Cover locators with and without an explicit tier.
- Verify human-readable and JSON CLI errors preserve `parse_not_required` and `param=locator`.
- Preserve existing read behavior for parsed documents.

Refs #5276
