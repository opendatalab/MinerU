# Tier Naming Design

Status: Approved

Date: 2026-07-21

## Goal

Replace the public parsing Tier vocabulary with four user-facing levels:

| Public literal | Display name | Chinese display name | Hybrid effort |
|---|---|---|---|
| `flash` | Flash | 极速解析 | N/A |
| `basic` | Basic | 基础解析 | `medium` |
| `standard` | Standard | 标准解析 | `high` |
| `advanced` | Advanced | 高级解析 | `xhigh` |

The public Tier and the Hybrid runtime effort are separate concepts. Hybrid keeps its existing `medium`, `high`, and `xhigh` effort values, internal branches, and effort-specific implementation names.

## Compatibility Policy

This is an intentional breaking change.

- Public inputs accept only `flash`, `basic`, `standard`, and `advanced`.
- Public outputs emit only those four literals.
- Retired public Tier literals are not accepted as aliases.
- Existing doclib databases and parsed cache directories are not migrated.
- Historical locators using retired Tier literals become invalid.
- Packaging extras are renamed directly, without compatibility extras.
- Users must clear old doclib data before using the renamed Tier system.

The initial database schema remains structurally valid because Tier columns are text fields. Code-backed defaults and all newly written values use the new literals; no schema-version increment or data migration is added.

## Selection Semantics

Tier quality ordering, from lowest to highest, is:

```text
flash < basic < standard < advanced
```

Default quality selection for a parse-server capability list is:

```text
standard -> basic -> error
```

The highest cached non-Flash result is selected in this order:

```text
advanced -> standard -> basic
```

Parsing-rule fallback remains quality-oriented but may end at Flash:

```text
standard -> basic -> flash
```

Direct local parsing without capability discovery defaults to Standard. Watch and lightweight-only file behavior continue to use Flash according to existing file-type rules.

## Runtime Architecture

The canonical Tier definition owns public validation and ordering. A separate explicit mapping translates model-backed Tiers into Hybrid effort:

```python
HYBRID_EFFORT_BY_TIER = {
    "basic": "medium",
    "standard": "high",
    "advanced": "xhigh",
}
```

Reverse inference from an expert-selected Hybrid effort returns the corresponding public Tier. Code must not validate an effort by passing it to the public Tier validator.

The backend mapping remains:

- Flash uses the Flash backend.
- Basic, Standard, and Advanced use the Hybrid backend with their mapped effort.
- Legacy expert backend aliases retain their runtime meaning but resolve to the new public Tier names.

Effort-specific implementation symbols and messages may retain an effort literal when they clearly describe Hybrid runtime behavior. Public Tier types, parameters, examples, package extras, locators, status responses, telemetry dimensions, and configuration values must use the new vocabulary.

## API and Discovery

The reusable local/self-hosted API server continues to expose whichever Tiers it is configured to run. Tier discovery and default selection logic do not encode the official cloud product policy.

The official MinerU API currently advertises only Standard. Clients continue to trust server discovery and existing mismatch handling. Custom remote servers may advertise any supported Tier.

The following public surfaces use the new literals:

- Python parser functions and `MinerUApiParser`
- Parse-job request and response schemas
- `GET /v1/tiers`
- API-server startup options
- doclib request and response models
- CLI and Gradio choices
- model download and verification commands

Tier-flavored public model metadata must not expose retired Tier terminology. Actual stable model repository identifiers remain unchanged when they identify a real repository rather than a Tier label.

## Persistence and Locators

New doclib records use the renamed Tier values in:

- parse records
- document metadata quality
- FTS content metadata
- parsing rules
- managed parse-server configuration
- telemetry dimensions

New parsed artifacts continue to use the Tier as a directory component, now with the new literal. No old directory is inspected, renamed, or merged.

Locator parsers accept only the new literals, and canonical locators emit only the new literals:

```text
doc:{short_id}/tier:{tier}
doc:{short_id}/tier:{tier}/page:{page_no}
doc:{short_id}/tier:{tier}/page:{page_no}/block:{block_no}
doc:{short_id}/tier:{tier}/page:{page_no}/block:{block_no}/char:{offset}
```

## Packaging

Optional dependency groups become:

- `mineru[basic]` for the Basic runtime dependency set
- `mineru[standard]` for Basic plus accelerated/VLM dependencies
- Advanced reuses `mineru[standard]`; no separate Advanced extra is exposed

Test, all-in-one, Docker, and documentation references use the renamed extras. Runtime dependency errors recommend Basic for Basic, and Standard for both Standard and Advanced.

## Documentation Rules

Normative documentation, examples, ADRs, implementation plans, the root README, and the bundled MinerU Skill use the new Tier vocabulary. Historical command examples are updated because they remain searchable and copyable.

References to Hybrid effort retain the effort literals and should explicitly say `effort` where ambiguity is possible. Ordinary English phrases such as “high quality” are not Tier literals and are not rewritten mechanically.

## Verification

Verification must cover:

1. Tier validation, order, defaults, and effort mappings.
2. Parser routing for all four public Tiers.
3. API schemas, discovery metadata, defaults, and invalid values.
4. Doclib parsing, searching, caching, locators, configuration, and telemetry.
5. CLI, mineru-kit, model commands, and Gradio choices.
6. Packaging extras and Docker commands.
7. Documentation and bundled Skill examples.
8. A final residue audit showing that `medium`, `high`, and `xhigh` occur only as Hybrid effort terminology or ordinary non-Tier English.

No compatibility, migration, or legacy-locator tests are added because those behaviors are intentionally unsupported.
