# document-extraction-manifest-v1 qualification matrix

Status: `deferred`

This record is the source-owner qualification boundary for the canonical
`document-extraction-manifest-v1` schema. It is a synthetic/public readiness
record only; it is not a claim that MinerU is qualified for all document types
or that any extraction derivative is source evidence.

## Profile disposition

| Profile | Intended scope | Status | Reason the gate remains open |
|---|---|---|---|
| `mineru_pipeline_native_text_v1` | Native-text PDF with local pipeline parsing and no VLM enrichment | `deferred` | The current Knowhere adapter still emits the observed `knowhere-mineru-artifacts/1.0` manifest and does not yet receive RA source identity fields required by the canonical manifest. |
| `mineru_pipeline_ocr_v1` | Scanned PDF with fixed OCR model and language-routing policy | `deferred` | Model-backed OCR gold fixtures and critical-token adjudication have not been run in this source-owner slice. |
| `mineru_hybrid_complex_layout_v1` | Complex tables, multi-column pages, images, and formulas | `deferred` | Local VLM/no-egress evidence and bounded meaning-drift results are not available. |

No profile enables unspecified automatic fallback. Any parser method,
backend, model, language policy, or fallback change must be recorded in the
canonical manifest and re-qualified as the affected profile.

## Acceptance matrix

| Acceptance target | Current evidence | Qualification disposition |
|---|---|---|
| Input SHA, page count, and source identity | Native adapter tests cover local input hashing and page-count artifact shape; canonical source identity fields are not emitted by the adapter yet. | `partial_mechanical_only` |
| Critical identifiers, numbers, units, and negation | No gold-question adjudication run in this slice. | `not_assessed` |
| Critical table cells and row/column association | Required artifact and table-output paths are mechanically checked; no gold table-cell run is recorded. | `partial_mechanical_only` |
| Page-to-block locator | Native content-list artifacts are preserved, but canonical page-block mapping is not emitted by the adapter. | `not_assessed` |
| Output inventory and SHA-256 values | `tests/unittest/test_knowhere_integration_contract.py` covers required artifact inventory, relative paths, and deterministic hashes. | `mechanical_pass` |
| Unmanifested outputs and silent fallback | Required output checks and offline HTTP-backend rejection are covered; full profile fallback characterization remains open. | `partial_mechanical_only` |
| Warning recording and fail-closed failures | Missing-artifact and parse-failure tests prove no completed legacy manifest is published; canonical warning/error mapping remains open. | `partial_mechanical_only` |

## Required fixture families before qualification

The next operator-run fixture batch must include native text, scanned OCR,
Traditional Chinese plus English, multi-column content, headers/footers,
footnotes, cross-page tables, image/caption pairs, formulas, rotated pages,
long documents, DOCX headings/tables, corrupt files, encrypted files,
duplicate filenames with different SHA-256 values, and superseded versions.

Qualification is recorded independently for each profile as `qualified`,
`qualified_with_bounded_limitation`, `rejected`, or `deferred`. A passing
adapter unit test does not change this disposition.
