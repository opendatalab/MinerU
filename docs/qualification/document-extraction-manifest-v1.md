# document-extraction-manifest-v1 qualification matrix

Status: `deferred`

This record is the source-owner qualification boundary for the canonical
`document-extraction-manifest-v1` schema. It is a synthetic/public readiness
record only; it is not a claim that MinerU is qualified for all document types
or that any extraction derivative is source evidence.

## Profile disposition

| Profile | Intended scope | Status | Reason the gate remains open |
|---|---|---|---|
| `mineru_pipeline_native_text_v1` | Native-text PDF with local pipeline parsing and no VLM enrichment | `deferred` | The adapter now has an opt-in canonical serializer, but source identity is caller-supplied and no gold-fixture or native-source adjudication run has qualified the profile. |
| `mineru_pipeline_ocr_v1` | Scanned PDF with fixed OCR model and language-routing policy | `deferred` | Model-backed OCR gold fixtures and critical-token adjudication have not been run in this source-owner slice. |
| `mineru_hybrid_complex_layout_v1` | Complex tables, multi-column pages, images, and formulas | `deferred` | Local VLM/no-egress evidence and bounded meaning-drift results are not available. |

No profile enables unspecified automatic fallback. Any parser method,
backend, model, language policy, or fallback change must be recorded in the
canonical manifest and re-qualified as the affected profile.

## Acceptance matrix

| Acceptance target | Current evidence | Qualification disposition |
|---|---|---|
| Input SHA, page count, and source identity | Native adapter tests cover local input hashing and page-count artifact shape; canonical serializer tests cover explicit source identity, input SHA, and page-count mapping. | `partial_mechanical_only` |
| Critical identifiers, numbers, units, and negation | No gold-question adjudication run in this slice. | `not_assessed` |
| Critical table cells and row/column association | Required artifact and table-output paths are mechanically checked; no gold table-cell run is recorded. | `partial_mechanical_only` |
| Page-to-block locator | Canonical serializer tests cover page/block IDs and JSON-pointer locators derived from `content_list_v2`; cross-profile locator fidelity remains unassessed. | `partial_mechanical_only` |
| Output inventory and SHA-256 values | `tests/unittest/test_knowhere_integration_contract.py` covers required artifact inventory, relative paths, and deterministic hashes. | `mechanical_pass` |
| Unmanifested outputs and silent fallback | Required output checks and offline HTTP-backend rejection are covered; full profile fallback characterization remains open. | `partial_mechanical_only` |
| Warning recording and fail-closed failures | Missing-artifact and parse-failure tests prove no completed legacy manifest is published; canonical warnings are carried forward and producer-revision failures are fail-closed. | `partial_mechanical_only` |

## Required fixture families before qualification

The next operator-run fixture batch must include native text, scanned OCR,
Traditional Chinese plus English, multi-column content, headers/footers,
footnotes, cross-page tables, image/caption pairs, formulas, rotated pages,
long documents, DOCX headings/tables, corrupt files, encrypted files,
duplicate filenames with different SHA-256 values, and superseded versions.

Qualification is recorded independently for each profile as `qualified`,
`qualified_with_bounded_limitation`, `rejected`, or `deferred`. A passing
adapter unit test does not change this disposition.

## Synthetic native smoke observations

On 2026-07-18, two public repository fixtures were run with MinerU
`3.4.4`, the local `pipeline` backend, and application offline mode on
producer revision `a63b3833e104c694bfec7bee24cb94741d6250ed`:

| Fixture | Method/profile characterization | Result |
|---|---|---|
| `tests/unittest/pdfs/test.pdf` | `auto`; native-text characterization; 1 logical page | Completed; source SHA and all five legacy artifact entries matched; zero warnings. |
| `demo/pdfs/small_ocr.pdf` | `ocr`; OCR characterization; 8 logical pages | Completed; source SHA and all five legacy artifact entries matched; zero warnings. |

These runs confirm only local execution and legacy artifact-integrity
behavior. `offline_verified` remains false because application flags do not
prove host-level network denial; no critical-token/table gold adjudication,
canonical manifest emission, or full fixture-family acceptance was performed.
The profile disposition therefore remains `deferred`.

## Canonical adapter implementation note

On 2026-07-18, the local Knowhere export adapter gained an opt-in
`document-extraction-manifest-v1` serializer. The `--canonical-manifest` path
requires explicit `source_id`, `source_version_id`, and `extraction_run_id`
values, preserves the existing `mineru_manifest.json` output, and maps the
validated `content_list_v2` artifact to page blocks, table records, and image
asset records with relative paths and SHA-256 values. It rejects incomplete
source identity and missing producer revision information before publishing a
canonical payload.

The implementation is a mechanical contract slice only. It does not establish
source sufficiency, native-source verification, critical-token accuracy, table
meaning fidelity, host-level no-egress, or profile qualification. The
qualification status remains `deferred` until the required fixture families
and source-owner review are completed.

## Synthetic canonical smoke observations

On 2026-07-18, the opt-in `--canonical-manifest` path was run against two
public repository fixtures with MinerU `3.4.4`, producer revision
`a1bf4d43147c6df0b88465ae6899433f74035216`, local `pipeline` execution, and
application offline mode. The completed manifests were retained under local
`.qa` output only and are not Git artifacts.

| Fixture | Input SHA-256 | Pages | Outputs | Page blocks | Tables | Images | Warnings / errors | Fallback |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `tests/unittest/pdfs/test.pdf` | `ae9e3f14cc3bea88dd0ce4e2715b3b03561378501318df61f0889df207aed25b` | 1 | 7 | 5 | 1 | 3 | 1 / 0 | `false` |
| `demo/pdfs/small_ocr.pdf` | `c48baa1997e719d414061bea6ca197ce36f1c47341837fbfbc976bbb1d226998` | 8 | 4 | 66 | 0 | 0 | 1 / 0 | `false` |

Both outputs carried the canonical contract version, explicit source and
source-version identity, `derivative_not_native_source_evidence: true`, and
`does_not_establish_source_sufficiency: true`. The only warning in each run
was `model_identifiers_not_exposed_by_adapter`; no parser error or fallback
was recorded. This is a public mechanical emission smoke only. It does not
qualify native text/OCR accuracy, critical-token or table-cell meaning,
host-level network denial, or any profile for private use; the profile status
remains `deferred`.
