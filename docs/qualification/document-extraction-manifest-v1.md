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

## Canonical payload boundary audit

The two local canonical manifests from the public-fixture smoke were also
checked against the producer schema root contract. Each had 19 root fields,
zero missing required fields, zero extra root fields, and zero occurrences of
RA authority fields such as `source_status`, `readiness_status`,
`evidence_status`, `regulatory_conclusion`, or
`source_sufficiency_decision` at any JSON level. Both retained
`derivative_not_native_source_evidence: true` and
`does_not_establish_source_sufficiency: true`. This confirms a payload
boundary only; it does not change the deferred qualification status.

## Synthetic fail-closed smoke observation

On 2026-07-18, six existing contract-test cases covering missing required
artifacts, output-root escape, parse failure, missing producer revision, and
missing canonical source identity passed with synthetic fixtures. The failure
paths assert that incomplete or failed exports do not publish a completed
manifest, and that canonical identity/revision requirements are checked before
payload emission. This is mechanical boundary evidence only; it does not
qualify native-source meaning, gold fixtures, or the deferred producer
profile.

## Bounded public fixture characterization

On 2026-07-19, a second public-fixture batch was run through the opt-in
canonical adapter inside the pinned MinerU GPU container. The runs used MinerU
`3.4.4`, producer revision
`6450b02c2d1c2fb0ef2c9369037bbe3c6663d052`, image digest
`mineru@sha256:e034f798206a8cdd384a6c3986693cbfe385fe2ed585952963eaeac84ec836c4`,
application offline flags, and the local Docker GPU profile. The PDF runs used
the `pipeline` backend; the DOCX run resolved to the `office` backend.

| Fixture | Run | Backend | Logical pages | Page blocks | Tables | Images | Outputs | Status | Errors | Fallback |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---|
| `demo/pdfs/demo1.pdf` | `EXT-WP03-20260719-DEMO1` | `pipeline` | 13 | 131 | 5 | 20 | 24 | `completed` | 0 | `false` |
| `demo/pdfs/demo2.pdf` | `EXT-WP03-20260719-DEMO2` | `pipeline` | 6 | 75 | 2 | 19 | 23 | `completed` | 0 | `false` |
| `demo/pdfs/demo3.pdf` | `EXT-WP03-20260719-DEMO3` | `pipeline` | 10 | 144 | 9 | 23 | 27 | `completed` | 0 | `false` |
| `demo/office_docs/docx_01.docx` | `EXT-WP03-20260719-DOCX01` | `office` | 3 | 109 | 7 | 10 | 14 | `completed` | 0 | `false` |

A repeat run of `demo/pdfs/demo2.pdf` with the same configuration and a new
run identity (`EXT-WP03-20260719-DEMO2-REPEAT`) produced the same 23 non-manifest
artifact files with identical SHA-256 values. Input SHA, page/block/table/image
counts, fallback state, and error count also matched. This is bounded artifact
reproducibility evidence only; it is not semantic gold or meaning-preservation
adjudication.

Each canonical manifest carried the expected derivative and non-sufficiency
flags. Each recorded the adapter limitation
`model_identifiers_not_exposed_by_adapter` as its only manifest warning; no
manifest error or fallback was recorded. The DOCX run completed with
parser-derived logical pages and required artifacts; its logical page count is
not a stable physical pagination claim. The isolated run outputs remain local
and ignored; no source contents or generated artifacts are retained in Git.

This batch expands mechanical coverage for public complex-layout PDFs and the
DOCX Office path only. It does not provide critical-token or table-cell gold
adjudication, native-source review, repeated-run reproducibility, host-level
network denial, model identity completeness, encrypted/corrupt-file handling,
duplicate or superseded-version handling, or the remaining required fixture
families. The profile dispositions therefore remain `deferred`.

## Updated native and OCR canonical emission observation

On 2026-07-19, the same pinned MinerU GPU image was used for one current
canonical emission run for the public native-text fixture and one for the
public OCR fixture. The implementation revision was
`6450b02c2d1c2fb0ef2c9369037bbe3c6663d052` and the image was
`mineru@sha256:e034f798206a8cdd384a6c3986693cbfe385fe2ed585952963eaeac84ec836c4`.
Application offline flags were requested; host-level network denial remains
unverified.

| Fixture | Profile characterization | Run | Logical pages | Page blocks | Tables | Images | Outputs | Status | Errors | Fallback | Input SHA-256 |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---|---|
| `tests/unittest/pdfs/test.pdf` | native text; `pipeline` / `auto` | `EXT-WP03-20260719-TESTPDF` | 1 | 5 | 1 | 3 | 7 | `completed` | 0 | `false` | `ae9e3f14cc3bea88dd0ce4e2715b3b03561378501318df61f0889df207aed25b` |
| `demo/pdfs/small_ocr.pdf` | OCR; `pipeline` / `ocr` / `en` | `EXT-WP03-20260719-SMALLOCR` | 8 | 66 | 0 | 0 | 4 | `completed` | 0 | `false` | `c48baa1997e719d414061bea6ca197ce36f1c47341837fbfbc976bbb1d226998` |

Both manifests carried `derivative_not_native_source_evidence: true` and
`does_not_establish_source_sufficiency: true`. The only warning in each run
was `model_identifiers_not_exposed_by_adapter`. These observations expand
current native/OCR mechanical coverage only; they do not provide critical
token, table-cell, native-source, or meaning-preservation adjudication, and
the profile dispositions remain `deferred`.

## Synthetic DOCX structure observation

On 2026-07-19, the existing Knowhere synthetic DOCX fixture generated by
`apps/worker/tests/fixtures/codex_export/generate_docx_fixture.py` was run
through the same pinned MinerU GPU image with the effective `office` backend.
The run used implementation revision
`6450b02c2d1c2fb0ef2c9369037bbe3c6663d052`, image digest
`mineru@sha256:e034f798206a8cdd384a6c3986693cbfe385fe2ed585952963eaeac84ec836c4`,
and run identity `EXT-WP03-20260719-DOCX-COMPLEX`.

The completed canonical manifest contained 12 page blocks, 2 tables, 1 image,
and 5 outputs; manifest errors were 0 and fallback was `false`. It retained
`derivative_not_native_source_evidence: true` and
`does_not_establish_source_sufficiency: true`. The Office parser reported one
logical page for this fixture; that is a parser-derived locator and is not a
physical pagination claim. The process also emitted the non-fatal warning
`No valid PDF or image files to process` after the Office artifacts were
written; this warning is retained as an operational limitation rather than
treated as a silent pass.

This observation expands mechanical coverage for DOCX headings, simple and
merged tables, an embedded image/caption, formula-like symbols, and a page
break. It does not establish native DOCX meaning fidelity, physical page
mapping, critical-cell gold, or profile qualification; the profile remains
`deferred`.

## Synthetic sentinel smoke observation

The same existing synthetic DOCX generator was also used for a bounded
content-sentinel smoke with run identity `EXT-WP03-20260719-DOCX-GOLD`. Twenty
predefined category checks derived from the generator passed: headings and
plain text, unit/symbol tokens, simple-table cells, merged-table labels and
numeric cells, image/caption presence, conclusion text, manifest table/image
counts, completed status, zero manifest errors, and no fallback. No source
content, generated document, or extracted body text was retained in Git.

This is `synthetic_sentinel_smoke_only`, not a human-adjudicated gold-question
set: the checks confirm expected synthetic tokens are visible in the
derivative, but do not establish native meaning fidelity, reviewer-critical
semantic correctness, table-cell association under adversarial layouts, or
profile qualification. The qualification disposition remains `deferred`.

## Chinese-only DOCX mechanical observation

On 2026-07-19, the existing Knowhere fixture
`apps/worker/tests/fixtures/sample_chinese_600chars.docx` was run through the
same pinned MinerU GPU image with run identity
`EXT-WP03-20260719-ZH-DOCX`. The command supplied the valid `pipeline` backend;
the DOCX path recorded `office` as the effective backend. The image contained
producer revision `6450b02c2d1c2fb0ef2c9369037bbe3c6663d052` and used digest
`mineru@sha256:e034f798206a8cdd384a6c3986693cbfe385fe2ed585952963eaeac84ec836c4`.

The canonical manifest completed with one parser-derived logical page, one
page block, zero tables, zero images, four outputs, zero errors, and no
fallback. It carried `derivative_not_native_source_evidence: true` and
`does_not_establish_source_sufficiency: true`; the only warning was
`model_identifiers_not_exposed_by_adapter`.

This adds Chinese-only mechanical coverage for the existing multilingual
fixture. The fixture is not a Traditional-Chinese-plus-English bilingual gold
set, and no critical-token adjudication, native-source review, semantic
meaning-preservation review, or profile qualification was performed. The
qualification disposition remains `deferred`.

## Synthetic PDF page-structure observation

On 2026-07-19, the existing Knowhere corpus fixture
`apps/worker/tests/fixtures/sample_3pages.pdf` was run through the same pinned
MinerU GPU image with run identity `EXT-WP03-20260719-SYNTHPDF`. The canonical
manifest used producer revision
`6450b02c2d1c2fb0ef2c9369037bbe3c6663d052` from image digest
`mineru@sha256:e034f798206a8cdd384a6c3986693cbfe385fe2ed585952963eaeac84ec836c4`.

The manifest completed with three parser-derived logical pages, three
paragraph blocks, four outputs, zero errors, and no fallback. It contained
zero table records and zero image records; the only warning was
`model_identifiers_not_exposed_by_adapter`. The source SHA-256 was
`b1dd4d86d2b7c6505e35d972d0074ec08a7431013dd95e926ba92c7fbf165b1d`.

This adds bounded page-count and paragraph-structure coverage for the
existing synthetic PDF only. It does not demonstrate table/image extraction
for this fixture, critical-token or native-source fidelity, semantic gold,
meaning preservation, or profile qualification; the qualification disposition
remains `deferred`.

## Native-source mechanical concordance observation

On 2026-07-19, the existing DOCX fixture generated by
`apps/worker/tests/fixtures/codex_export/generate_docx_fixture.py` was used to
derive source-side metadata and tokens before running the same pinned MinerU
GPU image. The run identity was
`EXT-WP03-20260719-DOCX-NATIVE-GOLD-RUN1`, source ID
`EXT-WP03-20260719-DOCX-NATIVE-GOLD`, source version
`20260719-native-gold-v1`, and input SHA-256
`70f9dabe1368d59714a678c928dfab60defb2a91a0d7b4b49b0e17f3715aed03`. The
producer revision was
`6450b02c2d1c2fb0ef2c9369037bbe3c6663d052` from image digest
`mineru@sha256:e034f798206a8cdd384a6c3986693cbfe385fe2ed585952963eaeac84ec836c4`.
The CLI requested the `pipeline` backend and the DOCX path resolved to the
effective `office` backend.

The source-side DOCX inspection found 9 document paragraphs, 2 top-level
tables, 15 table-cell entries including merged-cell traversal repeats, 1
footer paragraph, and 1 inline shape. The canonical manifest completed with 1
parser-derived logical page, 12 page blocks, 2 table records, 1 image record,
5 outputs, 0 errors, and `fallback_used: false`. Its only warning was
`model_identifiers_not_exposed_by_adapter`.

A source-derived concordance over 55 normalized tokens found 0 missing tokens
in the derivative Markdown/JSON outputs. The normalization preserved the
contents of `sub`/`sup` markup (for example, `H2O`) and separated other HTML
tags as field boundaries; table-cell token concordance likewise had 0 missing
tokens. This is mechanical source-to-derivative evidence for the existing
synthetic native DOCX fixture, not human semantic adjudication, adversarial
layout gold, retrieval top-N meaning preservation, or qualification. The
canonical disposition remains `deferred`; host-level network denial and model
identity completeness remain unresolved.

## Candidate gold-question packet preflight

On 2026-07-19, a separate preparation run used the same existing synthetic
DOCX generator and pinned image with run identity
`EXT-WP03-20260719-DOCX-GOLD-PREP-RUN1` and input SHA-256
`83c43e25c9e1359b2a9be605fd6ecf1308470531c0619ab589c59c49c1cb446e`. This is
a distinct generated DOCX instance from the earlier concordance run and is
tracked by its own source identity. The producer revision and image digest
were unchanged.

An in-memory candidate packet contained 7 bounded question categories:
heading hierarchy; formula, unit, and symbol tokens; simple-table cells;
embedded image and caption; merged-table structure; conclusion text; and
footer text. Every category had a source-side expected set and an explicit
`content_list_v2` artifact locator. The mechanical preflight passed 7/7
categories, including all 55 source-derived tokens, 6/6 table/image structure
checks, and the manifest counts of 1 logical page, 12 blocks, 2 tables, 1
image, 5 outputs, 0 errors, and no fallback.

The packet remains `pending_human_adjudication`; no adjudicator, gold decision,
retrieval top-N execution, active route, or qualification status was created.
The packet and generated outputs were temporary local QA material and were
removed after the preflight. This is preparation evidence only; the profile
disposition remains `deferred`.

## Required fixture-family mechanical batch

On 2026-07-19, a temporary local synthetic/public batch exercised the remaining
fixture-family inputs against the same pinned image and producer revision. The
batch contained 12 inputs: 10 valid documents and 2 negative-path documents.
All 10 valid canonical manifests completed with zero errors, no fallback, an
input SHA matching the host fixture, and the only warning
`model_identifiers_not_exposed_by_adapter`.

| Fixture family | Effective backend | Logical pages | Page blocks | Tables | Images | Outputs | Result |
|---|---|---:|---:|---:|---:|---:|---|
| Traditional Chinese plus English DOCX | `office` | 1 | 4 | 1 | 0 | 4 | `completed` |
| Multi-column plus header/footer PDF | `pipeline` | 2 | 8 | 0 | 0 | 4 | `completed` |
| Footnote PDF | `pipeline` | 1 | 1 | 0 | 0 | 4 | `completed` |
| Cross-page table PDF | `pipeline` | 2 | 11 | 0 | 0 | 4 | `completed` |
| Rotated-page PDF | `pipeline` | 2 | 3 | 0 | 0 | 4 | `completed` |
| Long synthetic PDF (12 pages) | `pipeline` | 12 | 420 | 0 | 0 | 4 | `completed` |
| Duplicate filename A/B | `pipeline` | 1 each | 1 each | 0 each | 0 each | 4 each | `completed` |
| Superseded version v1/v2 | `pipeline` | 1 each | 1 each | 0 each | 0 each | 4 each | `completed` |

The corrupt PDF and encrypted PDF both exited with code 2 and published no
canonical manifest. The corrupt path recorded a PDFium data-format failure;
the encrypted path recorded an incorrect-password failure. The duplicate
basename pair had distinct SHA-256 values
`d14246b4ce09ba78b15619a448a43e439fe1022e273c42db52e444d6c065c818` and
`715e1ccc95b33d3a1478d1d1e003dc433f17c40a7a867d7436e7a3560ae60bb1`; the
superseded v1/v2 pair likewise had distinct SHA-256 values
`f3019474d861d2d8d3a1d7ed79d1ca90e46845df924da293a78151b9047c4e09` and
`3dbd0d20e0d5a2b589570f5589dfb6dbf6f060f8e68a6da52f4478803599aa23`.

This batch expands mechanical input and fail-closed coverage only. The
cross-page-table fixture produced no canonical table record; rotated-page,
footnote, and multi-column semantics were not human-adjudicated as preserved
meaning; and duplicate/superseded handling was inventory evidence only, not a
source-version lifecycle or retrieval deletion decision. No gold adjudication,
active retrieval top-N test, source sufficiency decision, or profile
qualification was inferred. The profile dispositions remain `deferred`.

## Native page-render and derivative visual concordance observation

On 2026-07-19, a local rendered page for the public native-text fixture
`cross-edge-native-text-20260718-run3` was visually compared with its existing
derivative assets and structured blocks. The page visibly contained one
figure/caption, one displayed equation, one paragraph, one rotated complex
table/caption, and a page number. The derivative contained five corresponding
producer-owned blocks with page/bounding-box locators: image, interline
equation, paragraph, table, and page-number.

The figure asset and equation asset matched the visible page regions. The
table image preserved the rotated table presentation, while the HTML retained
the four data rows and three columns. The table metadata explicitly recorded
`csv_fidelity: lossy_complex` and warnings for detected rowspan/colspan
structure; therefore CSV output was not treated as a lossless semantic table
representation. This is an AI-assisted visual/mechanical concordance
observation only. It does not establish native-source adjudication, critical
table-cell correctness, meaning preservation, or profile qualification; the
profile disposition remains `deferred`.

## Current pinned contract regression recheck

On 2026-07-19, the pinned MinerU checkout at revision
`cebf5078a3ed2990260caa03110b0bab82a16b64` reran the producer-owned manifest
contract and Knowhere integration-contract selections with the repository
Python 3.13 environment. The schema/fixture contract and artifact-inventory
integration selection passed `18` tests in `20.30` seconds. This reconfirms
the current pinned mechanical contract baseline only; it does not qualify a
parser profile, establish semantic or native-source fidelity, prove host-level
no-egress, resolve model/legal provenance, or change the deferred disposition.
