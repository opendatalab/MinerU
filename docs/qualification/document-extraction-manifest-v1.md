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

## Current-head producer contract recheck

On 2026-07-19, the current qualification worktree at MinerU revision
`3f41eb76` reran the existing `test_document_extraction_manifest_v1_contract.py`
and `test_knowhere_integration_contract.py` selection with the project-local
Python 3.13 environment and portable PostgreSQL runtime. The selection passed
`18` tests in `13.61` seconds. This reconfirms the current mechanical
producer-manifest and Knowhere integration contract baseline only; parser
profile qualification, native/semantic fidelity, model/legal provenance,
host-level no-egress, source-owner disposition, and D3 remain deferred. No
runtime, edge, private-data, provider, or implementation status changed.

## OCR language-routing policy characterization

On 2026-07-19, the current MinerU qualification worktree at revision
`94118a79` ran an ephemeral pure-function smoke against
`mineru.utils.ocr_language` with the project-local Python 3.13 environment.
The check covered the `12` public OCR language choices, default-empty-list
fallback to `ch`, four public compatibility aliases mapping to `ch`, four
language-family normalizations (`east_slavic`, `arabic`, `devanagari`, and
CPU `seal` to `seal_lite`), and two invalid/unsupported inputs that failed
closed with `ValueError`.

The smoke did not initialize a model, open a source document, make a network
request, or write an output file. It characterizes the input policy layer only;
OCR model accuracy, language detection, critical-token/negation gold,
meaning preservation, no-egress enforcement, and `mineru_pipeline_ocr_v1`
qualification remain deferred. No implementation or runtime status changed.

## Latest-head producer contract recheck

On 2026-07-19, the latest qualification worktree at MinerU revision
`4fe85e50` reran `tests/contracts/test_document_extraction_manifest_v1_contract.py`
and `tests/unittest/test_knowhere_integration_contract.py` with the project
Python 3.13 environment. All `18` tests passed in `17.85` seconds.

This reconfirms the mechanical manifest schema, required artifact inventory,
relative-path and deterministic-hash, canonical identity, and Knowhere
integration boundaries at the latest documentation head. It does not qualify
a parser profile, OCR/model accuracy, native or semantic meaning, model/legal
provenance, host-level no-egress, source-owner disposition, or D3. No runtime,
edge, private-data, provider, or implementation status changed.

## Current-head producer contract recheck

On 2026-07-19, the current qualification worktree at MinerU revision
`43cf41d5bc0b05c35232ddf12fe0d42dbee11a3c` reran the same manifest and
Knowhere integration selections with the project `.venv` Python environment:
`tests/contracts/test_document_extraction_manifest_v1_contract.py` and
`tests/unittest/test_knowhere_integration_contract.py`. The command explicitly
overrode the repository coverage `addopts` because this environment did not
load the coverage plugin; the test selection itself was unchanged. All `18`
tests passed in `8.18` seconds.

This is direct current-head mechanical contract evidence. It does not qualify
a parser profile, OCR/model accuracy, native or semantic meaning, model/legal
provenance, host-level no-egress, source-owner disposition, or D3. No model,
source document, provider, network, private data, or runtime edge was used;
the profile and edge dispositions remain deferred.

## Latest current-head producer contract recheck

On 2026-07-19, the current qualification worktree at MinerU revision
`3f5a74e6` reran `tests/contracts/test_document_extraction_manifest_v1_contract.py`
and `tests/unittest/test_knowhere_integration_contract.py` with the project
`.venv` Python 3.13 environment and coverage addopts explicitly disabled for
this focused selection. All `18` tests passed in `7.44` seconds.

This is direct current-head mechanical evidence for the manifest schema,
required artifact inventory, relative-path and deterministic-hash checks,
canonical identity, and Knowhere integration boundary. It does not qualify a
parser profile, OCR/model accuracy, native or semantic meaning, model/legal
provenance, host-level no-egress, source-owner disposition, or D3. No model,
source document, provider, network, private data, or runtime edge was used;
the profile and edge dispositions remain deferred.

## Current image OCI/source-binding metadata recheck

On 2026-07-19, the existing `mineru-api` container and its local image metadata
were inspected read-only. The image ID and repository digest were both
`sha256:e034f798206a8cdd384a6c3986693cbfe385fe2ed585952963eaeac84ec836c4`.
The image labels exposed the vLLM base-image build commit
`ad7125a431e176d4161099480a66f0169609a690`, the vLLM build pipeline and URL,
and `vllm/vllm-openai:v0.21.0`; they did not expose a MinerU source revision,
MinerU release identifier, or a MinerU build-manifest label. The container
inspection also confirmed the presence of the model-source configuration key
and vLLM build metadata keys without recording their values; no secret or
credential value was read.

This recheck strengthens exact image/base-build identity evidence but confirms
that the required MinerU source-to-image release binding is still absent from
the observed OCI metadata. It does not infer that the image is unbound or
untrusted, and it does not close model notices, license/attribution terms,
build provenance, release-manifest, host-level no-egress, or source-owner
qualification. No container, image, source, model, network, or runtime
configuration was changed; the profile and edge dispositions remain deferred.

## Current public OCR/DOCX canonical export characterization

On 2026-07-19, two additional public-fixture runs exercised the canonical
export path with the local model/runtime environment. Dependency resolution
used `uv run --frozen --offline` with `UV_OFFLINE=1` and `UV_FROZEN=1`; both
commands also supplied the adapter's `--offline` flag. The completed manifests
and generated artifacts were written to a temporary directory outside the
repository and removed after integrity inspection.

| Fixture / run | Requested path | Result | Canonical counts | Warnings | Fallback |
|---|---|---|---|---:|---|
| `demo/pdfs/small_ocr.pdf` / `EXT-WP03-20260719-OCR-E2E-001` | `pipeline` / `ocr` / `en` | `completed`, 8 logical/native pages | 66 page blocks, 0 tables, 0 images, 4 outputs | 1 (`model_identifiers_not_exposed_by_adapter`) | `false` |
| `demo/office_docs/docx_01.docx` / `EXT-WP03-20260719-DOCX-E2E-001` | requested `pipeline` / `auto`; effective `office` | `completed`, 3 logical/native pages | 109 page blocks, 7 tables, 10 images, 14 outputs | 1 (`model_identifiers_not_exposed_by_adapter`) | `false` |

Both canonical payloads carried
`derivative_not_native_source_evidence: true` and
`does_not_establish_source_sufficiency: true`; both reported zero manifest
errors. This expands public OCR and Office model-backed mechanical coverage
and confirms the expected effective backend/manifest boundary. It does not
establish host-level egress denial, model identity completeness, license or
notice provenance, critical-token/table-cell meaning, native or semantic gold,
physical DOCX pagination, source sufficiency, or profile qualification. The
profile dispositions remain `deferred`; no implementation, provider, private
data, runtime configuration, or active edge was changed.

## Exact-image SBOM bounded recheck

On 2026-07-19, the pinned local `mineru` image digest
`sha256:e034f798206a8cdd384a6c3986693cbfe385fe2ed585952963eaeac84ec836c4`
was scanned read-only with the locally installed Syft `1.48.0` Docker source
path, `--scope all-layers`, and a `900`-second execution bound. The scan did
not return before the bound; the output path remained an empty `0`-byte file,
so no CycloneDX document, component count, license inventory, or SBOM hash was
accepted. The confirmed Syft process was stopped and the temporary output
directory was removed after the timeout. The image, container, source tree,
runtime configuration, and existing qualification artifacts were not changed.

This is bounded timeout evidence only. It does not establish dependency
completeness, image-license closure, model/data notices, source-to-image
release binding, owner/legal disposition, or any parser/profile qualification;
those gates remain `deferred`.

## Current source/model/image notice provenance audit

On 2026-07-19, a read-only source and image inventory was performed against
MinerU revision `76b886b1b35cb8e7d59f2a625fb987394d633935` and the running
`mineru-api` image. The scoped source tree contained the repository-level
`LICENSE.md` (SHA-256
`f7e50772426f4fd573c8690216e8e73936de456fc8979c35230187d93b6d51dd`) but no
separate `NOTICE`, `COPYING`, or third-party license inventory file. The
license declares Apache License 2.0 with additional MinerU terms, including
commercial-use thresholds and an online-service attribution obligation. The
package metadata declares `LicenseRef-MinerU-Open-Source-License` and points to
`LICENSE.md`.

The source model map identifies the VLM repositories
`opendatalab/MinerU2.5-Pro-2605-1.2B` /
`OpenDataLab/MinerU2.5-Pro-2605-1.2B` and the pipeline repositories
`opendatalab/PDF-Extract-Kit-1.0` / `OpenDataLab/PDF-Extract-Kit-1.0`, with
pipeline subpaths for PP-DocLayoutV2, UniMERNet, FormulaNet, PaddleOCR,
SlanetPlus, UnetStructure, and table classification. The read-only container
inventory found the corresponding Hugging Face model snapshots: VLM ref
`bff20d4ae2bf202df9f45284b4d43681555a97ed` and pipeline ref
`ed6b654c018d742e65a17671e379c5e6ecc87ec9`. It found seven expected pipeline
model subdirectories and no `LICENSE`, `NOTICE`, `COPYING`, `README`, or
`MODEL_CARD` files inside either downloaded snapshot directory. This absence
is an inventory observation, not a conclusion that the models lack licenses or
notices.

The image package metadata reports MinerU `3.4.4`, vLLM `0.21.0`, and Torch
`2.11.0+cu130`; the Docker recipe downloads all models from Hugging Face and
the runtime entrypoint selects local model paths. These observations narrow
the actual model/runtime inventory but do not close model-specific license or
notice terms, source-to-image release binding, model-data provenance,
owner/legal disposition, or qualification. No model file contents, private
configuration values, network download, container state, or source/runtime
implementation was changed.

## Public model-card license metadata cross-check

On 2026-07-19, the official public model cards were read-only cross-checked
against the model repositories declared by the current source map. The
`opendatalab/MinerU2.5-Pro-2605-1.2B` VLM model card reports `Apache-2.0`, while
the `opendatalab/PDF-Extract-Kit-1.0` pipeline model repository reports
`AGPL-3.0`. The upstream MinerU repository separately identifies its source
code as the MinerU Open Source License, based on Apache 2.0 with additional
conditions. References: [MinerU2.5-Pro-2605-1.2B model card](https://huggingface.co/opendatalab/MinerU2.5-Pro-2605-1.2B),
[PDF-Extract-Kit-1.0 model card](https://huggingface.co/opendatalab/PDF-Extract-Kit-1.0),
and [MinerU upstream license information](https://github.com/opendatalab/MinerU#license-information).

These are source-reported repository/model-card metadata only. They do not
prove that the observed local snapshot is byte-identical to the public card,
assign a single license to every individual model file or dependency, close
notice/attribution obligations, or provide owner/legal approval. The
AGPL-3.0 declaration on the pipeline model repository is a material legal and
provenance gate for the current profile and remains open for per-asset review,
source-to-image binding, and owner/legal disposition. No model was downloaded,
executed, or modified; no runtime, provider, private-data, or qualification
status changed.

## Selected pipeline submodel metadata cross-check

The public component-level metadata was then checked without downloading or
executing any model. The official `PaddlePaddle/PP-DocLayoutV2` model card
reports `Apache-2.0`, and the committed README for
`models/MFR/unimernet_hf_small_2503` inside the public PDF-Extract-Kit
repository also declares `Apache-2.0`. References: [PP-DocLayoutV2 model
card](https://huggingface.co/PaddlePaddle/PP-DocLayoutV2) and [the UniMERNet
component README](https://huggingface.co/opendatalab/PDF-Extract-Kit-1.0/blob/7f74df574f36014d56f800bae3f98852a2dbf896/models/MFR/unimernet_hf_small_2503/README.md).

The public pages inspected for the FormulaNet component, the bundled
`paddleocr_torch` OCR assets, and UnetStructure did not provide enough
independent license/notice metadata to close those individual assets; the
exact-name SLANetPlus and PP-LCNet results are recorded in the section below.
This is an evidence gap, not a conclusion that those assets have no license or
notice. The seven-path local cache match therefore
remains an inventory result only; per-asset license/notice mapping, exact
snapshot identity, attribution obligations, and owner/legal disposition remain
open. No model download, inference, runtime, provider, private-data, or
qualification status changed.

## Exact versus adjacent public component metadata

The remaining public checks were separated by identity. The exact local
`SlanetPlus/slanet-plus.onnx` and `PP-LCNet_x1_0_table_cls.onnx` names have
official cards reporting `Apache-2.0`: [SLANet_plus](https://huggingface.co/PaddlePaddle/SLANet_plus)
and [PP-LCNet_x1_0_table_cls](https://huggingface.co/PaddlePaddle/PP-LCNet_x1_0_table_cls).

The public PDF-Extract-Kit history contains the exact `PP-FormulaNet_plus-M.pth`
file, but no independent license metadata was found for that plus-M artifact;
a separate FormulaNet-L card is adjacent evidence only. Official PP-OCRv4
cards are likewise adjacent to the exact language/variant files in the local
OCR bundle. No independent metadata was found for `UnetStructure/unet.onnx`.
These distinctions leave exact snapshot identity, per-file notice/attribution
terms, and owner/legal disposition open. No model download, execution,
initialization, runtime, provider, or qualification status changed.

## Exact public snapshot ref binding recheck

On 2026-07-19, a read-only `git ls-remote` comparison returned the same refs
stored in the existing `mineru-api` image cache:

| Repository | Public `main` ref | Cached image ref | Result |
|---|---|---|---|
| `opendatalab/MinerU2.5-Pro-2605-1.2B` | `bff20d4ae2bf202df9f45284b4d43681555a97ed` | `bff20d4ae2bf202df9f45284b4d43681555a97ed` | exact match |
| `opendatalab/PDF-Extract-Kit-1.0` | `ed6b654c018d742e65a17671e379c5e6ecc87ec9` | `ed6b654c018d742e65a17671e379c5e6ecc87ec9` | exact match |

The image remained at digest
`sha256:e034f798206a8cdd384a6c3986693cbfe385fe2ed585952963eaeac84ec836c4`.
This narrows snapshot identity for the observed image only. It does not close
per-asset notices, license/attribution terms, signed release-manifest binding,
build provenance, or owner/legal qualification. No model download, inference,
provider execution, private-data processing, or implementation status change
occurred.

## Public parser E2E assertion recheck

On 2026-07-19, the existing `tests/unittest/test_e2e.py` was run once with
`MINERU_MODEL_SOURCE=local`, application/offline model flags, and locked
offline `uv` execution. The test passed (`1 passed` in `52.81` seconds) after
the already-declared test dependency `fuzzywuzzy==0.18.0` was installed into
the project `.venv`; the project lockfile was not changed.

The existing test exercised both `txt` and `ocr` pipeline parses of the public
`tests/unittest/pdfs/test.pdf` and asserted image caption text, table caption
and HTML structure, critical table values, equation markers, and text
similarity. This is a model-backed mechanical assertion against a public
fixture; the generated 14-file output remained local/ignored. Application
offline flags do not prove host-level egress denial, and the test does not
provide native-source adjudication, a human gold decision, profile-wide
coverage, or source sufficiency. All profile dispositions remain `deferred`;
no runtime, provider, private-data, or implementation status changed.

## Public fixture-family coverage inventory

On 2026-07-19, a read-only metadata probe inspected the existing public MinerU
fixtures with `pypdf`, `pdfplumber`, `python-docx`, and `openpyxl`. The probe
persisted no extracted content or new fixture files.

| Candidate family | Existing fixture observation | Evidence boundary |
|---|---|---|
| Native-text PDF | `tests/unittest/pdfs/test.pdf`: 1 page, 463 extracted characters, 1 detected table, 1 image | Structural/native smoke only; no critical-token adjudication |
| Scanned OCR PDF | `demo/pdfs/small_ocr.pdf`: 8 pages, 496 embedded images, 7 extracted characters | OCR execution/material presence only; no OCR gold or language adjudication |
| Text-rich / image PDF | `demo/pdfs/demo1.pdf`: 13 pages, 8 images; no tables detected by the probe | Layout characterization only; caption and locator fidelity unassessed |
| Table-bearing PDF | `demo/pdfs/demo2.pdf`: 6 pages, 12 detected tables, 10 images; `demo3.pdf`: 10 pages, 12 detected tables | Detector output only; no cross-page or critical-cell gold |
| DOCX headings/tables/images | `demo/office_docs/docx_01.docx`: 108 non-empty paragraphs, 26 headings, 8 tables, 9 inline shapes; 616 CJK and 2,159 Latin characters | Mixed-language/structure candidate only; Traditional-Chinese semantics and pagination unassessed |
| XLSX sheets/formulas | `demo/office_docs/xlsx_01.xlsx`: 4 sheets, 2 formulas, non-empty cells across sheets | Workbook structure only; formula/value and table meaning gold absent |

The narrow repository-public metadata probe itself did not establish
rotated-page, long-document, corrupt-file, encrypted-file,
duplicate-basename/different-SHA, or superseded-version coverage. The
separately recorded `Required fixture-family mechanical batch` in this record
already supplies bounded mechanical/fail-closed observations for those input
families. Neither the probe nor that batch establishes image/caption pairing,
formula meaning, critical identifiers, units, negation, table-cell meaning,
native-source adjudication, or human gold. All three profile dispositions
remain `deferred`; no qualification, source sufficiency, runtime, provider,
or implementation status changed.

## Public parser E2E dependency remediation recheck

After the first E2E run reported the optional slow pure-Python
`SequenceMatcher` warning, `python-Levenshtein==0.27.3` and its resolved
runtime dependencies were installed only into the project `.venv`. A second
locked/offline run of the same existing `tests/unittest/test_e2e.py` passed
`1` test in `45.81` seconds with no pytest warning. No `pyproject.toml`,
`uv.lock`, source file, model, image, or runtime configuration was changed.

This is environment/test-performance evidence only; the E2E assertions and
their public/local/offline boundary are unchanged, and all profile
dispositions remain `deferred`.

## Exact-image SBOM recheck

On 2026-07-19, Docker Scout CLI `1.23.1` generated a non-empty CycloneDX `1.5`
SBOM from `local://mineru:latest` at image digest
`sha256:e034f798206a8cdd384a6c3986693cbfe385fe2ed585952963eaeac84ec836c4`.
The report indexed `1,614` components; `777` carried at least one license
field and `837` did not. Package-URL namespaces were `deb=732`, `cargo=518`,
`pypi=308`, `rpm=23`, `golang=19`, `generic=11`, and `npm=3`.

This is accepted exact-image SBOM evidence, not complete license/notice or
model-data provenance evidence. The missing-license rows, model notices,
source-to-image binding, and owner/legal disposition remain open; no image,
container, model, source, runtime, provider, private-data, or qualification
status changed.

## Container Python distribution metadata cross-check

Read-only `importlib.metadata` inspection of the running `mineru-api`
container found `296` installed Python distributions. `195` exposed a
non-empty `License` field or `License ::` classifier and `101` exposed
neither. Selected results were:

| Distribution | Version | Metadata signal | Installed license-like files |
|---|---|---|---|
| `mineru` | `3.4.4` | No license field/classifier | `LICENSE.md` |
| `vllm` | `0.21.0` | No license field/classifier | `LICENSE` |
| `torch` | `2.11.0+cu130` | `BSD-3-Clause` | `LICENSE`, `NOTICE` |
| `torchvision` | `0.26.0+cu130` | `BSD` | `LICENSE` |
| `transformers` | `4.57.6` | Apache field/classifier | `LICENSE` |
| `pypdf` | `6.14.2` | No license field/classifier | `LICENSE` |

The installed distribution metadata does not exactly reproduce the Docker
Scout SBOM's normalized values for selected packages, including `mineru` and
`vllm`. This remains a packaging/normalization discrepancy, not a license or
notice closure. Installed license files do not by themselves establish
complete attribution, model/data terms, source-to-image provenance, or
owner/legal approval. No image, source, model, runtime, provider, or
qualification status changed.

## Exact-image SBOM versus Python metadata reconciliation

The cached exact-image CycloneDX SBOM contained `308` PyPI components. A
read-only normalized name/exact-version join against the running container's
Python distributions matched `296`; `12` had no exact direct distribution
match. Every matched component carried an SBOM license field; `195` also had a
direct Python `License`/`License ::` signal and `101` did not. No matched
component had a direct Python-only signal.

This is a license-presence comparison only, not identifier equivalence,
notice closure, model/data-term review, or owner/legal approval. The twelve
unmatched rows and the direct-metadata gaps remain open provenance items. The
temporary comparison report was removed; no image, source, model, runtime,
provider, or qualification status changed.

## Container license-file coverage audit

Read-only `importlib.metadata` inspection covered all `296` Python
distributions in the running `mineru-api` container. `242` declared at least
one `License-File` metadata entry. A bounded filename filter found `254`
distributions with `433` readable license/notice-like files; all `433` were
successfully SHA-256 hashed without retaining file contents. `96` had a file
but no direct license field/classifier, `37` had a direct signal but no file
found by the filter, and `18` had a declaration without a matching filtered
path.

This is bounded packaging/provenance evidence, not a complete notice
inventory or legal closure. The filename filter can miss differently named
files, and the declaration count does not prove absence. No image, source,
model, runtime, provider, or qualification status changed.

## Current canonical schema SHA recheck

The current `schemas/document-extraction-manifest-v1.schema.json` is
byte-identical to the RA compatibility profile's frozen SHA-256
`8d649b58e6748ae7d1fbd021d76b3c154d544044a37fd357b3e6bcdf03f25def` at
`7048473f0c51c2062e98e700c56286c94e8ea90a`, and that commit is an ancestor of
the current qualification head. This confirms contract-byte continuity only;
the source-owner qualification, runtime/edge promotion, native/semantic gold,
and release provenance gates remain open. No schema, source, runtime, provider,
private-data, or qualification status changed.

## Current qualification-head continuity recheck (2026-07-19)

The previously recorded runtime/source head `43cf41d5` remains an ancestor of
the current qualification head `52d7917011eb5a4f55d2635339e3f7befdaba6ed`.
The tracked diff between those heads contains only this qualification record
(`347` lines changed); all intervening commits are documentation-only
qualification updates. The branch remains `0` commits ahead/behind its
tracked remote, with the existing untracked `.qa/` directory and `uv.lock`
preserved and untouched.

No source, schema, model, image, runtime, provider, private-data, or upstream
state changed. Source-owner qualification, native/semantic gold, release
provenance, and runtime-edge promotion remain deferred.
