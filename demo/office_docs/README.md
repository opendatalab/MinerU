# Office regression fixtures

`rtf_01.rtf` was generated once with LibreOffice's `Rich Text Format` export from the MinerU-produced `2407.00079v4_origi-10.docx` review sample. It is checked in as a real producer fixture; runtime code and CI do not invoke LibreOffice.

The fixture intentionally exercises long English paragraphs and code-like text from a public paper excerpt. Deterministic edge cases for binary payloads, Unicode, lists, tables, equations, notes, links, and images are constructed separately in `tests/unittest/test_flash_rtf.py`.
