"""验证正文检索先应用过滤条件，再截取高分候选。"""

import asyncio
from pathlib import Path

from mineru.doclib.core.db import DatabaseManager
from mineru.doclib.core.fts import FTSManager, tokenize_for_index
from mineru.doclib.services.search_svc import SearchService


def test_search_filters_before_taking_top_ranked_matches(tmp_path: Path) -> None:
    """高分但不符合条件的 PDF 不会挤掉同样命中的高级 DOCX。"""

    async def run() -> None:
        db = DatabaseManager(str(tmp_path / "library.sqlite"))
        await db.initialize()
        try:
            pdf_text = tokenize_for_index("benchmark benchmark benchmark")
            docx_text = tokenize_for_index("monthly archive content benchmark results")
            statements = []
            for index in range(202):
                sha = f"{index:064x}"
                is_pdf = index < 200
                statements.extend(
                    [
                        (
                            "INSERT INTO docs (sha256, short_id, size_bytes, file_type, first_seen_at, updated_at) "
                            "VALUES (?, ?, 10, ?, 1, 1)",
                            (sha, str(index), "pdf" if is_pdf else "docx"),
                        ),
                        (
                            "INSERT INTO fts_contents (sha256, tier, text) VALUES (?, ?, ?)",
                            (sha, "basic" if is_pdf else "advanced", pdf_text if is_pdf else docx_text),
                        ),
                    ],
                )
            await db.execute_atomic(statements)
            fts = FTSManager(db)
            service = SearchService(db, fts)

            top = await fts.search("benchmark", limit=200)
            assert len(top) == 200
            assert all(row["tier"] == "basic" for row in top)
            for filters in ({"file_type": "DOCX"}, {"tier": "advanced"}, {"min_tier": "advanced"}):
                first, total = await service.search("benchmark", limit=1, **filters)
                second, second_total = await service.search("benchmark", limit=1, offset=1, **filters)
                assert total == second_total == 2
                assert {row["short_id"] for row in [*first, *second]} == {"200", "201"}
            assert await service.search("benchmark", tier="basic", min_tier="advanced") == ([], 0)
        finally:
            await db.close()

    asyncio.run(run())
