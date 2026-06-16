import asyncio

from mineru.doclib.core.db import DatabaseManager
from mineru.doclib.services.parse_svc import ensure_doc_record


def test_ensure_doc_record_extends_short_id_on_prefix_conflict(tmp_path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "mineru.db"))
        await db.initialize()
        now = 1000
        existing_sha = "abcdef0" + "0" * 57
        new_sha = "abcdef1" + "1" * 57

        await db.execute(
            "INSERT INTO docs (sha256, short_id, size_bytes, first_seen_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (existing_sha, existing_sha[:7], 1, now, now),
        )

        for sha256 in (existing_sha, new_sha, "abcdef0" + "2" * 57):
            await ensure_doc_record(
                db,
                sha256=sha256,
                size_bytes=1,
                file_type="pdf",
                page_count=1,
                title=None,
                author=None,
                subject=None,
                keywords=None,
                error_code=None,
                error_msg=None,
                first_seen_at=now,
                updated_at=now,
            )

        existing = await db.fetchone("SELECT short_id FROM docs WHERE sha256=?", (existing_sha,))
        new = await db.fetchone("SELECT short_id FROM docs WHERE sha256=?", (new_sha,))
        conflicting = await db.fetchone("SELECT short_id FROM docs WHERE sha256=?", ("abcdef0" + "2" * 57,))

        assert existing == {"short_id": existing_sha[:7]}
        assert new == {"short_id": new_sha[:7]}
        assert conflicting == {"short_id": ("abcdef0" + "2")}

    asyncio.run(_run())
