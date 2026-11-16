-- v003: Rebuild parsed-document indexes using image-free search text.
-- Pending rows hide old payload-bearing indexes until replacement succeeds.
CREATE TABLE IF NOT EXISTS fts_rebuild_queue (
    sha256 TEXT PRIMARY KEY NOT NULL REFERENCES docs(sha256) ON DELETE CASCADE,
    error TEXT
);

INSERT OR IGNORE INTO fts_rebuild_queue (sha256)
SELECT f.sha256 FROM fts_contents AS f JOIN docs AS d ON d.sha256=f.sha256
WHERE f.tier IS NOT NULL;
