-- v002: Add privacy / remote_url / via / output_path columns to parses;

ALTER TABLE parses ADD COLUMN privacy     TEXT DEFAULT 'local';
ALTER TABLE parses ADD COLUMN remote_url  TEXT;
ALTER TABLE parses ADD COLUMN via         TEXT;
ALTER TABLE parses ADD COLUMN output_path TEXT;
