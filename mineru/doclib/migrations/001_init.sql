-- v001: Initial schema
-- Applied to new databases.  For existing databases, see 002_add_parse_columns.sql.

CREATE TABLE IF NOT EXISTS files (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    path            TEXT    NOT NULL UNIQUE,
    filename        TEXT    NOT NULL,
    ext             TEXT    NOT NULL,
    size_bytes      INTEGER NOT NULL,
    mtime_ms        INTEGER NOT NULL,
    birthtime_ms    INTEGER,
    sha256          TEXT    REFERENCES docs(sha256),
    watch_id        INTEGER REFERENCES watch_targets(id),
    scan_status     TEXT    NOT NULL DEFAULT 'active',
    locked_at       INTEGER,
    error_code      TEXT,
    error_msg       TEXT,
    deleted_at      INTEGER,
    first_seen_at   INTEGER NOT NULL,
    updated_at      INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_files_sha256 ON files(sha256);
CREATE INDEX IF NOT EXISTS idx_files_watch_id ON files(watch_id);
CREATE INDEX IF NOT EXISTS idx_files_scan_status ON files(scan_status);

CREATE TABLE IF NOT EXISTS docs (
    sha256          TEXT    PRIMARY KEY,
    size_bytes      INTEGER NOT NULL,
    mime_type       TEXT,
    page_count      INTEGER,
    lang            TEXT,
    title           TEXT,
    author          TEXT,
    subject         TEXT,
    keywords        TEXT,
    is_encrypted    INTEGER NOT NULL DEFAULT 0,
    is_scanned      INTEGER NOT NULL DEFAULT 0,
    meta_tier       TEXT,
    first_seen_at   INTEGER NOT NULL,
    updated_at      INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS parses (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    sha256      TEXT    NOT NULL REFERENCES docs(sha256),
    tier        TEXT    NOT NULL,
    pages       TEXT    NOT NULL,
    status      TEXT    NOT NULL DEFAULT 'pending',
    priority    INTEGER NOT NULL DEFAULT 0,
    locked_at   INTEGER,
    error_code  TEXT,
    error_msg   TEXT,
    privacy     TEXT    NOT NULL DEFAULT 'local',
    remote_url  TEXT,
    via         TEXT,
    done_at     INTEGER,
    created_at  INTEGER NOT NULL,
    updated_at  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_parses_status ON parses(status, priority DESC, created_at ASC);
CREATE INDEX IF NOT EXISTS idx_parses_doc ON parses(sha256, tier);

CREATE VIRTUAL TABLE IF NOT EXISTS fts_contents USING fts5(
    sha256 UNINDEXED,
    tier UNINDEXED,
    text,
    title,
    author,
    filename,
    tokenize='unicode61'
);

CREATE VIRTUAL TABLE IF NOT EXISTS fts_filenames USING fts5(
    file_id UNINDEXED,
    filename,
    ext,
    tokenize='unicode61'
);

CREATE TABLE IF NOT EXISTS watch_targets (
    id              INTEGER PRIMARY KEY,
    path            TEXT    NOT NULL UNIQUE,
    label           TEXT,
    removable       INTEGER NOT NULL DEFAULT 0,
    enabled         INTEGER NOT NULL DEFAULT 1,
    recursive       INTEGER NOT NULL DEFAULT 0,
    watch_status    TEXT    NOT NULL DEFAULT 'active',
    unreachable_at  INTEGER,
    last_scan_at    INTEGER,
    last_scan_files INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS rules (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT,
    rule_type       TEXT    NOT NULL,
    pattern         TEXT    NOT NULL,
    tier            TEXT,
    pages           TEXT,
    remote          INTEGER NOT NULL DEFAULT 0,
    enabled         INTEGER NOT NULL DEFAULT 1,
    priority        INTEGER NOT NULL DEFAULT 0,
    hit_count       INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS config (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS _migrations (
    version     INTEGER PRIMARY KEY,
    applied_at  INTEGER NOT NULL,
    description TEXT
);
