-- v001: Initial schema
-- Applied to new databases. Existing databases advance through later numbered migrations.

-- STABLE
CREATE TABLE IF NOT EXISTS files (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    path            TEXT    NOT NULL UNIQUE,
    filename        TEXT    NOT NULL,
    ext             TEXT    NOT NULL,
    size_bytes      INTEGER NOT NULL,
    mtime_ms        INTEGER NOT NULL,
    sha256          TEXT    REFERENCES docs(sha256),
    watch_id        INTEGER REFERENCES watches(id),
    status          TEXT    NOT NULL DEFAULT 'active',
    locked_at       INTEGER,
    error_code      TEXT,
    error_msg       TEXT,
    deleted_at      INTEGER,
    first_seen_at   INTEGER NOT NULL,
    updated_at      INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_files_watch_id ON files(watch_id);
CREATE INDEX IF NOT EXISTS idx_files_status ON files(status);
CREATE INDEX IF NOT EXISTS idx_files_sha256_status ON files(sha256, status);

CREATE TABLE IF NOT EXISTS docs (
    sha256          TEXT    PRIMARY KEY NOT NULL,
    short_id        TEXT    NOT NULL UNIQUE,
    size_bytes      INTEGER NOT NULL,
    file_type       TEXT,
    page_count      INTEGER,
    language        TEXT,
    title           TEXT,
    author          TEXT,
    subject         TEXT,
    keywords        TEXT,
    is_image_based  INTEGER NOT NULL DEFAULT 0,
    meta_tier       TEXT,
    error_code      TEXT,
    error_msg       TEXT,
    first_seen_at   INTEGER NOT NULL,
    updated_at      INTEGER NOT NULL
);

-- STABLE
CREATE TABLE IF NOT EXISTS parses (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    sha256      TEXT    NOT NULL REFERENCES docs(sha256),
    tier        TEXT    NOT NULL,
    page_range  TEXT    NOT NULL,
    status      TEXT    NOT NULL DEFAULT 'pending',
    priority    INTEGER NOT NULL DEFAULT 0,
    locked_at   INTEGER,
    error_code  TEXT,
    error_msg   TEXT,
    privacy     TEXT    NOT NULL DEFAULT 'local',
    via         TEXT,
    done_at     INTEGER,
    created_at  INTEGER NOT NULL,
    updated_at  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_parses_status ON parses(status, priority DESC, created_at ASC);
CREATE INDEX IF NOT EXISTS idx_parses_doc_status ON parses(sha256, tier, status);

CREATE TABLE IF NOT EXISTS scans (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    path                TEXT    NOT NULL,
    kind                TEXT    NOT NULL,
    source              TEXT    NOT NULL DEFAULT 'unknown',
    watch_id            INTEGER REFERENCES watches(id),
    status              TEXT    NOT NULL DEFAULT 'pending',
    locked_at           INTEGER,
    files_seen          INTEGER NOT NULL DEFAULT 0,
    files_refreshed     INTEGER NOT NULL DEFAULT 0,
    files_new           INTEGER NOT NULL DEFAULT 0,
    files_changed       INTEGER NOT NULL DEFAULT 0,
    files_deleted       INTEGER NOT NULL DEFAULT 0,
    files_unreachable   INTEGER NOT NULL DEFAULT 0,
    files_error         INTEGER NOT NULL DEFAULT 0,
    files_unsupported   INTEGER NOT NULL DEFAULT 0,
    files_excluded      INTEGER NOT NULL DEFAULT 0,
    error_code          TEXT,
    error_msg           TEXT,
    started_at          INTEGER,
    finished_at         INTEGER,
    created_at          INTEGER NOT NULL,
    updated_at          INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_scans_status ON scans(status, created_at ASC);
CREATE INDEX IF NOT EXISTS idx_scans_kind_path_status ON scans(kind, path, status);
CREATE INDEX IF NOT EXISTS idx_scans_watch_id_status ON scans(watch_id, status);

CREATE VIRTUAL TABLE IF NOT EXISTS fts_contents USING fts5(
    sha256 UNINDEXED,
    tier UNINDEXED,
    text,
    title,
    author,
    tokenize='unicode61'
);

CREATE VIRTUAL TABLE IF NOT EXISTS fts_filenames USING fts5(
    file_id UNINDEXED,
    filename,
    ext,
    tokenize='unicode61'
);

CREATE TABLE IF NOT EXISTS watches (
    id              INTEGER PRIMARY KEY,
    path            TEXT    NOT NULL UNIQUE,
    label           TEXT,
    removable       INTEGER NOT NULL DEFAULT 0,
    enabled         INTEGER NOT NULL DEFAULT 1,
    recursive       INTEGER NOT NULL DEFAULT 0,
    status          TEXT    NOT NULL DEFAULT 'active',
    unreachable_at  INTEGER,
    last_scan_at    INTEGER,
    last_scan_files INTEGER NOT NULL DEFAULT 0,
    created_at      INTEGER NOT NULL,
    updated_at      INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS exclude_rules (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT,
    pattern         TEXT    NOT NULL,
    enabled         INTEGER NOT NULL DEFAULT 1,
    priority        INTEGER NOT NULL DEFAULT 0,
    hit_count       INTEGER NOT NULL DEFAULT 0,
    created_at      INTEGER NOT NULL,
    updated_at      INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS parsing_rules (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT,
    pattern         TEXT    NOT NULL,
    tier            TEXT,
    page_range      TEXT,
    remote          INTEGER NOT NULL DEFAULT 0,
    enabled         INTEGER NOT NULL DEFAULT 1,
    priority        INTEGER NOT NULL DEFAULT 0,
    hit_count       INTEGER NOT NULL DEFAULT 0,
    created_at      INTEGER NOT NULL,
    updated_at      INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS config (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS telemetry_state (
    key        TEXT PRIMARY KEY,
    value      TEXT    NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS telemetry_aggregates (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    period_start    INTEGER NOT NULL,
    period_end      INTEGER NOT NULL,
    metric_name     TEXT    NOT NULL,
    metric_value    INTEGER NOT NULL DEFAULT 0,
    dimensions      TEXT    NOT NULL,
    dimensions_hash TEXT    NOT NULL,
    created_at      INTEGER NOT NULL,
    updated_at      INTEGER NOT NULL,
    UNIQUE(period_start, period_end, metric_name, dimensions_hash)
);
CREATE INDEX IF NOT EXISTS idx_telemetry_aggregates_period
ON telemetry_aggregates(period_start, period_end);

CREATE TABLE IF NOT EXISTS _migrations (
    version     INTEGER PRIMARY KEY,
    applied_at  INTEGER NOT NULL,
    description TEXT
);
