-- OCR persistence schema.
--
-- Applied at ocr_service startup by storage.init_schema(). Idempotent because
-- of IF NOT EXISTS. The source of truth in v1 — future schema changes go in
-- database/migrations/NNN_*.sql once we actually need migrations.

CREATE TABLE IF NOT EXISTS documents (
    id              UUID        PRIMARY KEY,
    original_name   TEXT        NOT NULL,
    status          TEXT        NOT NULL,           -- queued | processing | ok | error
    scan_type       TEXT,                           -- native | scanned
    page_count      INT,
    source_key      TEXT        NOT NULL,           -- MinIO key for the original upload
    markdown_key    TEXT,                           -- MinIO key for output.md
    json_key        TEXT,                           -- MinIO key for layout.json
    docx_key        TEXT,                           -- MinIO key for output.docx
    layout          JSONB,                          -- inline copy for SQL queries
    elapsed_sec     NUMERIC,
    error           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS documents_created_at_idx ON documents(created_at DESC);
CREATE INDEX IF NOT EXISTS documents_status_idx     ON documents(status);

-- ── Auth: users + sessions ────────────────────────────────────────────────
-- Applied idempotently by both ocr_service and auth_service at startup.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS users (
    id                   UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    username             TEXT        NOT NULL UNIQUE,
    password_hash        TEXT        NOT NULL,
    role                 TEXT        NOT NULL CHECK (role IN ('user','admin','master')),
    must_change_password BOOLEAN     NOT NULL DEFAULT FALSE,
    disabled             BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS users_username_idx ON users (lower(username));

-- Idempotent widen of the role CHECK for installs created before 'master' existed.
DO $$
BEGIN
    ALTER TABLE users DROP CONSTRAINT IF EXISTS users_role_check;
    ALTER TABLE users ADD CONSTRAINT users_role_check
        CHECK (role IN ('user','admin','master'));
END $$;

CREATE TABLE IF NOT EXISTS sessions (
    session_id   UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    expires_at   TIMESTAMPTZ NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS sessions_user_idx       ON sessions (user_id);
CREATE INDEX IF NOT EXISTS sessions_expires_at_idx ON sessions (expires_at);

-- Document ownership. ON DELETE SET NULL so deleting a user keeps their
-- past documents around (admin can still see/delete the orphans).
ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS owner_id UUID REFERENCES users(id) ON DELETE SET NULL;
CREATE INDEX IF NOT EXISTS documents_owner_id_idx
    ON documents (owner_id, created_at DESC);

-- ── Translations ──────────────────────────────────────────────────────────
-- One row per translation job. `source_document_id` points at the OCR
-- documents row produced upstream of the translation step. Owner is the
-- user who requested the translation (same identity model as documents).

CREATE TABLE IF NOT EXISTS translations (
    id                   UUID        PRIMARY KEY,
    source_document_id   UUID        REFERENCES documents(id) ON DELETE CASCADE,
    original_name        TEXT        NOT NULL,
    target_lang          TEXT        NOT NULL DEFAULT 'pt-BR',
    status               TEXT        NOT NULL,   -- queued|ocr|translating|reconstructing|ok|error
    translated_json_key  TEXT,
    translated_docx_key  TEXT,
    elapsed_sec          NUMERIC,
    error                TEXT,
    owner_id             UUID        REFERENCES users(id) ON DELETE SET NULL,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS translations_owner_idx
    ON translations (owner_id, created_at DESC);
CREATE INDEX IF NOT EXISTS translations_source_idx
    ON translations (source_document_id);

-- Checkpointing for long translations. A 1000-page document takes hours, so
-- the work is persisted every TRANSLATE_PAGE_WINDOW pages: `pages_done` is how
-- many leading pages are fully translated, `checkpoint_key` is the MinIO object
-- holding the partially-translated layout JSON. A job that dies (timeout, GPU
-- swap, container restart) resumes from `pages_done` instead of restarting.
-- Both NULL/0 means "no checkpoint" — the normal state for short documents.
ALTER TABLE translations
    ADD COLUMN IF NOT EXISTS pages_done      INT,
    ADD COLUMN IF NOT EXISTS page_count      INT,
    ADD COLUMN IF NOT EXISTS checkpoint_key  TEXT;
