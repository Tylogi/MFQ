CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    mode TEXT NOT NULL,
    state TEXT NOT NULL,
    revision INTEGER NOT NULL CHECK (revision >= 0),
    title TEXT,
    runtime_instance_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    role TEXT NOT NULL,
    parts_json TEXT NOT NULL,
    parent_id TEXT REFERENCES messages(id),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS session_messages (
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 1),
    message_id TEXT NOT NULL REFERENCES messages(id),
    PRIMARY KEY (session_id, ordinal),
    UNIQUE (session_id, message_id)
);

CREATE INDEX IF NOT EXISTS session_messages_message_id
    ON session_messages(message_id);

CREATE TABLE IF NOT EXISTS responses (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    request_id TEXT NOT NULL,
    request_fingerprint TEXT NOT NULL,
    status TEXT NOT NULL,
    input_message_id TEXT NOT NULL REFERENCES messages(id),
    output_message_id TEXT REFERENCES messages(id),
    output_json TEXT NOT NULL,
    finish_reason TEXT,
    usage_json TEXT,
    performance_json TEXT,
    settings_json TEXT,
    error_json TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    UNIQUE (session_id, request_id)
);

CREATE INDEX IF NOT EXISTS responses_session_status
    ON responses(session_id, status);

CREATE TABLE IF NOT EXISTS media (
    id TEXT PRIMARY KEY,
    sha256 TEXT NOT NULL UNIQUE,
    mime_type TEXT NOT NULL,
    byte_size INTEGER NOT NULL CHECK (byte_size >= 0),
    storage_key TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    media_id TEXT PRIMARY KEY REFERENCES media(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    text TEXT NOT NULL,
    page_count INTEGER,
    extractor TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS generation_presets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    model TEXT,
    mode TEXT,
    settings_json TEXT NOT NULL,
    context_size INTEGER CHECK (context_size >= 512),
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runtime_profiles (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    load_json TEXT NOT NULL,
    sampling_json TEXT,
    artifact_id TEXT NOT NULL,
    artifact_modified_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifact_lineage (
    id TEXT PRIMARY KEY,
    artifact_uri TEXT NOT NULL UNIQUE,
    artifact_name TEXT NOT NULL,
    producer_job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    producer_kind TEXT NOT NULL,
    source_uris_json TEXT NOT NULL,
    parameters_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifact_validations (
    artifact_uri TEXT NOT NULL REFERENCES artifact_lineage(artifact_uri)
        ON DELETE CASCADE,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    PRIMARY KEY (artifact_uri, job_id)
);

CREATE TABLE IF NOT EXISTS api_keys (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    key_hash TEXT NOT NULL UNIQUE,
    prefix TEXT NOT NULL,
    scopes_json TEXT NOT NULL,
    role TEXT,
    expires_at TEXT,
    revoked_at TEXT,
    created_at TEXT NOT NULL,
    last_used_at TEXT
);

CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,
    artifact_uri TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    byte_size INTEGER NOT NULL CHECK (byte_size >= 0),
    source_uri TEXT,
    revision TEXT,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evaluation_results (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL UNIQUE REFERENCES jobs(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    model_id TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    parameters_json TEXT NOT NULL,
    dataset_id TEXT REFERENCES datasets(id) ON DELETE SET NULL,
    dataset_manifest_json TEXT NOT NULL,
    hardware_identity_json TEXT NOT NULL,
    runtime_identity_json TEXT NOT NULL,
    comparison_key TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS evaluation_results_comparison
    ON evaluation_results(comparison_key, created_at DESC);

CREATE TABLE IF NOT EXISTS remote_nodes (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    url TEXT NOT NULL UNIQUE,
    api_key_env TEXT,
    enabled INTEGER NOT NULL CHECK (enabled IN (0, 1)),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    progress REAL NOT NULL CHECK (progress >= 0 AND progress <= 1),
    progress_data_json TEXT NOT NULL DEFAULT '{}',
    cancel_requested INTEGER NOT NULL CHECK (cancel_requested IN (0, 1)),
    result_json TEXT,
    error_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    archived_at TEXT
);

CREATE INDEX IF NOT EXISTS jobs_status_created
    ON jobs(status, created_at);

CREATE TABLE IF NOT EXISTS job_events (
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    sequence INTEGER NOT NULL CHECK (sequence >= 1),
    type TEXT NOT NULL,
    level TEXT NOT NULL,
    message TEXT,
    progress REAL CHECK (progress IS NULL OR (progress >= 0 AND progress <= 1)),
    data_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (job_id, sequence)
);

CREATE TABLE IF NOT EXISTS runtime_metrics (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    instance_id TEXT,
    model TEXT,
    values_json TEXT NOT NULL,
    captured_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS runtime_metrics_captured
    ON runtime_metrics(captured_at DESC);

CREATE TABLE IF NOT EXISTS serving_requests (
    instance TEXT NOT NULL, request_id TEXT NOT NULL, model TEXT,
    completed_at REAL NOT NULL, prompt_tokens INTEGER NOT NULL,
    prefill_tokens INTEGER NOT NULL, completion_tokens INTEGER NOT NULL,
    cached_tokens INTEGER NOT NULL, prefill_ms REAL NOT NULL,
    decode_ms REAL NOT NULL, ttft_ms REAL NOT NULL, decode_tokens INTEGER NOT NULL,
    PRIMARY KEY(instance, request_id)
);
CREATE INDEX IF NOT EXISTS serving_requests_completed ON serving_requests(completed_at);

CREATE TABLE IF NOT EXISTS runtime_logs (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    instance_id TEXT,
    level TEXT NOT NULL,
    message TEXT NOT NULL,
    fields_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS runtime_logs_created
    ON runtime_logs(created_at DESC);
