-- 0001_initial_schema.sql
-- Esquema inicial de data/mcp_videos.sqlite conforme a docs/db/diseno_db.md.

CREATE TABLE IF NOT EXISTS voices (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    engine TEXT NOT NULL CHECK (
        engine IN ('kokoro', 'piper', 'xtts', 'f5', 'elevenlabs', 'openai')
    ),
    mode TEXT NOT NULL CHECK (mode IN ('preset', 'clone')),
    language TEXT NOT NULL DEFAULT 'es' CHECK (language IN ('es', 'en')),
    sample_wav_path TEXT,
    config_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    CHECK (mode = 'preset' OR sample_wav_path IS NOT NULL)
);

CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    width INTEGER NOT NULL DEFAULT 1080 CHECK (width > 0),
    height INTEGER NOT NULL DEFAULT 1920 CHECK (height > 0),
    fps INTEGER NOT NULL DEFAULT 60 CHECK (fps IN (24, 30, 60)),
    total_duration_ms INTEGER NOT NULL DEFAULT 0 CHECK (total_duration_ms >= 0),
    master_audio_path TEXT,
    status TEXT NOT NULL DEFAULT 'draft' CHECK (
        status IN ('draft', 'audio_ready', 'scenes_ready', 'rendered', 'failed')
    ),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dialogue_lines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL REFERENCES projects (id) ON DELETE CASCADE,
    line_index INTEGER NOT NULL CHECK (line_index >= 0),
    voice_id TEXT NOT NULL REFERENCES voices (id) ON DELETE RESTRICT,
    speaker_label TEXT NOT NULL,
    text_content TEXT NOT NULL CHECK (length(trim(text_content)) > 0),
    pause_after_ms INTEGER NOT NULL DEFAULT 150 CHECK (
        pause_after_ms BETWEEN -250 AND 2000
    ),
    start_ms INTEGER NOT NULL CHECK (start_ms >= 0),
    end_ms INTEGER NOT NULL CHECK (end_ms > start_ms),
    audio_path TEXT NOT NULL,
    UNIQUE (project_id, line_index)
);

CREATE TABLE IF NOT EXISTS word_timestamps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    line_id INTEGER NOT NULL REFERENCES dialogue_lines (id) ON DELETE CASCADE,
    word_index INTEGER NOT NULL CHECK (word_index >= 0),
    word TEXT NOT NULL,
    start_ms INTEGER NOT NULL CHECK (start_ms >= 0),
    end_ms INTEGER NOT NULL CHECK (end_ms > start_ms),
    confidence REAL NOT NULL DEFAULT 1.0 CHECK (confidence BETWEEN 0.0 AND 1.0),
    UNIQUE (line_id, word_index)
);

CREATE TABLE IF NOT EXISTS scenes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL REFERENCES projects (id) ON DELETE CASCADE,
    scene_index INTEGER NOT NULL CHECK (scene_index >= 0),
    title TEXT NOT NULL,
    start_ms INTEGER NOT NULL CHECK (start_ms >= 0),
    end_ms INTEGER NOT NULL CHECK (end_ms > start_ms),
    html_path TEXT NOT NULL,
    prompt_used TEXT NOT NULL DEFAULT '',
    validation_status TEXT NOT NULL DEFAULT 'pending' CHECK (
        validation_status IN ('pending', 'valid', 'invalid')
    ),
    validation_errors_json TEXT NOT NULL DEFAULT '[]',
    rendered_clip_path TEXT,
    updated_at TEXT NOT NULL,
    UNIQUE (project_id, scene_index)
);

CREATE TABLE IF NOT EXISTS render_jobs (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects (id) ON DELETE CASCADE,
    status TEXT NOT NULL CHECK (
        status IN ('queued', 'running', 'completed', 'failed')
    ),
    total_frames INTEGER NOT NULL CHECK (total_frames > 0),
    rendered_frames INTEGER NOT NULL DEFAULT 0 CHECK (rendered_frames >= 0),
    burn_subtitles INTEGER NOT NULL DEFAULT 1 CHECK (burn_subtitles IN (0, 1)),
    output_mp4_path TEXT,
    error_message TEXT,
    started_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_dialogue_lines_project_start
    ON dialogue_lines (project_id, start_ms);
CREATE INDEX IF NOT EXISTS idx_word_timestamps_line_start
    ON word_timestamps (line_id, start_ms);
CREATE INDEX IF NOT EXISTS idx_word_timestamps_window
    ON word_timestamps (start_ms, end_ms);
CREATE INDEX IF NOT EXISTS idx_scenes_project_status
    ON scenes (project_id, validation_status);
CREATE INDEX IF NOT EXISTS idx_render_jobs_project_status
    ON render_jobs (project_id, status);
