-- 0002_add_edge_tts_engine.sql
-- Habilita el motor edge_tts en el catálogo de voces sin perder filas existentes.

CREATE TABLE voices_0002 (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    engine TEXT NOT NULL CHECK (
        engine IN ('kokoro', 'piper', 'edge_tts', 'xtts', 'f5', 'elevenlabs', 'openai')
    ),
    mode TEXT NOT NULL CHECK (mode IN ('preset', 'clone')),
    language TEXT NOT NULL DEFAULT 'es' CHECK (language IN ('es', 'en')),
    sample_wav_path TEXT,
    config_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    CHECK (mode = 'preset' OR sample_wav_path IS NOT NULL)
);

INSERT INTO voices_0002 (
    id, name, engine, mode, language, sample_wav_path, config_json, created_at
)
SELECT
    id, name, engine, mode, language, sample_wav_path, config_json, created_at
FROM voices;

DROP TABLE voices;

ALTER TABLE voices_0002 RENAME TO voices;
