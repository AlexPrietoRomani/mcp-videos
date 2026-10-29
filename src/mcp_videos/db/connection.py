"""SQLite connection management, WAL configuration and migration runner."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Final

DEFAULT_DB_PATH: Final = Path("data/mcp_videos.sqlite")
DEFAULT_MIGRATIONS_DIR: Final = Path(__file__).resolve().parent / "migrations"

_MIGRATIONS_TABLE_DDL: Final = """
CREATE TABLE IF NOT EXISTS _schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
)
"""


def _split_sql_statements(script: str) -> list[str]:
    """Split a SQL script into complete statements understood by sqlite3."""

    statements: list[str] = []
    buffer = ""
    for line in script.splitlines(keepends=True):
        buffer += line
        if sqlite3.complete_statement(buffer):
            statement = buffer.strip()
            if statement:
                statements.append(statement)
            buffer = ""
    pending = buffer.strip()
    if pending and any(
        line.strip() and not line.strip().startswith("--") for line in pending.splitlines()
    ):
        statements.append(pending)
    return statements


class DatabaseManager:
    """Create configured SQLite connections and apply versioned migrations."""

    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB_PATH,
        migrations_dir: str | Path | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.migrations_dir = (
            Path(migrations_dir) if migrations_dir is not None else DEFAULT_MIGRATIONS_DIR
        )
        if not self.is_memory:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def is_memory(self) -> bool:
        """Whether the manager is backed by a transient in-memory database."""

        return str(self.db_path) == ":memory:"

    def connect(self) -> sqlite3.Connection:
        """Open a connection with WAL journaling and strict referential integrity."""

        conn = sqlite3.connect(str(self.db_path), timeout=5.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Yield a configured connection and always close it afterwards."""

        conn = self.connect()
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run a block inside an atomic write transaction."""

        conn = self.connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def apply_migrations(self) -> list[str]:
        """Apply pending ``.sql`` migrations once and return their versions."""

        if not self.migrations_dir.is_dir():
            raise FileNotFoundError(
                f"No existe el directorio de migraciones: {self.migrations_dir}"
            )

        applied: list[str] = []
        with self.connection() as conn:
            # SQLite cannot toggle foreign_keys inside a transaction, and table
            # rebuild migrations need it disabled before BEGIN.
            conn.execute("PRAGMA foreign_keys = OFF")
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(_MIGRATIONS_TABLE_DDL)
                known = {
                    row["version"] for row in conn.execute("SELECT version FROM _schema_migrations")
                }
                for migration_path in sorted(self.migrations_dir.glob("*.sql")):
                    version = migration_path.name
                    if version in known:
                        continue
                    script = migration_path.read_text(encoding="utf-8")
                    for statement in _split_sql_statements(script):
                        conn.execute(statement)
                    conn.execute(
                        "INSERT INTO _schema_migrations (version) VALUES (?)",
                        (version,),
                    )
                    applied.append(version)
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
            finally:
                conn.execute("PRAGMA foreign_keys = ON")
        return applied

    def migrate(self) -> list[str]:
        """Alias for :meth:`apply_migrations` used by higher-level services."""

        return self.apply_migrations()

    def initialize(self) -> list[str]:
        """Prepare the database schema for first use."""

        return self.apply_migrations()


__all__ = [
    "DEFAULT_DB_PATH",
    "DEFAULT_MIGRATIONS_DIR",
    "DatabaseManager",
]
