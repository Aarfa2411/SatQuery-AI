import json
import shutil
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional, Dict

from app.config import settings
from app.schemas import ValidationResult


class SessionManager:
    def __init__(self, db_path: Path = settings.DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        # Enable Write-Ahead Logging (WAL) and synchronous normal for high concurrency
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP,
                    expires_at TIMESTAMP,
                    mode TEXT,
                    validation_json TEXT,
                    query_history_json TEXT
                );
            """)
            conn.commit()

    def create_or_update_session(self, session_id: str, validation: ValidationResult) -> None:
        now = datetime.now(timezone.utc)
        expires = now + timedelta(hours=settings.SESSION_TTL_HOURS)
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO sessions (session_id, created_at, expires_at, mode, validation_json, query_history_json)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    expires_at=excluded.expires_at,
                    mode=excluded.mode,
                    validation_json=excluded.validation_json;
            """, (
                session_id,
                now.isoformat(),
                expires.isoformat(),
                validation.mode,
                validation.model_dump_json(),
                json.dumps([])
            ))
            conn.commit()

    def touch_session(self, session_id: str) -> None:
        """Sliding-window TTL reset on active user interaction."""
        now = datetime.now(timezone.utc)
        expires = now + timedelta(hours=settings.SESSION_TTL_HOURS)
        with self._get_connection() as conn:
            conn.execute("UPDATE sessions SET expires_at=? WHERE session_id=?", (expires.isoformat(), session_id))
            conn.commit()

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        self.touch_session(session_id)
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,))
            row = cur.fetchone()
            if not row:
                return None
            return {
                "session_id": row["session_id"],
                "created_at": row["created_at"],
                "expires_at": row["expires_at"],
                "mode": row["mode"],
                "validation": json.loads(row["validation_json"]) if row["validation_json"] else None,
                "query_history": json.loads(row["query_history_json"]) if row["query_history_json"] else []
            }

    def append_query_history(self, session_id: str, query_record: Dict[str, Any]) -> None:
        self.touch_session(session_id)
        session = self.get_session(session_id)
        if not session:
            return
        history = session.get("query_history", [])
        history.append(query_record)
        with self._get_connection() as conn:
            conn.execute("UPDATE sessions SET query_history_json=? WHERE session_id=?", (
                json.dumps(history), session_id
            ))
            conn.commit()

    def get_session_dir(self, session_id: str, subfolder: str = "uploads") -> Path:
        base = getattr(settings, f"{subfolder.upper()}_DIR", settings.STORAGE_DIR / subfolder)
        p = base / session_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def cleanup_expired_sessions(self) -> int:
        """Removes sessions past their sliding expiration time and purges storage."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cur = conn.execute("SELECT session_id FROM sessions WHERE expires_at < ?", (now,))
            expired = [row["session_id"] for row in cur.fetchall()]
            if expired:
                conn.execute("DELETE FROM sessions WHERE expires_at < ?", (now,))
                conn.commit()

        for sid in expired:
            for sub in [settings.UPLOADS_DIR, settings.ALIGNED_DIR, settings.OUTPUTS_DIR, settings.EXPORTS_DIR]:
                target = sub / sid
                if target.exists():
                    try:
                        shutil.rmtree(target)
                    except OSError:
                        pass
        return len(expired)

session_manager = SessionManager()
