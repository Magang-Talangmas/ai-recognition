from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS attendance_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id TEXT NOT NULL,
    track_id INTEGER NOT NULL,
    employee_id TEXT,
    original_candidate_id TEXT,
    direction TEXT NOT NULL,
    event_type TEXT NOT NULL,
    similarity REAL NOT NULL,
    status TEXT NOT NULL,
    employee_response TEXT,
    detected_at TEXT NOT NULL,
    responded_at TEXT
);
"""


class AttendanceDB:
    def __init__(self, path: str, duplicate_cooldown_seconds: int) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.duplicate_cooldown_seconds = duplicate_cooldown_seconds
        with self.connect() as conn:
            conn.execute(SCHEMA)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def create_pending_event(
        self,
        *,
        camera_id: str,
        track_id: int,
        employee_id: str,
        direction: str,
        event_type: str,
        similarity: float,
    ) -> int | None:
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=self.duplicate_cooldown_seconds)

        with self.connect() as conn:
            existing = conn.execute(
                """
                SELECT id
                FROM attendance_events
                WHERE camera_id = ?
                  AND employee_id = ?
                  AND event_type = ?
                  AND status = 'PENDING_CONFIRMATION'
                  AND detected_at >= ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (
                    camera_id,
                    employee_id,
                    event_type,
                    cutoff.isoformat(),
                ),
            ).fetchone()

            if existing:
                return None

            cursor = conn.execute(
                """
                INSERT INTO attendance_events (
                    camera_id,
                    track_id,
                    employee_id,
                    original_candidate_id,
                    direction,
                    event_type,
                    similarity,
                    status,
                    detected_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, 'PENDING_CONFIRMATION', ?)
                """,
                (
                    camera_id,
                    track_id,
                    employee_id,
                    employee_id,
                    direction,
                    event_type,
                    similarity,
                    now.isoformat(),
                ),
            )
            return int(cursor.lastrowid)

    def respond(self, event_id: int, action: str) -> None:
        now = datetime.now(timezone.utc)

        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM attendance_events WHERE id = ?",
                (event_id,),
            ).fetchone()

            if row is None:
                raise ValueError("Event not found")
            if row["status"] != "PENDING_CONFIRMATION":
                raise ValueError("Event is no longer pending")

            if action == "reject":
                conn.execute(
                    """
                    UPDATE attendance_events
                    SET employee_id = NULL,
                        status = 'UNRESOLVED_RECOGNITION',
                        employee_response = 'NOT_ME',
                        responded_at = ?
                    WHERE id = ?
                    """,
                    (now.isoformat(), event_id),
                )
                return

            event_mapping = {
                "confirm": "CHECK_IN",
                "break": "START_BREAK",
                "temporary_exit": "TEMPORARY_EXIT",
                "checkout": "CHECK_OUT",
                "return_break": "RETURN_FROM_BREAK",
                "return_temporary": "RETURN_FROM_TEMPORARY_EXIT",
            }
            resolved_event = event_mapping.get(action)
            if resolved_event is None:
                raise ValueError(f"Unsupported action: {action}")

            conn.execute(
                """
                UPDATE attendance_events
                SET event_type = ?,
                    status = 'CONFIRMED',
                    employee_response = ?,
                    responded_at = ?
                WHERE id = ?
                """,
                (
                    resolved_event,
                    action.upper(),
                    now.isoformat(),
                    event_id,
                ),
            )

    def list_events(self, limit: int = 100) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return list(
                conn.execute(
                    """
                    SELECT *
                    FROM attendance_events
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            )
