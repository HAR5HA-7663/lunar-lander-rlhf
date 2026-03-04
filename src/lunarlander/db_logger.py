"""SQLite experiment logger shared between Python notebooks and Spring Boot."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS experiments (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT    NOT NULL,
    exp_type     TEXT    NOT NULL,
    timestamp    TEXT    NOT NULL,
    mean_reward  REAL,
    std_reward   REAL,
    success_rate REAL,
    crash_rate   REAL,
    mean_ep_len  REAL,
    hyperparams  TEXT,
    notes        TEXT
);
"""


class ExperimentLogger:
    """Logs experiment results to a SQLite database."""

    def __init__(self, db_path: str | Path = "experiments.db"):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(CREATE_TABLE_SQL)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def log(
        self,
        name: str,
        exp_type: str,
        mean_reward: float | None = None,
        std_reward: float | None = None,
        success_rate: float | None = None,
        crash_rate: float | None = None,
        mean_ep_len: float | None = None,
        hyperparams: dict | None = None,
        notes: str | None = None,
    ) -> int:
        """Insert an experiment row and return the new row id."""
        timestamp = datetime.now(timezone.utc).isoformat()
        hyperparams_json = json.dumps(hyperparams) if hyperparams is not None else None

        sql = """
        INSERT INTO experiments
            (name, exp_type, timestamp, mean_reward, std_reward,
             success_rate, crash_rate, mean_ep_len, hyperparams, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._connect() as conn:
            cur = conn.execute(
                sql,
                (
                    name,
                    exp_type,
                    timestamp,
                    mean_reward,
                    std_reward,
                    success_rate,
                    crash_rate,
                    mean_ep_len,
                    hyperparams_json,
                    notes,
                ),
            )
            return cur.lastrowid

    def fetch_all(self) -> list[dict]:
        """Return all experiments as a list of dicts."""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM experiments ORDER BY timestamp DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def fetch_by_type(self, exp_type: str) -> list[dict]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM experiments WHERE exp_type=? ORDER BY timestamp DESC",
                (exp_type,),
            ).fetchall()
            return [dict(r) for r in rows]
