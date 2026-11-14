"""SQLite cache: every request and response is stored, keyed by a hash of the full request.

Re-running any script never pays for the same call twice, and the database is the
raw record we release with the paper.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path

from .data import ROOT

DB_PATH = ROOT / "cache" / "calls_v1.sqlite"
# Note: the laptop folder is a mounted Windows drive; WAL mode and journal-file deletion
# do not work there, so the journal is kept in memory. One writer process at a time.


def request_key(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class Cache:
    def __init__(self, path: Path = DB_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=60)
        self.db.execute("PRAGMA journal_mode=MEMORY")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS calls (
                 key TEXT PRIMARY KEY, created REAL, model TEXT, provider TEXT,
                 request TEXT, response TEXT, prompt_tokens INT, completion_tokens INT,
                 reasoning_tokens INT, cost REAL)"""
        )
        self.db.execute("CREATE TABLE IF NOT EXISTS prompts (hash TEXT PRIMARY KEY, text TEXT)")
        self.db.commit()

    def get(self, key: str):
        with self._lock:
            row = self.db.execute("SELECT response FROM calls WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, payload: dict, response: dict):
        u = response.get("usage") or {}
        rt = (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        # store the long shared system prompt once, by hash, to keep the database small
        stored = json.loads(json.dumps(payload))
        for m in stored["messages"]:
            if m["role"] == "system":
                h = hashlib.sha256(m["content"].encode()).hexdigest()
                with self._lock:
                    self.db.execute("INSERT OR IGNORE INTO prompts VALUES (?,?)", (h, m["content"]))
                m["content"] = "sha256:" + h
        payload = stored
        with self._lock:
            self.db.execute(
                "INSERT OR REPLACE INTO calls VALUES (?,?,?,?,?,?,?,?,?,?)",
                (key, time.time(), payload["model"], json.dumps(payload.get("provider")),
                 json.dumps(payload, ensure_ascii=False), json.dumps(response, ensure_ascii=False),
                 u.get("prompt_tokens"), u.get("completion_tokens"), rt, u.get("cost")),
            )
            self.db.commit()
