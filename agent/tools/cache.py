"""Exact, versioned SQLite cache for search, page, and evidence artifacts."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

CacheLayer = Literal["search", "page", "evidence"]
CacheMode = Literal["read_write", "read_only", "refresh", "disabled"]


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class SQLiteCache:
    def __init__(self, path: str | Path, namespace: str, mode: CacheMode = "read_write") -> None:
        if not namespace.strip():
            raise ValueError("cache namespace must not be blank")
        if mode not in {"read_write", "read_only", "refresh", "disabled"}:
            raise ValueError(f"unsupported cache mode: {mode}")
        self.path = Path(path)
        self.namespace = namespace
        self.mode = mode
        self._connection: sqlite3.Connection | None = None

    def open(self) -> "SQLiteCache":
        if self.mode == "disabled":
            return self
        if self.mode == "read_only" and not self.path.exists():
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.mode == "read_only":
            self._connection = sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True, timeout=15, check_same_thread=False)
            self._connection.execute("PRAGMA query_only=ON")
            return self
        self._connection = sqlite3.connect(self.path, timeout=15, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA busy_timeout=15000")
        self._connection.execute(
            """CREATE TABLE IF NOT EXISTS cache_entries (
                namespace TEXT NOT NULL, layer TEXT NOT NULL, cache_key TEXT NOT NULL,
                value_json TEXT NOT NULL, created_at TEXT NOT NULL,
                status_class TEXT, retries INTEGER NOT NULL, content_hash TEXT,
                PRIMARY KEY(namespace, layer, cache_key))"""
        )
        self._connection.execute(
            """CREATE TABLE IF NOT EXISTS serper_key_blacklist (
                key_fingerprint TEXT PRIMARY KEY, http_status INTEGER NOT NULL,
                reason TEXT NOT NULL, first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL, occurrences INTEGER NOT NULL
            )"""
        )
        self._connection.commit()
        return self

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> "SQLiteCache":
        return self.open()

    def __exit__(self, *_: object) -> None:
        self.close()

    def get(self, layer: CacheLayer, key_parts: Any) -> dict[str, Any] | None:
        if self.mode in {"disabled", "refresh"} or self._connection is None:
            return None
        row = self._connection.execute(
            "SELECT value_json FROM cache_entries WHERE namespace=? AND layer=? AND cache_key=?",
            (self.namespace, layer, canonical_hash(key_parts)),
        ).fetchone()
        return json.loads(row[0]) if row else None

    def put(
        self, layer: CacheLayer, key_parts: Any, value: dict[str, Any], *,
        status_class: str = "ok", retries: int = 0, content_hash: str | None = None,
    ) -> None:
        if self.mode not in {"read_write", "refresh"} or self._connection is None:
            return
        serialized = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        key = canonical_hash(key_parts)
        with self._connection:
            self._connection.execute(
                "INSERT OR IGNORE INTO cache_entries VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (self.namespace, layer, key, serialized, datetime.now(UTC).isoformat(),
                 status_class, retries, content_hash),
            )

    def manifest(self) -> dict[str, Any]:
        if self._connection is None:
            return {"namespace": self.namespace, "entries": [], "sha256": canonical_hash([])}
        rows = self._connection.execute(
            "SELECT layer, cache_key, content_hash FROM cache_entries WHERE namespace=? ORDER BY layer, cache_key",
            (self.namespace,),
        ).fetchall()
        entries = [{"layer": x[0], "key": x[1], "content_hash": x[2]} for x in rows]
        return {"namespace": self.namespace, "entries": entries, "sha256": canonical_hash(entries)}

    def serper_blacklisted_fingerprints(self) -> set[str]:
        if self._connection is None:
            return set()
        rows = self._connection.execute(
            "SELECT key_fingerprint FROM serper_key_blacklist"
        ).fetchall()
        return {str(row[0]) for row in rows}

    def blacklist_serper_key(self, fingerprint: str, *, http_status: int, reason: str) -> None:
        if self.mode not in {"read_write", "refresh"} or self._connection is None:
            return
        now = datetime.now(UTC).isoformat()
        with self._connection:
            self._connection.execute(
                """INSERT INTO serper_key_blacklist
                    (key_fingerprint, http_status, reason, first_seen_at, last_seen_at, occurrences)
                    VALUES (?, ?, ?, ?, ?, 1)
                    ON CONFLICT(key_fingerprint) DO UPDATE SET
                      http_status=excluded.http_status,
                      reason=excluded.reason,
                      last_seen_at=excluded.last_seen_at,
                      occurrences=serper_key_blacklist.occurrences + 1""",
                (fingerprint, http_status, reason[:300], now, now),
            )
