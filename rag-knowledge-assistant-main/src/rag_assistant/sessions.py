"""In-memory conversational memory for sessions (Phase 2.3).

Sessions track the last ``max_turns`` turns per ``session_id``; entries idle
longer than ``ttl_seconds`` are dropped lazily on access.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from threading import Lock


class SessionStore:
    """Thread-safe, TTL-bounded per-session turn history."""

    def __init__(self, max_turns: int = 10, ttl_seconds: float = 3600.0) -> None:
        self._lock = Lock()
        self._turns: dict[str, deque] = defaultdict(deque)
        self._last_seen: dict[str, float] = defaultdict(float)
        self._max_turns = max_turns
        self._ttl_seconds = ttl_seconds

    def history(self, session_id: str) -> list[dict[str, str]]:
        """Return the session's turns, oldest first (empty when expired/unknown)."""
        if not session_id:
            return []
        with self._lock:
            now = time.time()
            if session_id not in self._turns:
                return []
            if now - self._last_seen[session_id] > self._ttl_seconds:
                self._turns.pop(session_id, None)
                self._last_seen.pop(session_id, None)
                return []
            self._last_seen[session_id] = now
            return [dict(turn) for turn in self._turns[session_id]]

    def append(self, session_id: str, role: str, content: str) -> None:
        if not session_id:
            return
        with self._lock:
            self._last_seen[session_id] = time.time()
            self._turns[session_id].append({"role": role, "content": content})

    def clear(self, session_id: str) -> int:
        """Drop the session's history; returns the number of turns removed."""
        with self._lock:
            turns = self._turns.pop(session_id, None)
            self._last_seen.pop(session_id, None)
            return len(turns) if turns else 0

    def ids(self) -> list[str]:
        with self._lock:
            return sorted(self._turns)
