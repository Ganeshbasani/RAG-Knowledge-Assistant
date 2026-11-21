"""Shared test setup: keep test runs out of the repository's data directory."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# The API module reads AppConfig.from_env() at import time; redirect storage to
# a writable temp location unless the developer/CI explicitly set paths.
os.environ.setdefault("RAG_STORAGE_PATH", str(Path(tempfile.gettempdir()) / "rag-knowledge-assistant-test" / "index.json"))
os.environ.setdefault("RAG_UPLOAD_DIR", str(Path(tempfile.gettempdir()) / "rag-knowledge-assistant-test" / "uploads"))
