from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class OCRCache:
    """Disk cache keyed by (sha256, backend, model_version)."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_path(self, sha256: str, backend: str, model_version: str) -> Path:
        safe_ver = model_version.replace("/", "_").replace(" ", "_")
        name = f"{sha256[:16]}_{backend}_{safe_ver}.json"
        return self.cache_dir / name

    def get(self, sha256: str, backend: str, model_version: str) -> dict[str, Any] | None:
        path = self._key_path(sha256, backend, model_version)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Bad OCR cache entry %s: %s", path, exc)
            return None

    def set(
        self,
        sha256: str,
        backend: str,
        model_version: str,
        payload: dict[str, Any],
    ) -> Path:
        path = self._key_path(sha256, backend, model_version)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


class ResumeStore:
    """Tracks completed IDs for restartable stages."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._done: set[str] = set()
        if self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self._done.add(line.strip())

    def is_done(self, item_id: str) -> bool:
        return item_id in self._done

    def mark_done(self, item_id: str) -> None:
        if item_id in self._done:
            return
        self._done.add(item_id)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(item_id + "\n")
