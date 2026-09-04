from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from pipeline.env_loader import load_dotenv

_ROOT: Path | None = None


def get_project_root() -> Path:
    global _ROOT
    if _ROOT is None:
        _ROOT = Path(__file__).resolve().parents[1]
    return _ROOT


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    root = get_project_root()
    load_dotenv(root / ".env")
    cfg_path = Path(path) if path else root / "configs" / "config.yaml"
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Env overrides
    backend = os.environ.get("OCR_BACKEND")
    if backend:
        cfg["ocr_backend"] = backend.strip().lower()

    if os.environ.get("LLM_ENABLED"):
        cfg["llm_enabled"] = os.environ["LLM_ENABLED"].strip().lower() in {
            "1",
            "true",
            "yes",
        }
    if os.environ.get("LLM_API_URL"):
        cfg["llm_api_url"] = os.environ["LLM_API_URL"]
    if os.environ.get("LLM_API_KEY"):
        cfg["llm_api_key"] = os.environ["LLM_API_KEY"]
    if os.environ.get("LLM_MODEL"):
        cfg["llm_model"] = os.environ["LLM_MODEL"]
    if os.environ.get("ACTIVE_PROFILE"):
        cfg["active_profile"] = os.environ["ACTIVE_PROFILE"].strip()

    cfg["_root"] = str(root)
    return cfg


def resolve_path(cfg: dict[str, Any], key: str) -> Path:
    root = Path(cfg.get("_root") or get_project_root())
    rel = cfg.get("paths", {}).get(key, key)
    return (root / rel).resolve()
