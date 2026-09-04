from __future__ import annotations

import logging
import sys
from typing import Any


def setup_logging(cfg: dict[str, Any] | None = None) -> None:
    log_cfg = (cfg or {}).get("logging", {})
    level = getattr(logging, str(log_cfg.get("level", "INFO")).upper(), logging.INFO)
    fmt = log_cfg.get(
        "format", "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(handler)
    root.setLevel(level)
