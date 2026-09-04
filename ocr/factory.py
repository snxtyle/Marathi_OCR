from __future__ import annotations

import logging
from typing import Any

from ocr.base import OCRBackend

logger = logging.getLogger(__name__)


def create_backend(
    name: str | None = None,
    *,
    use_gpu: bool = True,
    lang: str = "mar",
) -> OCRBackend:
    """Pluggable OCR factory. Default Tesseract-mar until bakeoff sets winner."""
    key = (name or "tesseract").strip().lower()
    if key in {"tesseract", "tess", "mar", ""}:
        from ocr.tesseract_backend import TesseractBackend

        return TesseractBackend(use_gpu=False, lang=lang or "mar")
    if key in {"paddle", "paddleocr"}:
        from ocr.paddleocr_backend import PaddleOCRBackend

        return PaddleOCRBackend(use_gpu=use_gpu, lang=lang)
    if key == "surya":
        from ocr.surya_backend import SuryaBackend

        return SuryaBackend(use_gpu=use_gpu, lang=lang)
    raise ValueError(
        f"Unknown OCR backend: {name!r}. Use tesseract|paddleocr|surya"
    )
