from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PIL import Image

from ocr.base import OCRBackend

logger = logging.getLogger(__name__)


class PaddleOCRBackend(OCRBackend):
    """PaddleOCR Devanagari/Hindi path (Marathi script). Optional dependency."""

    name = "paddleocr"
    model_version = "paddleocr-devanagari"

    def __init__(self, *, use_gpu: bool = True, lang: str = "mar") -> None:
        self.use_gpu = use_gpu
        self.lang = lang
        self._ocr = None

    def _ensure(self) -> Any:
        if self._ocr is not None:
            return self._ocr
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise ImportError(
                "PaddleOCR is not installed. pip install paddleocr paddlepaddle"
            ) from exc
        last_err: Exception | None = None
        attempts = [
            {"lang": "hi", "use_textline_orientation": True},
            {"lang": "hi"},
            {"lang": "mr"},
            {"lang": "en"},
        ]
        for kwargs in attempts:
            try:
                self._ocr = PaddleOCR(**kwargs)
                self.model_version = f"paddleocr-{kwargs.get('lang', 'default')}"
                logger.info("Initialized PaddleOCR with %s", kwargs)
                return self._ocr
            except Exception as exc:  # noqa: BLE001
                last_err = exc
        raise RuntimeError(f"Could not initialize PaddleOCR: {last_err}")

    def detect(self, image: Image.Image | Any) -> list[dict[str, Any]]:
        return []

    def recognize(self, image: Image.Image | Any) -> str:
        ocr = self._ensure()
        img = image if isinstance(image, Image.Image) else Image.fromarray(image)
        arr = np.array(img.convert("RGB"))
        try:
            result = ocr.ocr(arr)
        except TypeError:
            result = ocr.predict(arr)
        texts: list[str] = []
        # Handle both classic and paddlex-style outputs
        if isinstance(result, list):
            for page in result:
                if not page:
                    continue
                if isinstance(page, dict) and "rec_texts" in page:
                    texts.extend(str(t) for t in (page.get("rec_texts") or []))
                    continue
                for line in page:
                    if isinstance(line, (list, tuple)) and len(line) >= 2:
                        piece = line[1][0] if isinstance(line[1], (list, tuple)) else line[1]
                        texts.append(str(piece))
        return " ".join(texts).strip()
