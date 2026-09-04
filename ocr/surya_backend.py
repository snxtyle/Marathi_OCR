from __future__ import annotations

import logging
import re
from typing import Any

from PIL import Image

from ocr.base import OCRBackend

logger = logging.getLogger(__name__)


def _html_to_text(html: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html or "")
    return " ".join(text.split())


class SuryaBackend(OCRBackend):
    """Surya OCR adapter (optional; needs surya-ocr + llama.cpp for recent versions)."""

    name = "surya"
    model_version = "surya-fullpage"

    def __init__(self, *, use_gpu: bool = True, lang: str = "mar") -> None:
        self.use_gpu = use_gpu
        self.lang = lang
        self._ready = False
        self._recognition_predictor = None

    def _ensure(self) -> None:
        if self._ready:
            return
        try:
            from surya.inference import SuryaInferenceManager
            from surya.recognition import RecognitionPredictor
        except ImportError as exc:
            raise ImportError(
                "Surya is not installed. Optional: pip install surya-ocr torch"
            ) from exc
        manager = SuryaInferenceManager()
        self._recognition_predictor = RecognitionPredictor(manager)
        self._ready = True
        logger.info("Initialized Surya OCR")

    def detect(self, image: Image.Image | Any) -> list[dict[str, Any]]:
        return []

    def recognize(self, image: Image.Image | Any) -> str:
        self._ensure()
        img = image if isinstance(image, Image.Image) else Image.fromarray(image)
        preds = self._recognition_predictor([img], full_page=True)
        if not preds:
            return ""
        texts: list[str] = []
        for block in getattr(preds[0], "blocks", []) or []:
            plain = _html_to_text(getattr(block, "html", "") or "")
            if plain:
                texts.append(plain)
        # Legacy text_lines API fallback
        if not texts:
            for line in getattr(preds[0], "text_lines", []) or []:
                texts.append(getattr(line, "text", str(line)))
        return " ".join(texts).strip()
