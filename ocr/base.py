from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from PIL import Image


class OCRBackend(ABC):
    """Pluggable OCR backend. Pipeline must not hardcode a library."""

    name: str = "base"
    model_version: str = "unknown"

    @abstractmethod
    def detect(self, image: Image.Image | Any) -> list[dict[str, Any]]:
        """Detect text regions. Each item: bbox, confidence (optional)."""

    @abstractmethod
    def recognize(self, image: Image.Image | Any) -> str:
        """Recognize text from a (typically cropped) image."""

    def process(self, image: Image.Image | Any) -> dict[str, Any]:
        """Full detect+recognize pipeline result."""
        text = self.recognize(image)
        boxes = self.detect(image)
        return {
            "text": text,
            "boxes": boxes,
            "backend": self.name,
            "model_version": self.model_version,
        }


def get_backend(name: str | None = None, *, use_gpu: bool = True, lang: str = "mar") -> OCRBackend:
    from ocr.factory import create_backend

    return create_backend(name, use_gpu=use_gpu, lang=lang)
