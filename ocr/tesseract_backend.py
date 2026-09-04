from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

from ocr.base import OCRBackend

logger = logging.getLogger(__name__)


class TesseractBackend(OCRBackend):
    """Tesseract OCR via CLI (requires `tesseract` + mar traineddata)."""

    name = "tesseract"
    model_version = "tesseract-mar"

    def __init__(self, *, use_gpu: bool = False, lang: str = "mar") -> None:
        self.lang = lang if lang else "mar"
        if not shutil.which("tesseract"):
            raise RuntimeError("tesseract binary not found on PATH")

    def detect(self, image: Image.Image | Any) -> list[dict[str, Any]]:
        # Tesseract CLI path does not expose rich detection for MVP; return empty.
        return []

    def recognize(self, image: Image.Image | Any) -> str:
        img = image if isinstance(image, Image.Image) else Image.fromarray(image)
        img = img.convert("RGB")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "crop.png"
            out_base = tmp_path / "out"
            img.save(in_path, format="PNG")
            best = ""
            for psm in (7, 6, 13):
                cmd = [
                    "tesseract",
                    str(in_path),
                    str(out_base),
                    "-l",
                    self.lang,
                    "--psm",
                    str(psm),
                ]
                subprocess.run(
                    cmd,
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                txt_path = out_base.with_suffix(".txt")
                if txt_path.is_file():
                    text = txt_path.read_text(encoding="utf-8", errors="replace").strip()
                    if text and len(text) > len(best):
                        best = text
            return " ".join(best.split())
