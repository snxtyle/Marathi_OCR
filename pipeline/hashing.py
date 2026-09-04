from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image
import imagehash


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def perceptual_hash_file(path: str | Path, hash_size: int = 16) -> str:
    with Image.open(path) as img:
        img = img.convert("RGB")
        return str(imagehash.phash(img, hash_size=hash_size))


def perceptual_hash_distance(a: str, b: str) -> int:
    return imagehash.hex_to_hash(a) - imagehash.hex_to_hash(b)
