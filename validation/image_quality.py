from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def laplacian_variance(img: Image.Image) -> float:
    gray = np.asarray(img.convert("L"), dtype=np.float64)
    # Simple Laplacian kernel
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    # Valid convolution via pad
    padded = np.pad(gray, 1, mode="edge")
    out = (
        k[0, 1] * padded[0:-2, 1:-1]
        + k[1, 0] * padded[1:-1, 0:-2]
        + k[1, 1] * padded[1:-1, 1:-1]
        + k[1, 2] * padded[1:-1, 2:]
        + k[2, 1] * padded[2:, 1:-1]
    )
    return float(out.var())


def check_image_quality(
    path: str | Path,
    *,
    min_width: int = 800,
    min_height: int = 100,
    min_aspect: float = 4.0,
    max_height: int = 400,
    min_blur_var: float = 20.0,
) -> dict[str, Any]:
    p = Path(path)
    issues: list[str] = []
    if not p.is_file():
        return {"ok": False, "issues": ["missing_file"], "width": 0, "height": 0}
    try:
        with Image.open(p) as img:
            img.load()
            w, h = img.size
            dpi = img.info.get("dpi", (0, 0))
            aspect = w / max(h, 1)
            blur = laplacian_variance(img)
            contrast = float(np.asarray(img.convert("L")).std())
    except OSError as exc:
        return {"ok": False, "issues": [f"corrupt:{exc}"], "width": 0, "height": 0}

    if w < min_width:
        issues.append("narrow_width")
    if h < min_height:
        issues.append("short_height")
    if h > max_height:
        issues.append("too_tall")
    if aspect < min_aspect:
        issues.append("low_aspect")
    if blur < min_blur_var:
        issues.append("blurry")
    if contrast < 15:
        issues.append("low_contrast")

    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "width": w,
        "height": h,
        "dpi": dpi,
        "aspect_ratio": round(aspect, 3),
        "blur_variance": round(blur, 2),
        "contrast_std": round(contrast, 2),
    }
