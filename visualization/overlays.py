"""Text bar drawn on a rendered frame. Display only."""

from __future__ import annotations

import numpy as np


def caption(frame: np.ndarray, text: str) -> np.ndarray:
    img = np.asarray(frame, np.uint8).copy()
    bar = 28
    img[:bar] = (18, 18, 18)
    try:
        from PIL import Image, ImageDraw

        im = Image.fromarray(img)
        ImageDraw.Draw(im).text((8, 6), text, fill=(255, 255, 255))
        return np.asarray(im)
    except Exception:
        return img
