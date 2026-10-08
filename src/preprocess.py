"""ONE preprocessing path used by training, evaluation, Grad-CAM, the guard and both web apps.

Anything PIL can open (8-bit, 16-bit, RGBA, palette, EXIF-rotated...) -> 8-bit greyscale ->
percentile contrast stretch -> crop the borders -> resize to IMG_SIZE x IMG_SIZE (uint8).
The contrast stretch removes brightness/exposure differences between scanners, which the old
model could use as a shortcut, and is also why X-rays from other sources behave better.
"""
import numpy as np
from PIL import Image, ImageOps

from .config import BORDER_CROP, IMG_SIZE

_HIGH_BIT = {"I", "I;16", "I;16L", "I;16B", "I;16N", "F"}


def to_gray(img: Image.Image) -> Image.Image:
    """Any PIL image -> 8-bit greyscale ('L')."""
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    if img.mode in _HIGH_BIT:                       # 16/32-bit medical PNG/TIFF exports
        a = np.asarray(img).astype(np.float32)
        lo, hi = np.percentile(a, [0.5, 99.5])
        if hi - lo < 1e-6:
            hi = lo + 1.0
        a = np.clip((a - lo) / (hi - lo), 0, 1) * 255.0
        return Image.fromarray(a.astype(np.uint8))
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        img = Image.alpha_composite(Image.new("RGBA", rgba.size, (0, 0, 0, 255)), rgba)
    return img.convert("L")


def prepare(img: Image.Image) -> np.ndarray:
    """PIL image -> uint8 array (IMG_SIZE, IMG_SIZE)."""
    g = ImageOps.autocontrast(to_gray(img), cutoff=1)
    w, h = g.size
    cx, cy = int(w * BORDER_CROP), int(h * BORDER_CROP)
    if BORDER_CROP > 0 and w - 2 * cx > 32 and h - 2 * cy > 32:
        g = g.crop((cx, cy, w - cx, h - cy))
    g = g.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    return np.asarray(g, dtype=np.uint8)


def to_model_input(x_u8: np.ndarray) -> np.ndarray:
    """uint8 (H,W) or (B,H,W) -> float32 (B,H,W,3) in 0..255. Scaling happens INSIDE the model."""
    x = np.asarray(x_u8, dtype=np.float32)
    if x.ndim == 2:
        x = x[None]
    return np.repeat(x[..., None], 3, axis=-1)
