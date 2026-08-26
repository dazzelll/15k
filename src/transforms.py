"""Official robustness protocol + training-time sampling of those ops."""

from __future__ import annotations

import io
import random
from typing import Callable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


JPEG_QUALITIES = (90, 70, 50, 30)
BLUR_SIGMAS = (0.5, 1.0, 2.0)
RESIZE_SCALES = (0.5, 0.25)
NOISE_SIGMAS = (0.02, 0.05, 0.10)
COLOR_JITTER = 0.20
CENTER_CROP = 0.80


def _to_rgb(img: Image.Image) -> Image.Image:
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def jpeg_compress(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    _to_rgb(img).save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def gaussian_blur(img: Image.Image, sigma: float) -> Image.Image:
    return _to_rgb(img).filter(ImageFilter.GaussianBlur(radius=float(sigma)))


def down_up_resize(img: Image.Image, scale: float) -> Image.Image:
    img = _to_rgb(img)
    w, h = img.size
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    small = img.resize((nw, nh), Image.Resampling.BILINEAR)
    return small.resize((w, h), Image.Resampling.BILINEAR)


def gaussian_noise(img: Image.Image, sigma: float) -> Image.Image:
    arr = np.asarray(_to_rgb(img), dtype=np.float32) / 255.0
    noise = np.random.normal(0.0, sigma, arr.shape).astype(np.float32)
    out = np.clip(arr + noise, 0.0, 1.0)
    return Image.fromarray((out * 255.0).round().astype(np.uint8), mode="RGB")


def color_jitter(img: Image.Image, amount: float = COLOR_JITTER) -> Image.Image:
    img = _to_rgb(img)
    b = 1.0 + random.uniform(-amount, amount)
    c = 1.0 + random.uniform(-amount, amount)
    s = 1.0 + random.uniform(-amount, amount)
    img = ImageEnhance.Brightness(img).enhance(b)
    img = ImageEnhance.Contrast(img).enhance(c)
    img = ImageEnhance.Color(img).enhance(s)
    return img


def center_crop(img: Image.Image, fraction: float = CENTER_CROP) -> Image.Image:
    img = _to_rgb(img)
    w, h = img.size
    nw, nh = max(1, int(w * fraction)), max(1, int(h * fraction))
    cropped = ImageOps.fit(img, (nw, nh), method=Image.Resampling.BILINEAR, centering=(0.5, 0.5))
    return cropped.resize((w, h), Image.Resampling.BILINEAR)


PROTOCOL: dict[str, list[tuple[str, Callable[[Image.Image], Image.Image]]]] = {
    "clean": [("clean", lambda im: _to_rgb(im))],
    "jpeg": [(f"jpeg_q{q}", lambda im, q=q: jpeg_compress(im, q)) for q in JPEG_QUALITIES],
    "blur": [(f"blur_s{s}", lambda im, s=s: gaussian_blur(im, s)) for s in BLUR_SIGMAS],
    "resize": [(f"resize_x{s}", lambda im, s=s: down_up_resize(im, s)) for s in RESIZE_SCALES],
    "noise": [(f"noise_s{s}", lambda im, s=s: gaussian_noise(im, s)) for s in NOISE_SIGMAS],
    "color_jitter": [("color_jitter", color_jitter)],
    "center_crop": [("center_crop", center_crop)],
}


def apply_named(img: Image.Image, name: str) -> Image.Image:
    for variants in PROTOCOL.values():
        for vname, fn in variants:
            if vname == name:
                return fn(img)
    raise KeyError(f"Unknown transform {name}")


def random_protocol_transform(img: Image.Image) -> Image.Image:
    """Sample one official transform family, then one parameter setting."""
    family = random.choice([k for k in PROTOCOL if k != "clean"])
    _name, fn = random.choice(PROTOCOL[family])
    return fn(img)


class ProtocolTrainTransform:
    """Match the evaluation protocol at train time (the main robustness lever)."""

    def __init__(self, p: float = 0.85):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        img = _to_rgb(img)
        if random.random() < self.p:
            img = random_protocol_transform(img)
        if random.random() < 0.3:
            img = random_protocol_transform(img)
        return img
