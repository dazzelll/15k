"""Official robustness protocol + training-time sampling of those ops."""

from __future__ import annotations

import io
import random
from dataclasses import dataclass
from typing import Callable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


JPEG_QUALITIES = (90, 70, 50, 30)
BLUR_SIGMAS = (0.5, 1.0, 2.0)
RESIZE_SCALES = (0.5, 0.25)
NOISE_SIGMAS = (0.02, 0.05, 0.10)
COLOR_JITTER = 0.20
CENTER_CROP = 0.80


@dataclass(frozen=True)
class ProtocolOp:
    name: str
    fn: Callable[[Image.Image], Image.Image]
    severity: float  # 0 = clean / intact traces, 1 = heavily degraded


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


def gaussian_noise(
    img: Image.Image, sigma: float, rng: random.Random | None = None
) -> Image.Image:
    arr = np.asarray(_to_rgb(img), dtype=np.float32) / 255.0
    if rng is None:
        noise = np.random.normal(0.0, sigma, arr.shape).astype(np.float32)
    else:
        rs = np.random.RandomState(rng.randint(0, 2**31 - 1))
        noise = rs.normal(0.0, sigma, arr.shape).astype(np.float32)
    out = np.clip(arr + noise, 0.0, 1.0)
    return Image.fromarray((out * 255.0).round().astype(np.uint8), mode="RGB")


def color_jitter(
    img: Image.Image, amount: float = COLOR_JITTER, rng: random.Random | None = None
) -> Image.Image:
    img = _to_rgb(img)
    r = rng if rng is not None else random
    b = 1.0 + r.uniform(-amount, amount)
    c = 1.0 + r.uniform(-amount, amount)
    s = 1.0 + r.uniform(-amount, amount)
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


# Severity maps roughly to how much forensic traces are expected to die.
PROTOCOL: dict[str, list[ProtocolOp]] = {
    "clean": [ProtocolOp("clean", lambda im: _to_rgb(im), 0.0)],
    "jpeg": [
        ProtocolOp(f"jpeg_q{q}", (lambda im, q=q: jpeg_compress(im, q)), sev)
        for q, sev in zip(JPEG_QUALITIES, (0.25, 0.5, 0.75, 1.0))
    ],
    "blur": [
        ProtocolOp(f"blur_s{s}", (lambda im, s=s: gaussian_blur(im, s)), sev)
        for s, sev in zip(BLUR_SIGMAS, (0.35, 0.7, 1.0))
    ],
    "resize": [
        ProtocolOp(f"resize_x{s}", (lambda im, s=s: down_up_resize(im, s)), sev)
        for s, sev in zip(RESIZE_SCALES, (0.55, 1.0))
    ],
    "noise": [
        ProtocolOp(f"noise_s{s}", (lambda im, s=s: gaussian_noise(im, s)), sev)
        for s, sev in zip(NOISE_SIGMAS, (0.3, 0.65, 1.0))
    ],
    "color_jitter": [ProtocolOp("color_jitter", color_jitter, 0.25)],
    "center_crop": [ProtocolOp("center_crop", center_crop, 0.2)],
}


def iter_protocol_ops() -> list[ProtocolOp]:
    return [op for variants in PROTOCOL.values() for op in variants]


def apply_named(img: Image.Image, name: str) -> Image.Image:
    for op in iter_protocol_ops():
        if op.name == name:
            return op.fn(img)
    raise KeyError(f"Unknown transform {name}")


def severity_for(name: str) -> float:
    for op in iter_protocol_ops():
        if op.name == name:
            return op.severity
    raise KeyError(f"Unknown transform {name}")


def _apply_op(op: ProtocolOp, img: Image.Image, rng: random.Random | None) -> Image.Image:
    """Apply a protocol op, threading rng into the two stochastic families."""
    if rng is None:
        return op.fn(img)
    if op.name == "color_jitter":
        return color_jitter(img, rng=rng)
    if op.name.startswith("noise_s"):
        sigma = float(op.name.split("noise_s", 1)[1])
        return gaussian_noise(img, sigma, rng=rng)
    return op.fn(img)


def random_protocol_transform(
    img: Image.Image, rng: random.Random | None = None
) -> tuple[Image.Image, str, float]:
    """Sample one official transform family, then one parameter setting."""
    r = rng if rng is not None else random
    family = r.choice([k for k in PROTOCOL if k != "clean"])
    op = r.choice(PROTOCOL[family])
    return _apply_op(op, img, rng), op.name, op.severity


class ProtocolTrainTransform:
    """Match the evaluation protocol at train time (the main robustness lever).

    Returns (image, transform_name, severity). Severity is max over stacked ops.
    """

    def __init__(self, p: float = 0.85):
        self.p = p

    def __call__(
        self, img: Image.Image, rng: random.Random | None = None
    ) -> tuple[Image.Image, str, float]:
        r = rng if rng is not None else random
        img = _to_rgb(img)
        name = "clean"
        severity = 0.0
        if r.random() < self.p:
            img, name, severity = random_protocol_transform(img, rng=rng)
        if r.random() < 0.3:
            img, name2, sev2 = random_protocol_transform(img, rng=rng)
            name = f"{name}+{name2}" if name != "clean" else name2
            severity = max(severity, sev2)
        return img, name, float(severity)
