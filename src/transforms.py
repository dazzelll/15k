"""Official robustness protocol (fixed, for eval) + a continuous, pipeline-based
training-time protocol that mimics real redistribution paths.

Design (per team discussion):
- Training-time params are sampled continuously, not from fixed buckets.
- Severity is MEASURED from actual high-frequency signal loss, not guessed.
- Most training samples go through a named, realistic multi-step pipeline
  (resize+recompress, screenshot+recompress, etc.) rather than one random op,
  because that's how images actually get degraded when redistributed.
- The sampling distribution is skewed toward mild/moderate degradation,
  since most real reposts aren't maximally destroyed.
- Eval-time fixed protocol (PROTOCOL / apply_named / severity_for) is
  UNCHANGED — judges need reproducible, named conditions, not randomness.
- Both the pipeline ops and the fixed eval ops accept an optional `rng`
  (a random.Random instance) so a specific sample's augmentation can be
  reproduced deterministically when needed; the module-level `random`
  state is used when rng is not supplied.
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


# ---------------------------------------------------------------------------
# Fixed-protocol constants — defined up front so nothing below references
# them before they exist (this ordering bug previously crashed the module
# on import).
# ---------------------------------------------------------------------------

JPEG_QUALITIES = (90, 70, 50, 30)
BLUR_SIGMAS = (0.5, 1.0, 2.0)
RESIZE_SCALES = (0.5, 0.25)
NOISE_SIGMAS = (0.02, 0.05, 0.10)
COLOR_JITTER = 0.20
CENTER_CROP = 0.80
SCREENSHOT_VARIANTS = ((0.92, 90), (0.85, 75), (0.75, 60))


# ---------------------------------------------------------------------------
# Shared low-level ops (used by both the fixed eval protocol and the
# continuous training protocol below)
# ---------------------------------------------------------------------------

def _to_rgb(img: Image.Image) -> Image.Image:
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def jpeg_compress(img: Image.Image, quality: float) -> Image.Image:
    buf = io.BytesIO()
    _to_rgb(img).save(buf, format="JPEG", quality=int(round(quality)))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def gaussian_blur(img: Image.Image, sigma: float) -> Image.Image:
    if sigma <= 0:
        return _to_rgb(img)
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
    if sigma <= 0:
        return _to_rgb(img)
    arr = np.asarray(_to_rgb(img), dtype=np.float32) / 255.0
    if rng is None:
        noise = np.random.normal(0.0, sigma, arr.shape).astype(np.float32)
    else:
        rs = np.random.RandomState(rng.randint(0, 2**31 - 1))
        noise = rs.normal(0.0, sigma, arr.shape).astype(np.float32)
    out = np.clip(arr + noise, 0.0, 1.0)
    return Image.fromarray((out * 255.0).round().astype(np.uint8), mode="RGB")


def color_jitter(
    img: Image.Image, amount: float, rng: random.Random | None = None
) -> Image.Image:
    img = _to_rgb(img)
    if amount <= 0:
        return img
    r = rng if rng is not None else random
    b = 1.0 + r.uniform(-amount, amount)
    c = 1.0 + r.uniform(-amount, amount)
    s = 1.0 + r.uniform(-amount, amount)
    img = ImageEnhance.Brightness(img).enhance(b)
    img = ImageEnhance.Contrast(img).enhance(c)
    img = ImageEnhance.Color(img).enhance(s)
    return img


def center_crop(img: Image.Image, fraction: float) -> Image.Image:
    img = _to_rgb(img)
    if fraction >= 0.999:
        return img
    w, h = img.size
    nw, nh = max(1, int(w * fraction)), max(1, int(h * fraction))
    cropped = ImageOps.fit(img, (nw, nh), method=Image.Resampling.BILINEAR, centering=(0.5, 0.5))
    return cropped.resize((w, h), Image.Resampling.BILINEAR)


def re_screenshot(img: Image.Image, scale: float, quality: float) -> Image.Image:
    """Approximate a screenshot-and-reupload cycle: display-resolution resize
    followed by whatever recompression the sharing app applies to the capture."""
    img = _to_rgb(img)
    w, h = img.size
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = img.resize((nw, nh), Image.Resampling.BILINEAR).resize((w, h), Image.Resampling.BILINEAR)
    return jpeg_compress(resized, quality)


# ---------------------------------------------------------------------------
# Measured severity: derive severity from actual signal loss instead of a
# hand-picked constant, using the same high-frequency-energy idea the gate's
# degradation_stats() relies on.
# ---------------------------------------------------------------------------

def _hf_lf_ratio(img: Image.Image) -> float:
    """Ratio of high-frequency to low-frequency energy in the luma channel.
    Degradation (blur/JPEG/downsampling) disproportionately removes high
    frequencies, so this ratio drops as forensic-relevant detail is lost."""
    arr = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    mag = torch.fft.rfft2(t).abs()
    h, w = mag.shape[-2:]
    cy, cx = max(1, h // 6), max(1, w // 6)
    lf = mag[:, :, :cy, :cx].mean().item() + 1e-8
    hf = mag[:, :, cy:, cx:].mean().item()
    return hf / lf


def measure_severity(original: Image.Image, transformed: Image.Image) -> float:
    """Fraction of relative high-frequency signal lost due to the transform(s)
    actually applied, clamped to [0, 1]. This is what feeds the gate's
    training target, so it reflects real measured damage rather than an
    author's guess about how harsh a given op "should" be."""
    before = _hf_lf_ratio(original)
    after = _hf_lf_ratio(transformed)
    if before <= 1e-8:
        return 0.0
    retained = max(0.0, min(after / before, 1.0))
    return float(1.0 - retained)


# ---------------------------------------------------------------------------
# Continuous-parameter single ops, for use inside pipelines below. Each
# takes a "strength" in [0, 1] (0 = no effect, 1 = harshest realistic
# setting) and maps it to the op's actual parameter range. Noise and
# color_jitter are handled separately in _apply_strength_op so rng can be
# threaded through their internal randomness.
# ---------------------------------------------------------------------------

def _strength_jpeg(img: Image.Image, strength: float) -> Image.Image:
    quality = 95 - strength * (95 - 15)  # strength 0 -> q95, strength 1 -> q15
    return jpeg_compress(img, quality)


def _strength_blur(img: Image.Image, strength: float) -> Image.Image:
    sigma = strength * 3.0
    return gaussian_blur(img, sigma)


def _strength_resize(img: Image.Image, strength: float) -> Image.Image:
    scale = 1.0 - strength * 0.8  # strength 0 -> scale 1.0, strength 1 -> scale 0.2
    return down_up_resize(img, scale)


def _strength_crop(img: Image.Image, strength: float) -> Image.Image:
    fraction = 1.0 - strength * 0.5  # strength 0 -> no crop, strength 1 -> 50% crop
    return center_crop(img, fraction)


def _strength_screenshot(img: Image.Image, strength: float) -> Image.Image:
    scale = 1.0 - strength * 0.3
    quality = 90 - strength * (90 - 40)
    return re_screenshot(img, scale, quality)


_STRENGTH_OPS: dict[str, Callable[[Image.Image, float], Image.Image]] = {
    "jpeg": _strength_jpeg,
    "blur": _strength_blur,
    "resize": _strength_resize,
    "crop": _strength_crop,
    "screenshot": _strength_screenshot,
    # "noise" and "color_jitter" are routed through _apply_strength_op below
    # so their internal randomness can take an rng; they're not in this dict.
}


def _apply_strength_op(
    name: str, img: Image.Image, strength: float, rng: random.Random | None
) -> Image.Image:
    """Route to the right strength-parameterized op, threading rng into the
    two stochastic ones (noise, color_jitter) that need their own randomness
    beyond the strength value itself."""
    if name == "noise":
        sigma = strength * 0.12
        return gaussian_noise(img, sigma, rng=rng)
    if name == "color_jitter":
        amount = strength * 0.35
        return color_jitter(img, amount, rng=rng)
    return _STRENGTH_OPS[name](img, strength)


_ALL_STEP_NAMES = list(_STRENGTH_OPS.keys()) + ["noise", "color_jitter"]


def _sample_strength(rng: random.Random | None = None) -> float:
    """Skewed toward mild/moderate degradation, matching how most real
    reposts look (a few heavy-JPEG-cycles images exist, but they're the
    minority, not the median case)."""
    r = rng if rng is not None else random
    return r.betavariate(2.0, 4.0)  # mean ~0.33, long tail toward 1.0


# ---------------------------------------------------------------------------
# Named realistic pipelines: multi-step sequences that mirror actual
# redistribution paths, each with a relative sampling weight reflecting how
# common that path is in practice.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Pipeline:
    name: str
    steps: tuple[str, ...]  # step names from _ALL_STEP_NAMES, applied in order
    weight: float


PIPELINES: list[Pipeline] = [
    Pipeline("clean_repost", ("jpeg",), weight=4.0),           # most common: just re-saved once
    Pipeline("social_repost", ("resize", "jpeg"), weight=3.0),  # platform resizes + recompresses
    Pipeline("screenshot_reshare", ("screenshot", "jpeg"), weight=1.5),
    Pipeline("multi_generation", ("jpeg", "resize", "jpeg"), weight=1.0),  # repost of a repost
    Pipeline("filtered_repost", ("blur", "jpeg"), weight=0.75),  # a soft-focus filter, then shared
    Pipeline("cropped_share", ("crop", "jpeg"), weight=1.0),     # cropped for a thumbnail/story
    Pipeline("noisy_capture", ("noise", "jpeg"), weight=0.5),    # low-light capture, then shared
    Pipeline("color_graded", ("color_jitter", "jpeg"), weight=0.75),  # filter app, then shared
]

_PIPELINE_WEIGHTS = [p.weight for p in PIPELINES]


def _run_pipeline(
    img: Image.Image, pipeline: Pipeline, rng: random.Random | None = None
) -> Image.Image:
    for step in pipeline.steps:
        strength = _sample_strength(rng)
        img = _apply_strength_op(step, img, strength, rng)
    return img


def _run_random_stack(
    img: Image.Image,
    min_ops: int = 1,
    max_ops: int = 3,
    rng: random.Random | None = None,
) -> Image.Image:
    """Fallback: a few independently-chosen ops, for coverage of unusual
    combinations no named pipeline represents."""
    r = rng if rng is not None else random
    k = r.randint(min_ops, min(max_ops, len(_ALL_STEP_NAMES)))
    for step in r.sample(_ALL_STEP_NAMES, k):
        img = _apply_strength_op(step, img, _sample_strength(rng), rng)
    return img


class ProtocolTrainTransform:
    """Training-time augmentation: mostly samples a named realistic pipeline
    (weighted by how common that redistribution path is), with a smaller
    chance of a pure random op stack for coverage. Severity is measured
    empirically from actual high-frequency signal loss, not assigned.

    Pass an rng (a random.Random instance) to __call__ for reproducible
    augmentation of a specific sample; otherwise the module-level random
    state is used.

    Returns (image, description, severity).
    """

    def __init__(self, p: float = 0.85, random_stack_prob: float = 0.15):
        self.p = p
        self.random_stack_prob = random_stack_prob

    def __call__(
        self, img: Image.Image, rng: random.Random | None = None
    ) -> tuple[Image.Image, str, float]:
        r = rng if rng is not None else random
        original = _to_rgb(img)
        if r.random() >= self.p:
            return original, "clean", 0.0

        if r.random() < self.random_stack_prob:
            transformed = _run_random_stack(original, rng=rng)
            description = "random_stack"
        else:
            pipeline = r.choices(PIPELINES, weights=_PIPELINE_WEIGHTS, k=1)[0]
            transformed = _run_pipeline(original, pipeline, rng=rng)
            description = pipeline.name

        severity = measure_severity(original, transformed)
        return transformed, description, severity


# ---------------------------------------------------------------------------
# FIXED eval protocol — UNCHANGED behavior. Used for the reproducible
# robustness table (Clean / JPEG q30 / Blur σ=2 / Crop 80% / Unseen gen.)
# and any --ablation reporting. Do not randomize this; judges need exact,
# repeatable named conditions.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProtocolOp:
    name: str
    fn: Callable[[Image.Image], Image.Image]
    severity: float


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
    "color_jitter": [
        ProtocolOp("color_jitter", (lambda im: color_jitter(im, COLOR_JITTER)), 0.25)
    ],
    "center_crop": [
        ProtocolOp("center_crop", (lambda im: center_crop(im, CENTER_CROP)), 0.2)
    ],
    "screenshot": [
        ProtocolOp(
            f"screenshot_{int(scale*100)}_{q}",
            (lambda im, scale=scale, q=q: re_screenshot(im, scale, q)),
            sev,
        )
        for (scale, q), sev in zip(SCREENSHOT_VARIANTS, (0.45, 0.7, 0.9))
    ],
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
