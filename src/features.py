"""Forensic maps that are less brittle than raw SRM, plus cheap degradation stats.

High-frequency SRM / PRNU dies under JPEG and blur. We keep:
- a mild high-pass residual (structure, not sensor fingerprint)
- neighboring pixel relationships (upsampling / generator grid)
- log-magnitude mid-band FFT of luma
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def _as_batch(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(0) if x.dim() == 3 else x


def luma(x: torch.Tensor) -> torch.Tensor:
    x = _as_batch(x)
    return 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]


def forensic_maps(x: torch.Tensor) -> torch.Tensor:
    """Bx3xHxW forensic tensor from RGB in [0, 1]. Also accepts 3xHxW."""
    single = x.dim() == 3
    x = _as_batch(x)
    y = luma(x).unsqueeze(1)
    blur = F.avg_pool2d(y, 5, stride=1, padding=2)
    hp = y - blur
    dx = F.pad(torch.abs(y[:, :, :, 1:] - y[:, :, :, :-1]), (0, 1, 0, 0))
    dy = F.pad(torch.abs(y[:, :, 1:, :] - y[:, :, :-1, :]), (0, 0, 0, 1))
    npr = 0.5 * (dx + dy)
    spec = torch.fft.fftshift(torch.fft.fft2(y), dim=(-2, -1))
    mag = torch.log1p(spec.abs())
    b, _, h, w = mag.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h, device=x.device),
        torch.linspace(-1, 1, w, device=x.device),
        indexing="ij",
    )
    radius = torch.sqrt(xx * xx + yy * yy)
    band = ((radius > 0.08) & (radius < 0.75)).float()
    mag = mag * band.view(1, 1, h, w)
    mag = mag / (mag.amax(dim=(-2, -1), keepdim=True) + 1e-6)
    maps = torch.cat([hp, npr, mag], dim=1)
    return maps[0] if single else maps


def degradation_stats(x: torch.Tensor) -> torch.Tensor:
    """Bx8 degradation features. High blur/blockiness => trust forensic stream less."""
    single = x.dim() == 3
    x = _as_batch(x)
    y = luma(x)
    b, h, w = y.shape
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    lap = F.conv2d(y.unsqueeze(1), kernel, padding=1)
    lap_var = lap.flatten(1).var(dim=1)
    blur_score = 1.0 / (1.0 + 50.0 * lap_var)

    spec = torch.fft.rfft2(y)
    mag = spec.abs()
    cy, cx = max(1, h // 6), max(1, mag.shape[-1] // 6)
    lf = mag[:, :cy, :cx].mean(dim=(1, 2)) + 1e-8
    hf = mag[:, cy:, cx:].mean(dim=(1, 2))
    hf_ratio = (hf / lf).clamp(0, 10) / 10.0

    if w > 8 and h > 8:
        dh = (y[:, :, 7::8] - y[:, :, 6::8]).abs().mean(dim=(1, 2))
        dv = (y[:, 7::8, :] - y[:, 6::8, :]).abs().mean(dim=(1, 2))
        blockiness = ((dh + dv) * 5.0).clamp(0, 1)
    else:
        blockiness = y.new_zeros(b)

    sat = (x.max(dim=1).values - x.min(dim=1).values).flatten(1).std(dim=1)
    contrast = y.flatten(1).std(dim=1)
    maps = forensic_maps(x)
    edge = maps[:, 1].flatten(1).mean(dim=1)
    residual = maps[:, 0].abs().flatten(1).mean(dim=1)
    denoise = F.avg_pool2d(y.unsqueeze(1), 3, stride=1, padding=1).squeeze(1)
    noise_est = (y - denoise).abs().flatten(1).mean(dim=1)
    stats = torch.stack(
        [blur_score, hf_ratio, blockiness, sat, contrast, edge, residual, noise_est],
        dim=1,
    )
    return stats[0] if single else stats
