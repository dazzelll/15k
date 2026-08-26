"""Temperature scaling calibration for ForgeGate logits."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.model import ForgeGate


@torch.no_grad()
def collect_logits(
    model: ForgeGate,
    loader: DataLoader,
    device: torch.device,
    use_transformed: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect raw (pre-temperature) logits and labels.

    Temporarily sets temperature=1 so we calibrate the unscaled head outputs.
    """
    model.eval()
    prev_t = model.temperature.detach().clone()
    model.temperature.fill_(1.0)
    logits, labels = [], []
    try:
        for batch in loader:
            x_clean, x_t, y = batch[0], batch[1], batch[2]
            x = (x_t if use_transformed else x_clean).to(device)
            y = y.to(device)
            out = model(x)
            logits.append(out.logit.detach().cpu())
            labels.append(y.detach().cpu())
    finally:
        model.temperature.copy_(prev_t)
    return torch.cat(logits), torch.cat(labels)


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_iter: int = 200,
    lr: float = 0.05,
) -> float:
    """Minimize BCEWithLogits on logits / T. Returns scalar T > 0."""
    log_t = torch.nn.Parameter(torch.zeros(1, device=logits.device))
    opt = torch.optim.Adam([log_t], lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    y = labels.float()
    for _ in range(max_iter):
        opt.zero_grad(set_to_none=True)
        t = log_t.exp().clamp_min(1e-4)
        loss = loss_fn(logits / t, y)
        loss.backward()
        opt.step()
    return float(log_t.exp().clamp_min(1e-4).item())


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """ECE with equal-width confidence bins on P(AIGC)."""
    confidences = np.where(probs >= 0.5, probs, 1.0 - probs)
    predictions = (probs >= 0.5).astype(np.float32)
    accs = (predictions == labels).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if not np.any(mask):
            continue
        ece += (mask.sum() / n) * abs(accs[mask].mean() - confidences[mask].mean())
    return float(ece)


def reliability_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (bin_centers, fraction_positive, counts) for P(AIGC) bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers, frac_pos, counts = [], [], []
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1] if i < n_bins - 1 else probs <= bins[i + 1])
        centers.append(0.5 * (bins[i] + bins[i + 1]))
        counts.append(int(mask.sum()))
        frac_pos.append(float(labels[mask].mean()) if mask.any() else float("nan"))
    return np.array(centers), np.array(frac_pos), np.array(counts)
