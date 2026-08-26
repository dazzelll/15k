#!/usr/bin/env python3
"""Train ForgeGate. Freeze CLIP; train forensic CNN + gate + head on protocol augs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.calibration import collect_logits, expected_calibration_error, fit_temperature
from src.dataset import AIGCFolderDataset
from src.model import ForgeGate


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: ForgeGate, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    ys, ps = [], []
    for x_clean, _x_t, y, _sev, _path in loader:
        x = x_clean.to(device)
        out = model(x)
        ys.append(y.numpy())
        ps.append(out.prob.cpu().numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    pred = (p >= 0.5).astype(np.float32)
    metrics = {
        "acc": float(accuracy_score(y, pred)),
        "ap": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "ece": expected_calibration_error(p, y),
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--val_dir", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs

    set_seed(cfg.get("seed", 42))
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    train_ds = AIGCFolderDataset(
        args.train_dir,
        image_size=cfg["image_size"],
        train=True,
        protocol_aug_prob=cfg.get("protocol_aug_prob", 0.85),
    )
    # Protocol-augmented val for consistency/gate monitoring + calibration.
    val_ds = AIGCFolderDataset(
        args.val_dir,
        image_size=cfg["image_size"],
        train=True,
        protocol_aug_prob=cfg.get("protocol_aug_prob", 0.85),
    )
    val_clean_ds = AIGCFolderDataset(
        args.val_dir, image_size=cfg["image_size"], train=False
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg.get("num_workers", 2),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg.get("num_workers", 2),
    )
    val_clean_loader = DataLoader(
        val_clean_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg.get("num_workers", 2),
    )

    model = ForgeGate(
        clip_model=cfg.get("clip_model", "ViT-B-16"),
        clip_pretrained=cfg.get("clip_pretrained", "openai"),
        freeze_clip=cfg.get("freeze_clip", True),
    ).to(device)
    # Critical: train() must keep CLIP in eval — assert once at startup.
    model.train()
    assert not model.clip.training, "CLIP must stay in eval while ForgeGate trains"
    model.train()  # restore train mode for forensic/gate/head

    counts = model.parameter_counts()
    print(f"params total={counts['total']:,} trainable={counts['trainable']:,}")

    opt = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()
    alpha = float(cfg.get("consistency_weight", 0.5))
    beta = float(cfg.get("gate_reg_weight", 0.3))
    use_amp = device.type == "cuda" and cfg.get("amp", True)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    ckpt_dir = Path(cfg.get("checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_auc = -1.0
    history = []

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        running = {"total": 0.0, "cls": 0.0, "cons": 0.0, "gate": 0.0}
        n = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{cfg['epochs']}")
        for x_clean, x_t, y, severity, _path in pbar:
            x_clean = x_clean.to(device)
            x_t = x_t.to(device)
            y = y.to(device)
            severity = severity.to(device).float()
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out_c = model(x_clean)
                out_t = model(x_t)
                cls_loss = loss_fn(out_c.logit, y) + loss_fn(out_t.logit, y)
                cons_loss = F.mse_loss(
                    torch.sigmoid(out_c.logit), torch.sigmoid(out_t.logit)
                )
                # Trust forensics when severity is low; clean target is 1.
                target_c = torch.ones_like(out_c.gate)
                target_t = (1.0 - severity).clamp(0.0, 1.0)
                gate_loss = F.binary_cross_entropy(
                    out_c.gate, target_c
                ) + F.binary_cross_entropy(out_t.gate, target_t)
                loss = cls_loss + alpha * cons_loss + beta * gate_loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            bs = x_clean.size(0)
            running["total"] += float(loss.item()) * bs
            running["cls"] += float(cls_loss.item()) * bs
            running["cons"] += float(cons_loss.item()) * bs
            running["gate"] += float(gate_loss.item()) * bs
            n += bs
            pbar.set_postfix(
                loss=f"{running['total'] / max(n, 1):.4f}",
                gate=f"{running['gate'] / max(n, 1):.3f}",
            )

        metrics = evaluate(model, val_clean_loader, device)
        metrics["epoch"] = epoch
        metrics["train_loss"] = running["total"] / max(n, 1)
        metrics["train_cls"] = running["cls"] / max(n, 1)
        metrics["train_cons"] = running["cons"] / max(n, 1)
        metrics["train_gate"] = running["gate"] / max(n, 1)
        history.append(metrics)
        print(
            f"val acc={metrics['acc']:.4f} auc={metrics['auc']:.4f} "
            f"ap={metrics['ap']:.4f} ece={metrics['ece']:.4f}"
        )

        payload = {
            "model": model.state_dict(),
            "cfg": cfg,
            "metrics": metrics,
            "param_counts": counts,
            "temperature": float(model.temperature.item()),
        }
        torch.save(payload, ckpt_dir / "last.pt")
        if metrics["auc"] >= best_auc or np.isnan(metrics["auc"]):
            best_auc = metrics["auc"] if not np.isnan(metrics["auc"]) else best_auc
            torch.save(payload, ckpt_dir / "best.pt")

    # Temperature scaling on protocol-augmented val (matches deployment mix).
    if cfg.get("calibrate", True):
        print("Fitting temperature on protocol-augmented val…")
        best_path = ckpt_dir / "best.pt"
        if best_path.exists():
            ckpt = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"], strict=False)
        logits, labels = collect_logits(model, val_loader, device, use_transformed=True)
        t_star = fit_temperature(logits.to(device), labels.to(device))
        model.temperature.fill_(t_star)
        cal_metrics = evaluate(model, val_clean_loader, device)
        cal_metrics["temperature"] = t_star
        print(
            f"temperature={t_star:.4f}  "
            f"clean val ece={cal_metrics['ece']:.4f} auc={cal_metrics['auc']:.4f}"
        )
        payload = {
            "model": model.state_dict(),
            "cfg": cfg,
            "metrics": cal_metrics,
            "param_counts": counts,
            "temperature": t_star,
        }
        torch.save(payload, ckpt_dir / "best.pt")
        torch.save(payload, ckpt_dir / "last.pt")
        history.append({"epoch": "calibrated", **cal_metrics})

    (ckpt_dir / "history.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
