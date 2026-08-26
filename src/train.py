#!/usr/bin/env python3
"""Train ForgeGate. Freeze CLIP; train forensic CNN + gate + head on protocol augs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

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
    for x, y, _ in loader:
        x = x.to(device)
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
    val_ds = AIGCFolderDataset(args.val_dir, image_size=cfg["image_size"], train=False)
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

    model = ForgeGate(
        clip_model=cfg.get("clip_model", "ViT-B-16"),
        clip_pretrained=cfg.get("clip_pretrained", "openai"),
        freeze_clip=cfg.get("freeze_clip", True),
    ).to(device)
    counts = model.parameter_counts()
    print(f"params total={counts['total']:,} trainable={counts['trainable']:,}")

    opt = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()
    use_amp = device.type == "cuda" and cfg.get("amp", True)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    ckpt_dir = Path(cfg.get("checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_auc = -1.0
    history = []

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        running = 0.0
        n = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{cfg['epochs']}")
        for x, y, _ in pbar:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(x)
                loss = loss_fn(out.logit, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += float(loss.item()) * x.size(0)
            n += x.size(0)
            pbar.set_postfix(loss=f"{running / max(n, 1):.4f}")

        metrics = evaluate(model, val_loader, device)
        metrics["epoch"] = epoch
        metrics["train_loss"] = running / max(n, 1)
        history.append(metrics)
        print(f"val acc={metrics['acc']:.4f} auc={metrics['auc']:.4f} ap={metrics['ap']:.4f}")

        payload = {
            "model": model.state_dict(),
            "cfg": cfg,
            "metrics": metrics,
            "param_counts": counts,
        }
        torch.save(payload, ckpt_dir / "last.pt")
        if metrics["auc"] >= best_auc or np.isnan(metrics["auc"]):
            best_auc = metrics["auc"] if not np.isnan(metrics["auc"]) else best_auc
            torch.save(payload, ckpt_dir / "best.pt")

    (ckpt_dir / "history.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
