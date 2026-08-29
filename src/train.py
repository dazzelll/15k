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


def _cls_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred = (p >= 0.5).astype(np.float32)
    return {
        "acc": float(accuracy_score(y, pred)),
        "ap": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "ece": expected_calibration_error(p, y),
    }


@torch.no_grad()
def evaluate_clean_and_protocol(
    model: ForgeGate, loader: DataLoader, device: torch.device
) -> tuple[dict, dict]:
    """One val pass: clean images and the frozen protocol mix."""
    model.eval()
    ys, p_clean, p_prot = [], [], []
    for x_clean, x_t, y, _sev, _path in loader:
        out_c = model(x_clean.to(device, non_blocking=True))
        out_t = model(x_t.to(device, non_blocking=True))
        ys.append(y.numpy())
        p_clean.append(out_c.prob.cpu().numpy())
        p_prot.append(out_t.prob.cpu().numpy())
    y = np.concatenate(ys)
    return _cls_metrics(y, np.concatenate(p_clean)), _cls_metrics(y, np.concatenate(p_prot))


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
    val_ds = AIGCFolderDataset(
        args.val_dir,
        image_size=cfg["image_size"],
        train=True,
        protocol_aug_prob=cfg.get("protocol_aug_prob", 0.85),
        protocol_seed=cfg.get("seed", 42),
    )

    num_workers = cfg.get("num_workers", 4)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    model = ForgeGate(
        clip_model=cfg.get("clip_model", "ViT-B-16"),
        clip_pretrained=cfg.get("clip_pretrained", "openai"),
        freeze_clip=cfg.get("freeze_clip", True),
    ).to(device)

    model.train()
    assert not model.clip.training, "CLIP must stay in eval while ForgeGate trains"
    model.train()

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
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    ckpt_dir = Path(cfg.get("checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_final_score = -1.0
    history = []

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        running = {
            "total": 0.0,
            "cls": 0.0,
            "cons": 0.0,
            "gate": 0.0,
            "gate_clean_mean": 0.0,
            "gate_t_mean": 0.0,
            "severity_mean": 0.0,
        }
        n = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{cfg['epochs']}")
        for x_clean, x_t, y, severity, _path in pbar:
            x_clean = x_clean.to(device, non_blocking=True)
            x_t = x_t.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            severity = severity.to(device, non_blocking=True).float()

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                out_c = model(x_clean)
                out_t = model(x_t)
                cls_loss = loss_fn(out_c.logit, y) + loss_fn(out_t.logit, y)
                cons_loss = F.mse_loss(
                    torch.sigmoid(out_c.logit), torch.sigmoid(out_t.logit)
                )

            gate_clean = out_c.gate.float()
            gate_t_out = out_t.gate.float()
            target_c = torch.ones_like(gate_clean)
            target_t = (1.0 - severity).clamp(0.0, 1.0).float()
            gate_loss = F.binary_cross_entropy(gate_clean, target_c) + F.binary_cross_entropy(
                gate_t_out, target_t
            )
            loss = cls_loss + alpha * cons_loss + beta * gate_loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            bs = x_clean.size(0)
            running["total"] += float(loss.item()) * bs
            running["cls"] += float(cls_loss.item()) * bs
            running["cons"] += float(cons_loss.item()) * bs
            running["gate"] += float(gate_loss.item()) * bs
            with torch.no_grad():
                running["gate_clean_mean"] += float(out_c.gate.mean().item()) * bs
                running["gate_t_mean"] += float(out_t.gate.mean().item()) * bs
                running["severity_mean"] += float(severity.mean().item()) * bs
            n += bs
            pbar.set_postfix(
                loss=f"{running['total'] / max(n, 1):.4f}",
                gate_loss=f"{running['gate'] / max(n, 1):.3f}",
            )

        clean_m, prot_m = evaluate_clean_and_protocol(model, val_loader, device)
        
        # Calculate Final Score: 0.50 * AUC_clean + 0.50 * AUC_robust
        final_score = 0.0
        if not np.isnan(clean_m["auc"]) and not np.isnan(prot_m["auc"]):
            final_score = 0.50 * clean_m["auc"] + 0.50 * prot_m["auc"]
        
        metrics = {
            "epoch": epoch,
            "acc": clean_m["acc"],
            "auc": clean_m["auc"],
            "ap": clean_m["ap"],
            "ece": clean_m["ece"],
            "protocol_acc": prot_m["acc"],
            "protocol_auc": prot_m["auc"],
            "protocol_ap": prot_m["ap"],
            "protocol_ece": prot_m["ece"],
            "final_score": final_score,
            "train_loss": running["total"] / max(n, 1),
            "train_cls": running["cls"] / max(n, 1),
            "train_cons": running["cons"] / max(n, 1),
            "train_gate": running["gate"] / max(n, 1),
            "gate_clean_mean": running["gate_clean_mean"] / max(n, 1),
            "gate_t_mean": running["gate_t_mean"] / max(n, 1),
            "severity_mean": running["severity_mean"] / max(n, 1),
        }
        history.append(metrics)
        print(
            f"val clean acc={metrics['acc']:.4f} auc={metrics['auc']:.4f} "
            f"ap={metrics['ap']:.4f} ece={metrics['ece']:.4f}  "
            f"protocol acc={metrics['protocol_acc']:.4f} auc={metrics['protocol_auc']:.4f} "
            f"ap={metrics['protocol_ap']:.4f}  "
            f"final_score={metrics['final_score']:.4f}  "
            f"gate_clean={metrics['gate_clean_mean']:.3f} "
            f"gate_t={metrics['gate_t_mean']:.3f} sev={metrics['severity_mean']:.3f}"
        )

        payload = {
            "model": model.state_dict(),
            "cfg": cfg,
            "metrics": metrics,
            "param_counts": counts,
            "temperature": float(model.temperature.item()),
        }
        torch.save(payload, ckpt_dir / "last.pt")
        # Select on Final Score: 0.50 * AUC_clean + 0.50 * AUC_robust
        # NaN AUC (e.g. single-class val split) must never count as "best".
        if not np.isnan(final_score) and final_score >= best_final_score:
            best_final_score = final_score
            torch.save(payload, ckpt_dir / "best.pt")

    if cfg.get("calibrate", True):
        print("Fitting temperature on protocol-augmented val…")
        best_path = ckpt_dir / "best.pt"
        if best_path.exists():
            ckpt = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"], strict=False)
        logits, labels = collect_logits(model, val_loader, device, use_transformed=True)
        t_star = fit_temperature(logits.to(device), labels.to(device))
        model.temperature.fill_(t_star)
        cal_clean, cal_prot = evaluate_clean_and_protocol(model, val_loader, device)
        
        # Calculate Final Score for calibrated model
        cal_final_score = 0.0
        if not np.isnan(cal_clean["auc"]) and not np.isnan(cal_prot["auc"]):
            cal_final_score = 0.50 * cal_clean["auc"] + 0.50 * cal_prot["auc"]
        
        cal_metrics = {
            **cal_clean,
            "protocol_acc": cal_prot["acc"],
            "protocol_auc": cal_prot["auc"],
            "protocol_ap": cal_prot["ap"],
            "protocol_ece": cal_prot["ece"],
            "final_score": cal_final_score,
            "temperature": t_star,
        }
        print(
            f"temperature={t_star:.4f}  "
            f"clean val ece={cal_metrics['ece']:.4f} auc={cal_metrics['auc']:.4f}  "
            f"protocol val ece={cal_metrics['protocol_ece']:.4f} "
            f"auc={cal_metrics['protocol_auc']:.4f}  "
            f"final_score={cal_metrics['final_score']:.4f}"
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