#!/usr/bin/env python3
"""Robustness table: clean vs each official transform family."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from torchvision import transforms as T
from tqdm import tqdm

from src.dataset import AIGCFolderDataset
from src.model import ForgeGate
from src.transforms import PROTOCOL


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    pred = (p >= 0.5).astype(np.float32)
    out = {"acc": float(accuracy_score(y, pred))}
    if len(np.unique(y)) > 1:
        out["auc"] = float(roc_auc_score(y, p))
        out["ap"] = float(average_precision_score(y, p))
    else:
        out["auc"] = float("nan")
        out["ap"] = float("nan")
    return out


@torch.no_grad()
def score_loader(model, paths_labels, transform_fn, device, image_size, zeroshot: bool):
    tfm = T.Compose([T.Resize((image_size, image_size), antialias=True), T.ToTensor()])
    ys, ps = [], []
    for path, label in tqdm(paths_labels, leave=False):
        img = Image.open(path).convert("RGB")
        img = transform_fn(img)
        x = tfm(img).unsqueeze(0).to(device)
        if zeroshot:
            p = model.zeroshot_prob(x).item()
        else:
            p = model(x).prob.item()
        ys.append(label)
        ps.append(p)
    return metrics(np.array(ys), np.array(ps))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--zeroshot", action="store_true")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--max_images", type=int, default=400)
    parser.add_argument("--output", default="outputs/robustness.csv")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = ForgeGate()
    zeroshot = args.zeroshot or not args.checkpoint
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state, strict=False)
        zeroshot = args.zeroshot
    model.to(device).eval()

    ds = AIGCFolderDataset(args.data_dir, image_size=args.image_size, train=False)
    pairs = ds.samples[: args.max_images]

    rows = []
    for family, variants in PROTOCOL.items():
        family_scores = []
        for name, fn in variants:
            m = score_loader(model, pairs, fn, device, args.image_size, zeroshot)
            m.update({"family": family, "transform": name})
            rows.append(m)
            family_scores.append(m["acc"])
        print(f"{family:12s} mean_acc={np.mean(family_scores):.3f}")

    df = pd.DataFrame(rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    (out.with_suffix(".json")).write_text(df.to_json(orient="records", indent=2))
    print(df.to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
