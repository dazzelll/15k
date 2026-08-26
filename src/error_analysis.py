#!/usr/bin/env python3
"""Dump representative false positives / false negatives with gate values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms as T

from src.dataset import AIGCFolderDataset
from src.model import ForgeGate


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="outputs/error_analysis.json")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = ForgeGate()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    ds = AIGCFolderDataset(args.data_dir, image_size=args.image_size, train=False)
    tfm = T.Compose([T.Resize((args.image_size, args.image_size), antialias=True), T.ToTensor()])

    rows = []
    for path, label in ds.samples:
        x = tfm(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        out = model(x)
        p = float(out.prob.item())
        rows.append(
            {
                "image_path": str(path),
                "label": int(label),
                "pred": p,
                "forensic_gate": float(out.gate.item()),
                "error": abs(p - label),
            }
        )

    fps = [r for r in rows if r["label"] == 0 and r["pred"] >= 0.5]
    fns = [r for r in rows if r["label"] == 1 and r["pred"] < 0.5]
    fps.sort(key=lambda r: -r["pred"])
    fns.sort(key=lambda r: r["pred"])

    note = {
        "summary": {
            "n": len(rows),
            "false_positives": len(fps),
            "false_negatives": len(fns),
            "mean_gate": sum(r["forensic_gate"] for r in rows) / max(len(rows), 1),
        },
        "tradeoffs": [
            "False positives are often highly stylised real photos (illustration, CGI-looking product shots) because CLIP semantics overlap with generator aesthetics.",
            "False negatives are often heavily JPEG'd / blurred fakes: the gate downweights forensic traces and the semantic stream can still look photographic.",
            "Raising the decision threshold cuts FPs on clean data but increases FNs after social-media re-encode. Calibrate on the protocol mix, not clean accuracy.",
        ],
        "top_false_positives": fps[: args.topk],
        "top_false_negatives": fns[: args.topk],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(note, indent=2))
    print(json.dumps(note["summary"], indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
