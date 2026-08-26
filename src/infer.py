#!/usr/bin/env python3
"""Score a directory of images. Writes JSON list of {image_path, pred}.

pred is P(AIGC-generated) in [0, 1].
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

from src.dataset import IMAGE_EXTS
from src.model import ForgeGate


def load_model(checkpoint: str | None, device: torch.device, zeroshot: bool) -> ForgeGate:
    model = ForgeGate()
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    model._zeroshot = zeroshot or not checkpoint  # type: ignore[attr-defined]
    return model


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="ForgeGate AIGC inference")
    parser.add_argument("--input_dir", required=True, help="Folder of images (recursive)")
    parser.add_argument("--output", default="outputs/preds.json")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--zeroshot", action="store_true", help="CLIP prompt baseline")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(args.checkpoint, device, args.zeroshot)
    use_zeroshot = bool(getattr(model, "_zeroshot", False))

    root = Path(args.input_dir)
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise SystemExit(f"No images found in {root}")

    tfm = T.Compose([T.Resize((args.image_size, args.image_size), antialias=True), T.ToTensor()])
    records = []
    batch_x, batch_p = [], []

    def flush() -> None:
        if not batch_x:
            return
        x = torch.stack(batch_x).to(device)
        if use_zeroshot:
            probs = model.zeroshot_prob(x)
            gates = [None] * len(batch_p)
        else:
            out = model(x)
            probs = out.prob
            gates = out.gate.cpu().tolist()
        for path, prob, gate in zip(batch_p, probs.cpu().tolist(), gates):
            rec = {"image_path": path, "pred": float(prob)}
            if gate is not None:
                rec["forensic_gate"] = float(gate)
            records.append(rec)
        batch_x.clear()
        batch_p.clear()

    for path in tqdm(paths, desc="infer"):
        img = Image.open(path).convert("RGB")
        batch_x.append(tfm(img))
        batch_p.append(str(path))
        if len(batch_x) >= args.batch_size:
            flush()
    flush()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2))
    print(f"wrote {len(records)} scores to {out_path}")


if __name__ == "__main__":
    main()
