#!/usr/bin/env python3
"""Gradio demo for the hackathon video: upload → P(AIGC) + forensic gate."""

from __future__ import annotations

import argparse

import gradio as gr
import torch
from PIL import Image
from torchvision import transforms as T

from src.model import ForgeGate
from src.transforms import PROTOCOL

STAT_NAMES = [
    "blur",
    "hf_ratio",
    "jpeg_blockiness",
    "saturation_std",
    "contrast",
    "edge_energy",
    "residual_energy",
    "noise_est",
]


def build(checkpoint: str | None, zeroshot: bool, device: torch.device):
    model = ForgeGate()
    use_zeroshot = zeroshot or not checkpoint
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state, strict=False)
        use_zeroshot = zeroshot
    model.to(device).eval()
    tfm = T.Compose([T.Resize((224, 224), antialias=True), T.ToTensor()])

    @torch.no_grad()
    def predict(image: Image.Image, transform_name: str):
        if image is None:
            return "No image", "", ""
        image = image.convert("RGB")
        if transform_name != "clean":
            for variants in PROTOCOL.values():
                for name, fn in variants:
                    if name == transform_name:
                        image = fn(image)
                        break
        x = tfm(image).unsqueeze(0).to(device)
        if use_zeroshot:
            p = float(model.zeroshot_prob(x).item())
            gate = None
            stats = None
        else:
            out = model(x)
            p = float(out.prob.item())
            gate = float(out.gate.item())
            stats = out.stats.squeeze(0).cpu().tolist()
        label = "AI-generated" if p >= 0.5 else "Authentic"
        lines = [f"{label}  ·  P(AIGC) = {p:.3f}"]
        if gate is not None:
            lines.append(f"Forensic gate (1 = trust traces) = {gate:.3f}")
        detail = ""
        if stats is not None:
            detail = "\n".join(f"{n}: {v:.4f}" for n, v in zip(STAT_NAMES, stats))
        return image, "\n".join(lines), detail

    names = ["clean"] + [n for variants in PROTOCOL.values() for n, _ in variants if n != "clean"]
    demo = gr.Interface(
        fn=predict,
        inputs=[
            gr.Image(type="pil", label="Image"),
            gr.Dropdown(choices=names, value="clean", label="Apply robustness transform"),
        ],
        outputs=[
            gr.Image(label="Maybe-transformed preview"),
            gr.Textbox(label="Prediction"),
            gr.Textbox(label="Degradation stats"),
        ],
        title="ForgeGate — robust AIGC image detection",
        description=(
            "Dual-stream detector: frozen CLIP semantics + a forensic CNN that is "
            "gated by blur / JPEG / noise statistics. Protocol transforms match the "
            "hackathon robustness table."
        ),
    )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--zeroshot", action="store_true")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    demo = build(args.checkpoint, args.zeroshot, device)
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
