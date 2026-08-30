#!/usr/bin/env python3
"""Robustness table, gate ablation, and gate-vs-severity audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from torchvision import transforms as T
from tqdm import tqdm

from src.calibration import expected_calibration_error, reliability_curve
from src.dataset import AIGCFolderDataset
from src.model import ForgeGate, GateMode
from src.transforms import PROTOCOL


def _stratified_subset(samples: list, max_images: int) -> list:
    """Take an even split across labels, not just the first N.

    ds.samples is built from a sorted file listing, so folder names like
    FAKE/REAL sort alphabetically and a naive samples[:max_images] slice can
    end up single-class (e.g. all FAKE). roc_auc_score / average_precision_score
    both return NaN when only one class is present, which silently breaks
    every AUC/AP/Final Score in this script. Stratifying guarantees both
    classes are represented regardless of file ordering.
    """
    by_label: dict = {}
    for path, label in samples:
        by_label.setdefault(label, []).append((path, label))
    n_labels = max(1, len(by_label))
    per_label = max(1, max_images // n_labels)
    subset = []
    for label, items in by_label.items():
        subset.extend(items[:per_label])
    return subset


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    pred = (p >= 0.5).astype(np.float32)
    out = {"acc": float(accuracy_score(y, pred))}
    if len(np.unique(y)) > 1:
        out["auc"] = float(roc_auc_score(y, p))
        out["ap"] = float(average_precision_score(y, p))
    else:
        out["auc"] = float("nan")
        out["ap"] = float("nan")
    out["ece"] = expected_calibration_error(p, y)
    return out


@torch.no_grad()
def score_loader(
    model: ForgeGate,
    paths_labels,
    transform_fn,
    device,
    image_size: int,
    zeroshot: bool,
    gate_mode: GateMode = "learned",
):
    tfm = T.Compose([T.Resize((image_size, image_size), antialias=True), T.ToTensor()])
    ys, ps, gates, sevs = [], [], [], []
    for path, label, severity in tqdm(paths_labels, leave=False):
        img = Image.open(path).convert("RGB")
        img = transform_fn(img)
        x = tfm(img).unsqueeze(0).to(device)
        if zeroshot:
            p = model.zeroshot_prob(x).item()
            g = float("nan")
        else:
            out = model(x, gate_mode=gate_mode)
            p = out.prob.item()
            g = out.gate.item()
        ys.append(label)
        ps.append(p)
        gates.append(g)
        sevs.append(severity)
    y = np.array(ys)
    p = np.array(ps)
    m = metrics(y, p)
    m["mean_gate"] = float(np.nanmean(gates))
    m["mean_severity"] = float(np.mean(sevs))
    return m, np.array(gates), np.array(sevs), y, p


def plot_gate_vs_severity(rows: list[dict], out_path: Path) -> None:
    df = pd.DataFrame(rows)
    df = df[df["gate_mode"] == "learned"].copy()
    if df.empty or df["mean_gate"].isna().all():
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for family, sub in df.groupby("family"):
        ax.scatter(sub["severity"], sub["mean_gate"], label=family, s=40, alpha=0.85)
    ax.set_xlabel("Transform severity (0=clean, 1=harsh)")
    ax.set_ylabel("Mean forensic gate g")
    ax.set_title("Gate vs degradation severity")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_reliability(y: np.ndarray, p: np.ndarray, out_path: Path, title: str) -> None:
    centers, frac, counts = reliability_curve(p, y)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    mask = counts > 0
    ax.plot(centers[mask], frac[mask], "o-", label="model")
    ax.set_xlabel("Predicted P(AIGC)")
    ax.set_ylabel("Empirical P(label=AIGC)")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--zeroshot", action="store_true")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--max_images", type=int, default=400)
    parser.add_argument("--output", default="outputs/robustness.csv")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run semantic-only / forensic-always / full ForgeGate",
    )
    parser.add_argument(
        "--aigc_benchmark_dir",
        default=None,
        help="Path to AIGC benchmark dataset for unseen generator evaluation",
    )
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = ForgeGate()
    zeroshot = args.zeroshot or not args.checkpoint
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state, strict=False)
        zeroshot = args.zeroshot
        if isinstance(ckpt, dict) and "temperature" in ckpt:
            model.temperature.fill_(float(ckpt["temperature"]))
    model.to(device).eval()

    ds = AIGCFolderDataset(args.data_dir, image_size=args.image_size, train=False)
    pairs = [(p, lab, 0.0) for p, lab in _stratified_subset(ds.samples, args.max_images)]

    gate_modes: list[GateMode] = ["learned"]
    if args.ablation and not zeroshot:
        gate_modes = ["zero", "one", "learned"]

    rows = []
    gate_audit = []
    clean_y, clean_p = None, None

    for gate_mode in gate_modes:
        for family, variants in PROTOCOL.items():
            family_scores = []
            for op in variants:
                tagged = [(p, lab, op.severity) for p, lab, _ in pairs]
                m, gates, sevs, y, p = score_loader(
                    model,
                    tagged,
                    op.fn,
                    device,
                    args.image_size,
                    zeroshot,
                    gate_mode=gate_mode,
                )
                row = {
                    "gate_mode": gate_mode,
                    "family": family,
                    "transform": op.name,
                    "severity": op.severity,
                    **m,
                }
                rows.append(row)
                family_scores.append(m["acc"])
                if gate_mode == "learned" and not zeroshot:
                    gate_audit.append(row)
                if family == "clean" and gate_mode == "learned":
                    clean_y, clean_p = y, p
            mode_tag = f"[{gate_mode}]" if args.ablation else ""
            print(f"{mode_tag:12s} {family:12s} mean_acc={np.mean(family_scores):.3f}")

    df = pd.DataFrame(rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    (out.with_suffix(".json")).write_text(df.to_json(orient="records", indent=2))

    # Calculate and display Final Score
    clean_auc = df[df["family"] == "clean"]["auc"].mean()
    robust_auc = df[df["family"] != "clean"]["auc"].mean()
    final_score = 0.0
    if not np.isnan(clean_auc) and not np.isnan(robust_auc):
        final_score = 0.50 * clean_auc + 0.50 * robust_auc

    print(f"\nFinal Score: {final_score:.4f} (0.50 * AUC_clean {clean_auc:.4f} + 0.50 * AUC_robust {robust_auc:.4f})")

    # Evaluate on AIGC benchmark for unseen generator performance if provided
    unseen_gen_metrics = None
    if args.aigc_benchmark_dir:
        print(f"\nEvaluating on unseen generators from {args.aigc_benchmark_dir}...")
        aigc_ds = AIGCFolderDataset(args.aigc_benchmark_dir, image_size=args.image_size, train=False)
        aigc_pairs = [(p, lab, 0.0) for p, lab in _stratified_subset(aigc_ds.samples, args.max_images)]
        
        # Evaluate clean condition on AIGC benchmark
        aigc_m, _, _, _, _ = score_loader(
            model,
            aigc_pairs,
            lambda x: x,  # No transform for clean evaluation
            device,
            args.image_size,
            zeroshot,
            gate_mode="learned",
        )
        unseen_gen_metrics = aigc_m
        print(f"Unseen generator (clean): acc={aigc_m['acc']:.4f}, auc={aigc_m['auc']:.4f}")

    # Generate simplified table with key conditions using actual transform names from PROTOCOL
    # These are the exact names that appear in the PROTOCOL dictionary
    key_conditions = {
        "clean": "Clean",
        "jpeg_q30": "JPEG q30",
        "blur_s2.0": "Blur σ=2",
        "center_crop": "Crop 80%",
        "resize_x0.5": "Resize 50%",
    }

    print("\nKey Conditions Table:")
    key_results = []
    for transform_name, display_name in key_conditions.items():
        matching_rows = df[df["transform"] == transform_name]
        if not matching_rows.empty:
            row = matching_rows.iloc[0]
            key_results.append({
                "Condition": display_name,
                "Acc.": f"{row['acc']:.4f}",
                "AUC": f"{row['auc']:.4f}" if not np.isnan(row['auc']) else "N/A"
            })
    
    # Add unseen generator row if AIGC benchmark was evaluated
    if unseen_gen_metrics is not None:
        key_results.append({
            "Condition": "Unseen gen.",
            "Acc.": f"{unseen_gen_metrics['acc']:.4f}",
            "AUC": f"{unseen_gen_metrics['auc']:.4f}" if not np.isnan(unseen_gen_metrics['auc']) else "N/A"
        })

    if key_results:
        key_df = pd.DataFrame(key_results)
        print(key_df.to_string(index=False))
        key_csv = out.parent / "key_conditions.csv"
        key_df.to_csv(key_csv, index=False)
        print(f"\nKey conditions table saved to: {key_csv}")

    print(df.to_string(index=False))
    print(f"wrote {out}")

    if gate_audit:
        plot_path = out.parent / "gate_vs_severity.png"
        plot_gate_vs_severity(gate_audit, plot_path)
        print(f"wrote {plot_path}")

    if clean_y is not None and clean_p is not None:
        rel_path = out.parent / "reliability_clean.png"
        plot_reliability(clean_y, clean_p, rel_path, "Reliability (clean)")
        print(f"wrote {rel_path}")

    if args.ablation and not zeroshot:
        # Compact three-row insight table: mean acc by severity bucket.
        insight = []
        for gate_mode in gate_modes:
            sub = df[df["gate_mode"] == gate_mode]
            insight.append(
                {
                    "gate_mode": gate_mode,
                    "meaning": {
                        "zero": "semantic-only (g=0)",
                        "one": "forensic-always (g=1)",
                        "learned": "full ForgeGate",
                    }[gate_mode],
                    "clean_acc": float(sub[sub["family"] == "clean"]["acc"].mean()),
                    "harsh_acc": float(
                        sub[sub["severity"] >= 0.7]["acc"].mean()
                        if (sub["severity"] >= 0.7).any()
                        else float("nan")
                    ),
                    "mean_gate_clean": float(
                        sub[sub["family"] == "clean"]["mean_gate"].mean()
                    ),
                    "mean_gate_harsh": float(
                        sub[sub["severity"] >= 0.7]["mean_gate"].mean()
                        if (sub["severity"] >= 0.7).any()
                        else float("nan")
                    ),
                }
            )
        insight_path = out.parent / "ablation_insight.json"
        insight_path.write_text(json.dumps(insight, indent=2))
        print(json.dumps(insight, indent=2))
        print(f"wrote {insight_path}")


if __name__ == "__main__":
    main()