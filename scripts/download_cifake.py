#!/usr/bin/env python3
"""Download CIFAKE with KaggleHub and point data/train + data/val at it.

Needs a Kaggle API token at ~/.kaggle/kaggle.json
(Kaggle → Account → Create New Token).

  python scripts/download_cifake.py
  python train.py --train_dir data/train --val_dir data/val --config configs/default.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

CIFAKE_HANDLE = "birdy654/cifake-real-and-ai-generated-synthetic-images"


def _has_real_fake(folder: Path) -> bool:
    names = {p.name for p in folder.iterdir() if p.is_dir()}
    return bool(names & {"REAL", "real", "FAKE", "fake"})


def find_split(root: Path, split: str) -> Path:
    candidates = [p for p in root.rglob(split) if p.is_dir() and _has_real_fake(p)]
    if not candidates:
        raise FileNotFoundError(
            f"No '{split}/' with REAL/FAKE under {root}. Listing:\n"
            + "\n".join(str(p.relative_to(root)) for p in sorted(root.rglob("*"))[:40])
        )
    return sorted(candidates, key=lambda p: len(p.parts))[0]


def link_split(src: Path, dest: Path, force: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        if not force:
            print(f"exists, skip: {dest} -> {dest.resolve()}")
            return
        dest.unlink()
    dest.symlink_to(src.resolve())
    print(f"linked {dest} -> {src}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo_root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--no-link",
        action="store_true",
        help="Only download and print paths (do not create data/train, data/val).",
    )
    parser.add_argument("--force", action="store_true", help="Replace existing data/train and data/val links.")
    args = parser.parse_args()

    import kagglehub

    cached = Path(kagglehub.dataset_download(CIFAKE_HANDLE))
    print(f"downloaded: {cached}")
    train_dir = find_split(cached, "train")
    val_dir = find_split(cached, "test")
    print(f"train: {train_dir}")
    print(f"val:   {val_dir}")

    if args.no_link:
        return

    link_split(train_dir, args.repo_root / "data" / "train", args.force)
    link_split(val_dir, args.repo_root / "data" / "val", args.force)


if __name__ == "__main__":
    main()
