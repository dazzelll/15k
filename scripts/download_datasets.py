#!/usr/bin/env python3
"""Download CIFAKE with KaggleHub and SID_Set from Hugging Face, point data/train + data/val at them.

Auth (first match wins):
  1. .env in the repo root with KAGGLE_USERNAME and KAGGLE_KEY
  2. already-exported env vars
  3. ~/.kaggle/kaggle.json

  python scripts/download_datasets.py
  python train.py --train_dir data/train --val_dir data/val --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

CIFAKE_HANDLE = "birdy654/cifake-real-and-ai-generated-synthetic-images"
SID_SET_HANDLE = "saberzl/SID_Set"


def load_dotenv(path: Path) -> None:
    """Load KEY=VALUE lines into os.environ without overwriting existing vars."""
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


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


def download_sid_set(repo_root: Path, sample_size: int = 5000) -> Path:
    """Download SID_Set from Hugging Face and convert to REAL/FAKE structure.
    
    Args:
        repo_root: Root directory of the repository
        sample_size: Number of images to sample per class (for hackathon-scale training)
    
    Returns:
        Path to the processed SID_Set data
    """
    from datasets import load_dataset
    from PIL import Image
    
    print(f"Downloading SID_Set from Hugging Face...")
    
    # Load dataset with streaming to avoid memory issues
    dataset = load_dataset(SID_SET_HANDLE, split="train")
    
    # Create output directory structure
    output_dir = repo_root / "data" / "sid_set"
    real_dir = output_dir / "REAL"
    fake_dir = output_dir / "FAKE"
    
    # Clean up existing directories
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)
    
    # Process and sample images
    real_count = 0
    fake_count = 0
    
    print(f"Sampling {sample_size} images per class from SID_Set...")
    
    for item in dataset:
        if real_count >= sample_size and fake_count >= sample_size:
            break
            
        try:
            label = item["label"]
            image = item["image"]
            img_id = item["img_id"]
            
            # Convert label to REAL/FAKE
            # 0: Real images -> REAL
            # 1: Full synthetic images -> FAKE
            # 2: Tampered images -> FAKE
            if label == 0:
                if real_count >= sample_size:
                    continue
                target_dir = real_dir
                real_count += 1
            else:
                if fake_count >= sample_size:
                    continue
                target_dir = fake_dir
                fake_count += 1
            
            # Save image
            if image is not None:
                # Convert to RGB if needed
                if image.mode != "RGB":
                    image = image.convert("RGB")
                
                # Save with unique filename
                safe_id = img_id.replace("/", "_").replace("\\", "_")
                image_path = target_dir / f"sid_{safe_id}.jpg"
                image.save(image_path, "JPEG", quality=95)
                
                if (real_count + fake_count) % 500 == 0:
                    print(f"Processed {real_count} REAL and {fake_count} FAKE images...")
        except Exception as e:
            print(f"Error processing item: {e}")
            continue
    
    print(f"Downloaded SID_Set: {real_count} REAL images, {fake_count} FAKE images")
    print(f"SID_Set data saved to: {output_dir}")
    
    return output_dir


def merge_datasets(cifake_dir: Path, sid_dir: Path, output_dir: Path) -> None:
    """Merge CIFAKE and SID_Set datasets into a single directory structure.
    
    Args:
        cifake_dir: Path to CIFAKE data (with REAL/FAKE subdirs)
        sid_dir: Path to SID_Set data (with REAL/FAKE subdirs)
        output_dir: Path to output merged directory
    """
    print(f"Merging CIFAKE and SID_Set datasets...")
    
    # Create output directories
    output_real = output_dir / "REAL"
    output_fake = output_dir / "FAKE"
    
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    output_real.mkdir(parents=True, exist_ok=True)
    output_fake.mkdir(parents=True, exist_ok=True)
    
    # Copy CIFAKE images
    cifake_real = cifake_dir / "REAL"
    cifake_fake = cifake_dir / "FAKE"
    
    if cifake_real.exists():
        for img in cifake_real.glob("*.jpg"):
            shutil.copy(img, output_real / f"cifake_{img.name}")
        print(f"Copied {len(list(output_real.glob('cifake_*.jpg')))} CIFAKE REAL images")
    
    if cifake_fake.exists():
        for img in cifake_fake.glob("*.jpg"):
            shutil.copy(img, output_fake / f"cifake_{img.name}")
        print(f"Copied {len(list(output_fake.glob('cifake_*.jpg')))} CIFAKE FAKE images")
    
    # Copy SID_Set images
    sid_real = sid_dir / "REAL"
    sid_fake = sid_dir / "FAKE"
    
    if sid_real.exists():
        for img in sid_real.glob("*.jpg"):
            shutil.copy(img, output_real / img.name)
        print(f"Copied {len(list(output_real.glob('sid_*.jpg')))} SID_Set REAL images")
    
    if sid_fake.exists():
        for img in sid_fake.glob("*.jpg"):
            shutil.copy(img, output_fake / img.name)
        print(f"Copied {len(list(output_fake.glob('sid_*.jpg')))} SID_Set FAKE images")
    
    total_real = len(list(output_real.glob("*.jpg")))
    total_fake = len(list(output_fake.glob("*.jpg")))
    
    print(f"Merged dataset: {total_real} REAL images, {total_fake} FAKE images")
    print(f"Merged data saved to: {output_dir}")


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
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5000,
        help="Number of images to sample per class from SID_Set (default: 5000 for hackathon-scale training)",
    )
    parser.add_argument(
        "--no-sid",
        action="store_true",
        help="Skip downloading SID_Set dataset (only use CIFAKE).",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge CIFAKE and SID_Set into single data/train directory.",
    )
    args = parser.parse_args()

    load_dotenv(args.repo_root / ".env")
    if not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        print(
            "No KAGGLE_USERNAME / KAGGLE_KEY in .env or the environment. "
            "KaggleHub will try ~/.kaggle/kaggle.json next."
        )

    import kagglehub

    # Download CIFAKE
    cached = Path(kagglehub.dataset_download(CIFAKE_HANDLE))
    print(f"downloaded CIFAKE: {cached}")
    train_dir = find_split(cached, "train")
    val_dir = find_split(cached, "test")
    print(f"CIFAKE train: {train_dir}")
    print(f"CIFAKE val:   {val_dir}")

    # Download SID_Set if not skipped
    sid_train_dir = None
    if not args.no_sid:
        try:
            sid_train_dir = download_sid_set(args.repo_root, args.sample_size)
        except Exception as e:
            print(f"Failed to download SID_Set: {e}")
            print("Continuing with CIFAKE only...")

    if args.no_link:
        return

    if args.merge and sid_train_dir:
        # Merge datasets if requested
        merge_datasets(train_dir, sid_train_dir, args.repo_root / "data" / "train")
        link_split(val_dir, args.repo_root / "data" / "val", args.force)
        print("Merged CIFAKE + SID_Set data available in data/train and data/val")
    else:
        # Link datasets separately
        link_split(train_dir, args.repo_root / "data" / "cifake_train", args.force)
        link_split(val_dir, args.repo_root / "data" / "cifake_val", args.force)
        
        if sid_train_dir:
            link_split(sid_train_dir, args.repo_root / "data" / "sid_train", args.force)
            print("Note: CIFAKE data in data/cifake_train, SID_Set data in data/sid_train")
            print("Use --merge flag to combine them into data/train")


if __name__ == "__main__":
    main()
