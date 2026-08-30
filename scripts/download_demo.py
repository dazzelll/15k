#!/usr/bin/env python3
"""Download WildFake demo subset (COCO val2017 + DALL-E Advanced) for demo video only.

This dataset is for demonstration purposes only and should NOT be used for training.
It contains:
- Non-AIGC: COCO val2017 (4998 images)
- AIGC: DALL-E Advanced (8843 images)

Usage:
  python scripts/download_demo.py
  python scripts/download_demo.py --force
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

WILDFAKE_HANDLE = "hy2628982280/WildFake"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def _has_real_fake(folder: Path) -> bool:
    names = {p.name for p in folder.iterdir() if p.is_dir()}
    return bool(names & {"REAL", "real", "FAKE", "fake"})


def source_ready(path: Path) -> bool:
    real, fake = path / "REAL", path / "FAKE"
    return (
        real.is_dir()
        and fake.is_dir()
        and any(real.iterdir())
        and any(fake.iterdir())
    )


def _safe_id(value: object, fallback: str) -> str:
    text = fallback if value is None else str(value)
    return text.replace("/", "_").replace("\\", "_")


def _suffix_from_name(name: str | None) -> str:
    if not name:
        return ""
    suf = Path(str(name)).suffix.lower()
    return suf if suf in IMAGE_EXTS else ""


def _field(item: dict, *names: str):
    lookup = {str(k).lower(): k for k in item}
    for name in names:
        key = lookup.get(name.lower())
        if key is not None:
            return item[key]
    return None


def _as_dict(raw: object) -> dict:
    if isinstance(raw, dict):
        return raw
    if hasattr(raw, "items"):
        return dict(raw.items())
    return dict(raw)


def is_wildfake_demo_row(item: dict) -> bool:
    """Check if this is a demo row: COCO val2017 + DALL-E Advanced only."""
    path = str(_field(item, "Image_path", "image_path", "path") or "")
    meta = " ".join(
        str(_field(item, k) or "")
        for k in ("Generator", "Architecture", "Weight", "Category")
    )
    blob = f"{path} {meta}".lower()
    if any(token in blob for token in ("coco_val2017", "coco-val", "val2017")):
        return True
    advanced = "advanced" in blob
    try:
        advanced = advanced or int(_field(item, "IsAdvanced", "is_advanced") or 0) == 1
    except (TypeError, ValueError):
        pass
    dalle = any(token in blob for token in ("dall-e", "dalle", "dall_e"))
    return bool(dalle and advanced)


def parse_binary_label(label: object) -> int | None:
    """Map dataset labels to 0=real, 1=fake."""
    if label is None:
        return None
    if isinstance(label, (bytes, bytearray)):
        label = label.decode("utf-8", errors="ignore")
    if isinstance(label, str):
        s = label.strip().lower()
        if s in {"0", "real", "authentic", "nature", "nonaigc", "non-aigc"}:
            return 0
        if s in {"1", "2", "fake", "aigc", "synthetic", "generated", "full_synthetic", "tampered"}:
            return 1
        return None
    if isinstance(label, bool):
        return int(label)
    if isinstance(label, (int, float)):
        value = int(label)
        if value == 0:
            return 0
        if value > 0:
            return 1
    return None


def _wildfake_relpath(image_path: str) -> str:
    text = image_path.strip()
    if text.startswith("./"):
        return text[2:]
    return text.lstrip("/")


def _locate_wildfake_root(rel: str) -> Path | None:
    parts = Path(rel).parts
    if not parts:
        return None
    marker = parts[0]
    bases = [
        Path.home() / ".cache" / "modelscope",
        Path.home() / ".cache" / "modelscope" / "hub" / "datasets",
    ]
    for base in bases:
        if not base.exists():
            continue
        for hit in base.rglob(marker):
            if hit.is_dir() and (hit.parent / rel).is_file():
                return hit.parent
    return None


def _resolve_wildfake_file(image_path: str, root: list[Path | None]) -> Path | None:
    p = Path(image_path)
    if p.is_file():
        return p
    rel = _wildfake_relpath(image_path)
    if root[0] is None:
        root[0] = _locate_wildfake_root(rel)
    if root[0] is not None:
        cand = root[0] / rel
        if cand.is_file():
            return cand
    return None


def download_wildfake_demo(repo_root: Path, force: bool) -> Path:
    """Download WildFake demo subset (COCO val2017 + DALL-E Advanced) for demo video only.
    
    This dataset is for demonstration purposes only and should NOT be used for training.
    It contains:
    - Non-AIGC: COCO val2017 (4998 images)
    - AIGC: DALL-E Advanced (8843 images)
    """
    from modelscope.msdatasets import MsDataset

    demo_out = repo_root / "data_demo" / "wildfake_demo"
    if not force and source_ready(demo_out):
        print(f"WildFake demo already present at {demo_out}; skip (--force to redo)")
        return demo_out

    print("Downloading WildFake demo subset from ModelScope...")
    try:
        dataset = MsDataset.load(WILDFAKE_HANDLE, subset_name='default', split='train')
    except Exception as e:
        print(f"WildFake demo failed ({e}); loading without subset/split")
        dataset = MsDataset.load(WILDFAKE_HANDLE)

    if demo_out.exists():
        shutil.rmtree(demo_out)
    real_dir, fake_dir = demo_out / "REAL", demo_out / "FAKE"
    real_dir.mkdir(parents=True)
    fake_dir.mkdir(parents=True)

    n_real = n_fake = 0
    root: list[Path | None] = [None]

    for raw in dataset:
        item = _as_dict(raw)
        # Only include demo rows (COCO val2017 + DALL-E Advanced)
        if not is_wildfake_demo_row(item):
            continue
        
        mapped = parse_binary_label(_field(item, "IsFake", "is_fake", "label"))
        if mapped is None:
            continue
            
        image_path = _field(item, "Image_path", "image_path", "path")
        src = None
        if image_path:
            src = _resolve_wildfake_file(str(image_path), root)
        if src is None:
            image = _field(item, "image", "img", "Image")
            if image is not None and not isinstance(image, (str, Path)):
                continue
        if src is None or not src.is_file():
            continue

        arch = _field(item, "Architecture", "Category", "Generator") or "demo"
        num = _field(item, "Num", "num")
        stem = _safe_id(f"{arch}_{num}_{src.stem}", src.stem)
        
        target_dir = fake_dir if mapped == 1 else real_dir
        shutil.copy2(src, target_dir / f"demo_{stem}{src.suffix.lower() or '.jpg'}")
        
        if mapped == 1:
            n_fake += 1
        else:
            n_real += 1

    print(f"WildFake demo downloaded: {n_real} REAL (COCO val2017), {n_fake} FAKE (DALL-E Advanced)")
    print(f"WARNING: This dataset is for demo video only. DO NOT use for training.")
    return demo_out


def main() -> None:
    import os
    
    parser = argparse.ArgumentParser(description="Download WildFake demo subset for demo video")
    parser.add_argument("--repo_root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--force", action="store_true", help="Rebuild even if demo exists.")
    args = parser.parse_args()

    demo_dir = download_wildfake_demo(args.repo_root, args.force)
    
    print(f"\nDemo dataset ready at: {demo_dir}")
    print(f"Demo eval: python evaluate.py --data_dir {demo_dir} --checkpoint checkpoints/best.pt")


if __name__ == "__main__":
    main()
