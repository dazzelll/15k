#!/usr/bin/env python3
"""Build ForgeGate train/val plus a separate AIGC test set.

data/train <- CIFAKE(train) + SID_Set(train) + WildFake(train), sampled and balanced.
data/val   <- CIFAKE(test) + SID_Set(val) + WildFake(val)  # checkpoint selection only
data/aigc_benchmark <- AIGC-Detection-Benchmark(test)     # unseen-generator test; never in train/val

Auth (first match wins):
  1. .env in the repo root with KAGGLE_USERNAME and KAGGLE_KEY
  2. already-exported env vars
  3. ~/.kaggle/kaggle.json

Usage:
  python scripts/download_datasets.py
  python scripts/download_datasets.py --no-wildfake
  python scripts/download_datasets.py --no-aigc-benchmark
  python train.py --train_dir data/train --val_dir data/val --config configs/default.yaml
  python evaluate.py --data_dir data/val --checkpoint checkpoints/best.pt
  python evaluate.py --data_dir data/aigc_benchmark --checkpoint checkpoints/best.pt

Notes:
  - WildFake rows are Generator / Architecture / Weight / Category / IsAdvanced /
    IsFake / Image_path / Num. Label is IsFake (0=real, 1=fake). The table is
    grouped by generator, so we reservoir-sample instead of taking the first N
    (which would be almost all BigGAN). Demo subset (COCO val2017 + DALL-E
    Advanced) is excluded from train and val.
  - SID_Set is sequential (no .shuffle()). Shuffling a streaming HF dataset
    backed by large parquet shards can spike RAM by several GB per row.
    DataLoader(shuffle=True) in train.py reshuffles at train time.
  - Original image bytes are written when available so we do not add a JPEG pass
    on real photos (the forensic stream looks at compression traces).
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path

CIFAKE_HANDLE = "birdy654/cifake-real-and-ai-generated-synthetic-images"
SID_SET_HANDLE = "saberzl/SID_Set"
WILDFAKE_HANDLE = "hy2628982280/WildFake"
AIGC_BENCHMARK_HANDLE = "TheKernel01/AIGC-Detection-Benchmark"

CIFAKE_TRAIN_SAMPLE = 6000
CIFAKE_VAL_SAMPLE = 1200
SID_TRAIN_SAMPLE = 6000
SID_VAL_SAMPLE = 1200
WILDFAKE_TRAIN_SAMPLE = 6000
WILDFAKE_VAL_SAMPLE = 1200
AIGC_PER_GENERATOR = 100

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


def find_split(root: Path, split: str) -> Path:
    candidates = [p for p in root.rglob(split) if p.is_dir() and _has_real_fake(p)]
    if not candidates:
        listing = "\n".join(str(p.relative_to(root)) for p in sorted(root.rglob("*"))[:40])
        raise FileNotFoundError(f"No '{split}/' with REAL/FAKE under {root}. Listing:\n{listing}")
    return sorted(candidates, key=lambda p: len(p.parts))[0]


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
    """Drop organiser demo refs: COCO val2017 + DALL-E Advanced only."""
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
    """Map dataset labels to 0=real, 1=fake. None if unknown (caller must skip)."""
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


def save_original_or_png(image: object, dest_stem: Path) -> bool:
    """Write original bytes when present; otherwise PNG (no extra JPEG pass)."""
    dest_stem.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(image, dict):
        raw = image.get("bytes")
        src_path = image.get("path")
        suffix = _suffix_from_name(src_path) or ".jpg"
        if raw:
            dest_stem.with_suffix(suffix).write_bytes(raw)
            return True
        if src_path and Path(src_path).is_file():
            suffix = _suffix_from_name(src_path) or Path(src_path).suffix or ".jpg"
            shutil.copy2(src_path, dest_stem.with_suffix(suffix))
            return True
        return False

    if isinstance(image, (bytes, bytearray)):
        dest_stem.with_suffix(".jpg").write_bytes(image)
        return True

    if isinstance(image, (str, Path)) and Path(image).is_file():
        src = Path(image)
        suffix = _suffix_from_name(src.name) or src.suffix or ".png"
        shutil.copy2(src, dest_stem.with_suffix(suffix))
        return True

    if hasattr(image, "save"):
        pil = image.convert("RGB") if getattr(image, "mode", "RGB") != "RGB" else image
        pil.save(dest_stem.with_suffix(".png"))
        return True

    return False


def sample_folder(src_class_dir: Path, dest_dir: Path, n: int, prefix: str, seed: int = 42) -> int:
    dest_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in src_class_dir.iterdir() if p.is_file())
    rng = random.Random(seed)
    rng.shuffle(files)
    chosen = files[:n]
    for f in chosen:
        shutil.copy2(f, dest_dir / f"{prefix}_{f.name}")
    return len(chosen)


def sample_cifake(cached: Path, repo_root: Path, force: bool) -> tuple[Path, Path]:
    train_dir = find_split(cached, "train")
    test_dir = find_split(cached, "test")

    def class_subdir(root: Path, wanted: set[str]) -> Path:
        for p in root.iterdir():
            if p.is_dir() and p.name in wanted:
                return p
        raise FileNotFoundError(f"No {wanted} folder under {root}")

    out_train = repo_root / "data" / "_sources" / "cifake_train"
    out_val = repo_root / "data" / "_sources" / "cifake_val"
    if not force and source_ready(out_train) and source_ready(out_val):
        print(f"CIFAKE samples already present under {out_train.parent}; skip (--force to redo)")
        return out_train, out_val

    for split_root, out_root, n in (
        (train_dir, out_train, CIFAKE_TRAIN_SAMPLE),
        (test_dir, out_val, CIFAKE_VAL_SAMPLE),
    ):
        if out_root.exists():
            shutil.rmtree(out_root)
        real_src = class_subdir(split_root, {"REAL", "real"})
        fake_src = class_subdir(split_root, {"FAKE", "fake"})
        n_real = sample_folder(real_src, out_root / "REAL", n, "cifake")
        n_fake = sample_folder(fake_src, out_root / "FAKE", n, "cifake")
        print(f"CIFAKE {out_root.name}: {n_real} REAL, {n_fake} FAKE")

    return out_train, out_val


def _dump_sid_split(dataset_split: str, repo_root: Path, out_name: str, sample_size: int) -> Path:
    from datasets import Image as HFImage
    from datasets import load_dataset

    ds = load_dataset(SID_SET_HANDLE, split=dataset_split, streaming=True)
    try:
        ds = ds.remove_columns(["mask"])
    except Exception:
        pass
    ds = ds.cast_column("image", HFImage(decode=False))

    output_dir = repo_root / "data" / "_sources" / out_name
    if output_dir.exists():
        shutil.rmtree(output_dir)
    real_dir, fake_dir = output_dir / "REAL", output_dir / "FAKE"
    real_dir.mkdir(parents=True)
    fake_dir.mkdir(parents=True)

    real_count = fake_count = 0
    for item in ds:
        if real_count >= sample_size and fake_count >= sample_size:
            break
        mapped = parse_binary_label(item.get("label"))
        if mapped is None:
            continue
        if mapped == 0:
            if real_count >= sample_size:
                continue
            target_dir, kind = real_dir, "real"
        else:
            if fake_count >= sample_size:
                continue
            target_dir, kind = fake_dir, "fake"

        stem = target_dir / f"sid_{_safe_id(item.get('img_id'), f'{kind}_{real_count + fake_count + 1}')}"
        if not save_original_or_png(item.get("image"), stem):
            continue
        if mapped == 0:
            real_count += 1
        else:
            fake_count += 1
        if (real_count + fake_count) % 1000 == 0:
            print(f"SID_Set {out_name}: {real_count} REAL, {fake_count} FAKE so far...")

    print(f"SID_Set {out_name}: done, {real_count} REAL, {fake_count} FAKE")
    return output_dir


def download_sid_set(repo_root: Path, force: bool) -> tuple[Path, Path]:
    train_out = repo_root / "data" / "_sources" / "sid_train"
    val_out = repo_root / "data" / "_sources" / "sid_val"
    if not force and source_ready(train_out) and source_ready(val_out):
        print("SID_Set samples already present; skip (--force to redo)")
        return train_out, val_out

    train_out = _dump_sid_split("train", repo_root, "sid_train", SID_TRAIN_SAMPLE)
    val_split = "val"
    try:
        val_out = _dump_sid_split(val_split, repo_root, "sid_val", SID_VAL_SAMPLE)
    except Exception as first:
        print(f"SID_Set split='val' failed ({first}); trying 'validation'")
        val_out = _dump_sid_split("validation", repo_root, "sid_val", SID_VAL_SAMPLE)
    return train_out, val_out


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


def _reservoir_push(buf: list, item, seen: int, k: int, rng: random.Random) -> int:
    seen += 1
    if len(buf) < k:
        buf.append(item)
    else:
        j = rng.randrange(seen)
        if j < k:
            buf[j] = item
    return seen


def _write_wildfake_split(rows: list[tuple[Path, bool, str]], dest: Path) -> tuple[int, int]:
    if dest.exists():
        shutil.rmtree(dest)
    real_dir, fake_dir = dest / "REAL", dest / "FAKE"
    real_dir.mkdir(parents=True)
    fake_dir.mkdir(parents=True)
    n_real = n_fake = 0
    for src, is_fake, stem in rows:
        target_dir = fake_dir if is_fake else real_dir
        shutil.copy2(src, target_dir / f"wf_{stem}{src.suffix.lower() or '.jpg'}")
        if is_fake:
            n_fake += 1
        else:
            n_real += 1
    print(f"WildFake {dest.name}: {n_real} REAL, {n_fake} FAKE")
    return n_real, n_fake


def download_wildfake(repo_root: Path, force: bool) -> tuple[Path, Path | None]:
    from modelscope.msdatasets import MsDataset

    train_out = repo_root / "data" / "_sources" / "wildfake_train"
    val_out = repo_root / "data" / "_sources" / "wildfake_val"
    if not force and source_ready(train_out) and source_ready(val_out):
        print("WildFake samples already present; skip (--force to redo)")
        return train_out, val_out

    try:
        dataset = MsDataset.load(WILDFAKE_HANDLE, subset_name='default', split='train')
    except Exception as e:
        print(f"WildFake default subset failed ({e}); loading without subset/split")
        dataset = MsDataset.load(WILDFAKE_HANDLE)

    need_real = WILDFAKE_TRAIN_SAMPLE + WILDFAKE_VAL_SAMPLE
    need_fake = WILDFAKE_TRAIN_SAMPLE + WILDFAKE_VAL_SAMPLE
    rng = random.Random(42)
    real_buf: list[tuple[Path, str]] = []
    fake_buf: list[tuple[Path, str]] = []
    seen_real = seen_fake = 0
    root: list[Path | None] = [None]
    scanned = 0

    for raw in dataset:
        scanned += 1
        item = _as_dict(raw)
        if is_wildfake_demo_row(item):
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
                # decoded image with no usable path — skip; we copy originals only
                continue
        if src is None or not src.is_file():
            continue
        arch = _field(item, "Architecture", "Category", "Generator") or "wf"
        num = _field(item, "Num", "num")
        stem = _safe_id(f"{arch}_{num}_{src.stem}", src.stem)
        if mapped == 0:
            seen_real = _reservoir_push(real_buf, (src, stem), seen_real, need_real, rng)
        else:
            seen_fake = _reservoir_push(fake_buf, (src, stem), seen_fake, need_fake, rng)
        if scanned % 50000 == 0:
            print(
                f"WildFake scanned {scanned}: kept pool REAL={len(real_buf)}/{seen_real} "
                f"FAKE={len(fake_buf)}/{seen_fake}"
            )

    rng.shuffle(real_buf)
    rng.shuffle(fake_buf)
    train_rows = [(p, False, s) for p, s in real_buf[:WILDFAKE_TRAIN_SAMPLE]]
    train_rows += [(p, True, s) for p, s in fake_buf[:WILDFAKE_TRAIN_SAMPLE]]
    val_rows = [(p, False, s) for p, s in real_buf[WILDFAKE_TRAIN_SAMPLE:need_real]]
    val_rows += [(p, True, s) for p, s in fake_buf[WILDFAKE_TRAIN_SAMPLE:need_fake]]

    n_tr_real, n_tr_fake = _write_wildfake_split(train_rows, train_out)
    n_va_real, n_va_fake = _write_wildfake_split(val_rows, val_out)
    if n_tr_real == 0 or n_tr_fake == 0:
        raise RuntimeError(
            f"WildFake train empty class (REAL={n_tr_real}, FAKE={n_tr_fake}) after {scanned} rows. "
            "Check Image_path resolution against the ModelScope cache."
        )
    if n_va_real == 0 or n_va_fake == 0:
        print("WildFake val is missing a class; val will skip WildFake")
        return train_out, None
    return train_out, val_out


def download_aigc_benchmark(repo_root: Path, force: bool) -> Path:
    from datasets import Image as HFImage
    from datasets import load_dataset

    output_dir = repo_root / "data" / "aigc_benchmark"
    if not force and source_ready(output_dir):
        print(f"AIGC benchmark already present at {output_dir}; skip (--force to redo)")
        return output_dir

    ds = load_dataset(AIGC_BENCHMARK_HANDLE, split="test", streaming=True)
    ds = ds.cast_column("image", HFImage(decode=False))

    generator_names = [
        "Real", "ADM", "BigGAN", "CycleGAN", "DALLE2", "GauGAN", "GLIDE",
        "Midjourney", "ProGAN", "SD14", "SD15", "SDXL", "StarGAN",
        "StyleGAN", "StyleGAN2", "VQDM", "WhichFaceIsReal", "Wukong",
    ]
    if output_dir.exists():
        shutil.rmtree(output_dir)
    real_dir, fake_dir = output_dir / "REAL", output_dir / "FAKE"
    real_dir.mkdir(parents=True)
    fake_dir.mkdir(parents=True)

    counts = {name: 0 for name in generator_names}
    for item in ds:
        if all(c >= AIGC_PER_GENERATOR for c in counts.values()):
            break
        try:
            gid = int(item["generator"])
            gname = generator_names[gid]
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        if counts[gname] >= AIGC_PER_GENERATOR:
            continue
        mapped = parse_binary_label(item.get("label"))
        if mapped is None:
            continue
        stem = (real_dir if mapped == 0 else fake_dir) / f"aigc_{gname}_{counts[gname]}"
        if not save_original_or_png(item.get("image"), stem):
            continue
        counts[gname] += 1

    n_real = sum(1 for _ in real_dir.iterdir())
    n_fake = sum(1 for _ in fake_dir.iterdir())
    print(f"AIGC benchmark test: {n_real} REAL, {n_fake} FAKE across {len(generator_names)} generators")
    print(f"  per generator: {counts}")
    return output_dir


def merge_into(sources: list[Path | None], dest: Path) -> None:
    if dest.exists() or dest.is_symlink():
        if dest.is_symlink() or dest.is_file():
            dest.unlink()
        else:
            shutil.rmtree(dest)
    dest_real, dest_fake = dest / "REAL", dest / "FAKE"
    dest_real.mkdir(parents=True)
    dest_fake.mkdir(parents=True)
    n_real = n_fake = 0
    for src in sources:
        if src is None or not src.exists():
            continue
        for cls, dest_cls in (("REAL", dest_real), ("FAKE", dest_fake)):
            src_cls = src / cls
            if not src_cls.exists():
                continue
            for f in src_cls.iterdir():
                if not f.is_file():
                    continue
                shutil.copy2(f, dest_cls / f.name)
                n_real += cls == "REAL"
                n_fake += cls == "FAKE"
    print(f"merged -> {dest}: {n_real} REAL, {n_fake} FAKE")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--force", action="store_true", help="Rebuild sampled sources even if they exist.")
    parser.add_argument("--no-cifake", action="store_true", help="Skip CIFAKE.")
    parser.add_argument("--no-sid", action="store_true", help="Skip SID_Set.")
    parser.add_argument(
        "--wildfake",
        action="store_true",
        help="Include WildFake in training data (requires modelscope).",
    )
    parser.add_argument(
        "--no-aigc-benchmark",
        action="store_true",
        help="Skip the held-out AIGC-Detection-Benchmark test folder.",
    )
    args = parser.parse_args()

    load_dotenv(args.repo_root / ".env")
    if not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        print("No KAGGLE_USERNAME/KAGGLE_KEY found; kagglehub will try ~/.kaggle/kaggle.json next.")

    import kagglehub

    train_sources: list[Path | None] = []
    val_sources: list[Path | None] = []

    if not args.no_cifake:
        cifake_cached = Path(kagglehub.dataset_download(CIFAKE_HANDLE))
        cifake_train, cifake_val = sample_cifake(cifake_cached, args.repo_root, args.force)
        train_sources.append(cifake_train)
        val_sources.append(cifake_val)

    if not args.no_sid:
        try:
            sid_train, sid_val = download_sid_set(args.repo_root, args.force)
            train_sources.append(sid_train)
            val_sources.append(sid_val)
        except Exception as e:
            print(f"SID_Set failed ({e}); continuing without it.")

    if args.wildfake:
        try:
            wf_train, wf_val = download_wildfake(args.repo_root, args.force)
            train_sources.append(wf_train)
            if wf_val is not None:
                val_sources.append(wf_val)
        except Exception as e:
            print(
                f"WildFake failed ({e}); continuing without it. "
                "Install with `pip install modelscope[datasets]` or pass --no-wildfake."
            )

    merge_into(train_sources, args.repo_root / "data" / "train")
    merge_into(val_sources, args.repo_root / "data" / "val")

    aigc_dir = None
    if not args.no_aigc_benchmark:
        try:
            aigc_dir = download_aigc_benchmark(args.repo_root, args.force)
        except Exception as e:
            print(f"AIGC-Detection-Benchmark failed ({e}); continuing without the test set.")

    train_dest = args.repo_root / "data" / "train"
    val_dest = args.repo_root / "data" / "val"
    print(f"\nTrain/val: python train.py --train_dir {train_dest} --val_dir {val_dest} --config configs/default.yaml")
    print(f"In-distribution eval: python evaluate.py --data_dir {val_dest} --checkpoint checkpoints/best.pt")
    if aigc_dir is not None:
        print(
            f"Unseen-generator test: python evaluate.py --data_dir {aigc_dir} --checkpoint checkpoints/best.pt"
        )
    print(f"\nDemo dataset: python scripts/download_demo.py")


if __name__ == "__main__":
    main()
