"""Folder dataset: real/ vs fake/ (or REAL/FAKE CIFAKE layout)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from .transforms import ProtocolTrainTransform

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
REAL_NAMES = {"real", "REAL", "authentic", "nature", "coco", "non-aigc", "nonaigc"}
FAKE_NAMES = {"fake", "FAKE", "aigc", "synthetic", "ai", "generated"}


def list_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def infer_label_from_path(path: Path) -> int | None:
    parts = {p.lower() for p in path.parts}
    if parts & {n.lower() for n in FAKE_NAMES}:
        return 1
    if parts & {n.lower() for n in REAL_NAMES}:
        return 0
    return None


class AIGCFolderDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        image_size: int = 224,
        train: bool = True,
        protocol_aug_prob: float = 0.85,
    ):
        self.root = Path(root)
        self.samples: list[tuple[Path, int]] = []
        for path in list_images(self.root):
            label = infer_label_from_path(path)
            if label is None:
                continue
            self.samples.append((path, label))
        if not self.samples:
            raise FileNotFoundError(
                f"No labelled images under {self.root}. "
                "Expected folder names like real/ fake/ or REAL/ FAKE/."
            )
        self.protocol = ProtocolTrainTransform(p=protocol_aug_prob) if train else None
        self.to_tensor = T.Compose(
            [
                T.Resize((image_size, image_size), antialias=True),
                T.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.protocol is not None:
            img = self.protocol(img)
        x = self.to_tensor(img)
        return x, float(label), str(path)
