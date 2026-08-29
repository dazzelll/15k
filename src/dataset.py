"""Folder dataset: real/ vs fake/ (or REAL/FAKE CIFAKE layout)."""

from __future__ import annotations

from pathlib import Path
from random import Random

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
    """Returns (x_clean, x_transformed, label, severity, path).

    At train time, x_transformed is protocol-augmented and severity is the
    known degradation strength in [0, 1]. At eval time both tensors are clean
    and severity is 0.
    """

    def __init__(
        self,
        root: str | Path,
        image_size: int = 224,
        train: bool = True,
        protocol_aug_prob: float = 0.85,
        protocol_seed: int | None = None,
    ):
        self.root = Path(root)
        self.protocol_seed = protocol_seed
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
        x_clean = self.to_tensor(img)
        if self.protocol is not None:
            rng = Random(self.protocol_seed + idx) if self.protocol_seed is not None else None
            img_t, _name, severity = self.protocol(img, rng=rng)
            x_t = self.to_tensor(img_t)
        else:
            x_t = x_clean
            severity = 0.0
        return x_clean, x_t, float(label), float(severity), str(path)
