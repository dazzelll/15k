import torch

from src.features import degradation_stats, forensic_maps
from src.transforms import PROTOCOL, random_protocol_transform
from PIL import Image


def test_forensic_batch_shapes():
    x = torch.rand(4, 3, 224, 224)
    maps = forensic_maps(x)
    stats = degradation_stats(x)
    assert maps.shape == (4, 3, 224, 224)
    assert stats.shape == (4, 8)
    assert torch.isfinite(maps).all()
    assert torch.isfinite(stats).all()


def test_protocol_keeps_size():
    img = Image.fromarray((torch.rand(64, 80, 3).numpy() * 255).astype("uint8"))
    for variants in PROTOCOL.values():
        for name, fn in variants:
            out = fn(img)
            assert out.size == img.size, name
            assert out.mode == "RGB"
    out = random_protocol_transform(img)
    assert out.size == img.size
