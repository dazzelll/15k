import torch
from PIL import Image

from src.features import degradation_stats, forensic_maps
from src.transforms import (
    PROTOCOL,
    ProtocolTrainTransform,
    gaussian_noise,
    jpeg_compress,
    measure_severity,
    severity_for,
)


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
        for op in variants:
            out = op.fn(img)
            assert out.size == img.size, op.name
            assert out.mode == "RGB"
            assert 0.0 <= op.severity <= 1.0
            assert severity_for(op.name) == op.severity


def test_protocol_train_returns_severity():
    img = Image.fromarray((torch.rand(64, 80, 3).numpy() * 255).astype("uint8"))
    tfm = ProtocolTrainTransform(p=1.0)
    out, name, sev = tfm(img)
    assert out.size == img.size
    assert isinstance(name, str)
    assert 0.0 <= sev <= 1.0


def test_measure_severity_noise_and_jpeg():
    import random

    rng = random.Random(0)
    arr = (torch.rand(96, 96, 3).numpy() * 255).astype("uint8")
    img = Image.fromarray(arr)
    noisy = gaussian_noise(img, 0.10, rng=rng)
    jpeg = jpeg_compress(img, 30)
    sev_noise = measure_severity(img, noisy)
    sev_jpeg = measure_severity(img, jpeg)
    sev_same = measure_severity(img, img)
    assert 0.0 <= sev_noise <= 1.0
    assert 0.0 <= sev_jpeg <= 1.0
    assert sev_same < 1e-6
    assert sev_noise > 0.02
    assert sev_jpeg > 0.02


def test_measure_severity_tiny_image_is_finite():
    img = Image.fromarray((torch.rand(1, 1, 3).numpy() * 255).astype("uint8"))
    other = Image.fromarray((torch.rand(1, 1, 3).numpy() * 255).astype("uint8"))
    sev = measure_severity(img, other)
    assert 0.0 <= sev <= 1.0


def test_protocol_rng_is_deterministic():
    import random

    img = Image.fromarray((torch.rand(64, 80, 3).numpy() * 255).astype("uint8"))
    tfm = ProtocolTrainTransform(p=1.0)
    a = tfm(img, rng=random.Random(0))
    b = tfm(img, rng=random.Random(0))
    c = tfm(img, rng=random.Random(1))
    assert a[1] == b[1] and a[2] == b[2]
    assert list(a[0].getdata()) == list(b[0].getdata())
    assert a[1] != c[1] or list(a[0].getdata()) != list(c[0].getdata())


def test_forgegate_train_keeps_clip_eval():
    from src.model import ForgeGate

    model = ForgeGate()
    model.train()
    assert model.training
    assert not model.clip.training
    out = model(torch.rand(1, 3, 224, 224))
    assert out.logit.shape == (1,)
    assert out.gate.shape == (1,)
    out0 = model(torch.rand(1, 3, 224, 224), gate_mode="zero")
    out1 = model(torch.rand(1, 3, 224, 224), gate_mode="one")
    assert out0.gate.shape == (1,)
    assert out1.gate.shape == (1,)
