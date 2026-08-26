import torch
from PIL import Image

from src.features import degradation_stats, forensic_maps
from src.transforms import PROTOCOL, ProtocolTrainTransform, random_protocol_transform, severity_for


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
    out, name, sev = random_protocol_transform(img)
    assert out.size == img.size
    assert severity_for(name) == sev
    assert 0.0 <= sev <= 1.0


def test_protocol_train_returns_severity():
    img = Image.fromarray((torch.rand(64, 80, 3).numpy() * 255).astype("uint8"))
    tfm = ProtocolTrainTransform(p=1.0)
    out, name, sev = tfm(img)
    assert out.size == img.size
    assert isinstance(name, str)
    assert 0.0 <= sev <= 1.0


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
