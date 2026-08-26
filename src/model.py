"""ForgeGate dual-stream detector: frozen CLIP + gated forensic CNN."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torchvision.models as tvm

from .features import degradation_stats, forensic_maps


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def clip_normalize(x: torch.Tensor) -> torch.Tensor:
    mean = x.new_tensor(CLIP_MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(CLIP_STD).view(1, 3, 1, 1)
    return (x - mean) / std


@dataclass
class ForgeGateOutput:
    logit: torch.Tensor
    prob: torch.Tensor
    gate: torch.Tensor
    stats: torch.Tensor


class ForgeGate(nn.Module):
    """P(AIGC) from semantic CLIP features + gated forensic residual stream.

    CLIP stays frozen (UnivFD-style generalisation). The forensic CNN is small
    and only trusted when degradation stats say traces are still intact.
    """

    def __init__(
        self,
        clip_model: str = "ViT-B-16",
        clip_pretrained: str = "openai",
        freeze_clip: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        import open_clip

        self.clip_arch = clip_model
        self.clip, _, _ = open_clip.create_model_and_transforms(
            clip_model, pretrained=clip_pretrained
        )
        self.clip.eval()
        self.freeze_clip = freeze_clip
        if freeze_clip:
            for p in self.clip.parameters():
                p.requires_grad = False

        visual = self.clip.visual
        self.sem_dim = getattr(visual, "output_dim", 512)

        forensic = tvm.resnet18(weights=None)
        forensic.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        forensic.fc = nn.Identity()
        self.forensic = forensic
        self.for_dim = 512

        self.gate = nn.Sequential(
            nn.Linear(self.sem_dim + self.for_dim + 8, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(self.sem_dim + self.for_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

        self.zeroshot_prompts = (
            "a real photograph taken by a camera",
            "an AI-generated synthetic image",
        )

    def encode_semantic(self, rgb: torch.Tensor) -> torch.Tensor:
        x = clip_normalize(rgb)
        ctx = torch.no_grad() if self.freeze_clip else torch.enable_grad()
        with ctx:
            z = self.clip.encode_image(x)
        z = z.float()
        return z / z.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def forward(self, rgb: torch.Tensor) -> ForgeGateOutput:
        """rgb: Bx3xHxW in [0, 1]."""
        z_s = self.encode_semantic(rgb)
        maps = forensic_maps(rgb)
        stats = degradation_stats(rgb)
        z_f = self.forensic(maps)
        z_f = z_f / z_f.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        gate = torch.sigmoid(self.gate(torch.cat([z_s, z_f, stats], dim=1)))
        fused = torch.cat([z_s, gate * z_f], dim=1)
        logit = self.head(fused).squeeze(-1)
        return ForgeGateOutput(
            logit=logit,
            prob=torch.sigmoid(logit),
            gate=gate.squeeze(-1),
            stats=stats,
        )

    @torch.no_grad()
    def zeroshot_prob(self, rgb: torch.Tensor) -> torch.Tensor:
        """CLIP prompt baseline so a demo runs before any training."""
        import open_clip

        z = self.encode_semantic(rgb)
        tokenizer = open_clip.get_tokenizer(self.clip_arch)
        tokens = tokenizer(list(self.zeroshot_prompts)).to(rgb.device)
        text = self.clip.encode_text(tokens).float()
        text = text / text.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        sims = z @ text.T
        return torch.softmax(sims, dim=-1)[:, 1]

    def trainable_parameters(self):
        for p in self.parameters():
            if p.requires_grad:
                yield p

    def parameter_counts(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}
