"""CP-BG-Bench A1 인코더 (사양 A.2): 채널 어댑터 + DINOv3 ViT-B/16(LoRA) + projection head,
perturbation 인코더(고정 임베딩 표 또는 학습 가능 표) + perturbation head.

[원문 없음] 선택(decisions.md 2026-10-14): head hidden = 입력 차원, skip = 선형 사영,
어댑터 1×1 conv 초기값 = 채널 평균(1/C), 학습 가능 표 차원 128.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ChannelAdapter(nn.Module):
    """C채널 → 3채널 1×1 conv(bias 없음) + ImageNet 통계로 초기화한 학습 가능 affine 정규화."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
        nn.init.constant_(self.conv.weight, 1.0 / in_channels)
        std = torch.tensor(IMAGENET_STD)
        self.scale = nn.Parameter((1.0 / std).view(1, 3, 1, 1))
        self.shift = nn.Parameter((-torch.tensor(IMAGENET_MEAN) / std).view(1, 3, 1, 1))

    def forward(self, x):
        return self.conv(x) * self.scale + self.shift


class GeGLUHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or in_dim
        self.fc1 = nn.Linear(in_dim, 2 * hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.skip = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        a, g = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(a * F.gelu(g)) + self.skip(x)


class SiLUHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or in_dim
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.skip = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        return self.fc2(F.silu(self.fc1(x))) + self.skip(x)


class ImageEncoder(nn.Module):
    def __init__(self, in_channels: int, backbone: str, lora: dict, out_dim: int, pretrained: bool = True):
        super().__init__()
        import timm
        from peft import LoraConfig, get_peft_model

        self.adapter = ChannelAdapter(in_channels)
        bb = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        cfg = LoraConfig(r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                         target_modules=lora["target_modules"])
        self.backbone = get_peft_model(bb, cfg)        # LoRA 외 trunk는 고정된다
        self.head = GeGLUHead(bb.num_features, out_dim)

    def features(self, x):
        """CLS 토큰 768차원 (timm DINOv3의 기본 global_pool은 avg라 직접 CLS를 꺼낸다)."""
        return self.backbone.forward_features(self.adapter(x))[:, 0]

    def forward(self, x):
        return self.head(self.features(x))


class PertEncoder(nn.Module):
    """label 인덱스 0..P-1 = perturbation, P = 대조군(전체 perturbation 입력 임베딩 평균).

    table: 고정 임베딩 [P, D](ESM2·ECFP4) 또는 None(학습 가능 표, learned_dim 차원).
    """

    def __init__(self, n_perts: int, out_dim: int, table: torch.Tensor | None = None, learned_dim: int = 128):
        super().__init__()
        self.n_perts = n_perts
        if table is not None:
            assert table.shape[0] == n_perts
            self.register_buffer("table", table.float())
            self.learned = None
            dim = table.shape[1]
        else:
            self.learned = nn.Embedding(n_perts, learned_dim)
            nn.init.normal_(self.learned.weight, std=0.02)
            dim = learned_dim
        self.head = SiLUHead(dim, out_dim)

    def inputs(self) -> torch.Tensor:
        t = self.table if self.learned is None else self.learned.weight
        return torch.cat([t, t.mean(0, keepdim=True)], 0)          # [P+1, D]

    def forward(self, idx: torch.Tensor | None = None) -> torch.Tensor:
        e = self.inputs()
        return self.head(e if idx is None else e[idx])
