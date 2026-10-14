"""view 생성과 증강 (사양 A.1, A.2). GPU 배치 연산.

입력은 crop 저장소의 원본(uint8 [B,C,150,150])과 중심 세포의 이진 마스크(핵, 세포)다.
순서: 리사이즈(이미지 bilinear, 마스크 nearest) → 밝기·대비 → view 마스킹 → 기하 증강.
밝기·대비를 마스킹 전에 적용해 S view의 마스크 밖이 정확히 0으로 남게 한다.
기하 증강은 파라미터를 따로 뽑아(`sample_geometry`) 같은 세포의 여러 view에 같은 변환을 줄 수 있다.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

VIEWS = ("C", "S", "CD", "SD", "N", "B")


def to_input(img_u8: torch.Tensor, nuc: torch.Tensor, cell: torch.Tensor, size: int, scale: float = 255.0):
    """uint8 [B,C,h,w], bool [B,h,w] ×2 → float [B,C,size,size] ∈ [0,1], bool 마스크 [B,size,size] ×2."""
    x = F.interpolate(img_u8.float() / scale, size=(size, size), mode="bilinear", align_corners=False)
    m = torch.stack([nuc, cell], 1).float()
    m = F.interpolate(m, size=(size, size), mode="nearest") > 0.5
    return x, m[:, 0], m[:, 1]


def make_view(x: torch.Tensor, nuc: torch.Tensor, cell: torch.Tensor, view: str, dna_idx: int) -> torch.Tensor:
    """x [B,C,H,W], 중심 세포 마스크 nuc/cell [B,H,W] → view 이미지.

    C: 원본. S: DNA 채널은 핵 마스크 밖을 0, 나머지 채널은 세포 마스크 밖을 0.
    (E0은 원문대로 핵 마스크를 세포 마스크로 자르지 않는다. decisions.md 2026-10-14)
    """
    if view == "C":
        return x
    if view == "S":
        keep = cell[:, None].expand_as(x).clone()
        keep[:, dna_idx] = nuc
        return x * keep
    if view in ("CD", "SD"):
        # TODO: 22×22 패치의 적용 해상도(150 crop 기준 / 224 기준)가 원문에 없다. E0 C·S 이후에 정한다
        raise NotImplementedError(f"{view} view는 아직 구현하지 않았다")
    if view in ("N", "B"):
        raise NotImplementedError(f"{view} view는 E2용 스텁이다")
    raise ValueError(f"알 수 없는 view: {view}")


def photometric(x: torch.Tensor, brightness: float, contrast: float, gen: torch.Generator | None = None) -> torch.Tensor:
    """이미지별 밝기(곱) ±brightness, 대비(이미지 평균 기준) ±contrast."""
    b = x.shape[0]
    u = torch.rand(b, 2, device=x.device, generator=gen) * 2 - 1
    bf = (1 + brightness * u[:, 0]).view(b, 1, 1, 1)
    cf = (1 + contrast * u[:, 1]).view(b, 1, 1, 1)
    mean = x.mean(dim=(2, 3), keepdim=True)
    return ((x - mean) * cf + mean) * bf


def sample_geometry(b: int, max_deg: float, hflip: bool, vflip: bool, device, gen: torch.Generator | None = None) -> dict:
    u = torch.rand(b, 3, device=device, generator=gen)
    return {"angle": u[:, 0] * math.radians(max_deg),
            "hflip": (u[:, 1] < 0.5) if hflip else torch.zeros(b, dtype=torch.bool, device=device),
            "vflip": (u[:, 2] < 0.5) if vflip else torch.zeros(b, dtype=torch.bool, device=device)}


def apply_geometry(x: torch.Tensor, g: dict) -> torch.Tensor:
    """flip 후 중심 기준 회전(bilinear, 밖은 0)."""
    x = torch.where(g["hflip"].view(-1, 1, 1, 1), x.flip(-1), x)
    x = torch.where(g["vflip"].view(-1, 1, 1, 1), x.flip(-2), x)
    cos, sin = torch.cos(g["angle"]), torch.sin(g["angle"])
    zero = torch.zeros_like(cos)
    theta = torch.stack([torch.stack([cos, -sin, zero], 1), torch.stack([sin, cos, zero], 1)], 1)
    grid = F.affine_grid(theta.to(x.dtype), list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def build_batch(img_u8, nuc, cell, *, view: str, dna_idx: int, size: int, augment: dict | None,
                gen: torch.Generator | None = None) -> torch.Tensor:
    """학습·추론 공용. augment가 None이면 증강 없이 view만 만든다."""
    x, n, c = to_input(img_u8, nuc, cell, size)
    if augment:
        x = photometric(x, augment["brightness"], augment["contrast"], gen)
    x = make_view(x, n, c, view, dna_idx)
    if augment:
        g = sample_geometry(x.shape[0], augment["rotate_deg"][1], augment["hflip"], augment["vflip"], x.device, gen)
        x = apply_geometry(x, g)
    return x
