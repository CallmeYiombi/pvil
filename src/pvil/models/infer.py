"""추론 공용 함수: crop 임베딩, validation image→perturbation R@1."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from pvil.data.dataset import CropStore
from pvil.data.views import build_batch


@torch.no_grad()
def embed_cells(model, store: CropStore, idx: np.ndarray, *, view: str, dna_idx: int, size: int,
                batch: int = 1024, device="cuda") -> np.ndarray:
    """idx 순서대로 증강 없이 128차원 projection 임베딩을 낸다."""
    model.eval()
    out = []
    for a in range(0, len(idx), batch):
        img, nuc, cell, _ = store.load(idx[a:a + batch])
        x = build_batch(img.to(device), nuc.to(device), cell.to(device), view=view, dna_idx=dna_idx, size=size, augment=None)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(x).float().cpu())
    model.train()
    return torch.cat(out).numpy()


@torch.no_grad()
def img2pert_macro_r1(z_img: np.ndarray, labels: np.ndarray, pert_encoder, n_perts: int, device="cuda") -> float:
    """대조군을 뺀 세포마다 전체 perturbation 임베딩 중 최근접이 자기 label인지 → label별 평균 → macro."""
    keep = labels < n_perts
    zi = F.normalize(torch.from_numpy(z_img[keep]).to(device), dim=-1)
    pert_encoder.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        zp = pert_encoder()[:n_perts].float()
    pert_encoder.train()
    zp = F.normalize(zp, dim=-1)
    pred = torch.cat([(zi[a:a + 8192] @ zp.T).argmax(1) for a in range(0, len(zi), 8192)]).cpu().numpy()
    lab = labels[keep]
    hit = (pred == lab).astype(float)
    per_label = np.bincount(lab, hit, minlength=n_perts) / np.bincount(lab, minlength=n_perts).clip(1)
    return float(per_label[np.unique(lab)].mean())
