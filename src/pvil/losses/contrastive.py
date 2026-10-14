"""대칭 in-log multi-positive InfoNCE (사양 A.2).

ℓ_ij = z_i^img · z_j^pert / τ (둘 다 L2 정규화), P(i) = 배치에서 같은 label인 표본 전부.
L_i = −log Σ_{j∈P(i)} exp(ℓ_ij) / Σ_k exp(ℓ_ik). image→pert와 pert→image를 같은 가중치로 평균한다.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def multi_positive_infonce(z_img: torch.Tensor, z_pert: torch.Tensor, labels: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    z_img = F.normalize(z_img.float(), dim=-1)
    z_pert = F.normalize(z_pert.float(), dim=-1)
    logits = z_img @ z_pert.T / tau
    pos = labels[:, None] == labels[None, :]
    neg_inf = torch.finfo(logits.dtype).min

    def one_side(lg):
        return -(torch.logsumexp(lg.masked_fill(~pos, neg_inf), 1) - torch.logsumexp(lg, 1)).mean()

    return 0.5 * (one_side(logits) + one_side(logits.T))


class LearnableTemperature(nn.Module):
    """τ를 직접 학습한다(초기 0.1). 사용 시 [lo, hi]로 clamp."""

    def __init__(self, init: float, clamp: tuple[float, float]):
        super().__init__()
        self.tau = nn.Parameter(torch.tensor(float(init)))
        self.lo, self.hi = clamp

    def forward(self) -> torch.Tensor:
        return self.tau.clamp(self.lo, self.hi)
