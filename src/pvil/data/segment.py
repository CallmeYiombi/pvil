"""Cellpose-SAM 핵·세포 분할, 핵-세포 매칭, 세포 표 (사양 A.1).

- 핵 = 핵 채널 단독, 세포 = [세포 채널, 핵 채널] (channel_axis=-1). 그 밖에는 Cellpose 기본 파라미터.
- 매칭: 겹치는 픽셀 수 기준으로 서로가 최대 겹침 상대인 핵-세포 쌍만 남긴다. 매칭되지 않은 객체는 제거.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def match_nuclei_cells(nuc: np.ndarray, cell: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """상호 최대 겹침 쌍에 같은 라벨 1..K를 붙인 (nuc, cell) 마스크를 돌려준다. 나머지는 0."""
    both = (nuc > 0) & (cell > 0)
    out_n = np.zeros(nuc.shape, np.uint16)
    out_c = np.zeros(cell.shape, np.uint16)
    if not both.any():
        return out_n, out_c
    pairs, counts = np.unique(np.stack([nuc[both], cell[both]]).astype(np.int64), axis=1, return_counts=True)
    n_ids, c_ids = pairs
    order = np.lexsort((-counts, n_ids))                     # 핵별로 겹침 큰 순
    best_c = dict(zip(n_ids[order][::-1], c_ids[order][::-1]))   # 마지막(=최대)이 남도록 역순으로 덮어쓰기
    order = np.lexsort((-counts, c_ids))
    best_n = dict(zip(c_ids[order][::-1], n_ids[order][::-1]))
    matched = sorted((n, c) for n, c in best_c.items() if best_n.get(c) == n)
    lut_n = np.zeros(int(nuc.max()) + 1, np.uint16)
    lut_c = np.zeros(int(cell.max()) + 1, np.uint16)
    for k, (n, c) in enumerate(matched, 1):
        lut_n[n] = k
        lut_c[c] = k
    return lut_n[nuc], lut_c[cell]


def cell_table(nuc: np.ndarray, cell: np.ndarray, crop_size: int) -> pd.DataFrame:
    """매칭된 마스크(같은 라벨)에서 세포별 centroid·면적·경계 플래그를 계산한다."""
    k = int(cell.max())
    if k == 0:
        return pd.DataFrame(columns=["label", "centroid_y", "centroid_x", "cell_area", "nuc_area", "cyto_area",
                                     "nuc_cyto_ratio", "border_ok"])
    h, w = cell.shape
    yy, xx = np.indices(cell.shape)
    flat = cell.ravel()
    cell_area = np.bincount(flat, minlength=k + 1)[1:]
    cy = np.bincount(flat, weights=yy.ravel(), minlength=k + 1)[1:] / cell_area
    cx = np.bincount(flat, weights=xx.ravel(), minlength=k + 1)[1:] / cell_area
    nuc_area = np.bincount(nuc.ravel(), minlength=k + 1)[1:]
    inside = nuc == cell                                     # 같은 라벨 = 자기 핵이 세포 안에 있는 픽셀
    nuc_in_cell = np.bincount(cell[inside & (cell > 0)], minlength=k + 1)[1:]
    cyto = cell_area - nuc_in_cell
    half = crop_size // 2
    y0 = np.rint(cy).astype(int) - half
    x0 = np.rint(cx).astype(int) - half
    border_ok = (y0 >= 0) & (x0 >= 0) & (y0 + crop_size <= h) & (x0 + crop_size <= w)
    return pd.DataFrame({
        "label": np.arange(1, k + 1, dtype=np.int32), "centroid_y": cy, "centroid_x": cx,
        "cell_area": cell_area, "nuc_area": nuc_area, "cyto_area": cyto,
        "nuc_cyto_ratio": np.where(cyto > 0, nuc_area / np.maximum(cyto, 1), np.nan), "border_ok": border_ok,
    })


class Segmenter:
    """Cellpose-SAM 두 단계 추론. 모델은 프로세스당 한 번 올린다."""

    def __init__(self, pretrained_model: str, use_bfloat16: bool, batch_size: int = 64):
        from cellpose import models

        self.model = models.CellposeModel(gpu=True, pretrained_model=pretrained_model, use_bfloat16=use_bfloat16)
        self.batch_size = batch_size

    def __call__(self, images: list[np.ndarray], nuc_idx: int, cell_idx: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """images: [C,H,W] 목록 → 매칭된 (nuc, cell) 마스크 목록."""
        nuc_in = [im[nuc_idx] for im in images]
        cell_in = [np.stack([im[cell_idx], im[nuc_idx]], axis=-1) for im in images]
        nucs = self.model.eval(nuc_in, batch_size=self.batch_size)[0]
        cells = self.model.eval(cell_in, batch_size=self.batch_size, channel_axis=-1)[0]
        return [match_nuclei_cells(n, c) for n, c in zip(nucs, cells)]
