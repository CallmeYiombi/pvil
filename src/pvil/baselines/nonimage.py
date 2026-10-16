"""비이미지 하한선 G1–G3 (research_plan.md G 그룹). well 단위 특징을 만든다.

G1 = well당 세포 수, G2 = well 위치(행·열), G3 = 채널별 강도 통계(crop 기반 근사).
검색은 cross_batch_recall(metric="euclidean")로 한다. 저차원이라 후처리(sphering·PCA·Harmony)는 하지 않는다.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

_WELL = re.compile(r"^([A-Za-z]+)(\d+)$")


def parse_well(well: str) -> tuple[int, int]:
    """'B02' → (2, 2), 'AA26' → (27, 26). 행은 엑셀 열 이름처럼 A=1, Z=26, AA=27."""
    m = _WELL.match(str(well))
    if not m:
        raise ValueError(f"well 형식이 아니다: {well!r}")
    row = 0
    for ch in m.group(1).upper():
        row = row * 26 + (ord(ch) - ord("A") + 1)
    return row, int(m.group(2))


def g1_cell_count(cells: pd.DataFrame, wells: pd.Index) -> np.ndarray:
    """well에 속한 고유 FOV의 fov_n_cells 합 (분할된 세포 수, 샘플링 전). crop이 하나도 없는 FOV는 빠진다."""
    fov = cells.drop_duplicates("fov_id")
    return fov.groupby("well_key").fov_n_cells.sum().reindex(wells).to_numpy(np.float64)[:, None]


def g2_position(cells: pd.DataFrame, wells: pd.Index) -> np.ndarray:
    """행 one-hot과 열 one-hot을 이어 붙인다."""
    w = cells.drop_duplicates("well_key").set_index("well_key").well.reindex(wells)
    rc = np.array([parse_well(x) for x in w])
    rows = pd.get_dummies(rc[:, 0]).to_numpy(np.float64)
    cols = pd.get_dummies(rc[:, 1]).to_numpy(np.float64)
    return np.concatenate([rows, cols], 1)


def well_plain_mean(x: np.ndarray, cells: pd.DataFrame, wells: pd.Index) -> np.ndarray:
    """정규화 없이 well 평균 (G3: crop 강도 통계)."""
    codes = pd.Index(wells).get_indexer(cells.well_key)
    assert (codes >= 0).all()
    sums = np.zeros((len(wells), x.shape[1]))
    np.add.at(sums, codes, x)
    return sums / np.bincount(codes, minlength=len(wells))[:, None]


def standardize(x: np.ndarray, fit_mask: np.ndarray, *, jitter: float = 1e-6, seed: int = 0) -> np.ndarray:
    """fit_mask(학습 배치 well)로 z-score를 fit하고, 동률을 깨는 작은 잡음을 더한다."""
    mu = x[fit_mask].mean(0)
    sd = x[fit_mask].std(0)
    z = (x - mu) / np.where(sd > 0, sd, 1.0)
    return z + jitter * np.random.default_rng(seed).standard_normal(z.shape)
