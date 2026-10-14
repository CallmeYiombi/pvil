"""QC와 세포 샘플링 (사양 A.1, decisions.md 2026-10-14).

QC: 핵 면적, 세포질 면적, 핵/세포질 비율 중 하나라도 풀 전체 [q_lo, q_hi] 분위수 밖이면 제거.
샘플링:
  처리군 — (perturbation label, source)마다 QC 통과 세포에서 균일하게 n개(부족하면 전부).
  대조군 — well마다, 처리군 well당 평균 샘플 수만큼(부족하면 전부).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

QC_COLS = ("nuc_area", "cyto_area", "nuc_cyto_ratio")


def qc_filter(cells: pd.DataFrame, q_lo: float, q_hi: float, cols=QC_COLS) -> tuple[pd.Series, dict]:
    """통과 여부(bool Series)와 사용한 분위수 경계를 돌려준다. 입력은 경계 제외를 마친 세포 풀."""
    keep = pd.Series(True, index=cells.index)
    bounds = {}
    for c in cols:
        lo, hi = cells[c].quantile([q_lo, q_hi])
        bounds[c] = (float(lo), float(hi))
        keep &= cells[c].between(lo, hi) & cells[c].notna()
    return keep, bounds


def sample_cells(cells: pd.DataFrame, n_per_pert_source: int, seed: int) -> tuple[pd.DataFrame, dict]:
    """cells: role, label, source, well_key 열을 가진 QC 통과 세포. role == exclude는 미리 빼 둔다."""
    rng = np.random.default_rng(seed)
    cells = cells.assign(_u=rng.random(len(cells))).sort_values("_u")
    treat = cells[cells.role == "perturbation"].groupby(["label", "source"], sort=False).head(n_per_pert_source)
    n_treat_wells = cells.loc[cells.role == "perturbation", "well_key"].nunique()
    per_well = int(round(len(treat) / max(n_treat_wells, 1)))
    ctrl = cells[cells.role == "control"].groupby("well_key", sort=False).head(per_well)
    out = pd.concat([treat, ctrl]).drop(columns="_u")
    short = treat.groupby(["label", "source"]).size()
    info = {
        "n_treatment_cells": int(len(treat)), "n_control_cells": int(len(ctrl)),
        "control_cells_per_well": per_well, "n_treatment_wells": int(n_treat_wells),
        "n_control_wells_covered": int(ctrl.well_key.nunique()),
        "n_pert_source_pairs": int(len(short)),
        "n_pert_source_pairs_short": int((short < n_per_pert_source).sum()),
    }
    return out.sort_index(), info
