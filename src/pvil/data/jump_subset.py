"""JUMP-CP(cpg0016) 근사 부분집합 선택 규칙. decisions.md 2026-10-14.

원문은 "화합물 1,466개, source 10개, wells 28,003(대조군 포함), held-out source_4"만 밝힌다.
규칙:
  1. source·plate 유형으로 거른다(source_2–11, COMPOUND·TARGET2).
  2. 화합물 = TARGET2 비대조군 화합물 ∪ min_sources개 이상 source에 나오는 화합물. 대조군은 제외.
  3. 처리군: (화합물, source)마다 최대 cap개 well을 무작위 선택.
  4. 음성대조군(DMSO): 처리군 well이 있는 plate에서 plate별로 고르게 뽑아 total_wells에 맞춘다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KEY = ["Metadata_Source", "Metadata_Plate", "Metadata_Well"]


def select_subset(wells: pd.DataFrame, plates: pd.DataFrame, compound_ids: set[str], control_ids: set[str], *,
                  negcon_id: str, sources: list[str], plate_types: list[str], target_plate_type: str,
                  min_sources: int, cap: int, total_wells: int, seed: int) -> pd.DataFrame:
    """wells: Metadata_Source/Plate/Well/JCP2022. plates: Metadata_Source/Plate/Batch/PlateType.

    반환: 선택된 well(+Metadata_Batch, Metadata_PlateType, role ∈ {treatment, negcon}, is_target2_compound).
    """
    w = wells.merge(plates[["Metadata_Source", "Metadata_Plate", "Metadata_Batch", "Metadata_PlateType"]],
                    on=["Metadata_Source", "Metadata_Plate"], how="inner")
    w = w[w.Metadata_Source.isin(sources) & w.Metadata_PlateType.isin(plate_types)]
    w = w.sort_values(KEY).reset_index(drop=True)

    treated = w[w.Metadata_JCP2022.isin(compound_ids) & ~w.Metadata_JCP2022.isin(control_ids)]
    target2 = set(treated.loc[treated.Metadata_PlateType == target_plate_type, "Metadata_JCP2022"])
    n_src = treated.groupby("Metadata_JCP2022").Metadata_Source.nunique()
    chosen = target2 | set(n_src[n_src >= min_sources].index)

    pool = treated[treated.Metadata_JCP2022.isin(chosen)]
    pool = pool.sample(frac=1.0, random_state=seed)
    treat = pool.groupby(["Metadata_JCP2022", "Metadata_Source"], sort=False).head(cap)

    n_neg = total_wells - len(treat)
    if n_neg < 0:
        raise ValueError(f"처리군 well {len(treat)}개가 total_wells {total_wells}보다 많다. cap을 줄여야 한다")
    neg_pool = w[(w.Metadata_JCP2022 == negcon_id) & w.Metadata_Plate.isin(treat.Metadata_Plate.unique())]
    if len(neg_pool) < n_neg:
        raise ValueError(f"DMSO 후보 {len(neg_pool)}개가 필요한 {n_neg}개보다 적다")
    # plate별로 섞은 뒤 plate 안 순번이 작은 것부터 뽑아 plate마다 고르게 분배한다
    rng = np.random.default_rng(seed)
    neg_pool = neg_pool.assign(_u=rng.random(len(neg_pool)))
    neg_pool = neg_pool.sort_values(["Metadata_Plate", "_u"])
    neg_pool["_rank"] = neg_pool.groupby("Metadata_Plate").cumcount()
    neg = neg_pool.sort_values(["_rank", "_u"]).head(n_neg).drop(columns=["_u", "_rank"])

    out = pd.concat([treat.assign(role="treatment"), neg.assign(role="negcon")], ignore_index=True)
    out["is_target2_compound"] = out.Metadata_JCP2022.isin(target2)
    return out.sort_values(KEY).reset_index(drop=True)


def summarize(sel: pd.DataFrame, held_out_source: str) -> dict:
    t = sel[sel.role == "treatment"]
    held = t[t.Metadata_Source == held_out_source]
    return {
        "n_wells": int(len(sel)),
        "n_treatment_wells": int(len(t)),
        "n_negcon_wells": int((sel.role == "negcon").sum()),
        "n_compounds": int(t.Metadata_JCP2022.nunique()),
        "n_target2_compounds": int(t.loc[t.is_target2_compound, "Metadata_JCP2022"].nunique()),
        "n_sources": int(sel.Metadata_Source.nunique()),
        "n_plates": int(sel.Metadata_Plate.nunique()),
        "held_out_source": held_out_source,
        "held_out_query_compounds": int(held.Metadata_JCP2022.nunique()),
        "held_out_wells": int((sel.Metadata_Source == held_out_source).sum()),
        "wells_per_source": {k: int(v) for k, v in sel.groupby("Metadata_Source").size().items()},
    }
