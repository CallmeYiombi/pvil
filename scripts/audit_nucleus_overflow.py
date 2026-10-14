"""S view 결정용: 매칭된 핵 마스크가 자기 세포 마스크 밖으로 나오는 정도를 잰다.

사용: source scripts/env.sh && python scripts/audit_nucleus_overflow.py configs/preprocess/rxrx3_core.yaml \
        [--max-plates 150] [--seed 0]
입력: Stage A 결과(pvil_data/seg/<dataset>/*.npz, *.cells.parquet). 끝난 plate만 쓴다.
대상 세포: held-out 배치 제외, 경계 조건 통과(border_ok) + 역할이 exclude가 아닌 well. QC 전.
출력: results/E0/data_audit/nucleus_overflow_<dataset>.json, 세포별 표 nucleus_overflow_<dataset>.parquet

세포별 지표 (핵 픽셀 = 자기 핵 마스크):
  out_frac      = 자기 세포 마스크 밖 핵 픽셀 / 핵 면적
  out_neighbor  = 그중 이웃 세포 마스크(다른 라벨) 위에 놓인 픽셀 / 밖 핵 픽셀
  out_bg        = 그중 배경(세포 마스크 0) 위에 놓인 픽셀 / 밖 핵 픽셀
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from pvil.data.readers import load_wells  # noqa: E402


def fov_overflow(nuc: np.ndarray, cell: np.ndarray) -> pd.DataFrame:
    """매칭된 마스크(같은 라벨 체계) → 라벨별 핵 면적, 밖 픽셀, 이웃 위 픽셀."""
    k = int(max(nuc.max(), cell.max()))
    if k == 0:
        return pd.DataFrame(columns=["label", "nuc_px", "out_px", "out_neighbor_px"])
    n = nuc.ravel().astype(np.int64)
    c = cell.ravel().astype(np.int64)
    isn = n > 0
    out = isn & (c != n)
    nb = out & (c > 0)
    return pd.DataFrame({
        "label": np.arange(1, k + 1),
        "nuc_px": np.bincount(n[isn], minlength=k + 1)[1:],
        "out_px": np.bincount(n[out], minlength=k + 1)[1:],
        "out_neighbor_px": np.bincount(n[nb], minlength=k + 1)[1:],
    })


def quantiles(s: pd.Series, qs=(0.5, 0.75, 0.9, 0.95, 0.99)) -> dict:
    return {f"q{int(q * 100)}": float(s.quantile(q)) for q in qs} | {"mean": float(s.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--max-plates", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)
    seg_dir = Path(OmegaConf.load(REPO / "configs/paths.yaml").seg_dir) / cfg.dataset

    wells = load_wells(cfg)
    wells = wells[~wells.is_heldout_batch]       # 설계 판단용이므로 테스트 배치는 읽지 않는다 (절대 규칙 1)
    plates =sorted(p for p in wells.plate_key.unique() if (seg_dir / f"{p}.cells.parquet").exists())
    rng = np.random.default_rng(args.seed)
    if len(plates) > args.max_plates:
        plates = sorted(rng.choice(plates, args.max_plates, replace=False))
    keep_wells = set(wells.loc[wells.role != "exclude", "well_key"])

    rows = []
    for i, p in enumerate(plates, 1):
        cells = pd.read_parquet(seg_dir / f"{p}.cells.parquet")
        cells = cells[cells.border_ok & cells.well_key.isin(keep_wells)]
        z = np.load(seg_dir / f"{p}.npz")
        nucs, cms = z["nuc"], z["cell"]          # npz는 접근할 때마다 압축을 푼다. plate당 한 번만 읽는다
        for j, fov in enumerate(z["fov_id"]):
            sel = cells[cells.fov_id == fov]
            if sel.empty:
                continue
            o = fov_overflow(nucs[j], cms[j])
            rows.append(sel[["fov_id", "well_key", "source", "label", "fov_n_cells"]].merge(o, on="label"))
        if i % 25 == 0:
            print(f"[{i}/{len(plates)}] {p}", flush=True)
    t = pd.concat(rows, ignore_index=True)
    t["out_frac"] = t.out_px / t.nuc_px
    has = t.out_px > 0
    t["out_neighbor"] = np.where(has, t.out_neighbor_px / t.out_px.clip(lower=1), np.nan)
    t["out_bg"] = 1 - t.out_neighbor

    o = t[has]
    summary = {
        "dataset": cfg.dataset, "n_plates": len(plates), "n_cells": int(len(t)),
        "frac_cells_with_overflow": float(has.mean()),
        "frac_cells_out_frac_gt": {str(x): float((t.out_frac > x).mean()) for x in (0.01, 0.05, 0.1, 0.25, 0.5)},
        "out_frac_among_overflow": quantiles(o.out_frac),
        "out_neighbor_among_overflow": quantiles(o.out_neighbor),
        "pixel_level": {
            "overflow_px_over_all_nuc_px": float(t.out_px.sum() / t.nuc_px.sum()),
            "neighbor_px_over_overflow_px": float(t.out_neighbor_px.sum() / max(t.out_px.sum(), 1)),
        },
        "frac_cells_with_overflow_by_density_quartile": {
            str(k): float(v) for k, v in has.groupby(pd.qcut(t.fov_n_cells, 4, duplicates="drop"), observed=True).mean().items()},
        "frac_cells_with_overflow_by_source": {str(k): float(v) for k, v in has.groupby(t.source).mean().items()},
    }
    out = REPO / "results/E0/data_audit"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"nucleus_overflow_{cfg.dataset}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    t.to_parquet(out / f"nucleus_overflow_{cfg.dataset}.parquet", index=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
