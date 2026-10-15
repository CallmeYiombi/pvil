"""진단용: replicate mAP의 negative 정의 4개를 비교한다 (progress.md 이슈 0, 2026-10-15).

고정 평가 정의(scripts/evaluate.py)는 바꾸지 않는다. evaluate.py가 저장한 wells.parquet·wells.npz를 읽어
정의마다 Harmony 전·후·무작위 임베딩의 mAP를 계산하고 results/.../map_defs.json에 저장한다.

  A: negative = 같은 plate 처리군–대조군 (현행, 원문 본문 표기)
  B: negative = plate 무관 처리군–대조군
  C: copairs 예제(reference index): positive = 같은 label(plate 조건 없음), negative = 처리군–대조군
  D: negative = 다른 plate의 label이 다른 well 전부

사용: source scripts/env.sh && python scripts/diagnose_map_defs.py configs/E0/dino_c_rxrx1.yaml
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
from pvil.eval import run_info  # noqa: E402

DEFS = {
    "A_same_plate_ctrl": dict(pos_sameby=["label"], pos_diffby=["plate"], neg_sameby=["plate"], neg_diffby=["is_control"]),
    "B_any_plate_ctrl": dict(pos_sameby=["label"], pos_diffby=["plate"], neg_sameby=[], neg_diffby=["is_control"]),
    "C_copairs_ref_index": dict(pos_sameby=["label", "ref"], pos_diffby=[], neg_sameby=[], neg_diffby=["label", "ref"]),
    "D_all_other_label": dict(pos_sameby=["label"], pos_diffby=["plate"], neg_sameby=[], neg_diffby=["label", "plate"]),
}


def macro_map(m: pd.DataFrame, x: np.ndarray, kw: dict) -> float:
    from copairs.map import average_precision

    ap = average_precision(m, np.asarray(x, dtype=np.float64), progress_bar=False, **kw)
    ap = ap[~ap.is_control]
    return float(ap.groupby("label").average_precision.mean().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--defs", nargs="+", default=list(DEFS))
    args = ap.parse_args()
    exp = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(OmegaConf.load(REPO / exp.base), exp)
    paths = OmegaConf.load(REPO / "configs/paths.yaml")
    rel = Path(cfg.experiment) / cfg.method / cfg.dataset / f"seed{cfg.seed}"
    emb_dir = Path(paths.embeddings_dir) / rel
    w = pd.read_parquet(emb_dir / "wells.parquet")
    z = np.load(emb_dir / "wells.npz")
    ctrl = (w.role == "control").to_numpy()
    m = pd.DataFrame({"label": w.label.to_numpy(), "plate": w.plate_key.to_numpy(), "is_control": ctrl})
    m["ref"] = np.where(ctrl, np.arange(len(m)), -1)
    rnd = np.random.default_rng(cfg.seed).normal(size=z["X_pca"].shape)

    neg_ctrl = w[~ctrl].plate_key.map(w[ctrl].groupby("plate_key").size()).fillna(0)
    out = {"config_path": args.config, "dataset": cfg.dataset, "method": cfg.method, "view": cfg.view,
           "n_wells": int(len(w)), "n_control_wells": int(ctrl.sum()), "n_plates": int(w.plate_key.nunique()),
           "same_plate_ctrl_per_treated_well": {"mean": float(neg_ctrl.mean()), "min": float(neg_ctrl.min()),
                                                "max": float(neg_ctrl.max())},
           "defs": {}, "run_info": run_info.collect(args.config, cfg.seed, protocol=cfg.protocol)}
    for name in args.defs:
        r = {k: macro_map(m, x, DEFS[name]) for k, x in
             (("post_harmony", z["X_pca_harmony"]), ("pre_harmony", z["X_pca"]), ("random", rnd))}
        out["defs"][name] = r
        print(name, {k: round(v, 3) for k, v in r.items()}, flush=True)
    dst = REPO / paths.results_dir / rel / "map_defs.json"
    dst.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    print("saved", dst)


if __name__ == "__main__":
    main()
