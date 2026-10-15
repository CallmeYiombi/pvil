"""후처리 + 평가 (사양 §5, .claude/rules/evaluation.md). held-out 배치를 쓰는 최종 평가 스크립트다.

사용: source scripts/env.sh && python scripts/evaluate.py configs/E0/dino_c_rxrx1.yaml [--null-size 10000]
입력: <embeddings_dir>/<exp>/<method>/<dataset>/seed<k>/cells.npy (train.py가 저장), crop 표
출력: results/<exp>/<method>/<dataset>/seed<k>/metrics.json, per_perturbation.csv
      <embeddings_dir>/.../wells.npz (X_well, X_pca, X_pca_harmony, well 표는 wells.parquet)
이번 범위: cross-batch R@1–10, replicate mAP (Harmony 전후). CP prediction·scIB는 TODO(미측정으로 기록).
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
from pvil.data.dataset import CONTROL_LABEL  # noqa: E402
from pvil.eval import run_info  # noqa: E402
from pvil.eval.metrics import cross_batch_recall, replicate_map  # noqa: E402
from pvil.eval.postprocess import postprocess, well_mean  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--null-size", type=int, default=None, help="기본값 = config eval.map.null_size")
    ap.add_argument("--smoke", action="store_true", help="train.py --max-steps 결과(seed<k>_smoke)를 평가")
    args = ap.parse_args()
    exp = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(OmegaConf.load(REPO / exp.base), exp)
    paths = OmegaConf.load(REPO / "configs/paths.yaml")
    rel = Path(cfg.experiment) / cfg.method / cfg.dataset / (f"seed{cfg.seed}" + ("_smoke" if args.smoke else ""))
    out = REPO / paths.results_dir / rel
    emb_dir = Path(paths.embeddings_dir) / rel

    cells = pd.read_parquet(Path(paths.crop_cache_dir) / cfg.dataset / "cells.parquet")
    cells["label"] = cells.label.where(cells.role != "control", CONTROL_LABEL)
    emb = np.load(emb_dir / "cells.npy")
    assert len(emb) == len(cells)

    x_well, meta = well_mean(emb, cells)
    pp = postprocess(x_well, meta, batch_col="source", pca_dim=cfg.postprocess.pca_dim,
                     harmony=OmegaConf.to_container(cfg.postprocess.harmony))
    np.savez(emb_dir / "wells.npz", X_well=x_well, X_pca=pp["X_pca"], X_pca_harmony=pp["X_pca_harmony"])
    meta.to_parquet(emb_dir / "wells.parquet", index=False)

    null = args.null_size or cfg.eval.map.null_size
    metrics, per = {}, []
    for key, x in (("pre_harmony", pp["X_pca"]), ("post_harmony", pp["X_pca_harmony"])):
        rec, rec_pp = cross_batch_recall(x, meta, cfg.eval.recall_k, allow_test=True)
        mp, mp_pp = replicate_map(x, meta, null_size=null, alpha=cfg.eval.map.alpha, seed=cfg.seed, allow_test=True)
        metrics[key] = {"recall": rec, "replicate_map": mp}
        per.append(rec_pp.merge(mp_pp[["label", "mean_average_precision", "corrected_p_value"]], on="label", how="outer")
                   .assign(stage=key))
        print(key, {k: round(v, 4) for k, v in rec.items() if k.startswith("R@")}, "mAP", round(mp["mAP"], 4), flush=True)
    pd.concat(per).to_csv(out / "per_perturbation.csv", index=False)

    info = run_info.collect(args.config, cfg.seed, protocol=cfg.protocol, fold=None, dataset=cfg.dataset,
                            method=cfg.method, map_null_size=null)
    result = {"protocol": cfg.protocol, "fold": None, "seed": cfg.seed, "git_hash": info["git_hash"],
              "config_path": args.config, "dataset": cfg.dataset, "method": cfg.method, "view": cfg.view,
              "metrics": metrics, "not_measured": ["cp_prediction", "scib"], "postprocess": pp["info"],
              "n_wells_by_role": meta.role.value_counts().to_dict(), "run_info": info}
    (out / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    print("saved", out / "metrics.json")


if __name__ == "__main__":
    main()
