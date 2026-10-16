"""G0 진단 1단계 평가 (docs/tasks/P0_g0_diagnosis.md D1–D4). 고정 결과(results/E0)는 건드리지 않는다.

사용: source scripts/env.sh && python scripts/diag_eval.py configs/E0/dino_c_rxrx1.yaml --variant head
      python scripts/diag_eval.py --summarize   → results/E0_diag/summary.md
variant:
  head, cls, frozen_cls : 임베딩. 고정 후처리(post-Harmony) recall + D4 단계별 recall
                          head = E0 임베딩(<embeddings_dir>/E0/...), 나머지 = diag_embed.py 출력
  g1, g2, g3            : 비이미지 하한선(D3). 데이터셋당 1회, 유클리드 거리, 후처리 없음
출력: results/E0_diag/<variant>/<method|nonimage>/<dataset>/seed<k>/metrics.json
recall만 계산한다(replicate mAP 정의 문제는 diagnose_map_defs.py). held-out 배치 평가 → decisions.md 기록.
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
sys.path.insert(0, str(REPO / "scripts"))
from pvil.baselines import nonimage  # noqa: E402
from pvil.data.dataset import CONTROL_LABEL  # noqa: E402
from pvil.eval import run_info  # noqa: E402
from pvil.eval.metrics import cross_batch_recall  # noqa: E402
from pvil.eval.postprocess import postprocess, well_mean  # noqa: E402
from summarize_e0 import REF  # noqa: E402

OUT = REPO / "results/E0_diag"
EMB_VARIANTS = ("head", "cls", "frozen_cls")
G_VARIANTS = ("g1", "g2", "g3")
STAGES = ("a_raw_well", "b_sphering_only", "c_pca_only", "d_pre_harmony", "e_post_harmony")


def stage_features(x_well: np.ndarray, meta: pd.DataFrame, cfg) -> tuple[dict, dict]:
    """D4: (a) raw well 평균 (b) sphering만 (c) PCA만 (d)(e) 고정 후처리 postprocess()의 Harmony 전후."""
    from pycytominer.operations import Spherize
    from sklearn.decomposition import PCA

    cols = [f"f{i}" for i in range(x_well.shape[1])]
    ctrl = (meta.role == "control").to_numpy()
    sph = Spherize(method="ZCA-cor", return_numpy=True).fit(pd.DataFrame(x_well[ctrl], columns=cols))
    xs = np.asarray(sph.transform(pd.DataFrame(x_well, columns=cols)), dtype=np.float64)
    xp = PCA(cfg.postprocess.pca_dim, random_state=0).fit_transform(x_well)
    pp = postprocess(x_well, meta, batch_col="source", pca_dim=cfg.postprocess.pca_dim,
                     harmony=OmegaConf.to_container(cfg.postprocess.harmony))
    feats = {"a_raw_well": x_well, "b_sphering_only": xs, "c_pca_only": xp,
             "d_pre_harmony": pp["X_pca"], "e_post_harmony": pp["X_pca_harmony"]}
    return feats, pp["info"]


def evaluate(args, cfg, paths):
    cells = pd.read_parquet(Path(paths.crop_cache_dir) / cfg.dataset / "cells.parquet")
    cells["label"] = cells.label.where(cells.role != "control", CONTROL_LABEL)
    v = args.variant
    is_g = v in G_VARIANTS
    method = "nonimage" if is_g else cfg.method
    out = OUT / v / method / cfg.dataset / f"seed{cfg.seed}"
    out.mkdir(parents=True, exist_ok=True)
    result = {"protocol": cfg.protocol, "fold": None, "seed": cfg.seed, "config_path": args.config,
              "dataset": cfg.dataset, "method": method, "view": None if is_g else cfg.view, "variant": v}

    if is_g:
        _, meta = well_mean(np.zeros((len(cells), 1)), cells)
        wells = pd.Index(meta.well_key)
        if v == "g1":
            x = nonimage.g1_cell_count(cells, wells)
        elif v == "g2":
            x = nonimage.g2_position(cells, wells)
        else:
            emb_dir = Path(paths.embeddings_dir) / "E0_diag/intensity" / cfg.method / cfg.dataset / f"seed{cfg.seed}"
            emb = np.load(emb_dir / "cells.npy")
            assert len(emb) == len(cells)
            x = nonimage.well_plain_mean(emb, cells, wells)
            result["note"] = "G3 = crop(150×150) 채널별 평균·표준편차의 well 평균. FOV 전체 통계가 아닌 근사"
        z = nonimage.standardize(x, ~meta.is_heldout_batch.to_numpy(), seed=cfg.seed)
        rec, per = cross_batch_recall(z, meta, cfg.eval.recall_k, allow_test=True, metric="euclidean")
        result |= {"metric": "euclidean", "feature_dim": int(x.shape[1]), "recall": {"raw": rec}}
        print(v, cfg.dataset, {k: round(val, 4) for k, val in rec.items() if k.startswith("R@")}, flush=True)
        per.assign(stage="raw").to_csv(out / "per_perturbation.csv", index=False)
    else:
        sub = Path(cfg.experiment) if v == "head" else Path("E0_diag") / v
        emb_dir = Path(paths.embeddings_dir) / sub / cfg.method / cfg.dataset / f"seed{cfg.seed}"
        emb = np.load(emb_dir / "cells.npy")
        assert len(emb) == len(cells)
        x_well, meta = well_mean(emb, cells)
        feats, pinfo = stage_features(x_well, meta, cfg)
        recall, per = {}, []
        for s in STAGES:
            rec, rp = cross_batch_recall(feats[s], meta, cfg.eval.recall_k, allow_test=True)
            recall[s] = rec
            per.append(rp.assign(stage=s))
            print(v, cfg.dataset, cfg.view, s, "R@1", round(rec["R@1"], 4), "R@10", round(rec["R@10"], 4), flush=True)
        pd.concat(per).to_csv(out / "per_perturbation.csv", index=False)
        result |= {"metric": "cosine", "embedding_dim": int(emb.shape[1]), "recall": recall, "postprocess": pinfo}
        if v == "head":     # 재현성 확인: 고정 평가(results/E0)와 같아야 한다
            e0 = json.loads((REPO / paths.results_dir / cfg.experiment / cfg.method / cfg.dataset /
                             f"seed{cfg.seed}" / "metrics.json").read_text())
            ref = e0["metrics"]["post_harmony"]["recall"]["R@10"]
            result["e0_post_harmony_R@10"] = ref
            print("E0 대조: R@10", ref, "vs", round(recall["e_post_harmony"]["R@10"], 4), flush=True)

    info = run_info.collect(args.config, cfg.seed, protocol=cfg.protocol, fold=None, dataset=cfg.dataset,
                            method=method, variant=v)
    result |= {"git_hash": info["git_hash"], "run_info": info}
    (out / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    print("saved", out / "metrics.json")


def _load(variant: str, method: str, ds: str, seed: int = 0) -> dict | None:
    p = OUT / variant / method / ds / f"seed{seed}" / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def _r10(m: dict | None, stage: str) -> float | None:
    return None if m is None else m["recall"][stage]["R@10"]


def _f(x) -> str:
    return "—" if x is None else f"{x:.3f}"


def summarize():
    lines = ["# E0 G0 진단 1단계 요약 (protocol: P-orig, seed 0)", "",
             "`scripts/diag_eval.py --summarize`가 생성한다. 손으로 고치지 않는다. 계획: `docs/tasks/P0_g0_diagnosis.md`.", "",
             "- 지표 = held-out 배치 cross-batch R@10. 임베딩 variant는 고정 후처리(post-Harmony), G는 유클리드·후처리 없음.",
             "- head = E0 고정 평가 임베딩(projection 128), cls = 같은 best.pt의 백본 CLS(768), frozen_cls = 학습 전 DINOv3 CLS.",
             "", "## D1·D2: 평가 층과 학습 여부 (post-Harmony R@10)", "",
             "| 데이터셋 | view | 원문 | head | cls | frozen_cls |", "| --- | --- | --- | --- | --- | --- |"]
    ratio = {}
    for ds in ("rxrx1", "rxrx3_core"):
        for view, meth in (("C", "dino_c"), ("S", "dino_s")):
            vals = {v: _r10(_load(v, meth, ds), "e_post_harmony") for v in EMB_VARIANTS}
            ref = REF[ds]["R@10"][view]
            ratio.setdefault(ds, {})[view] = {"원문": ref, **vals}
            lines.append(f"| {ds} | {view} | {ref:.3f} | " + " | ".join(_f(vals[v]) for v in EMB_VARIANTS) + " |")
    lines += ["", "## S/C 비율 (R@10)", "", "| 데이터셋 | 원문 | head | cls | frozen_cls |", "| --- | --- | --- | --- | --- |"]
    for ds, r in ratio.items():
        cells = []
        for k in ("원문", *EMB_VARIANTS):
            c, s = r["C"][k], r["S"][k]
            cells.append(_f(s / c) if c and s is not None else "—")
        lines.append(f"| {ds} | " + " | ".join(cells) + " |")
    lines += ["", "## D3: 비이미지 하한선 R@10", "", "| 데이터셋 | G1 세포 수 | G2 위치 | G3 강도(crop 근사) |",
              "| --- | --- | --- | --- |"]
    for ds in ("rxrx1", "rxrx3_core"):
        lines.append(f"| {ds} | " + " | ".join(_f(_r10(_load(g, "nonimage", ds), "raw")) for g in G_VARIANTS) + " |")
    lines += ["", "## D4: 후처리 단계별 R@10", "",
              "| variant | 데이터셋 | view | (a) raw well | (b) sphering만 | (c) PCA만 | (d) pre-Harmony | (e) post-Harmony |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for v in EMB_VARIANTS:
        for ds in ("rxrx1", "rxrx3_core"):
            for view, meth in (("C", "dino_c"), ("S", "dino_s")):
                m = _load(v, meth, ds)
                if m is not None:
                    lines.append(f"| {v} | {ds} | {view} | " + " | ".join(_f(_r10(m, s)) for s in STAGES) + " |")
    lines += ["", "## 재현성 확인 (head (e) vs results/E0)", "", "| 데이터셋 | view | E0 | 재계산 |", "| --- | --- | --- | --- |"]
    for ds in ("rxrx1", "rxrx3_core"):
        for view, meth in (("C", "dino_c"), ("S", "dino_s")):
            m = _load("head", meth, ds)
            if m is not None:
                lines.append(f"| {ds} | {view} | {m['e0_post_harmony_R@10']:.4f} | {_r10(m, 'e_post_harmony'):.4f} |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("saved", OUT / "summary.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?")
    ap.add_argument("--variant", choices=EMB_VARIANTS + G_VARIANTS)
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    if args.summarize:
        return summarize()
    if not (args.config and args.variant):
        ap.error("config와 --variant가 필요하다")
    exp = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(OmegaConf.load(REPO / exp.base), exp)
    if cfg.protocol != "P-orig":
        raise ValueError("G0 진단은 E0 P-orig run에만 쓴다")
    evaluate(args, cfg, OmegaConf.load(REPO / "configs/paths.yaml"))


if __name__ == "__main__":
    main()
