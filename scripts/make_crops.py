"""Stage B: 경계 제외 → QC → 샘플링 → 150×150 crop zarr.

사용: source scripts/env.sh && python scripts/make_crops.py configs/preprocess/rxrx3_core.yaml [--dry-run] [--workers 16]
입력: Stage A 결과(pvil_data/seg/<dataset>/*.cells.parquet, *.npz). 모든 plate가 끝나 있어야 한다.
출력: pvil_data/crops/<dataset>/{crops.zarr, cells.parquet, sampling_summary.json}
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from pvil.data.crops import CropWriter, extract_crop  # noqa: E402
from pvil.data.qc_sample import qc_filter, sample_cells  # noqa: E402
from pvil.data.readers import iter_plate_fovs, load_wells  # noqa: E402

_G: dict = {}


def _init(cfg_path: str, crop_size: int, seg_dir: str):
    _G["cfg"] = OmegaConf.load(cfg_path)
    _G["wells"] = load_wells(_G["cfg"])
    _G["crop_size"] = crop_size
    _G["seg_dir"] = Path(seg_dir)


def _plate_crops(args):
    plate, sel = args
    cfg, crop = _G["cfg"], _G["crop_size"]
    z = np.load(_G["seg_dir"] / f"{plate}.npz")
    pos = {f: i for i, f in enumerate(z["fov_id"])}
    need = sel.groupby("fov_id")
    imgs, nucs, cells, centers, order = [], [], [], [], []
    for meta, image in iter_plate_fovs(cfg, _G["wells"], plate):
        if meta["fov_id"] not in need.groups:
            continue
        nm, cm = z["nuc"][pos[meta["fov_id"]]], z["cell"][pos[meta["fov_id"]]]
        for r in need.get_group(meta["fov_id"]).itertuples():
            ci, cn, cc, center = extract_crop(image, nm, cm, r.centroid_y, r.centroid_x, r.cell_label, crop)
            imgs.append(ci); nucs.append(cn); cells.append(cc); centers.append(center); order.append(r.crop_index)
    o = np.argsort(order)
    return (np.asarray(order)[o], np.stack(imgs)[o], np.stack(nucs)[o], np.stack(cells)[o], np.asarray(centers)[o])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--dry-run", action="store_true", help="QC·샘플링 수치만 출력하고 crop은 만들지 않는다")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--allow-partial", action="store_true", help="시험용: Stage A가 끝난 plate만 사용")
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)
    base = OmegaConf.load(REPO / cfg.base)
    paths = OmegaConf.load(REPO / "configs/paths.yaml")
    seg_dir = Path(paths.seg_dir) / cfg.dataset
    out_dir = Path(paths.crops_dir) / cfg.dataset
    crop_size = base.data.crop_size

    wells = load_wells(cfg)
    plates = sorted(wells.plate_key.unique())
    missing = [p for p in plates if not (seg_dir / f"{p}.cells.parquet").exists()]
    if missing and not args.allow_partial:
        raise RuntimeError(f"Stage A 미완료 plate {len(missing)}개 (예: {missing[:3]})")
    if args.allow_partial:      # 시험용: 끝난 plate만 쓰고 결과는 _partial 폴더에 둔다
        plates = [p for p in plates if p not in set(missing)]
        wells = wells[wells.plate_key.isin(plates)]
        out_dir = out_dir.with_name(out_dir.name + "_partial")
    cells = pd.concat([pd.read_parquet(seg_dir / f"{p}.cells.parquet") for p in plates], ignore_index=True)
    cells = cells.rename(columns={"label": "cell_label"})
    extra = [c for c in wells.columns if c not in cells.columns or c in ("well_key",)]
    cells = cells.merge(wells[extra], on="well_key", how="left", validate="many_to_one")

    n_all = len(cells)
    pool = cells[cells.border_ok & (cells.role != "exclude")]
    q_lo, q_hi = base.data.qc_quantiles
    keep, bounds = qc_filter(pool, q_lo, q_hi)
    passed = pool[keep]
    sampled, info = sample_cells(passed, cfg.sampling.n_per_pert_source, cfg.sampling.seed)
    sampled = sampled.sort_values(["plate_key", "fov_id", "cell_label"]).reset_index(drop=True)
    sampled["crop_index"] = np.arange(len(sampled))

    tw = wells[wells.role != "exclude"]
    summary = {
        "dataset": cfg.dataset, "n_fovs": int(cells.fov_id.nunique()), "n_cells_matched": int(n_all),
        "n_cells_border_ok_non_excluded": int(len(pool)), "n_cells_qc_pass": int(len(passed)), "qc_bounds": bounds,
        "n_wells_non_excluded": int(len(tw)), "n_perturbations": int(tw[tw.role == "perturbation"].label.nunique()),
        "n_sources": int(tw.source.nunique()), **info, "n_crops": int(len(sampled)),
        "n_wells_with_crops": int(sampled.well_key.nunique()),
        "crops_heldout_batch": int(sampled.is_heldout_batch.sum()),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.dry_run:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sampling_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    first_plate = sampled.plate_key.iloc[0]
    _, img0 = next(iter_plate_fovs(cfg, wells, first_plate))
    writer = CropWriter(out_dir / "crops.zarr", len(sampled), img0.shape[0], crop_size, img0.dtype)
    centers = np.zeros(len(sampled), np.int32)
    jobs = [(p, g) for p, g in sampled.groupby("plate_key", sort=True)]
    t0, done = time.time(), 0
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(args.config, crop_size, str(seg_dir))) as ex:
        for k, (idx, im, nm, cm, ce) in enumerate(ex.map(_plate_crops, jobs), 1):
            assert np.all(np.diff(idx) == 1), "plate 안 crop_index가 연속이 아니다"
            writer.write(int(idx[0]), im, nm, cm)
            centers[idx] = ce
            done += len(idx)
            if k % 20 == 0 or k == len(jobs):
                print(f"[{k}/{len(jobs)}] {done:,}/{len(sampled):,} crops, {(time.time() - t0) / 60:.0f}분", flush=True)
    sampled["center_label"] = centers
    sampled.to_parquet(out_dir / "cells.parquet", index=False)
    print("DONE", out_dir, flush=True)


if __name__ == "__main__":
    main()
