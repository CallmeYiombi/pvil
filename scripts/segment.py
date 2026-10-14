"""Stage A: FOV 분할 → plate별 마스크(.npz)와 세포 표(.cells.parquet).

사용: source scripts/env.sh && CUDA_VISIBLE_DEVICES=0 python scripts/segment.py configs/preprocess/rxrx3_core.yaml --shard 0/8
      [--plates P1 P2 ...] [--max-plates N]
plate 단위로 저장하며, 이미 끝난 plate(.cells.parquet 존재)는 건너뛴다. 다음 plate의 이미지는 미리 읽어 둔다.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from pvil.data.readers import iter_plate_fovs, load_wells  # noqa: E402
from pvil.data.segment import Segmenter, cell_table  # noqa: E402


def load_cfg(path: str):
    cfg = OmegaConf.load(path)
    base = OmegaConf.load(REPO / cfg.base)
    paths = OmegaConf.load(REPO / "configs/paths.yaml")
    return cfg, base, paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--shard", default="0/1", help="i/n: 정렬된 plate 목록에서 i번째부터 n개 간격")
    ap.add_argument("--plates", nargs="*")
    ap.add_argument("--max-plates", type=int)
    ap.add_argument("--batch-fovs", type=int, default=16)
    args = ap.parse_args()
    cfg, base, paths = load_cfg(args.config)
    seg = base.data.segmentation
    crop_size = base.data.crop_size
    out_dir = Path(paths.seg_dir) / cfg.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    wells = load_wells(cfg)
    plates = sorted(wells.plate_key.unique())
    if args.plates:
        plates = [p for p in plates if p in set(args.plates)]
    i, n = map(int, args.shard.split("/"))
    plates = plates[i::n]
    plates = [p for p in plates if not (out_dir / f"{p}.cells.parquet").exists()]
    if args.max_plates:
        plates = plates[: args.max_plates]
    names = list(cfg.channels.names)
    nuc_idx, cell_idx = names.index(cfg.channels.nucleus), names.index(cfg.channels.cell)
    print(f"[{cfg.dataset} shard {args.shard}] {len(plates)} plates, GPU {os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    if not plates:
        return

    segmenter = Segmenter(seg.pretrained_model, seg.use_bfloat16)
    pool = ThreadPoolExecutor(1)
    load = lambda p: list(iter_plate_fovs(cfg, wells, p))
    nxt = pool.submit(load, plates[0])
    t_all = time.time()
    for k, plate in enumerate(plates):
        t0 = time.time()
        fovs = nxt.result()
        if k + 1 < len(plates):
            nxt = pool.submit(load, plates[k + 1])
        t_load = time.time() - t0
        metas, nucs, cells, tables = [], [], [], []
        for b in range(0, len(fovs), args.batch_fovs):
            batch = fovs[b: b + args.batch_fovs]
            for (meta, _), (nm, cm) in zip(batch, segmenter([im for _, im in batch], nuc_idx, cell_idx)):
                t = cell_table(nm, cm, crop_size)
                t = t.assign(**meta, fov_n_cells=len(t))
                metas.append(meta["fov_id"]); nucs.append(nm); cells.append(cm); tables.append(t)
        tmp = out_dir / f"{plate}.tmp.npz"
        np.savez_compressed(tmp, fov_id=np.array(metas), nuc=np.stack(nucs), cell=np.stack(cells))
        os.replace(tmp, out_dir / f"{plate}.npz")
        tab = pd.concat(tables, ignore_index=True)
        tab.to_parquet(out_dir / f"{plate}.cells.tmp.parquet", index=False)
        os.replace(out_dir / f"{plate}.cells.tmp.parquet", out_dir / f"{plate}.cells.parquet")
        dt = time.time() - t0
        print(f"[{k + 1}/{len(plates)}] {plate}: {len(fovs)} FOV, {len(tab)} cells, {dt:.0f}s "
              f"(load {t_load:.0f}s, {1000 * dt / max(len(fovs), 1):.0f} ms/FOV), 경과 {(time.time() - t_all) / 60:.0f}분", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
