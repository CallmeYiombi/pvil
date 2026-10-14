"""crop zarr를 학습용 memmap으로 푼다 (overlay, 같은 데이터셋의 run들이 page cache를 공유).

사용: source scripts/env.sh && python scripts/cache_crops.py rxrx1 [--chunk 8192]
입력: pvil_data/crops/<dataset>/{crops.zarr, cells.parquet}
출력: <crop_cache_dir>/<dataset>/{images.u8, masks.u1, cells.parquet, meta.json}
  masks.u1 [N, 2, s, s]: 0 = 중심 세포 핵, 1 = 중심 세포 (center_label과 같은 라벨). 이웃 마스크는 zarr에만 있다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--chunk", type=int, default=8192)
    args = ap.parse_args()
    import zarr

    paths = OmegaConf.load(REPO / "configs/paths.yaml")
    src = Path(paths.crops_dir) / args.dataset
    dst = Path(paths.crop_cache_dir) / args.dataset
    dst.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(src / "crops.zarr"), mode="r")
    cells = pd.read_parquet(src / "cells.parquet")
    n, c, s, _ = root["images"].shape
    assert len(cells) == n and root["images"].dtype == np.uint8, (len(cells), n, root["images"].dtype)
    center = cells.center_label.to_numpy()
    assert (center > 0).all(), "중심 라벨이 0인 crop이 있다"

    img = np.memmap(dst / "images.u8", dtype=np.uint8, mode="w+", shape=(n, c, s, s))
    msk = np.memmap(dst / "masks.u1", dtype=np.uint8, mode="w+", shape=(n, 2, s, s))
    t0 = time.time()
    for a in range(0, n, args.chunk):
        b = min(a + args.chunk, n)
        img[a:b] = root["images"][a:b]
        lab = center[a:b, None, None]
        msk[a:b, 0] = root["nuc_mask"][a:b] == lab
        msk[a:b, 1] = root["cell_mask"][a:b] == lab
        print(f"{b:,}/{n:,} ({(time.time() - t0) / 60:.1f}분)", flush=True)
    img.flush()
    msk.flush()
    shutil.copy2(src / "cells.parquet", dst / "cells.parquet")
    summ = src / "sampling_summary.json"
    meta = {"dataset": args.dataset, "n": int(n), "n_channels": int(c), "crop_size": int(s), "source": str(src),
            "sampling_summary_sha256": hashlib.sha256(summ.read_bytes()).hexdigest(),
            "cells_parquet_sha256": hashlib.sha256((src / "cells.parquet").read_bytes()).hexdigest()}
    (dst / "meta.json").write_text(json.dumps(meta, indent=2))
    print("DONE", dst, flush=True)


if __name__ == "__main__":
    main()
