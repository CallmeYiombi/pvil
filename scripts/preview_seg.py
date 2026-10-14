"""Stage A 분할 결과 점검: source마다 대표 FOV 하나에 핵·세포 윤곽을 그리고 source별 통계를 출력한다.

사용: source scripts/env.sh && python scripts/preview_seg.py configs/preprocess/rxrx1.yaml [--per-source 1] [--quantile 0.5]
대표 FOV = 그 source에서 FOV당 세포 수가 지정 분위수에 가장 가까운 FOV.
출력: pvil_data/seg/<dataset>/_preview/by_source_q<quantile>.png (빨강 = 세포 채널, 초록 = 핵 채널,
      노랑 = 세포 윤곽, 청록 = 핵 윤곽)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from skimage.segmentation import find_boundaries

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from pvil.data.readers import iter_plate_fovs, load_wells  # noqa: E402


def norm(a: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(a, [1, 99.5])
    return np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)


def render(cfg, img: np.ndarray, nuc: np.ndarray, cell: np.ndarray, size: int) -> np.ndarray:
    names = list(cfg.channels.names)
    ni, ci = names.index(cfg.channels.nucleus), names.index(cfg.channels.cell)
    rgb = np.stack([norm(img[ci]), norm(img[ni]), np.zeros(cell.shape, np.float32)], -1)
    rgb[find_boundaries(cell)] = [1, 1, 0]
    rgb[find_boundaries(nuc)] = [0, 1, 1]
    im = Image.fromarray((rgb * 255).astype(np.uint8))
    return np.asarray(im.resize((size, size), Image.BILINEAR)) if im.size[0] != size else np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--per-source", type=int, default=1)
    ap.add_argument("--quantile", type=float, default=0.5)
    ap.add_argument("--tile", type=int, default=384)
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)
    seg_dir = Path(OmegaConf.load(REPO / "configs/paths.yaml").seg_dir) / cfg.dataset
    wells = load_wells(cfg)

    files = sorted(seg_dir.glob("*.cells.parquet"))
    cells = pd.concat([pd.read_parquet(f, columns=["fov_id", "plate_key", "source", "fov_n_cells", "border_ok"])
                       for f in files], ignore_index=True)
    fov = cells.groupby(["source", "plate_key", "fov_id"], as_index=False).agg(
        n=("fov_n_cells", "first"), border_ok=("border_ok", "mean"))
    stats = fov.groupby("source").n.describe(percentiles=[0.1, 0.5, 0.9])[["count", "10%", "50%", "90%"]]
    stats.columns = ["n_fov", "cells_p10", "cells_p50", "cells_p90"]
    stats["border_ok_frac"] = fov.groupby("source").border_ok.mean()
    print(f"[{cfg.dataset}] plates {len(files)}, FOV {len(fov):,}, 매칭 세포 {len(cells):,}")
    print(stats.round(2).to_string())

    tiles, labels = [], []
    for src, g in fov.groupby("source", sort=True):
        target = g.n.quantile(args.quantile)
        picks = g.iloc[(g.n - target).abs().argsort()[: args.per_source]]
        for r in picks.itertuples():
            z = np.load(seg_dir / f"{r.plate_key}.npz")
            i = int(np.where(z["fov_id"] == r.fov_id)[0][0])
            img = next(im for m, im in iter_plate_fovs(cfg, wells, r.plate_key) if m["fov_id"] == r.fov_id)
            tiles.append(render(cfg, img, z["nuc"][i], z["cell"][i], args.tile))
            labels.append(f"{src}  {r.fov_id}  n={r.n}")
    ncol = min(4, len(tiles))
    nrow = -(-len(tiles) // ncol)
    canvas = Image.new("RGB", (ncol * args.tile, nrow * args.tile), "white")
    draw = ImageDraw.Draw(canvas)
    for k, (t, lab) in enumerate(zip(tiles, labels)):
        x, y = (k % ncol) * args.tile, (k // ncol) * args.tile
        canvas.paste(Image.fromarray(t), (x, y))
        draw.rectangle([x, y, x + 8 * len(lab), y + 14], fill="black")
        draw.text((x + 2, y + 1), lab, fill="white")
    out = seg_dir / "_preview" / f"by_source_q{args.quantile:g}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    stats.to_csv(out.with_suffix(".csv"))
    print("saved:", out)


if __name__ == "__main__":
    main()
