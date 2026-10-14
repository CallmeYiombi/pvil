"""원본 이미지 색인을 만들고 메타데이터와의 커버리지를 확인한다.

사용: source scripts/env.sh && python scripts/index_raw.py {rxrx3_core|rxrx1}
  rxrx3_core: parquet 키 → (shard, row_group, row) 색인. <raw>/rxrx3_core/image_index.parquet
  rxrx1:      PNG 경로 색인. <raw>/rxrx1/image_index.parquet
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
RAW = Path(OmegaConf.load(REPO / "configs/paths.yaml").raw_dir)


def index_rxrx3_core() -> None:
    root = RAW / "rxrx3_core"
    rows = []
    for f in sorted(glob.glob(str(root / "data/train-*.parquet"))):
        pf = pq.ParquetFile(f)
        for g in range(pf.metadata.num_row_groups):
            keys = pf.read_row_group(g, columns=["__key__"]).column(0).to_pylist()
            rows += [(k, Path(f).name, g, i) for i, k in enumerate(keys)]
    idx = pd.DataFrame(rows, columns=["key", "shard", "row_group", "row_in_group"])
    p = idx.key.str.extract(r"^(?P<experiment_name>[^/]+)/Plate(?P<plate>\d+)/(?P<address>[A-Z]+\d+)_s(?P<site>\d+)_(?P<channel>\d)$")
    assert not p.isna().any(axis=None), "키 파싱 실패"
    idx = pd.concat([idx, p], axis=1).astype({"plate": int, "site": int, "channel": int})
    idx["well_id"] = idx.experiment_name + "_" + idx.plate.astype(str) + "_" + idx.address
    meta = pd.read_csv(root / "metadata_rxrx3_core.csv", low_memory=False)
    report(idx, meta.well_id, n_channels=6)
    idx.to_parquet(root / "image_index.parquet", index=False)


def index_rxrx1() -> None:
    root = RAW / "rxrx1"
    files = []
    for dirpath, _, names in os.walk(root / "rxrx1/images"):
        files += [os.path.join(dirpath, n) for n in names if n.endswith(".png")]
    idx = pd.DataFrame({"path": sorted(os.path.relpath(f, root) for f in files)})
    p = idx.path.str.extract(r"images/(?P<experiment>[^/]+)/Plate(?P<plate>\d+)/(?P<well>[A-Z]\d+)_s(?P<site>\d)_w(?P<channel>\d)\.png$")
    assert not p.isna().any(axis=None), "경로 파싱 실패"
    idx = pd.concat([idx, p], axis=1).astype({"plate": int, "site": int, "channel": int})
    idx["well_id"] = idx.experiment + "_" + idx.plate.astype(str) + "_" + idx.well
    meta = pd.read_csv(root / "metadata/rxrx1/metadata.csv")
    report(idx, meta.well_id, n_channels=6)
    sites = idx.groupby(["well_id", "site"]).channel.nunique()
    print("site 6채널 완비:", int((sites == 6).sum()), "/", len(sites), "| 메타데이터 site:", len(meta))
    idx.to_parquet(root / "image_index.parquet", index=False)


def report(idx: pd.DataFrame, meta_wells: pd.Series, n_channels: int) -> None:
    per_well = idx.groupby("well_id").channel.nunique()
    meta_wells = set(meta_wells)
    print(f"이미지 {len(idx):,}장, well {len(per_well):,}개 ({n_channels}채널 완비 {(per_well == n_channels).sum():,})")
    print(f"메타데이터 well {len(meta_wells):,}개 중 이미지 없음 {len(meta_wells - set(per_well.index)):,}, "
          f"메타데이터에 없는 이미지 well {len(set(per_well.index) - meta_wells):,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["rxrx3_core", "rxrx1"])
    {"rxrx3_core": index_rxrx3_core, "rxrx1": index_rxrx1}[ap.parse_args().dataset]()
