"""JUMP-CP 근사 부분집합의 선택 well 원본 이미지를 S3에서 받는다.

사용: source scripts/env.sh && python scripts/download_jump.py configs/data/jump_cp.yaml [--sources source_4] [--max-plates N] [--dry-run]

plate마다 load_data_with_illum.csv를 읽어 선택 well의 모든 site × 형광 채널 TIFF와 채널별 illum .npy를 받는다.
S3 키 구조(cpg0016-jump/ 이하)를 raw_image_dir 아래에 그대로 둔다. 완료된 파일은 건너뛰므로 중단 후 다시 실행하면 이어 받는다.
plate별 manifest(JSON)를 manifest_dir에 남긴다.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]


def s3_client(workers: int):
    cfg = Config(signature_version=UNSIGNED, max_pool_connections=workers, retries={"max_attempts": 10, "mode": "adaptive"})
    return boto3.client("s3", config=cfg, region_name="us-east-1")


def plate_keys(s3, d, plate_row, wells: set[str]) -> tuple[list[str], dict]:
    key = f"{d.prefix}/{plate_row.Metadata_Source}/workspace/load_data_csv/{plate_row.Metadata_Batch}/{plate_row.Metadata_Plate}/load_data_with_illum.csv"
    ld = pd.read_csv(io.BytesIO(s3.get_object(Bucket=d.bucket, Key=key)["Body"].read()))
    ld = ld[ld.Metadata_Well.isin(wells)]
    keys = []
    for ch in d.channels:
        keys += ld[f"URL_Orig{ch}"].tolist()
        if d.illum:
            keys += sorted(set(ld[f"URL_Illum{ch}"]))
    keys = [k.replace(f"s3://{d.bucket}/", "") for k in keys]
    info = {"n_wells_selected": len(wells), "n_wells_found": int(ld.Metadata_Well.nunique()),
            "missing_wells": sorted(wells - set(ld.Metadata_Well)), "sites": sorted(int(x) for x in ld.Metadata_Site.unique())}
    return keys, info


def fetch(s3, bucket: str, key: str, root: Path) -> int:
    dst = root / key.split("/", 1)[1]          # cpg0016-jump/ 접두어 제거
    if dst.exists():
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    tmp.write_bytes(body)
    os.replace(tmp, dst)
    return len(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--sources", nargs="*")
    ap.add_argument("--max-plates", type=int)
    ap.add_argument("--dry-run", action="store_true", help="파일 수만 세고 받지 않는다")
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)
    d, s = cfg.download, cfg.subset

    sel = pd.read_csv(REPO / s.wells_csv)
    if args.sources:
        sel = sel[sel.Metadata_Source.isin(args.sources)]
    plates = sel[["Metadata_Source", "Metadata_Batch", "Metadata_Plate"]].drop_duplicates()
    # held-out source를 먼저 받는다
    plates = plates.assign(_o=(plates.Metadata_Source != s.held_out_source)).sort_values(
        ["_o", "Metadata_Source", "Metadata_Plate"]).drop(columns="_o")
    if args.max_plates:
        plates = plates.head(args.max_plates)
    wells_by_plate = sel.groupby("Metadata_Plate").Metadata_Well.apply(set).to_dict()

    root, mdir = Path(d.raw_image_dir), Path(d.manifest_dir)
    root.mkdir(parents=True, exist_ok=True)
    mdir.mkdir(parents=True, exist_ok=True)
    s3 = s3_client(d.max_workers)
    t0, total_bytes, total_files = time.time(), 0, 0
    print(f"{len(plates)} plates, {len(sel)} wells -> {root}", flush=True)

    with ThreadPoolExecutor(d.max_workers) as ex:
        for i, row in enumerate(plates.itertuples(index=False), 1):
            mpath = mdir / f"{row.Metadata_Source}__{row.Metadata_Plate}.json"
            if not args.dry_run and mpath.exists() and json.loads(mpath.read_text()).get("complete"):
                continue
            keys, info = plate_keys(s3, d, row, wells_by_plate[row.Metadata_Plate])
            total_files += len(keys)
            if args.dry_run:
                print(f"[{i}/{len(plates)}] {row.Metadata_Source} {row.Metadata_Plate}: {len(keys)} files", flush=True)
                continue
            nbytes = sum(ex.map(lambda k: fetch(s3, d.bucket, k, root), keys))
            total_bytes += nbytes
            mpath.write_text(json.dumps({**row._asdict(), **info, "n_files": len(keys), "complete": True,
                                         "keys": keys}, indent=1))
            el = time.time() - t0
            print(f"[{i}/{len(plates)}] {row.Metadata_Source} {row.Metadata_Plate}: {len(keys)} files, "
                  f"누적 {total_bytes / 1e9:.1f} GB, 평균 {total_bytes / 1e6 / max(el, 1):.0f} MB/s"
                  + (f", 누락 well {len(info['missing_wells'])}" if info["missing_wells"] else ""), flush=True)
    print(f"DONE files={total_files} bytes={total_bytes}", flush=True)


if __name__ == "__main__":
    main()
