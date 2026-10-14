"""데이터셋별 well 표와 FOV 읽기. 모든 데이터셋을 같은 형식으로 맞춘다.

well 표 열: well_key, plate_key, source, well, label, role(perturbation|control|exclude), is_heldout_batch, + 데이터셋별 부가 열
FOV: (meta dict, image ndarray [C, H, W]) — 채널 순서는 config의 channels.names.
작업 단위는 plate_key다(분할 결과도 plate 단위로 저장).
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from omegaconf import DictConfig

REPO = Path(__file__).resolve().parents[3]


def _assign_roles(df: pd.DataFrame, roles: DictConfig) -> pd.Series:
    role = pd.Series("exclude", index=df.index)
    for name in ("perturbation", "control"):
        mask = pd.Series(True, index=df.index)
        for col, values in roles[name].items():
            mask &= df[col].astype(str).isin([str(v) for v in values])
        role[mask] = name
    return role


def _finish(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    w = cfg.wells
    df = df.assign(source=df[w.source_col].astype(str), label=df[w.label_col].astype(str),
                   role=_assign_roles(df, w.roles))
    df["is_heldout_batch"] = df.source == str(w.heldout_source)
    return df.sort_values(["plate_key", "well"]).reset_index(drop=True)


# ---------------------------------------------------------------- well 표

def load_wells(cfg: DictConfig) -> pd.DataFrame:
    return {"rxrx1": _wells_rxrx1, "rxrx3_core": _wells_rxrx3_core, "jump_cp": _wells_jump_cp}[cfg.dataset](cfg)


def _filter(df: pd.DataFrame, filt) -> pd.DataFrame:
    for col, values in (filt or {}).items():
        df = df[df[col].astype(str).isin([str(v) for v in values])]
    return df


def _wells_rxrx1(cfg):
    m = pd.read_csv(Path(cfg.raw_root) / "metadata/rxrx1/metadata.csv").drop_duplicates("well_id")
    m = _filter(m, cfg.wells.filter)
    m = m.assign(well_key=m.well_id, plate_key=m.experiment + "_P" + m.plate.astype(str))
    return _finish(m, cfg)


def _wells_rxrx3_core(cfg):
    m = pd.read_csv(Path(cfg.raw_root) / "metadata_rxrx3_core.csv", low_memory=False)
    m = _filter(m, cfg.wells.filter)
    m = m.assign(well_key=m.well_id, well=m.address, plate_key=m.experiment_name + "_P" + m.plate.astype(str))
    return _finish(m, cfg)


def _wells_jump_cp(cfg):
    m = pd.read_csv(REPO / cfg.wells_csv)
    m = m.assign(plate_key=m.Metadata_Source + "__" + m.Metadata_Plate, well=m.Metadata_Well)
    m["well_key"] = m.plate_key + "_" + m.well
    m["Metadata_InChIKey"] = m.Metadata_InChIKey.where(m.role != "negcon", "DMSO")
    return _finish(m, cfg)


# ---------------------------------------------------------------- FOV 읽기

def iter_plate_fovs(cfg: DictConfig, wells: pd.DataFrame, plate_key: str) -> Iterator[tuple[dict, np.ndarray]]:
    """plate 하나의 FOV를 (meta, image[C,H,W])로 낸다. wells는 load_wells 결과(이 plate의 행만 써도 된다)."""
    pw = wells[wells.plate_key == plate_key]
    fn = {"rxrx1": _fovs_rxrx1, "rxrx3_core": _fovs_rxrx3_core, "jump_cp": _fovs_jump_cp}[cfg.dataset]
    yield from fn(cfg, pw)


def _meta(row, site: int) -> dict:
    return {"fov_id": f"{row.well_key}_s{site}", "well_key": row.well_key, "plate_key": row.plate_key,
            "source": row.source, "well": row.well, "site": int(site)}


_INDEX_CACHE: dict[str, pd.DataFrame] = {}


def _index(cfg) -> pd.DataFrame:
    p = str(Path(cfg.raw_root) / "image_index.parquet")
    if p not in _INDEX_CACHE:
        _INDEX_CACHE[p] = pd.read_parquet(p)
    return _INDEX_CACHE[p]


def _fovs_rxrx1(cfg, pw):
    from PIL import Image

    idx = _index(cfg)
    idx = idx[idx.well_id.isin(set(pw.well_key))]
    chan = {int(n[1:]): i for i, n in enumerate(cfg.channels.names)}       # w1 -> 0
    rows = {r.well_key: r for r in pw.itertuples(index=False)}
    for (well_id, site), g in idx.groupby(["well_id", "site"], sort=True):
        img = [None] * len(chan)
        for r in g.itertuples():
            img[chan[r.channel]] = np.asarray(Image.open(Path(cfg.raw_root) / r.path))
        yield _meta(rows[well_id], site), np.stack(img)


def _fovs_rxrx3_core(cfg, pw):
    import imagecodecs
    import pyarrow.parquet as pq

    idx = _index(cfg)
    idx = idx[idx.well_id.isin(set(pw.well_key))]
    chan = {int(n[2:]): i for i, n in enumerate(cfg.channels.names)}       # ch1 -> 0
    blobs = {}
    for (shard, rg), g in idx.groupby(["shard", "row_group"]):
        col = pq.ParquetFile(Path(cfg.raw_root) / "data" / shard).read_row_group(rg, columns=["jp2"]).column(0)
        for r in g.itertuples():
            blobs[r.key] = col[r.row_in_group].as_py()["bytes"]
    rows = {r.well_key: r for r in pw.itertuples(index=False)}
    for (well_id, site), g in idx.groupby(["well_id", "site"], sort=True):
        img = [None] * len(chan)
        for r in g.itertuples():
            img[chan[r.channel]] = imagecodecs.jpeg2k_decode(blobs[r.key])
        yield _meta(rows[well_id], site), np.stack(img)


def jump_load_data(cfg, source: str, batch: str, plate: str) -> pd.DataFrame:
    """plate의 load_data_with_illum.csv(형광 채널 URL·site)를 S3에서 읽어 캐시한다."""
    cache = Path(cfg.load_data_cache) / f"{source}__{plate}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED), region_name="us-east-1")
    key = f"{cfg.s3.prefix}/{source}/workspace/load_data_csv/{batch}/{plate}/load_data_with_illum.csv"
    ld = pd.read_csv(io.BytesIO(s3.get_object(Bucket=cfg.s3.bucket, Key=key)["Body"].read()))
    cols = ["Metadata_Well", "Metadata_Site"] + [f"URL_Orig{c}" for c in cfg.channels.names] + [f"URL_Illum{c}" for c in cfg.channels.names]
    ld = ld[cols]
    cache.parent.mkdir(parents=True, exist_ok=True)
    ld.to_parquet(cache, index=False)
    return ld


def _fovs_jump_cp(cfg, pw):
    import tifffile

    first = pw.iloc[0]
    ld = jump_load_data(cfg, first.Metadata_Source, first.Metadata_Batch, first.Metadata_Plate)
    rows = {r.well: r for r in pw.itertuples(index=False)}
    ld = ld[ld.Metadata_Well.isin(set(rows))].sort_values(["Metadata_Well", "Metadata_Site"])
    strip = f"s3://{cfg.s3.bucket}/{cfg.s3.prefix}/"
    for r in ld.itertuples(index=False):
        img = np.stack([tifffile.imread(Path(cfg.raw_root) / getattr(r, f"URL_Orig{c}").replace(strip, ""))
                        for c in cfg.channels.names])
        yield _meta(rows[r.Metadata_Well], r.Metadata_Site), img
