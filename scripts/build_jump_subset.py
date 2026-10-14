"""JUMP-CP 근사 부분집합 well 목록을 만든다. 사용: python scripts/build_jump_subset.py configs/data/jump_cp.yaml

메타데이터 CSV를 pvil_data로 복사하고(재현성), 규칙(src/pvil/data/jump_subset.py)을 실행해
고정된 정의 configs/data/jump_cp_wells.csv.gz와 요약 JSON을 쓴다.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from pvil.data.jump_subset import select_subset, summarize  # noqa: E402

FILES = ["plate.csv.gz", "well.csv.gz", "compound.csv.gz", "perturbation_control.csv", "microscope_config.csv"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    cfg = OmegaConf.load(ap.parse_args().config)
    m, s = cfg.metadata, cfg.subset

    src = Path(m.repo_dir) / "metadata"
    head = subprocess.run(["git", "-C", m.repo_dir, "rev-parse", "--short=7", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    if head != str(m.commit):
        raise RuntimeError(f"메타데이터 commit {head} != config {m.commit}")
    dst = Path(m.copy_to)
    dst.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        shutil.copy2(src / f, dst / f)
    (dst / "COMMIT").write_text(f"{head}\n")

    plates = pd.read_csv(dst / "plate.csv.gz")
    wells = pd.read_csv(dst / "well.csv.gz")
    compounds = pd.read_csv(dst / "compound.csv.gz")
    controls = pd.read_csv(dst / "perturbation_control.csv")

    sel = select_subset(
        wells, plates, set(compounds.Metadata_JCP2022), set(controls.Metadata_JCP2022),
        negcon_id=s.negcon_id, sources=list(s.sources), plate_types=list(s.plate_types),
        target_plate_type=s.target_plate_type, min_sources=s.min_sources,
        cap=s.cap_wells_per_compound_source, total_wells=s.total_wells, seed=s.seed,
    )
    sel = sel.merge(compounds[["Metadata_JCP2022", "Metadata_InChIKey", "Metadata_SMILES"]],
                    on="Metadata_JCP2022", how="left")
    cols = ["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well", "Metadata_PlateType",
            "Metadata_JCP2022", "Metadata_InChIKey", "Metadata_SMILES", "role", "is_target2_compound"]
    out = REPO / s.wells_csv
    sel[cols].to_csv(out, index=False, compression={"method": "gzip", "mtime": 0})

    summary = {"metadata_commit": head, "rule": OmegaConf.to_container(s, resolve=True),
               **summarize(sel, s.held_out_source)}
    Path(s.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(s.summary_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "rule"}, indent=2, ensure_ascii=False))
    print("saved:", out)


if __name__ == "__main__":
    main()
