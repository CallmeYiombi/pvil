"""RxRx3-core 역할 정의 적용 후 원문(735 perturbation / 104,166 wells)과의 차이를 분해한다.

사용: source scripts/env.sh && python scripts/audit_rxrx3_core_roles.py configs/preprocess/rxrx3_core.yaml \
        --uniprot-tsv <UP000005640_geneid.tsv>
입력: Stage A 결과(pvil_data/seg/rxrx3_core/*.cells.parquet), 메타데이터
출력: results/E0/data_audit/rxrx3_core_roles.json (+ 표준 출력 요약)

(a) 유전자 수: CRISPR wells의 gene 값 전체 vs 역할별, 유전자마다 UniProt 정규 서열(ESM2 입력) 유무
(b) well 수: well_type_label별 well 수, 세포 0개 well 수(매칭·경계·QC 단계별)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from pvil.data.qc_sample import qc_filter  # noqa: E402
from pvil.data.readers import load_wells  # noqa: E402

PAPER = {"n_perturbations": 735, "n_wells": 104_166}


def uniprot_status(genes: list[str], tsv: Path) -> pd.DataFrame:
    u = pd.read_csv(tsv, sep="\t")
    u = u.rename(columns={"Gene Names (primary)": "gene"})
    rev = u[u.Reviewed == "reviewed"].groupby("gene").agg(n_reviewed=("Entry", "size"), length=("Length", "max"))
    unrev = u[u.Reviewed != "reviewed"].groupby("gene").size().rename("n_unreviewed")
    out = pd.DataFrame(index=pd.Index(sorted(genes), name="gene")).join(rev).join(unrev)
    out = out.fillna({"n_reviewed": 0, "n_unreviewed": 0}).astype({"n_reviewed": int, "n_unreviewed": int})
    out["has_sequence"] = (out.n_reviewed > 0) | (out.n_unreviewed > 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--uniprot-tsv", required=True, help="UniProt 인간 참조 프로테옴 gene 표 (Entry, Reviewed, Gene Names (primary), Length)")
    ap.add_argument("--out", default="results/E0/data_audit/rxrx3_core_roles.json")
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)
    base = OmegaConf.load(REPO / cfg.base)
    seg_dir = Path(OmegaConf.load(REPO / "configs/paths.yaml").seg_dir) / cfg.dataset

    wells = load_wells(cfg)
    wells["wtl"] = wells.well_type_label.fillna("(빈 값: EMPTY_control)")
    cells = pd.concat([pd.read_parquet(p) for p in sorted(seg_dir.glob("*.cells.parquet"))], ignore_index=True)
    cells = cells.merge(wells[["well_key", "role"]], on="well_key", how="left", validate="many_to_one")
    assert cells.role.notna().all(), "메타데이터에 없는 well이 분할 결과에 있다"

    pool = cells[cells.border_ok & (cells.role != "exclude")]
    keep, bounds = qc_filter(pool, *base.data.qc_quantiles)
    per_well = pd.DataFrame({
        "n_matched": cells.groupby("well_key").size(),
        "n_border_ok": cells[cells.border_ok].groupby("well_key").size(),
        "n_qc_pass": pool[keep].groupby("well_key").size(),
    })
    w = wells.set_index("well_key").join(per_well).fillna({"n_matched": 0, "n_border_ok": 0, "n_qc_pass": 0})
    w.loc[w.role == "exclude", "n_qc_pass"] = float("nan")   # QC 풀에 들어가지 않는 well

    # (b) well_type_label별 분해
    by_type = w.groupby("wtl").agg(
        role=("role", "first"), n_wells=("role", "size"), n_genes=("gene", "nunique"),
        zero_matched=("n_matched", lambda s: int((s == 0).sum())),
        zero_border_ok=("n_border_ok", lambda s: int((s == 0).sum())),
        zero_qc_pass=("n_qc_pass", lambda s: int((s == 0).sum()) if s.notna().any() else None),
        median_cells=("n_matched", "median"),
    )
    used = w[w.role != "exclude"]
    used_nonzero = used[used.n_qc_pass > 0]

    # (a) 유전자 분해
    all_genes = sorted(set(w.gene.dropna()))
    real_genes = [g for g in all_genes if g != "EMPTY_control"]
    pert_genes = sorted(set(used[used.role == "perturbation"].gene))
    pert_genes_nonzero = sorted(set(used_nonzero[used_nonzero.role == "perturbation"].gene))
    gene_roles = w.dropna(subset=["gene"]).groupby("gene").wtl.agg(lambda s: sorted(set(s)))
    up = uniprot_status(real_genes, Path(args.uniprot_tsv))
    no_seq = up[~up.has_sequence].index.tolist()
    long_seq = up[up.length > 1022].index.tolist()          # ESM2 최대 1,024 토큰(BOS/EOS 포함)

    out = {
        "paper": PAPER,
        "roles": OmegaConf.to_container(cfg.wells.roles),
        "qc_bounds": bounds,
        "genes": {
            "n_gene_values_all_crispr": len(all_genes),
            "non_gene_values": [g for g in all_genes if g not in real_genes],
            "n_real_genes": len(real_genes),
            "n_perturbation_genes": len(pert_genes),
            "n_perturbation_genes_with_qc_cells": len(pert_genes_nonzero),
            "real_genes_not_in_perturbation": {g: gene_roles[g] for g in real_genes if g not in pert_genes},
            "control_genes_also_query": {g: gene_roles[g] for g in real_genes if len(gene_roles[g]) > 1},
            "uniprot_no_sequence": no_seq,
            "uniprot_unreviewed_only": up[(up.n_reviewed == 0) & up.has_sequence].index.tolist(),
            "uniprot_multiple_reviewed": up[up.n_reviewed > 1].index.tolist(),
            "uniprot_length_gt_1022": {g: int(up.length[g]) for g in long_seq},
        },
        "wells": {
            "n_crispr_wells": int(len(w)),
            "by_well_type_label": by_type.reset_index().to_dict(orient="records"),
            "n_used_wells": int(len(used)),
            "n_used_wells_qc_cells_gt0": int(len(used_nonzero)),
            "n_used_perturbation_wells": int((used.role == "perturbation").sum()),
            "n_used_control_wells": int((used.role == "control").sum()),
            "diff_used_vs_paper": int(len(used) - PAPER["n_wells"]),
            "diff_all_crispr_vs_paper": int(len(w) - PAPER["n_wells"]),
        },
    }
    p = REPO / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    print(by_type.to_string())
    print(json.dumps({k: out[k] for k in ("genes",)}, indent=2, ensure_ascii=False, default=str))
    print(json.dumps({k: v for k, v in out["wells"].items() if k != "by_well_type_label"}, indent=2))


if __name__ == "__main__":
    main()
