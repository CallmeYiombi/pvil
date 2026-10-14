"""유전자 perturbation 임베딩(ESM2) 사전 계산 (사양 A.2, decisions.md 2026-10-14).

사용: source scripts/env.sh && CUDA_VISIBLE_DEVICES=1 python scripts/make_pert_embeddings.py configs/pert_emb/rxrx3_core.yaml
유전자 기호 → UniProt reviewed 정규 서열. 기호가 UniProt 기본 이름에 없으면 NCBI gene_info의 Synonyms로
현재 기호를 찾는다(예: C7orf26 → INTS15). 후보가 0개나 2개 이상이면 오류.
ESM2 토큰 평균은 BOS/EOS를 뺀 잔기 토큰으로 낸다. 1,022 aa를 넘으면 stride 간격의 겹치는 창으로 나눠
창별 평균을 창 길이로 가중 평균한다.
출력: <pert_emb_dir>/<name>.npz (labels, emb[P, 1280], uniprot_entry, resolved_symbol) + .json(출처·해시)
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from pvil.data.readers import load_wells  # noqa: E402


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_fasta(path: Path) -> dict[str, str]:
    seqs, acc, buf = {}, None, []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if acc:
                    seqs[acc] = "".join(buf)
                acc, buf = line[1:].split("|")[1], []
            else:
                buf.append(line.strip())
    if acc:
        seqs[acc] = "".join(buf)
    return seqs


def resolve(genes: list[str], geneid_tsv: Path, gene_info: Path) -> pd.DataFrame:
    u = pd.read_csv(geneid_tsv, sep="\t").rename(columns={"Gene Names (primary)": "symbol"})
    rev = u[u.Reviewed == "reviewed"]
    by_sym = rev.groupby("symbol").Entry.apply(list).to_dict()
    with gzip.open(gene_info, "rt") as f:
        ni = pd.read_csv(f, sep="\t", usecols=["Symbol", "Synonyms"])
    rows = []
    for g in genes:
        cand, via = by_sym.get(g, []), "uniprot_primary"
        if not cand:
            cur = ni.loc[ni.Synonyms.str.split("|").apply(lambda s: g in s), "Symbol"].unique().tolist()
            cand = [e for s in cur for e in by_sym.get(s, [])]
            via = f"ncbi_synonym:{'|'.join(cur)}"
        if len(cand) != 1:
            raise RuntimeError(f"{g}: reviewed UniProt 후보 {len(cand)}개 ({via}, {cand})")
        rows.append({"gene": g, "entry": cand[0], "via": via})
    return pd.DataFrame(rows)


@torch.no_grad()
def embed(seq: str, tok, model, window: int, stride: int, device) -> np.ndarray:
    starts = [0] if len(seq) <= window else list(range(0, len(seq) - window, stride)) + [len(seq) - window]
    acc, tot = 0.0, 0
    for s in starts:
        sub = seq[s:s + window]
        x = tok(sub, return_tensors="pt").to(device)
        h = model(**x).last_hidden_state[0, 1:-1].float()      # BOS/EOS 제외
        assert h.shape[0] == len(sub)
        acc = acc + h.mean(0) * len(sub)
        tot += len(sub)
    return (acc / tot).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)
    paths = OmegaConf.load(REPO / "configs/paths.yaml")
    wells = load_wells(OmegaConf.load(REPO / cfg.preprocess_config))
    genes = sorted(wells.loc[wells.role == "perturbation", "label"].unique())

    src = {k: Path(v) for k, v in cfg.sources.items()}
    res = resolve(genes, src["uniprot_geneid_tsv"], src["ncbi_gene_info"])
    seqs = read_fasta(src["uniprot_fasta"])
    res["length"] = res.entry.map(lambda e: len(seqs[e]))

    from transformers import AutoTokenizer, EsmModel

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(cfg.model)
    model = EsmModel.from_pretrained(cfg.model).to(device).eval()
    emb = np.stack([embed(seqs[e], tok, model, cfg.window, cfg.stride, device) for e in res.entry])

    out = Path(paths.pert_emb_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{cfg.name}.npz", labels=np.array(genes), emb=emb.astype(np.float32),
             uniprot_entry=res.entry.to_numpy(), via=res.via.to_numpy())
    info = {"name": cfg.name, "model": cfg.model, "window": cfg.window, "stride": cfg.stride,
            "n_genes": len(genes), "dim": int(emb.shape[1]),
            "n_windowed": int((res.length > cfg.window).sum()),
            "resolved_via_synonym": res[res.via != "uniprot_primary"].to_dict(orient="records"),
            "sources_sha256": {k: sha256(v) for k, v in src.items()}}
    (out / f"{cfg.name}.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
