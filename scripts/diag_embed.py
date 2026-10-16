"""G0 진단 1단계 임베딩 (docs/tasks/P0_g0_diagnosis.md D1·D2·D3). 고정 결과(results/E0)는 건드리지 않는다.

사용: source scripts/env.sh && CUDA_VISIBLE_DEVICES=1 python scripts/diag_embed.py configs/E0/dino_c_rxrx1.yaml --variant cls
  cls        : E0 best.pt의 백본 CLS(768) (D1)
  frozen_cls : best.pt를 로드하지 않은 초기 모델의 CLS. LoRA B = 0이라 사전학습 DINOv3와 같다 (D2)
  intensity  : view 적용 전 원본 crop의 채널별 평균·표준편차 (G3 crop 기반 근사, view와 무관)
출력: <embeddings_dir>/E0_diag/<variant>/<method>/<dataset>/seed<k>/cells.npy, run_info.json
held-out 배치 crop도 임베딩한다(P-orig E0 진단 예외, decisions.md 기록).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from pvil.data.dataset import CropStore  # noqa: E402
from pvil.eval import run_info  # noqa: E402
from pvil.models.cpbg import ImageEncoder  # noqa: E402
from pvil.models.infer import embed_cells  # noqa: E402
from train import load_config  # noqa: E402

VARIANTS = ("cls", "frozen_cls", "intensity")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--variant", choices=VARIANTS, required=True)
    ap.add_argument("--batch", type=int, default=1024)
    args = ap.parse_args()
    t0 = time.time()
    cfg = load_config(args.config)
    if cfg.protocol != "P-orig":
        raise ValueError("G0 진단은 E0 P-orig run에만 쓴다")
    rel = Path(cfg.experiment) / cfg.method / cfg.dataset / f"seed{cfg.seed}"
    out = Path(cfg.paths.embeddings_dir) / "E0_diag" / args.variant / cfg.method / cfg.dataset / f"seed{cfg.seed}"
    out.mkdir(parents=True, exist_ok=True)

    pre = OmegaConf.load(REPO / cfg.preprocess_config)
    store = CropStore(Path(cfg.paths.crop_cache_dir) / cfg.dataset)
    dna_idx = list(pre.channels.names).index(pre.channels.nucleus)
    size = cfg.data.input_size
    patch = {"size": cfg.data.views.cd_patch_size, "inset": cfg.data.views.cd_patch_inset}

    model, ckpt_info = None, None
    if args.variant != "intensity":
        torch.manual_seed(cfg.seed)
        m = cfg.model
        model = ImageEncoder(store.meta["n_channels"], m.backbone, OmegaConf.to_container(m.lora),
                             m.image_head.out_dim).cuda()
        if args.variant == "cls":
            ck_path = Path(cfg.paths.ckpt_dir) / rel / "best.pt"
            ck = torch.load(ck_path, map_location="cuda")
            part = {k[len("model."):]: v for k, v in ck["state"].items() if k.startswith("model.")}
            missing, unexpected = model.load_state_dict(part, strict=False)
            assert not unexpected, unexpected
            assert all(k.startswith("backbone.") and "lora_" not in k for k in missing), "학습 파라미터 누락"
            ckpt_info = {"path": str(ck_path), "epoch": ck["epoch"], "val_r1": ck["val_r1"]}

    idx = store.indices("all", allow_test=True)
    emb = embed_cells(model, store, idx, view=cfg.view, dna_idx=dna_idx, size=size, batch=args.batch,
                      patch=patch, layer="intensity" if args.variant == "intensity" else "cls")
    np.save(out / "cells.npy", emb.astype(np.float32))
    info = run_info.collect(args.config, cfg.seed, protocol=cfg.protocol, fold=None, dataset=cfg.dataset,
                            method=cfg.method, view=cfg.view, variant=args.variant, checkpoint=ckpt_info,
                            data_version=store.meta, n_cells=int(len(emb)), dim=int(emb.shape[1]),
                            gpu_name=torch.cuda.get_device_name(0), embed_sec=time.time() - t0)
    (out / "run_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False, default=str))
    print("saved", out / "cells.npy", emb.shape, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
