"""Phase 0 환경 점검. 사용: source scripts/env.sh && python scripts/check_env.py [--out DIR]

각 항목을 실행해 PASS/FAIL과 측정값을 JSON으로 저장한다. 랜덤·합성 입력만 쓴다(실데이터 없음).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from pvil.eval import run_info  # noqa: E402

HOME_WATCH = ["~/.cache", "~/.cellpose", "~/.triton", "~/.local/share/uv", "~/.local/bin", "~/.config/matplotlib"]


def check_gpus():
    import torch

    assert torch.cuda.is_available(), "CUDA 사용 불가"
    out = {"torch": torch.__version__, "cuda": torch.version.cuda, "devices": []}
    for i in range(torch.cuda.device_count()):
        a = torch.randn(4096, 4096, device=f"cuda:{i}", dtype=torch.bfloat16)
        torch.cuda.synchronize(i)
        t0 = time.perf_counter()
        for _ in range(20):
            a @ a
        torch.cuda.synchronize(i)
        tflops = 20 * 2 * 4096**3 / (time.perf_counter() - t0) / 1e12
        out["devices"].append({
            "index": i,
            "name": torch.cuda.get_device_name(i),
            "capability": list(torch.cuda.get_device_capability(i)),
            "bf16_matmul_tflops": round(tflops, 1),
        })
    return out


def check_backbone_lora(batch: int = 64, n_channels: int = 6):
    """DINOv3 ViT-B/16 + 1×1 conv 채널 어댑터 + peft LoRA, bf16 fwd/bwd 1 step."""
    import timm
    import torch
    from peft import LoraConfig, get_peft_model

    dev = "cuda:0"
    backbone = timm.create_model("vit_base_patch16_dinov3.lvd1689m", pretrained=True, num_classes=0)
    lora_cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, target_modules=["qkv", "proj", "fc1", "fc2"])
    backbone = get_peft_model(backbone, lora_cfg)
    adapter = torch.nn.Conv2d(n_channels, 3, kernel_size=1, bias=False)
    model = torch.nn.Sequential(adapter, backbone).to(dev)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    lora_modules = sorted({n.split(".")[-2] for n, _ in backbone.named_modules() if n.endswith("lora_A")})

    x = torch.randn(batch, n_channels, 224, 224, device=dev)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    times = []
    torch.cuda.reset_peak_memory_stats(dev)
    for _ in range(6):
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = model(x)
        z.float().pow(2).mean().backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(dev)
        times.append(time.perf_counter() - t0)
    step = float(np.median(times[2:]))
    return {
        "embed_dim": int(z.shape[-1]),
        "trainable_params": n_train,
        "total_params": n_total,
        "lora_target_leaf_names": lora_modules,
        "batch": batch,
        "step_sec_median": round(step, 4),
        "images_per_sec": round(batch / step, 1),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated(dev) / 1e9, 2),
        "note": "torch.compile 미사용, 단일 GPU. 정식 T는 E0에서 측정",
    }


def check_cellpose():
    from cellpose import models

    rng = np.random.default_rng(0)
    img = np.zeros((256, 256), np.float32)
    yy, xx = np.mgrid[:256, :256]
    centers = rng.integers(30, 226, size=(12, 2))
    for cy, cx in centers:
        img[(yy - cy) ** 2 + (xx - cx) ** 2 < 12**2] = 1.0
    img += rng.normal(0, 0.05, img.shape).astype(np.float32)
    model = models.CellposeModel(gpu=True)
    t0 = time.perf_counter()
    masks, *_ = model.eval(img)
    return {
        "pretrained_model": str(getattr(model, "pretrained_model", "?")),
        "n_objects_found": int(masks.max()),
        "n_objects_drawn": len(centers),
        "eval_sec": round(time.perf_counter() - t0, 2),
    }


def check_ecfp():
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    mol = Chem.MolFromSmiles("C[C@H](N)C(=O)O")  # L-alanine
    fp = np.log1p(gen.GetCountFingerprintAsNumPy(mol).astype(np.float32))
    return {"dim": int(fp.shape[0]), "nonzero": int((fp > 0).sum())}


def check_esm2():
    import torch
    from transformers import AutoModel, AutoTokenizer

    name = "facebook/esm2_t33_650M_UR50D"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to("cuda:0").eval()
    batch = tok(["MKTAYIAKQRQISFVKSHFSRQ"], return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        h = model(**batch).last_hidden_state
    return {"dim": int(h.shape[-1]), "n_tokens": int(h.shape[1])}


def check_postprocessing():
    import harmonypy
    import pandas as pd
    from pycytominer.operations import Spherize
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(0)
    n, d = 300, 80
    x = rng.normal(size=(n, d))
    batch = np.repeat(["b0", "b1", "b2"], n // 3)
    x += (batch == "b1")[:, None] * 0.5
    ctrl = rng.random(n) < 0.3
    cols = [f"f{i}" for i in range(d)]
    sph = Spherize(method="ZCA-cor").fit(pd.DataFrame(x[ctrl], columns=cols))
    xs = sph.transform(pd.DataFrame(x, columns=cols))
    xp = PCA(50, random_state=0).fit_transform(np.asarray(xs))
    ho = harmonypy.run_harmony(xp, pd.DataFrame({"batch": batch}), ["batch"], theta=2, lamb=1, sigma=0.1,
                               max_iter_harmony=100, verbose=False, random_state=0)
    z = np.asarray(ho.Z_corr)
    z = z.T if z.shape[0] != n else z
    return {"pca_shape": list(xp.shape), "harmony_shape": list(z.shape)}


def check_copairs():
    import pandas as pd
    from copairs.map import average_precision, mean_average_precision

    rng = np.random.default_rng(0)
    pert = np.repeat([f"p{i}" for i in range(10)] + ["ctrl"] * 2, 6)
    plate = np.tile(np.arange(6), len(pert) // 6).astype(str)
    meta = pd.DataFrame({"pert": pert, "plate": plate, "is_ctrl": pert == "ctrl"})
    feats = rng.normal(size=(len(meta), 16)) + (pd.factorize(pert)[0][:, None] % 4)
    ap = average_precision(meta, feats, pos_sameby=["pert"], pos_diffby=["plate"],
                           neg_sameby=["plate"], neg_diffby=["is_ctrl"], progress_bar=False)
    m = mean_average_precision(ap[~ap["is_ctrl"]], ["pert"], null_size=1000, threshold=0.05, seed=0, progress_bar=False)
    return {"n_ap": int(ap["average_precision"].notna().sum()), "mean_map": round(float(m["mean_average_precision"].mean()), 3)}


def check_scib():
    import scib_metrics
    from scib_metrics.benchmark import BatchCorrection, BioConservation

    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 20)).astype(np.float32)
    labels = np.repeat(np.arange(10), 30)
    x += labels[:, None] * 0.3
    batch = np.tile(np.arange(3), 100)
    s = scib_metrics.silhouette_label(x, labels)
    return {
        "silhouette_label": round(float(s), 3),
        "batch_metrics_default": list(BatchCorrection.__dataclass_fields__),
        "bio_metrics_default": list(BioConservation.__dataclass_fields__),
        "has_bras": hasattr(scib_metrics, "bras"),
        "has_pcr_comparison": hasattr(scib_metrics, "pcr_comparison"),
        "n_batches": int(len(set(batch))),
    }


def check_cp_measure():
    from cp_measure.bulk import get_core_measurements, get_correlation_measurements

    rng = np.random.default_rng(0)
    masks = np.zeros((100, 100), np.int32)
    masks[20:50, 20:50] = 1
    masks[60:90, 55:85] = 2
    pix = rng.random((100, 100)).astype(np.float32)
    pix2 = rng.random((100, 100)).astype(np.float32)
    core = get_core_measurements()
    n_core = sum(len(f(masks, pix)) for f in core.values())
    corr = get_correlation_measurements()
    n_corr = sum(len(f(pix, pix2, masks)) for f in corr.values())
    return {"core_groups": sorted(core), "n_core_features_one_channel": n_core,
            "corr_groups": sorted(corr), "n_corr_features_one_pair": n_corr}


CHECKS = {
    "gpus": check_gpus,
    "backbone_lora": check_backbone_lora,
    "cellpose_sam": check_cellpose,
    "ecfp4": check_ecfp,
    "esm2": check_esm2,
    "postprocessing": check_postprocessing,
    "copairs": check_copairs,
    "scib_metrics": check_scib,
    "cp_measure": check_cp_measure,
}


def home_writes(since: float) -> list[str]:
    hits = []
    for d in HOME_WATCH:
        root = Path(os.path.expanduser(d))
        if not root.exists():
            continue
        for p in root.rglob("*"):
            try:
                if p.lstat().st_mtime >= since:
                    hits.append(str(p))
            except OSError:
                pass
            if len(hits) >= 50:
                return hits
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "results/E0/env_check"))
    ap.add_argument("--only", nargs="*", default=None, help="실행할 항목 이름")
    args = ap.parse_args()

    start = time.time()
    report = {"run_info": run_info.collect(), "checks": {}}
    for name, fn in CHECKS.items():
        if args.only and name not in args.only:
            continue
        t0 = time.perf_counter()
        try:
            res = {"status": "PASS", **fn()}
        except Exception as e:  # 점검 스크립트이므로 실패를 기록하고 다음 항목으로 넘어간다
            res = {"status": "FAIL", "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(limit=5)}
        res["sec"] = round(time.perf_counter() - t0, 1)
        report["checks"][name] = res
        print(f"[{res['status']}] {name} ({res['sec']}s)", flush=True)

    writes = home_writes(start)
    report["checks"]["no_home_writes"] = {"status": "PASS" if not writes else "FAIL", "files": writes}
    print(f"[{report['checks']['no_home_writes']['status']}] no_home_writes")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "env_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print(f"saved: {out / 'env_report.json'}")
    sys.exit(0 if all(c["status"] == "PASS" for c in report["checks"].values()) else 1)


if __name__ == "__main__":
    main()
