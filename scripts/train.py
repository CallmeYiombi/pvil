"""A1 학습 (CP-BG-Bench 재현, 사양 A.2) + 학습 비용 T 측정.

사용: source scripts/env.sh && CUDA_VISIBLE_DEVICES=1 python scripts/train.py configs/E0/dino_c_rxrx1.yaml [--max-steps 50]
출력:
  results/<experiment>/<method>/<dataset>/seed<k>/  config.yaml, run_info.json, git_hash.txt, train_log.csv,
                                                    epoch_log.csv, gpu_log.csv, timing.json
  <ckpt_dir>/<experiment>/<method>/<dataset>/seed<k>/best.pt   (학습되는 파라미터만)
  <embeddings_dir>/<experiment>/<method>/<dataset>/seed<k>/cells.npy  (best 체크포인트, 전체 crop 순서)
P-orig: held-out 배치를 validation(early stopping)에 쓴다(절대 규칙 1의 유일한 예외). P-strict는 아직 없다.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from pvil.data.dataset import BalancedBatchSampler, CropStore, steps_per_epoch  # noqa: E402
from pvil.data.views import build_batch  # noqa: E402
from pvil.eval import run_info  # noqa: E402
from pvil.losses.contrastive import LearnableTemperature, multi_positive_infonce  # noqa: E402
from pvil.models.cpbg import ImageEncoder, PertEncoder  # noqa: E402
from pvil.models.infer import embed_cells, img2pert_macro_r1  # noqa: E402

GPU_QUERY = "index,name,utilization.gpu,memory.used,memory.total"


def load_config(path: str):
    exp = OmegaConf.load(path)
    cfg = OmegaConf.merge(OmegaConf.load(REPO / exp.base), exp)
    cfg.paths = OmegaConf.to_container(OmegaConf.load(REPO / "configs/paths.yaml"), resolve=True)
    return cfg


class GpuLogger(threading.Thread):
    """interval초마다 nvidia-smi로 GPU 사용률·메모리와 GPU별 프로세스 목록을 기록한다(공유 서버 상황 기록용)."""

    def __init__(self, path: Path, interval: float):
        super().__init__(daemon=True)
        self.path, self.interval, self.stop = path, interval, threading.Event()

    def snapshot(self):
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        gpus = subprocess.run(["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.strip().splitlines()
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip().splitlines()
        uuids = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                               capture_output=True, text=True).stdout.strip().splitlines()
        idx_of = {u.split(", ")[1]: u.split(", ")[0] for u in uuids}
        n_apps = {}
        for a in apps:
            gi = idx_of.get(a.split(", ")[0])
            n_apps[gi] = n_apps.get(gi, 0) + 1
        with open(self.path, "a") as f:
            for g in gpus:
                f.write(f"{ts}, {g}, {n_apps.get(g.split(', ')[0], 0)}\n")

    def run(self):
        self.path.write_text(f"time, {GPU_QUERY}, n_processes\n")
        while not self.stop.is_set():
            self.snapshot()
            self.stop.wait(self.interval)


def param_groups(model, pert, tau, lr, wd):
    """[원문 없음] 어댑터와 학습 가능 perturbation 표는 head LR, WD 0 (decisions.md 2026-10-14)."""
    lora = [p for n, p in model.backbone.named_parameters() if p.requires_grad]
    head = list(model.head.parameters()) + list(pert.head.parameters())
    other = list(model.adapter.parameters()) + (list(pert.learned.parameters()) if pert.learned is not None else [])
    return [{"name": "head", "params": head, "lr": lr.head, "weight_decay": wd.head},
            {"name": "lora", "params": lora, "lr": lr.lora, "weight_decay": wd.lora},
            {"name": "adapter_table", "params": other, "lr": lr.head, "weight_decay": 0.0},
            {"name": "tau", "params": list(tau.parameters()), "lr": lr.tau, "weight_decay": wd.tau}]


def trainable_state(model, pert, tau) -> dict:
    sd = {f"model.{k}": v for k, v in model.state_dict().items() if "lora_" in k or not k.startswith("backbone.")}
    sd |= {f"pert.{k}": v for k, v in pert.state_dict().items() if k != "table"}
    sd |= {f"tau.{k}": v for k, v in tau.state_dict().items()}
    return {k: v.detach().cpu().clone() for k, v in sd.items()}


def load_trainable(model, pert, tau, sd: dict):
    for prefix, mod in (("model.", model), ("pert.", pert), ("tau.", tau)):
        part = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        missing, unexpected = mod.load_state_dict(part, strict=False)
        assert not unexpected, unexpected


class Batches(torch.utils.data.Dataset):
    def __init__(self, store: CropStore):
        self.store = store

    def __getitem__(self, idx):
        return self.store.load(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--max-steps", type=int, default=None, help="스모크 테스트용. 결과는 <seed>_smoke에 쓴다")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    t_wall0 = time.time()
    cfg = load_config(args.config)
    tr = cfg.train
    smoke = args.max_steps is not None
    tag = f"seed{cfg.seed}" + ("_smoke" if smoke else "")
    rel = Path(cfg.experiment) / cfg.method / cfg.dataset / tag
    out = REPO / cfg.paths.results_dir / rel
    ckpt_dir = Path(cfg.paths.ckpt_dir) / rel
    emb_dir = Path(cfg.paths.embeddings_dir) / rel
    for d in (out, ckpt_dir, emb_dir):
        d.mkdir(parents=True, exist_ok=True)

    if cfg.protocol != "P-orig":
        raise NotImplementedError("P-strict validation 배치 정의는 아직 없다 (P0 작업 순서 10)")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    gen = torch.Generator(device="cuda").manual_seed(cfg.seed)

    pre = OmegaConf.load(REPO / cfg.preprocess_config)
    store = CropStore(Path(cfg.paths.crop_cache_dir) / cfg.dataset)
    dna_idx = list(pre.channels.names).index(pre.channels.nucleus)
    train_idx = store.indices("train")
    val_idx = store.indices("heldout", allow_test=True)       # P-orig에서만 (위에서 확인)
    spe = steps_per_epoch(len(train_idx), tr.batch_size)
    total_steps = tr.max_epochs * spe
    n_slots = tr.perts_per_batch - (1 if tr.include_controls_every_batch else 0)
    sampler = BalancedBatchSampler(store.cells, train_idx, n_perts=n_slots, k=tr.samples_per_pert,
                                   n_control=tr.samples_per_pert if tr.include_controls_every_batch else 0,
                                   steps_per_epoch=spe if not smoke else args.max_steps, seed=cfg.seed)

    table = None
    if cfg.pert_embedding != "learned":
        z = np.load(Path(cfg.paths.pert_emb_dir) / f"{cfg.pert_embedding}.npz")
        pos = {l: i for i, l in enumerate(z["labels"])}
        table = torch.from_numpy(z["emb"][[pos[p] for p in store.perts]])
    m = cfg.model
    model = ImageEncoder(store.meta["n_channels"], m.backbone, OmegaConf.to_container(m.lora), m.image_head.out_dim,
                         grad_checkpointing=tr.get("grad_checkpointing", False)).cuda()
    pert = PertEncoder(store.n_perts, m.pert_head.out_dim, table=table, learned_dim=cfg.learned_pert_dim).cuda()
    tau = LearnableTemperature(cfg.loss.tau_init, tuple(cfg.loss.tau_clamp)).cuda()
    groups = param_groups(model, pert, tau, tr.lr, tr.weight_decay)
    opt = torch.optim.AdamW(groups, betas=tuple(tr.optimizer.betas))
    warm = max(1, int(tr.schedule.warmup_frac * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total_steps - warm))))
    params = [p for g in groups for p in g["params"]]

    info = run_info.collect(args.config, cfg.seed, protocol=cfg.protocol, fold=None, dataset=cfg.dataset,
                            method=cfg.method, view=cfg.view, smoke=smoke, data_version=store.meta,
                            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                            gpu_name=torch.cuda.get_device_name(0), n_train_crops=int(len(train_idx)),
                            n_val_crops=int(len(val_idx)), steps_per_epoch=spe, n_perts=store.n_perts,
                            n_trainable_params=int(sum(p.numel() for p in params)))
    run_info.save(out, info)
    OmegaConf.save(cfg, out / "config.yaml")
    gpu_log = GpuLogger(out / "gpu_log.csv", cfg.gpu_log_interval_sec)
    gpu_log.start()

    loader = torch.utils.data.DataLoader(Batches(store), batch_size=None, sampler=sampler, num_workers=args.workers,
                                         pin_memory=True, persistent_workers=True, prefetch_factor=4)
    aug = OmegaConf.to_container(tr.augment)
    size = cfg.data.input_size
    tlog = open(out / "train_log.csv", "w", newline="")
    tw = csv.writer(tlog)
    tw.writerow(["epoch", "step", "loss", "tau", "lr_head", "lr_lora", "samples_per_sec", "data_wait_frac"])
    elog = open(out / "epoch_log.csv", "w", newline="")
    ew = csv.writer(elog)
    ew.writerow(["epoch", "train_loss", "val_macro_img2pert_r1", "train_sec", "val_sec", "best"])

    best, bad, step = -1.0, 0, 0
    t_train = t_val = 0.0
    max_epochs = 1 if smoke else tr.max_epochs
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        losses, t_wait, t_win, n_win = [], 0.0, time.time(), 0
        t_fetch = time.time()
        for img, nuc, cell, lab in loader:
            t_wait += time.time() - t_fetch
            img, nuc, cell, lab = (t.cuda(non_blocking=True) for t in (img, nuc, cell, lab))
            x = build_batch(img, nuc, cell, view=cfg.view, dna_idx=dna_idx, size=size, augment=aug, gen=gen)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z_img = model(x)
                z_pert = pert(lab)
            loss = multi_positive_infonce(z_img, z_pert, lab, tau())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, tr.grad_clip)
            opt.step()
            sched.step()
            step += 1
            n_win += len(lab)
            losses.append(loss.item())
            if step % cfg.log_every == 0:
                torch.cuda.synchronize()
                dt = time.time() - t_win
                tw.writerow([epoch, step, np.mean(losses[-cfg.log_every:]), tau().item(), opt.param_groups[0]["lr"],
                             opt.param_groups[1]["lr"], n_win / dt, t_wait / dt])
                tlog.flush()
                print(f"ep {epoch} step {step} loss {np.mean(losses[-cfg.log_every:]):.4f} "
                      f"{n_win / dt:.0f} samples/s (data wait {t_wait / dt:.0%})", flush=True)
                t_win, n_win, t_wait = time.time(), 0, 0.0
            t_fetch = time.time()
        torch.cuda.synchronize()
        t1 = time.time()
        t_train += t1 - t0

        z_val = embed_cells(model, store, val_idx, view=cfg.view, dna_idx=dna_idx, size=size)
        r1 = img2pert_macro_r1(z_val, store.cells.label_idx.to_numpy()[val_idx], pert, store.n_perts)
        ep_val = time.time() - t1
        t_val += ep_val
        improved = r1 > best
        if improved:
            best, bad = r1, 0
            torch.save({"state": trainable_state(model, pert, tau), "epoch": epoch, "val_r1": r1}, ckpt_dir / "best.pt")
        else:
            bad += 1
        ew.writerow([epoch, float(np.mean(losses)), r1, t1 - t0, ep_val, improved])
        elog.flush()
        print(f"== epoch {epoch}: loss {np.mean(losses):.4f}, val R@1 {r1:.4f} (best {best:.4f}, bad {bad})", flush=True)
        if bad >= tr.early_stopping.patience:
            break

    # best 체크포인트로 전체 crop 임베딩 (평가용, held-out 포함 → P-orig 최종 평가)
    t2 = time.time()
    ck = torch.load(ckpt_dir / "best.pt", map_location="cuda")
    load_trainable(model, pert, tau, ck["state"])
    emb = embed_cells(model, store, store.indices("all", allow_test=True), view=cfg.view, dna_idx=dna_idx, size=size)
    np.save(emb_dir / "cells.npy", emb.astype(np.float32))
    t_embed = time.time() - t2

    gpu_log.stop.set()
    gpu_log.snapshot()
    timing = {"protocol": cfg.protocol, "epochs_run": epoch, "best_epoch": ck["epoch"], "best_val_r1": ck["val_r1"],
              "steps": step, "batch_size": tr.batch_size, "train_sec": t_train, "val_sec": t_val, "embed_sec": t_embed,
              "wall_sec": time.time() - t_wall0, "gpu_hours_T": (time.time() - t_wall0) / 3600,
              "train_samples_per_sec": step * tr.batch_size / max(t_train, 1e-9),
              "note": "공유 GPU(다른 작업과 동시 실행)에서 잰 상한값. 점유 상황은 gpu_log.csv"}
    (out / "timing.json").write_text(json.dumps(timing, indent=2, ensure_ascii=False))
    print(json.dumps(timing, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # 예외 후에도 DataLoader worker 등이 남아 GPU 메모리를 쥔 채 멈추는 일이 있었다(2026-10-15).
        # 공유 GPU를 바로 돌려주도록 traceback만 남기고 즉시 종료한다
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
