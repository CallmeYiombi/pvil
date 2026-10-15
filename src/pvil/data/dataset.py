"""crop 저장소 읽기와 학습 미니배치 샘플러 (사양 A.2, decisions.md 2026-10-14).

CropStore: scripts/cache_crops.py가 만든 memmap(이미지 uint8, 중심 세포 핵·세포 이진 마스크)과
crop 표(cells.parquet). held-out 배치는 allow_test=True 없이 꺼낼 수 없다(.claude/rules/evaluation.md).
CD/SD 패치 강도(`density`)는 FOV 세포 수를 min-max로 [0, 255] 정수에 맞춘 뒤 /255 한 값이다.
min·max는 학습 배치 세포에서만 구한다(held-out 배치를 전처리 상수에도 쓰지 않는다. decisions.md 2026-10-15).

BalancedBatchSampler: step마다 perturbation n_perts개 × k개 + 대조군 k개.
perturbation 안에서는 배치(source)를 무작위 순서로 돌아가며 뽑아 cross-batch 커버리지를 확보한다.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CONTROL_LABEL = "__control__"


class HeldOutAccessError(RuntimeError):
    pass


class CropStore:
    def __init__(self, cache_dir: str | Path):
        self.dir = Path(cache_dir)
        meta = json.loads((self.dir / "meta.json").read_text())
        n = meta["n"]
        self.meta = meta
        self._mm: dict = {}
        cells = pd.read_parquet(cache_dir / "cells.parquet")
        assert len(cells) == n and (cells.crop_index.to_numpy() == np.arange(n)).all()
        cells["label"] = cells.label.where(cells.role != "control", CONTROL_LABEL)
        self.perts = sorted(cells.loc[cells.role == "perturbation", "label"].unique())
        lut = {p: i for i, p in enumerate(self.perts)} | {CONTROL_LABEL: len(self.perts)}
        cells["label_idx"] = cells.label.map(lut).astype(np.int64)
        self.cells = cells
        n_cells = cells.fov_n_cells.to_numpy(np.float64)
        tr = n_cells[~cells.is_heldout_batch.to_numpy()]
        self.density_range = (float(tr.min()), float(tr.max()))
        lo, hi = self.density_range
        self.density = (np.round(np.clip((n_cells - lo) / max(hi - lo, 1.0), 0, 1) * 255) / 255).astype(np.float32)

    def __getstate__(self):                     # DataLoader worker로 memmap 내용을 복사하지 않는다
        return {**self.__dict__, "_mm": {}}

    def _memmap(self, name: str, n_ch: int) -> np.memmap:
        if name not in self._mm:                # 프로세스마다 처음 쓸 때 연다
            s = self.meta["crop_size"]
            self._mm[name] = np.memmap(self.dir / name, dtype=np.uint8, mode="r", shape=(self.meta["n"], n_ch, s, s))
        return self._mm[name]

    @property
    def images(self) -> np.memmap:
        return self._memmap("images.u8", self.meta["n_channels"])

    @property
    def masks(self) -> np.memmap:
        return self._memmap("masks.u1", 2)

    @property
    def n_perts(self) -> int:
        return len(self.perts)

    def indices(self, split: str, allow_test: bool = False) -> np.ndarray:
        """split: train(held-out 배치 제외) | heldout | all."""
        h = self.cells.is_heldout_batch.to_numpy()
        if split in ("heldout", "all") and not allow_test:
            raise HeldOutAccessError(f"split={split}는 held-out 배치를 포함한다. allow_test=True가 필요하다")
        return {"train": np.flatnonzero(~h), "heldout": np.flatnonzero(h), "all": np.arange(len(h))}[split]

    def load(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """정렬된 순서로 읽어 원래 순서로 돌려준다 → (img uint8, nuc bool, cell bool, label_idx)."""
        idx = np.asarray(idx)
        o = np.argsort(idx)
        inv = np.empty_like(o)
        inv[o] = np.arange(len(o))
        s = idx[o]
        img = self.images[s][inv]
        m = self.masks[s][inv]
        lab = self.cells.label_idx.to_numpy()[idx]
        return (torch.from_numpy(np.ascontiguousarray(img)), torch.from_numpy(m[:, 0] > 0),
                torch.from_numpy(m[:, 1] > 0), torch.from_numpy(lab))


class BalancedBatchSampler:
    """index 배열 목록을 내는 반복자. 한 epoch = steps_per_epoch step."""

    def __init__(self, cells: pd.DataFrame, pool: np.ndarray, n_perts: int, k: int, n_control: int,
                 steps_per_epoch: int, seed: int):
        sub = pd.DataFrame({"pos": pool, "label_idx": cells.label_idx.to_numpy()[pool],
                            "source": cells.source.to_numpy()[pool], "label": cells.label.to_numpy()[pool]})
        self.groups: dict[int, list[np.ndarray]] = {}
        for (lab, _src), g in sub.groupby(["label_idx", "source"], sort=True):
            self.groups.setdefault(int(lab), []).append(g.pos.to_numpy())
        ctrl = sub.loc[sub.label == CONTROL_LABEL, "label_idx"]
        self.control = self.groups.pop(int(ctrl.iloc[0])) if len(ctrl) else None
        self.pert_ids = np.array(sorted(self.groups))
        if len(self.pert_ids) < n_perts:
            raise ValueError(f"perturbation {len(self.pert_ids)}개 < 배치당 {n_perts}개")
        self.n_perts, self.k, self.n_control = n_perts, k, n_control
        self.steps_per_epoch = steps_per_epoch
        self.rng = np.random.default_rng(seed)
        self._queue = np.array([], dtype=int)

    def _draw(self, per_source: list[np.ndarray], n: int) -> np.ndarray:
        order = self.rng.permutation(len(per_source))
        out = []
        for j in range(n):                      # 배치를 무작위 순서로 돌아가며 하나씩
            src = per_source[order[j % len(order)]]
            out.append(src[self.rng.integers(len(src))])
        return np.asarray(out)

    def _next_perts(self) -> np.ndarray:
        if len(self._queue) < self.n_perts:     # perturbation을 고르게 돌도록 순열을 이어 붙인다
            self._queue = np.concatenate([self._queue, self.rng.permutation(self.pert_ids)])
        out, self._queue = self._queue[:self.n_perts], self._queue[self.n_perts:]
        if len(np.unique(out)) < len(out):      # 순열 경계에서 중복이 생기면 다시 뽑는다
            out = self.rng.choice(self.pert_ids, self.n_perts, replace=False)
        return out

    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            idx = [self._draw(self.groups[p], self.k) for p in self._next_perts()]
            if self.control is not None and self.n_control:
                idx.append(self._draw(self.control, self.n_control))
            yield np.concatenate(idx)

    def __len__(self):
        return self.steps_per_epoch


def steps_per_epoch(n_train: int, batch_size: int) -> int:
    return math.ceil(n_train / batch_size)
