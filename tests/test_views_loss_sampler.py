import json

import numpy as np
import pandas as pd
import pytest
import torch

from pvil.data.dataset import CONTROL_LABEL, BalancedBatchSampler, CropStore, HeldOutAccessError
from pvil.data.views import apply_geometry, build_batch, make_view, sample_geometry, to_input
from pvil.eval.metrics import cross_batch_recall
from pvil.losses.contrastive import multi_positive_infonce


def _masks(b=2, s=32):
    yy, xx = np.indices((s, s))
    cell = ((yy - 16) ** 2 + (xx - 16) ** 2 <= 10 ** 2)
    nuc = ((yy - 16) ** 2 + (xx - 22) ** 2 <= 6 ** 2)      # 핵 일부가 세포 밖으로 나온다(x 28 근처)
    return torch.from_numpy(np.stack([nuc] * b)), torch.from_numpy(np.stack([cell] * b))


def test_s_view_zeros_outside_masks_and_dna_uses_nucleus():
    nuc, cell = _masks()
    x = torch.rand(2, 6, 32, 32) + 0.1
    y = make_view(x, nuc, cell, "S", dna_idx=0)
    assert (y[:, 0][~nuc] == 0).all() and (y[:, 0][nuc] > 0).all()        # DNA: 핵 밖 0, 세포 밖 핵 픽셀은 유지
    assert (y[:, 1:][~cell[:, None].expand(-1, 5, -1, -1)] == 0).all()
    assert (y[:, 1:][cell[:, None].expand(-1, 5, -1, -1)] > 0).all()
    assert torch.equal(make_view(x, nuc, cell, "C", 0), x)


def test_density_patches_position_size_value():
    nuc, cell = _masks(3, 64)
    x = torch.rand(3, 6, 64, 64) + 0.1
    dens = torch.tensor([0.0, 0.5, 1.0])
    patch = {"size": 22, "inset": 7}
    for view, base in (("CD", "C"), ("SD", "S")):
        y = make_view(x, nuc, cell, view, 0, dens, patch)
        b = make_view(x, nuc, cell, base, 0)
        on = torch.zeros(64, 64, dtype=torch.bool)
        for r in (7, 64 - 7 - 22):
            for c in (7, 64 - 7 - 22):
                on[r:r + 22, c:c + 22] = True
        assert on.sum() == 4 * 22 * 22
        for i in range(3):                                  # 모든 채널, 패치 안은 density 값, 밖은 기본 view 그대로
            assert (y[i][:, on] == dens[i]).all()
            assert torch.equal(y[i][:, ~on], b[i][:, ~on])
    with pytest.raises(ValueError):
        make_view(x, nuc, cell, "CD", 0)


def test_density_patches_drawn_at_input_resolution():
    nuc, cell = _masks(2, 150)
    img = torch.randint(0, 255, (2, 6, 150, 150), dtype=torch.uint8)
    y = build_batch(img, nuc, cell, view="CD", dna_idx=0, size=224, augment=None,
                    density=torch.tensor([0.25, 0.75]), patch={"size": 22, "inset": 7})
    assert y.shape[-1] == 224
    assert (y[0, :, 7:29, 7:29] == 0.25).all() and (y[1, :, 224 - 29:224 - 7, 224 - 29:224 - 7] == 0.75).all()


def test_to_input_resizes_masks_nearest():
    nuc, cell = _masks()
    img = torch.randint(0, 255, (2, 6, 32, 32), dtype=torch.uint8)
    x, n, c = to_input(img, nuc, cell, 64)
    assert x.shape == (2, 6, 64, 64) and n.dtype == torch.bool and x.max() <= 1
    assert abs(c.float().mean() - cell.float().mean()) < 0.02


def test_same_geometry_for_paired_views():
    x = torch.rand(4, 6, 32, 32)
    nuc, cell = _masks(4)
    g = sample_geometry(4, 90, True, True, "cpu", torch.Generator().manual_seed(0))
    a = apply_geometry(make_view(x, nuc, cell, "S", 0), g)
    b = make_view(apply_geometry(x, g), *(apply_geometry(m[:, None].float(), g)[:, 0] > 0.5 for m in (nuc, cell)), "S", 0)
    assert (a - b).abs().mean() < 0.05        # 같은 변환이면 마스킹·변환 순서와 무관하게 거의 같다


def _loss_reference(zi, zp, lab, tau):
    zi = zi / zi.norm(dim=1, keepdim=True)
    zp = zp / zp.norm(dim=1, keepdim=True)
    L = (zi @ zp.T / tau).numpy()
    out = []
    for M in (L, L.T):
        for i in range(len(lab)):
            pos = lab == lab[i]
            out.append(-np.log(np.exp(M[i, pos]).sum() / np.exp(M[i]).sum()) / len(lab))
    return sum(out) / 2


def test_multi_positive_loss_matches_reference_and_is_symmetric():
    g = torch.Generator().manual_seed(0)
    zi, zp = torch.randn(12, 8, generator=g), torch.randn(12, 8, generator=g)
    lab = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3])
    got = multi_positive_infonce(zi, zp, lab, torch.tensor(0.1)).item()
    assert got == pytest.approx(_loss_reference(zi, zp, lab.numpy(), 0.1), rel=1e-4)
    assert multi_positive_infonce(zp, zi, lab, torch.tensor(0.1)).item() == pytest.approx(got, rel=1e-5)


def _cells():
    rows = []
    for src in ("b1", "b2", "b3"):
        for lab in [f"p{i}" for i in range(6)] + [CONTROL_LABEL]:
            for _ in range(5):
                rows.append({"label": lab, "source": src, "is_heldout_batch": src == "b3",
                             "role": "control" if lab == CONTROL_LABEL else "perturbation"})
    c = pd.DataFrame(rows)
    lut = {f"p{i}": i for i in range(6)} | {CONTROL_LABEL: 6}
    c["label_idx"] = c.label.map(lut)
    return c


def test_sampler_composition_and_batch_stratification():
    c = _cells()
    pool = np.flatnonzero(~c.is_heldout_batch.to_numpy())
    s = BalancedBatchSampler(c, pool, n_perts=3, k=4, n_control=4, steps_per_epoch=20, seed=0)
    for idx in s:
        assert len(idx) == 3 * 4 + 4
        sub = c.iloc[idx]
        assert not sub.is_heldout_batch.any()
        assert (sub.label == CONTROL_LABEL).sum() == 4
        assert sub[sub.label != CONTROL_LABEL].label.nunique() == 3
        for _, g in sub.groupby("label"):                 # 4개를 배치 2개에 돌아가며 → 각 2개
            assert g.source.value_counts().tolist() == [2, 2]


def test_heldout_guards(tmp_path):
    c = _cells().assign(crop_index=lambda d: np.arange(len(d)), well_key="w", plate_key="p",
                        fov_n_cells=lambda d: np.arange(len(d)) % 7 + 10)
    c.drop(columns="label_idx").to_parquet(tmp_path / "cells.parquet")
    (tmp_path / "meta.json").write_text(json.dumps({"n": len(c), "n_channels": 6, "crop_size": 8}))
    store = CropStore(tmp_path)
    assert len(store.indices("train")) == (~c.is_heldout_batch).sum()
    with pytest.raises(HeldOutAccessError):
        store.indices("heldout")
    assert len(store.indices("heldout", allow_test=True)) == c.is_heldout_batch.sum()
    tr = c.fov_n_cells[~c.is_heldout_batch]                # density min-max는 학습 배치에서만
    assert store.density_range == (tr.min(), tr.max())
    assert store.density.min() == 0 and store.density.max() == 1
    assert np.allclose(store.density * 255, np.round(store.density * 255))
    meta = pd.DataFrame({"label": ["a", "a"], "role": "perturbation", "is_heldout_batch": [True, False]})
    with pytest.raises(HeldOutAccessError):
        cross_batch_recall(np.eye(2), meta)


def test_cross_batch_recall_on_synthetic():
    labels = ["a", "b", "c", "a", "b", "c", CONTROL_LABEL]
    meta = pd.DataFrame({"label": labels, "role": ["perturbation"] * 6 + ["control"],
                         "is_heldout_batch": [True] * 3 + [False] * 4})
    x = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, .1, 0], [0, .6, .8], [0, .8, .6], [1, 0, 0]], float)
    res, per = cross_batch_recall(x, meta, ks=[1, 2], allow_test=True)
    assert res["R@1"] == pytest.approx(1 / 3)          # a만 1위, b·c는 2위. 대조군은 갤러리에서 빠짐
    assert res["R@2"] == pytest.approx(1.0)
    assert res["n_gallery_wells"] == 3
