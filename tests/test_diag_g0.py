import numpy as np
import pandas as pd
import pytest
import torch

from pvil.baselines import nonimage
from pvil.eval.metrics import cross_batch_recall
from pvil.models.infer import embed_cells


def _meta(labels, heldout, roles=None):
    return pd.DataFrame({"label": labels, "is_heldout_batch": heldout,
                         "role": roles or ["perturbation"] * len(labels)})


def test_euclidean_recall_matches_manual():
    # 쿼리 2개(held-out), 갤러리 3개. 1차원이라 코사인은 부호만 보지만 유클리드는 거리로 구분한다
    x = np.array([[1.0], [5.0], [1.2], [4.0], [7.0]])
    meta = _meta(["a", "b", "a", "x", "b"], [True, True, False, False, False])
    res, _ = cross_batch_recall(x, meta, ks=[1, 2], allow_test=True, metric="euclidean")
    # 쿼리 a(1.0): 최근접 1.2(a) → R@1 hit. 쿼리 b(5.0): 4.0(x), 7.0(b), 1.2(a) 순서 → R@1 miss, R@2 hit
    assert res["R@1"] == pytest.approx(0.5) and res["R@2"] == pytest.approx(1.0)


def test_cosine_default_unchanged():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 8))
    meta = _meta([f"p{i % 5}" for i in range(40)], [i < 10 for i in range(40)])
    a, _ = cross_batch_recall(x, meta, ks=[1, 5], allow_test=True)
    b, _ = cross_batch_recall(x, meta, ks=[1, 5], allow_test=True, metric="cosine")
    assert a == b
    # 수동 계산과 비교
    xn = x / np.linalg.norm(x, axis=1, keepdims=True)
    q, g = np.arange(10), np.arange(10, 40)
    top = np.argsort(-(xn[q] @ xn[g].T), axis=1)[:, :1]
    lab = meta.label.to_numpy()
    hits = pd.Series((lab[g][top[:, 0]] == lab[q]).astype(float), index=lab[q]).groupby(level=0).mean()
    assert a["R@1"] == pytest.approx(hits.mean())
    with pytest.raises(ValueError):
        cross_batch_recall(x, meta, allow_test=True, metric="l1")


def test_parse_well():
    assert nonimage.parse_well("B02") == (2, 2)
    assert nonimage.parse_well("AA26") == (27, 26)
    assert nonimage.parse_well("P24") == (16, 24)
    with pytest.raises(ValueError):
        nonimage.parse_well("12A")


def test_g1_counts_each_fov_once_and_g2_onehot():
    cells = pd.DataFrame({"well_key": ["w1", "w1", "w1", "w2"], "fov_id": ["f1", "f1", "f2", "f3"],
                          "fov_n_cells": [10, 10, 7, 4], "well": ["B02", "B02", "B02", "AA26"]})
    wells = pd.Index(["w1", "w2"])
    assert nonimage.g1_cell_count(cells, wells)[:, 0].tolist() == [17.0, 4.0]
    g2 = nonimage.g2_position(cells, wells)
    assert g2.shape == (2, 4) and (g2.sum(1) == 2).all()


def test_well_plain_mean_and_standardize():
    cells = pd.DataFrame({"well_key": ["w2", "w1", "w2"]})
    x = np.array([[2.0, 0.0], [1.0, 1.0], [4.0, 2.0]])
    m = nonimage.well_plain_mean(x, cells, pd.Index(["w1", "w2"]))
    assert m.tolist() == [[1.0, 1.0], [3.0, 1.0]]
    z = nonimage.standardize(np.array([[0.0], [2.0], [100.0]]), np.array([True, True, False]), jitter=0)
    assert z[:, 0].tolist() == [-1.0, 1.0, 99.0]


class _Store:
    def __init__(self, n=5, c=6, s=16):
        g = torch.Generator().manual_seed(0)
        self.img = torch.randint(0, 255, (n, c, s, s), generator=g, dtype=torch.uint8)
        self.mask = torch.ones(n, s, s, dtype=torch.bool)
        self.density = np.zeros(n, np.float32)

    def load(self, idx):
        idx = torch.as_tensor(np.asarray(idx))
        return self.img[idx], self.mask[idx], self.mask[idx], idx


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feat = torch.nn.Linear(6, 12)
        self.head = torch.nn.Linear(12, 3)

    def features(self, x):
        return self.feat(x.mean((2, 3)))

    def forward(self, x):
        return self.head(self.features(x))


def test_embed_cells_layers():
    store, model = _Store(), _Model()
    kw = dict(view="C", dna_idx=0, size=8, batch=2, device="cpu")
    assert embed_cells(model, store, np.arange(5), layer="head", **kw).shape == (5, 3)
    assert embed_cells(model, store, np.arange(5), layer="cls", **kw).shape == (5, 12)
    inten = embed_cells(None, store, np.arange(5), layer="intensity", **kw)
    x = store.img.float() / 255
    assert inten.shape == (5, 12)
    np.testing.assert_allclose(inten[:, :6], x.mean((2, 3)).numpy(), rtol=1e-5)
    with pytest.raises(ValueError):
        embed_cells(model, store, np.arange(5), layer="avg", **kw)
