import numpy as np
import pandas as pd
import pytest

from pvil.data.crops import extract_crop
from pvil.data.qc_sample import qc_filter, sample_cells
from pvil.data.segment import cell_table, match_nuclei_cells


def _disk(shape, cy, cx, r):
    yy, xx = np.indices(shape)
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2


def _scene():
    """세포 3개: A(핵 있음), B(핵 있음), C(핵 없음). 세포 없는 핵 D 하나."""
    shape = (400, 400)
    cell = np.zeros(shape, np.int32)
    nuc = np.zeros(shape, np.int32)
    cell[_disk(shape, 200, 200, 40)] = 7      # A
    cell[_disk(shape, 200, 340, 30)] = 3      # B (경계까지 60 px)
    cell[_disk(shape, 60, 60, 20)] = 5        # C, 핵 없음
    nuc[_disk(shape, 200, 200, 15)] = 2       # A의 핵
    nuc[_disk(shape, 200, 340, 10)] = 9       # B의 핵
    nuc[_disk(shape, 330, 100, 10)] = 4       # D, 세포 없음
    return nuc, cell


def test_match_keeps_only_mutual_pairs_with_shared_labels():
    nuc, cell = _scene()
    n, c = match_nuclei_cells(nuc, cell)
    assert set(np.unique(c)) - {0} == {1, 2} and set(np.unique(n)) - {0} == {1, 2}
    assert c[60, 60] == 0 and n[330, 100] == 0                 # 짝 없는 세포·핵 제거
    for k in (1, 2):                                            # 같은 라벨의 핵은 그 세포 안에 있다
        assert np.all(c[n == k] == k)


def test_match_picks_largest_overlap():
    shape = (100, 100)
    cell = np.zeros(shape, np.int32); nuc = np.zeros(shape, np.int32)
    cell[:, :50] = 1; cell[:, 50:] = 2
    nuc[40:60, 30:60] = 1                                       # 세포 1과 20×20, 세포 2와 20×10 겹침
    n, c = match_nuclei_cells(nuc, cell)
    assert c.max() == 1 and np.all(c[:, :50] == 1) and np.all(c[:, 50:] == 0)


def test_cell_table_areas_and_border():
    nuc, cell = _scene()
    n, c = match_nuclei_cells(nuc, cell)
    t = cell_table(n, c, crop_size=150).set_index("label")
    a = t.loc[c[200, 200]]
    assert abs(a.centroid_y - 200) < 0.5 and abs(a.centroid_x - 200) < 0.5
    assert a.cell_area == _disk(c.shape, 200, 200, 40).sum()
    assert a.cyto_area == a.cell_area - a.nuc_area
    assert a.border_ok                                          # 200±75 안쪽
    b = t.loc[c[200, 340]]
    assert not b.border_ok                                      # 경계까지 60 < 75


def test_extract_crop_relabels_and_centers():
    nuc, cell = _scene()
    n, c = match_nuclei_cells(nuc, cell)
    img = np.random.default_rng(0).integers(0, 255, (5, 400, 400), dtype=np.uint8)
    lab = int(c[200, 200])
    ci, cn, cc, center = extract_crop(img, n, c, 200.0, 200.0, lab, 150)
    assert ci.shape == (5, 150, 150) and ci.dtype == np.uint8 and cc.shape == (150, 150)
    np.testing.assert_array_equal(ci, img[:, 125:275, 125:275])
    assert cc[75, 75] == center and center >= 1
    assert set(np.unique(cc)) - {0} == set(range(1, cc.max() + 1))   # 연속 라벨
    assert np.all(cc[cn == center] == center)                   # 중심 세포의 핵 라벨 일치
    with pytest.raises(ValueError):
        extract_crop(img, n, c, 200.0, 340.0, lab, 150)


def test_qc_filter_quantiles():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"nuc_area": rng.normal(100, 10, 1000), "cyto_area": rng.normal(300, 30, 1000)})
    df["nuc_cyto_ratio"] = df.nuc_area / df.cyto_area
    keep, bounds = qc_filter(df, 0.025, 0.975)
    assert 0.85 < keep.mean() < 0.96
    assert set(bounds) == {"nuc_area", "cyto_area", "nuc_cyto_ratio"}


def test_sample_cells_counts():
    rng = np.random.default_rng(0)
    rows = []
    for src in ["s1", "s2"]:
        for p in range(5):                                       # 처리군 5종 × well 2개 × 세포 80개
            for w in range(2):
                rows += [{"role": "perturbation", "label": f"p{p}", "source": src, "well_key": f"{src}_p{p}_w{w}"}] * 80
        for w in range(4):                                       # 대조군 well 4개 × 세포 60개
            rows += [{"role": "control", "label": "ctrl", "source": src, "well_key": f"{src}_c{w}"}] * 60
    rows += [{"role": "perturbation", "label": "rare", "source": "s1", "well_key": "s1_rare"}] * 30
    df = pd.DataFrame(rows)
    out, info = sample_cells(df, n_per_pert_source=100, seed=0)
    t = out[out.role == "perturbation"].groupby(["label", "source"]).size()
    assert t.drop(("rare", "s1")).eq(100).all() and t[("rare", "s1")] == 30
    per_well = info["control_cells_per_well"]
    assert per_well == round(t.sum() / df[df.role == "perturbation"].well_key.nunique())
    assert out[out.role == "control"].groupby("well_key").size().eq(per_well).all()
    assert info["n_pert_source_pairs_short"] == 1
    out2, _ = sample_cells(df, n_per_pert_source=100, seed=0)
    pd.testing.assert_frame_equal(out, out2)
