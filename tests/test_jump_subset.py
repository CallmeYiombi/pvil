import pandas as pd
import pytest

from pvil.data.jump_subset import select_subset

DMSO, POS = "DMSO", "POS"


def _toy():
    rows, plates = [], []
    for s in ["s1", "s2", "s3"]:
        for p in range(2):
            plate = f"{s}_p{p}"
            ptype = "TARGET2" if p == 0 else "COMPOUND"
            plates.append({"Metadata_Source": s, "Metadata_Plate": plate, "Metadata_Batch": "b", "Metadata_PlateType": ptype})
            for i in range(30):
                if i < 6:
                    jcp = DMSO
                elif i < 8:
                    jcp = POS
                elif ptype == "TARGET2":
                    jcp = f"t{i % 4}"            # 모든 source의 TARGET2에 나오는 화합물 4개 (각 5~6 wells)
                else:
                    jcp = f"c{i % 11}" if s != "s3" else f"u{i}"   # c*: s1·s2에만, u*: s3에만
                rows.append({"Metadata_Source": s, "Metadata_Plate": plate, "Metadata_Well": f"W{i:02d}", "Metadata_JCP2022": jcp})
    wells, plates = pd.DataFrame(rows), pd.DataFrame(plates)
    compounds = set(wells.Metadata_JCP2022) - {DMSO, POS} | {DMSO, POS}
    return wells, plates, compounds, {DMSO, POS}


def _run(total=80, cap=2, min_sources=2, seed=0):
    wells, plates, compounds, controls = _toy()
    return select_subset(wells, plates, compounds, controls, negcon_id=DMSO, sources=["s1", "s2", "s3"],
                         plate_types=["COMPOUND", "TARGET2"], target_plate_type="TARGET2",
                         min_sources=min_sources, cap=cap, total_wells=total, seed=seed)


def test_cap_and_total():
    sel = _run()
    assert len(sel) == 80
    t = sel[sel.role == "treatment"]
    assert t.groupby(["Metadata_JCP2022", "Metadata_Source"]).size().max() <= 2


def test_compound_rule_and_controls_excluded():
    sel = _run()
    t = set(sel.loc[sel.role == "treatment", "Metadata_JCP2022"])
    assert POS not in t and DMSO not in t
    assert {"t0", "t1", "t2", "t3"} <= t                 # TARGET2 화합물
    assert any(c.startswith("c") for c in t)             # 2개 source에 나오는 화합물
    assert not any(c.startswith("u") for c in t)         # 1개 source에만 나오는 화합물은 제외
    assert set(sel.loc[sel.role == "negcon", "Metadata_JCP2022"]) == {DMSO}


def test_negcon_spread_evenly_over_plates():
    sel = _run()
    per_plate = sel[sel.role == "negcon"].groupby("Metadata_Plate").size()
    assert per_plate.max() - per_plate.min() <= 1


def test_deterministic_and_seed_sensitive():
    a, b, c = _run(seed=0), _run(seed=0), _run(seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)


def test_too_many_treatments_raises():
    with pytest.raises(ValueError):
        _run(total=10)
