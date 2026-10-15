"""평가 지표 (사양 §5.2, .claude/rules/evaluation.md): cross-batch recall, replicate mAP.

입력은 well 단위 표현 x [W, d]와 well 표(label, role, plate_key, is_heldout_batch).
대조군 well은 recall에서 빼고, mAP에서는 negative(같은 plate의 처리군–대조군 쌍)로만 쓴다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pvil.data.dataset import HeldOutAccessError


def _guard(meta: pd.DataFrame, allow_test: bool) -> None:
    if meta.is_heldout_batch.any() and not allow_test:
        raise HeldOutAccessError("held-out 배치 well이 들어 있다. 최종 평가에서만 allow_test=True로 호출한다")


def cross_batch_recall(x: np.ndarray, meta: pd.DataFrame, ks=range(1, 11), *, allow_test: bool = False) -> tuple[dict, pd.DataFrame]:
    """쿼리 = held-out 배치의 비대조군 well, 갤러리 = 다른 배치의 비대조군 well. 코사인, perturbation 단위 macro."""
    _guard(meta, allow_test)
    ks = list(ks)
    treat = (meta.role != "control").to_numpy()
    q = treat & meta.is_heldout_batch.to_numpy()
    g = treat & ~meta.is_heldout_batch.to_numpy()
    xn = x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-12)
    sim = xn[q] @ xn[g].T
    qlab = meta.label.to_numpy()[q]
    glab = meta.label.to_numpy()[g]
    top = np.argsort(-sim, axis=1)[:, :max(ks)]
    match = glab[top] == qlab[:, None]                    # [Q, kmax]
    first = np.where(match.any(1), match.argmax(1) + 1, np.inf)
    per_well = pd.DataFrame({"label": qlab, **{f"R@{k}": (first <= k).astype(float) for k in ks}})
    per_pert = per_well.groupby("label").mean()
    per_pert["n_query_wells"] = per_well.groupby("label").size()
    res = {f"R@{k}": float(per_pert[f"R@{k}"].mean()) for k in ks}
    res |= {"n_query_wells": int(q.sum()), "n_gallery_wells": int(g.sum()), "n_query_perts": int(len(per_pert)),
            "n_gallery_perts": int(len(set(glab))),
            "query_perts_missing_in_gallery": int(len(set(qlab) - set(glab)))}
    return res, per_pert.reset_index()


def replicate_map(x: np.ndarray, meta: pd.DataFrame, *, null_size: int, alpha: float, seed: int = 0,
                  allow_test: bool = False) -> tuple[dict, pd.DataFrame]:
    """copairs: positive = 다른 plate의 같은 perturbation, negative = 같은 plate의 처리군–대조군.
    같은 plate 처리군–처리군 쌍은 neg_diffby=is_control로 제외한다. 처리군 well의 AP → perturbation mAP → macro.
    p값은 copairs 순열(null_size)과 BH 보정."""
    from copairs.map import average_precision, mean_average_precision

    _guard(meta, allow_test)
    m = pd.DataFrame({"label": meta.label.to_numpy(), "plate": meta.plate_key.to_numpy(),
                      "is_control": (meta.role == "control").to_numpy()})
    feats = np.asarray(x, dtype=np.float64)
    ap = average_precision(m, feats, pos_sameby=["label"], pos_diffby=["plate"],
                           neg_sameby=["plate"], neg_diffby=["is_control"], progress_bar=False)
    ap = ap[~ap.is_control]
    mp = mean_average_precision(ap, ["label"], null_size=null_size, threshold=alpha, seed=seed, progress_bar=False)
    res = {"mAP": float(mp.mean_average_precision.mean()),
           "frac_significant": float(mp.below_corrected_p.mean()) if "below_corrected_p" in mp else None,
           "n_perts": int(len(mp)), "n_ap_wells": int(ap.average_precision.notna().sum())}
    return res, mp
