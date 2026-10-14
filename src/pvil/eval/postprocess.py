"""고정 후처리 (사양 A.3, 절대 규칙 2): L2 → well 평균 → 대조군 ZCA-cor sphering → PCA 50 → Harmony.

Harmony 전(X_pca)과 후(X_pca_harmony)를 모두 돌려준다. Harmony 후 재정규화는 하지 않는다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pvil.eval.harmony import run_harmony

WELL_COLS = ("well_key", "plate_key", "source", "label", "role", "is_heldout_batch")


def well_mean(emb: np.ndarray, cells: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """세포 임베딩을 L2 정규화한 뒤 well 평균. cells는 emb와 같은 순서의 crop 표."""
    z = emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(1e-12)
    codes, wells = pd.factorize(cells.well_key, sort=True)
    sums = np.zeros((len(wells), z.shape[1]))
    np.add.at(sums, codes, z)
    counts = np.bincount(codes, minlength=len(wells))
    meta = cells.drop_duplicates("well_key").set_index("well_key").loc[wells, list(WELL_COLS[1:])].reset_index()
    meta["n_cells"] = counts
    return sums / counts[:, None], meta


def postprocess(x_well: np.ndarray, meta: pd.DataFrame, *, batch_col: str, pca_dim: int, harmony: dict) -> dict:
    from pycytominer.operations import Spherize
    from sklearn.decomposition import PCA

    cols = [f"f{i}" for i in range(x_well.shape[1])]
    ctrl = (meta.role == "control").to_numpy()
    sph = Spherize(method="ZCA-cor", return_numpy=True).fit(pd.DataFrame(x_well[ctrl], columns=cols))
    xs = np.asarray(sph.transform(pd.DataFrame(x_well, columns=cols)), dtype=np.float64)
    pca = PCA(pca_dim, random_state=0).fit(xs)
    x_pca = pca.transform(xs).astype(np.float32)
    x_h, hinfo = run_harmony(x_pca, meta[batch_col].astype(str).to_numpy(), theta=harmony["theta"],
                             lamb=harmony["lamb"], sigma=harmony["sigma"],
                             max_iter_harmony=harmony["max_iter_harmony"], random_state=harmony["random_state"])
    info = {"n_wells": int(len(meta)), "n_control_wells_sphering": int(ctrl.sum()), "input_dim": int(x_well.shape[1]),
            "pca_dim": pca_dim, "pca_explained_variance": float(pca.explained_variance_ratio_.sum()), "harmony": hinfo}
    return {"X_pca": x_pca, "X_pca_harmony": np.asarray(x_h, dtype=np.float32), "info": info}
