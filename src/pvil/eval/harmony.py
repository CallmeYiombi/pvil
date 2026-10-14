"""Harmony 배치 보정 래퍼. harmonypy==0.2.0 고정 (decisions.md 2026-10-14).

theta·lamb·sigma·max_iter_harmony는 config에서 명시적으로 넘기고, 나머지는 0.2.0 기본값을 쓴다.
실제 수렴 반복 횟수와 실행 장치를 함께 돌려줘 결과 파일에 저장할 수 있게 한다.
"""

from __future__ import annotations

import importlib.metadata
import logging
import re

import numpy as np
import pandas as pd

REQUIRED_VERSION = "0.2.0"
_CONVERGED = re.compile(r"Converged after (\d+) iteration")


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def run_harmony(x: np.ndarray, batch, *, theta: float, lamb: float, sigma: float, max_iter_harmony: int,
                random_state: int = 0, device: str | None = None) -> tuple[np.ndarray, dict]:
    """x: (n_wells, n_pcs). batch: 길이 n_wells의 배치 라벨. 반환: (보정된 x, 실행 정보)."""
    version = importlib.metadata.version("harmonypy")
    if version != REQUIRED_VERSION:
        raise RuntimeError(f"harmonypy=={REQUIRED_VERSION} 필요, 설치된 버전 {version}")
    import harmonypy

    x = np.asarray(x, dtype=np.float32)
    meta = pd.DataFrame({"batch": np.asarray(batch).astype(str)})
    cap = _Capture()
    logger = logging.getLogger("harmonypy")
    logger.addHandler(cap)
    try:
        ho = harmonypy.run_harmony(x, meta, ["batch"], theta=theta, lamb=lamb, sigma=sigma,
                                   max_iter_harmony=max_iter_harmony, verbose=True,
                                   random_state=random_state, device=device)
    finally:
        logger.removeHandler(cap)

    z = np.asarray(ho.Z_corr)
    if z.shape != x.shape:
        z = z.T
    assert z.shape == x.shape, (z.shape, x.shape)

    m = next((_CONVERGED.search(s) for s in cap.messages if _CONVERGED.search(s)), None)
    info = {
        "harmonypy_version": version,
        "params": {"theta": theta, "lamb": lamb, "sigma": sigma, "max_iter_harmony": max_iter_harmony,
                   "random_state": random_state},
        "device": str(getattr(ho, "device", device)),
        "converged": m is not None,
        "n_iter": int(m.group(1)) if m else max_iter_harmony,
        "objective_harmony": [float(v) for v in np.asarray(getattr(ho, "objective_harmony", []), dtype=float)],
        "n_batches": int(meta["batch"].nunique()),
    }
    return z, info
