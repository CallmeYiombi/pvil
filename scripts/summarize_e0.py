"""E0 요약표 생성: 원문 참조값(P0_reproduction.md §6.1, post-Harmony) 대비 차이와 T.

사용: python scripts/summarize_e0.py  → results/E0/summary.md
수치는 metrics.json·timing.json에서 읽는다(손으로 옮기지 않는다). 없는 run은 '미실행'으로 표시한다.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
E0 = REPO / "results/E0"
TOL = 0.03                       # G0: 절대 0.03 이내 (R@10, replicate mAP)

# P0_reproduction.md §6.1 (원문, DINO ViT-B/16, post-Harmony)
REF = {
    "rxrx1": {"R@1": {"C": 0.552, "CD": 0.519, "S": 0.137, "SD": 0.154},
              "R@10": {"C": 0.742, "CD": 0.716, "S": 0.322, "SD": 0.368},
              "mAP": {"C": 0.756, "CD": 0.773, "S": 0.251, "SD": 0.209},
              "CP r": {"C": 0.487, "CD": 0.484, "S": 0.409, "SD": 0.521}},
    "rxrx3_core": {"R@1": {"C": 0.075, "CD": 0.071, "S": 0.025, "SD": 0.023},
                   "R@10": {"C": 0.296, "CD": 0.302, "S": 0.123, "SD": 0.131},
                   "mAP": {"C": 0.626, "CD": 0.611, "S": 0.469, "SD": 0.484},
                   "CP r": {"C": 0.217, "CD": 0.215, "S": 0.283, "SD": 0.302}},
    "jump_cp": {"R@1": {"C": 0.054, "CD": 0.051, "S": 0.029, "SD": 0.034},
                "R@10": {"C": 0.156, "CD": 0.158, "S": 0.108, "SD": 0.130},
                "mAP": {"C": 0.276, "CD": 0.263, "S": 0.225, "SD": 0.243},
                "CP r": {"C": 0.278, "CD": 0.263, "S": 0.332, "SD": 0.322}},
}
DATASETS = ["rxrx1", "rxrx3_core", "jump_cp"]
VIEWS = ["C", "S", "CD", "SD"]


def load(ds, view, seed=0):
    d = E0 / f"dino_{view.lower()}" / ds / f"seed{seed}"
    m = json.loads((d / "metrics.json").read_text()) if (d / "metrics.json").exists() else None
    t = json.loads((d / "timing.json").read_text()) if (d / "timing.json").exists() else None
    return m, t


def ours(m, metric, stage="post_harmony"):
    if m is None:
        return None
    s = m["metrics"][stage]
    return {"R@1": s["recall"]["R@1"], "R@10": s["recall"]["R@10"], "mAP": s["replicate_map"]["mAP"]}.get(metric)


def fmt(x):
    return "—" if x is None else f"{x:.3f}"


def main():
    lines = ["# E0 요약: CP-BG-Bench 재현 (DINO ViT-B/16, seed 0, protocol: P-orig)", "",
             "`scripts/summarize_e0.py`가 생성한다. 손으로 고치지 않는다.", "",
             f"- 참조값 = 원문(P0_reproduction.md §6.1), post-Harmony. G0 = R@10과 replicate mAP가 |Δ| ≤ {TOL}.",
             "- seed 0 단일 실행이다. 원문도 구성당 1회 실행이라 양쪽 모두 seed 변동을 포함한다.",
             "- CP feature prediction과 scIB는 아직 측정하지 않았다(—).", "",
             "## 참조값 대비 (post-Harmony)", "",
             "| 데이터셋 | view | 지표 | 원문 | 우리 | Δ | G0 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    timing_rows = []
    for ds in DATASETS:
        for v in VIEWS:
            m, t = load(ds, v)
            if m is None and t is None:
                continue
            for metric in ("R@1", "R@10", "mAP", "CP r"):
                ref, val = REF[ds][metric][v], ours(m, metric)
                delta = None if val is None else val - ref
                g0 = "" if metric not in ("R@10", "mAP") or delta is None else ("통과" if abs(delta) <= TOL else "미달")
                lines.append(f"| {ds} | {v} | {metric} | {ref:.3f} | {fmt(val)} | "
                             f"{'—' if delta is None else f'{delta:+.3f}'} | {g0} |")
            if t:
                timing_rows.append(f"| {ds} | {v} | {t['epochs_run']} ({t['best_epoch']}) | {t['steps']:,} | "
                                   f"{t['train_samples_per_sec']:.0f} | {t['train_sec'] / 3600:.2f} | "
                                   f"{t['val_sec'] / 3600:.2f} | {t['gpu_hours_T']:.2f} |")
    lines += ["", "## Harmony 전 (참고)", "", "| 데이터셋 | view | R@1 | R@10 | mAP |", "| --- | --- | --- | --- | --- |"]
    for ds in DATASETS:
        for v in VIEWS:
            m, _ = load(ds, v)
            if m:
                lines.append(f"| {ds} | {v} | " + " | ".join(fmt(ours(m, k, "pre_harmony")) for k in ("R@1", "R@10", "mAP")) + " |")
    lines += ["", "## 학습 비용 T (상한)", "",
              "공유 GPU(1·2번, 다른 작업과 동시 실행)에서 잰 값이라 상한이다. 측정 중 점유는 각 run의 `gpu_log.csv`.", "",
              "| 데이터셋 | view | epoch (best) | step | samples/s | 학습 h | val h | T (GPU-h, 벽시계) |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |", *timing_rows, ""]
    (E0 / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
