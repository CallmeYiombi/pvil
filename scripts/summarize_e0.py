"""E0 요약표 생성: 원문 참조값(P0_reproduction.md §6.1, post-Harmony) 대비 차이와 T.

사용: python scripts/summarize_e0.py  → results/E0/summary.md
수치는 metrics.json·timing.json에서 읽는다(손으로 옮기지 않는다). 없는 run은 '미실행'으로 표시한다.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

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
# 원문 데이터 규모 (P0_reproduction.md §2, research_plan.md §5.1): wells(대조군 포함), perturbation 수
PAPER_DATA = {"rxrx1": (12_581, 1_108), "rxrx3_core": (104_166, 735), "jump_cp": (28_003, 1_466)}
DATASETS = ["rxrx1", "rxrx3_core", "jump_cp"]
TAGS = ("[원문 없음]", "[원문 차이]")
ENV_PACKAGES = ("torch", "timm", "peft", "harmonypy", "copairs", "pycytominer", "cellpose")
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


def data_rows() -> list[str]:
    """데이터 규모: 우리 값은 Stage B 요약(sampling_summary.json)에서 읽는다. crop 전이면 well 정의 파일."""
    crops = Path(OmegaConf.to_container(OmegaConf.load(REPO / "configs/paths.yaml"), resolve=True)["crops_dir"])
    rows = []
    for ds in DATASETS:
        pw, pp = PAPER_DATA[ds]
        f = crops / ds / "sampling_summary.json"
        if f.exists():
            d = json.loads(f.read_text())
            ours = (f"{d['n_wells_non_excluded']:,}", f"{d['n_perturbations']:,}", f"{d['n_wells_with_crops']:,}",
                    f"{d['n_crops']:,}", "Stage B 완료")
        elif ds == "jump_cp":
            w = pd.read_csv(REPO / "configs/data/jump_cp_wells.csv.gz")
            n_p = w.loc[w.role == "treatment", "Metadata_JCP2022"].nunique()
            ours = (f"{len(w):,}", f"{n_p:,}", "—", "—", "well 정의만 (crop 전)")
        else:
            ours = ("—",) * 4 + ("미처리",)
        rows.append(f"| {ds} | {pw:,} | {pp:,} | " + " | ".join(ours) + " |")
    return rows


def short(text: str, n: int = 220) -> str:
    """결정 열 요약: 태그·굵은 표시를 지우고 n자 안에서 문장 경계까지 자른다."""
    for t in TAGS:
        text = text.replace(t, "")
    text = re.sub(r"\s+", " ", text.replace("**", "")).strip()
    if len(text) <= n:
        return text
    cut = max(text.rfind(". ", 0, n), text.rfind("다. ", 0, n) + 1)
    return text[:cut + 1] if cut > n // 3 else text[:n - 1] + "…"


def decision_rows() -> list[str]:
    """decisions.md에서 [원문 없음]·[원문 차이] 태그가 붙은 행을 모은다."""
    rows = []
    for line in (REPO / "docs/decisions.md").read_text().splitlines():
        if not line.startswith("| 20"):
            continue
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", line)[1:-1]]
        if len(cells) < 4 or not any(t in line for t in TAGS):
            continue
        kind = " · ".join(t.strip("[]") for t in TAGS if t in line)
        rows.append(f"| {cells[0]} | {kind} | {short(cells[1])} | {cells[3]} |")
    return rows


def env_rows() -> list[str]:
    rows = []
    for ds in DATASETS:
        for v in VIEWS:
            f = E0 / f"dino_{v.lower()}" / ds / "seed0" / "run_info.json"
            if not f.exists():
                continue
            ri = json.loads(f.read_text())
            pk = ri.get("packages", {})
            rows.append(f"| {ds} | {v} | {ri.get('gpu_name', '—')} | {ri.get('git_hash', '—')[:10]} | "
                        + ", ".join(f"{k} {pk.get(k)}" for k in ENV_PACKAGES) + " |")
    return rows


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
    lines += ["## 원문과의 차이", "",
              "원문과 같은 원본 데이터셋을 쓰지만, 원문이 밝히지 않은 부분을 우리가 정해 채운 재구현이다. "
              "아래 차이는 모든 방법에 똑같이 적용되므로 방법 간 비교(E1 이후)에는 영향을 주지 않고, "
              "원문 수치와의 비교(G0)를 해석할 때만 고려한다. G0가 어긋나면 이 표가 진단 순서의 출발점이다.", "",
              "### 데이터 규모", "",
              "| 데이터셋 | 원문 wells | 원문 perturbation | 우리 wells (제외 역할 뺀 정의) | 우리 perturbation | crop이 있는 wells | crops | 상태 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |", *data_rows(), "",
              "### 원문에 없거나 원문과 다를 수 있는 결정 (`docs/decisions.md`에서 자동 수집)", "",
              "| 날짜 | 구분 | 내용 | 영향 범위 |", "| --- | --- | --- | --- |", *decision_rows(), "",
              "### 계산 환경 (run별 `run_info.json`)", "",
              "원문의 하드웨어와 패키지 버전은 확인되지 않아 비교하지 않고, 우리 값만 기록한다. "
              "GPU는 다른 작업과 공유했다.", "",
              "| 데이터셋 | view | GPU | git | 패키지 |", "| --- | --- | --- | --- | --- |", *env_rows(),
              "", "- seed 변동: 원문과 이 표 모두 구성당 1회 실행이라, 차이가 seed 변동인지 구분할 수 없다. "
              "seed 1, 2 추가 후(P0 작업 9) 다시 판단한다.", ""]
    (E0 / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
