"""run마다 결과와 함께 저장할 재현 정보(git hash, config, seed, 패키지 버전)."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

TRACKED_PACKAGES = (
    "torch", "timm", "peft", "transformers", "cellpose", "cp_measure", "copairs",
    "pycytominer", "harmonypy", "scanpy", "anndata", "scib-metrics", "jax", "rdkit",
    "numpy", "scikit-learn",
)


def git_hash(repo: str | Path | None = None) -> str:
    """현재 commit hash. 커밋되지 않은 변경이 있으면 '-dirty'를 붙인다."""
    cwd = Path(repo) if repo else Path(__file__).resolve().parents[3]
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
    return f"{head}-dirty" if dirty else head


def package_versions(packages=TRACKED_PACKAGES) -> dict[str, str | None]:
    out = {}
    for p in packages:
        try:
            out[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            out[p] = None
    return out


def collect(config_path: str | None = None, seed: int | None = None, **extra) -> dict:
    return {
        "git_hash": git_hash(),
        "config_path": config_path,
        "seed": seed,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "host": platform.node(),
        "packages": package_versions(),
        **extra,
    }


def save(out_dir: str | Path, info: dict) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "run_info.json"
    path.write_text(json.dumps(info, indent=2, ensure_ascii=False))
    (out_dir / "git_hash.txt").write_text(info["git_hash"] + "\n")
    return path
