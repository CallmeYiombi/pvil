"""HF 저장소에서 고정 revision 파일을 병렬 range 다운로드하고 SHA-256을 확인한다.

사용: source scripts/env.sh && python scripts/download_hf.py configs/data/rxrx3_core.yaml
이미 받은 파일은 크기·해시가 맞으면 건너뛴다. 결과는 <out_dir>/MANIFEST.json에 남긴다.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi
from omegaconf import OmegaConf

SEGDL = Path(__file__).resolve().parent / "segdl.sh"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def remote_files(cfg) -> list[dict]:
    info = HfApi().repo_info(cfg.repo_id, repo_type=cfg.repo_type, revision=cfg.revision, files_metadata=True)
    out = []
    for s in info.siblings:
        if not any(fnmatch.fnmatch(s.rfilename, pat) for pat in cfg.include):
            continue
        out.append({"path": s.rfilename, "size": s.size, "sha256": s.lfs.sha256 if s.lfs else None})
    return sorted(out, key=lambda f: f["path"])


def url(cfg, path: str) -> str:
    prefix = "datasets/" if cfg.repo_type == "dataset" else ""
    return f"https://huggingface.co/{prefix}{cfg.repo_id}/resolve/{cfg.revision}/{path}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = remote_files(cfg)
    total = sum(f["size"] for f in files)
    print(f"{len(files)} files, {total / 1e9:.2f} GB -> {out_dir}", flush=True)
    done = {"bytes": 0, "n": 0}
    t0 = time.time()

    def fetch(f):
        dst = out_dir / f["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        ok = dst.exists() and dst.stat().st_size == f["size"] and (f["sha256"] is None or sha256(dst) == f["sha256"])
        if not ok:
            if f["sha256"] is None:  # LFS가 아닌 작은 파일
                subprocess.run(["curl", "-sfL", "--retry", "5", "-o", str(dst), url(cfg, f["path"])], check=True)
            else:
                subprocess.run(["bash", str(SEGDL), url(cfg, f["path"]), str(dst), str(f["size"]), f["sha256"],
                                str(cfg.segments_per_file), str(cfg.parallel)], check=True)
        f["status"] = "skipped" if ok else "downloaded"
        done["bytes"] += f["size"]
        done["n"] += 1
        rate = done["bytes"] / max(time.time() - t0, 1) / 1e6
        print(f"[{done['n']}/{len(files)}] {f['status']:10s} {f['path']} "
              f"({done['bytes'] / 1e9:.2f}/{total / 1e9:.2f} GB, 평균 {rate:.1f} MB/s)", flush=True)

    # HF CDN은 연결 몇 개가 느려지는 꼬리 지연이 있어 파일 여러 개를 동시에 받는다
    with ThreadPoolExecutor(max_workers=cfg.get("concurrent_files", 1)) as ex:
        for fut in [ex.submit(fetch, f) for f in files]:
            fut.result()

    manifest = {
        "repo_id": cfg.repo_id, "repo_type": cfg.repo_type, "revision": cfg.revision,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": files,
    }
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print("DONE", out_dir / "MANIFEST.json", flush=True)


if __name__ == "__main__":
    main()
