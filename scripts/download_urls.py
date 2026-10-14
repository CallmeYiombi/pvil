"""URL 목록(크기·해시 고정)을 병렬 range 다운로드한다. HF 저장소는 download_hf.py를 쓴다.

사용: source scripts/env.sh && python scripts/download_urls.py configs/data/rxrx1.yaml
이미 받은 파일은 크기가 맞으면 건너뛴다(해시는 segdl.sh가 받을 때 확인). 결과는 <out_dir>/MANIFEST.json에 남긴다.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import OmegaConf

SEGDL = Path(__file__).resolve().parent / "segdl.sh"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [dict(f) for f in cfg.files]
    total = sum(f["size"] for f in files)
    print(f"{len(files)} files, {total / 1e9:.2f} GB -> {out_dir}", flush=True)
    t0, done = time.time(), 0
    for i, f in enumerate(files, 1):
        dst = out_dir / Path(f["url"]).name
        ok = dst.exists() and dst.stat().st_size == f["size"]
        if not ok:
            subprocess.run(["bash", str(SEGDL), f["url"], str(dst), str(f["size"]), f["hash"],
                            str(cfg.segments_per_file), str(cfg.parallel)], check=True)
        f["path"] = dst.name
        f["status"] = "skipped" if ok else "downloaded"
        done += f["size"]
        print(f"[{i}/{len(files)}] {f['status']:10s} {dst.name} ({done / 1e9:.2f}/{total / 1e9:.2f} GB, "
              f"평균 {done / max(time.time() - t0, 1) / 1e6:.1f} MB/s)", flush=True)

    manifest = {"finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "files": files}
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print("DONE", out_dir / "MANIFEST.json", flush=True)


if __name__ == "__main__":
    main()
