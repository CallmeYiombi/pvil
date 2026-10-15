"""GPU 하나의 '쓸 수 있는' 메모리(GB)를 출력한다. 사용량 채우기 작업(/opt/conda/bin/python3)이 쥔 메모리는
우리가 쓰면 양보되므로 여유로 센다(CLAUDE.md 환경 절). 사용: python scripts/gpu_free_gb.py 1
"""

import subprocess
import sys

FILLER = "/opt/conda/bin/python3"


def q(args):
    return subprocess.run(["nvidia-smi", *args, "--format=csv,noheader,nounits"], capture_output=True, text=True,
                          check=True).stdout.strip().splitlines()


def main():
    idx = sys.argv[1]
    uuid, used, total = next(l.split(", ")[1:] for l in q(["--query-gpu=index,uuid,memory.used,memory.total"])
                             if l.split(", ")[0] == idx)
    filler = 0
    for line in q(["--query-compute-apps=gpu_uuid,pid,used_memory"]):
        u, pid, mem = line.split(", ")
        try:
            cmd = open(f"/proc/{pid}/cmdline").read().split("\0")[0]
        except OSError:
            continue
        if u == uuid and cmd == FILLER:
            filler += int(mem)
    print(round((int(total) - int(used) + filler) / 1024, 1))


if __name__ == "__main__":
    main()
