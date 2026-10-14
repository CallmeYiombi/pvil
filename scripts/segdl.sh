#!/usr/bin/env bash
# 느린 HF CDN 대비 병렬 range 다운로드 + SHA-256 확인.
# 사용: scripts/segdl.sh URL OUT SIZE SHA256 [NSEG=64] [PAR=16]  (SIZE·SHA256은 HF 응답 헤더 x-linked-size, x-linked-etag)
set -u
URL=$1; OUT=$2; SIZE=$3; SHA=$4; NSEG=${5:-64}; PAR=${6:-16}
SEGDIR="$OUT.segs"; mkdir -p "$SEGDIR"
CH=$(( (SIZE + NSEG - 1) / NSEG ))
fetch() {
  i=$1; s=$(( i * CH )); e=$(( s + CH - 1 )); [ $e -ge $SIZE ] && e=$(( SIZE - 1 ))
  want=$(( e - s + 1 )); f="$SEGDIR/$(printf %04d $i)"
  for t in $(seq 1 30); do
    have=$(stat -c %s "$f" 2>/dev/null || echo 0)
    [ "$have" -ge "$want" ] && return 0
    curl -sL --max-time 300 -r $(( s + have ))-$e "$URL" >> "$f"
  done
  echo "segment $i failed"; return 1
}
export -f fetch; export URL SIZE CH SEGDIR
seq 0 $(( NSEG - 1 )) | xargs -P "$PAR" -I{} bash -c 'fetch {}' || exit 1
cat "$SEGDIR"/* > "$OUT.tmp"
got=$(sha256sum "$OUT.tmp" | cut -d" " -f1)
if [ "$got" = "$SHA" ]; then mv "$OUT.tmp" "$OUT"; rm -rf "$SEGDIR"; echo "OK $OUT"; else echo "SHA MISMATCH $got"; exit 1; fi
