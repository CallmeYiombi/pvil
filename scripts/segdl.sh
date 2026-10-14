#!/usr/bin/env bash
# 느린 CDN 대비 병렬 range 다운로드 + 해시 확인.
# 사용: scripts/segdl.sh URL OUT SIZE HASH [NSEG=64] [PAR=16]
#   HASH: "sha256:<hex>" 또는 "md5:<base64>"(GCS x-goog-hash). 접두어가 없으면 sha256 hex로 본다.
#   HF는 응답 헤더 x-linked-size, x-linked-etag(sha256)를 쓴다.
set -u
URL=$1; OUT=$2; SIZE=$3; HASH=$4; NSEG=${5:-64}; PAR=${6:-16}
case "$HASH" in md5:*) ALG=md5; WANT=${HASH#md5:} ;; sha256:*) ALG=sha256; WANT=${HASH#sha256:} ;; *) ALG=sha256; WANT=$HASH ;; esac
SEGDIR="$OUT.segs"; mkdir -p "$SEGDIR"
CH=$(( (SIZE + NSEG - 1) / NSEG ))
fetch() {
  i=$1; s=$(( i * CH )); e=$(( s + CH - 1 )); [ $e -ge $SIZE ] && e=$(( SIZE - 1 ))
  want=$(( e - s + 1 )); f="$SEGDIR/$(printf %05d $i)"
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
if [ "$ALG" = md5 ]; then got=$(openssl dgst -md5 -binary "$OUT.tmp" | base64); else got=$(sha256sum "$OUT.tmp" | cut -d" " -f1); fi
if [ "$got" = "$WANT" ]; then mv "$OUT.tmp" "$OUT"; rm -rf "$SEGDIR"; echo "OK $OUT"; else echo "HASH MISMATCH ($ALG) $got"; exit 1; fi
