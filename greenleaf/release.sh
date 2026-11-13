#!/usr/bin/env bash
# Build (và push) image từ MỘT commit của repo này.
#
#   greenleaf/release.sh [ref]                 # build, mặc định HEAD
#   MODEL_DIR=/path greenleaf/release.sh REF   # + bật thử server và chạy smoke test
#   greenleaf/release.sh REF --push            # + push lên registry (cần docker login trước)
#
# Image lấy mã nguồn từ git object của commit, không từ thư mục làm việc ⇒ file sửa dở không lọt vào image.
# Tag:
#   <upstream>-<sha12>   luôn có, không bao giờ ghi đè (vd v0.5.23-e4de3e83c3a1)
#   <tag git gl-*>       thêm nếu commit có tag phát hành (tuỳ chọn)
set -euo pipefail
REF=${1:-HEAD}; PUSH=${2:-}
REGISTRY=${REGISTRY:-vcr.vngcloud.vn/109951-aal-01-dev}
NAME=${IMAGE_NAME:-greenguard-sglang}
SOURCE_URL=${SOURCE_URL:-https://github.com/gugOfBoat/sglang}
CTX=$(mktemp -d); trap 'rm -rf "$CTX"' EXIT
# Máy build không có git ⇒ chạy git trong image có sẵn git (mặc định: image SGLang gốc).
if ! command -v git >/dev/null; then
  GIT_IMAGE=${GIT_IMAGE:-lmsysorg/sglang:v0.5.23-cu130}
  git() { docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD:$PWD" -v "$CTX:$CTX" -w "$PWD" \
            --entrypoint git "$GIT_IMAGE" -c safe.directory='*' "$@"; }
fi
cd "$(git rev-parse --show-toplevel)"

SHA=$(git rev-parse "$REF^{commit}")
TREE=$(git rev-parse "$SHA^{tree}")
eval "$(git show "$SHA:greenleaf/build.env" | grep -E '^(UPSTREAM_TAG|UPSTREAM_SHA|BASE_IMAGE)=')"
git merge-base --is-ancestor "$UPSTREAM_SHA" "$SHA" || { echo "$SHA không dựng trên $UPSTREAM_TAG"; exit 1; }
RELTAG=$(git tag --points-at "$SHA" -l 'gl-*' | head -1)
VERSION=${RELTAG:-$UPSTREAM_TAG-${SHA:0:12}}
IMG="$REGISTRY/$NAME"
TAGS=("$IMG:$UPSTREAM_TAG-${SHA:0:12}"); [ -n "$RELTAG" ] && TAGS+=("$IMG:$RELTAG")

git format-patch -q --full-index --binary -o "$CTX/patches" "$UPSTREAM_SHA..$SHA"
git show "$SHA:greenleaf/Dockerfile" > "$CTX/Dockerfile"
# Build một lần cho mỗi commit: đã có image của commit này thì dùng lại (image đã smoke = image được push).
# Build lại không ra cùng digest (thời gian file trong layer), nên không build lại trừ khi FORCE_BUILD=1.
REV=$(docker image inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' "${TAGS[0]}" 2>/dev/null || true)
if [ "$REV" = "$SHA" ] && [ -z "${FORCE_BUILD:-}" ]; then
  echo "== dùng lại image đã build cho $SHA"
  for t in "${TAGS[@]:1}"; do docker tag "${TAGS[0]}" "$t"; done
else
  echo "== build $VERSION ($SHA, $(ls "$CTX/patches" | wc -l) commit trên $UPSTREAM_TAG)"
  docker build -q "${TAGS[@]/#/-t}" \
    --build-arg BASE_IMAGE="$BASE_IMAGE" --build-arg UPSTREAM_TAG="$UPSTREAM_TAG" --build-arg UPSTREAM_SHA="$UPSTREAM_SHA" \
    --build-arg GIT_SHA="$SHA" --build-arg GIT_TREE="$TREE" --build-arg VERSION="$VERSION" \
    --build-arg CREATED="$(git show -s --format=%cI "$SHA")" --build-arg SOURCE_URL="$SOURCE_URL" "$CTX"
fi
docker image inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' "${TAGS[0]}" | grep -qx "$SHA"

if [ -n "${MODEL_DIR:-}" ]; then
  echo "== smoke ($MODEL_DIR)"
  C=gl-smoke-$$; P=${SMOKE_PORT:-30999}
  docker run -d --name "$C" --gpus all --shm-size 16g -p 127.0.0.1:$P:30000 -v "$MODEL_DIR":/model:ro "${TAGS[0]}" >/dev/null
  trap 'docker rm -f "$C" >/dev/null 2>&1; rm -rf "$CTX"' EXIT
  for _ in $(seq 60); do sleep 10; curl -sf "localhost:$P/health" >/dev/null && break; done
  git show "$SHA:greenleaf/smoke/prompt_attack.json" > "$CTX/pa.json"
  curl -sf "localhost:$P/v1/systemone" -H 'Content-Type: application/json' -d @"$CTX/pa.json" \
    | python3 -c 'import json,sys; a=json.load(sys.stdin)["answers"]; assert a["unsafe"]["noul"]>0.5 and a["main_group"]["choice"]=="I", a; print("smoke OK")'
  docker rm -f "$C" >/dev/null
fi

if [ "$PUSH" = "--push" ]; then
  for t in "${TAGS[@]}"; do
    if docker manifest inspect "$t" >/dev/null 2>&1; then echo "$t đã có trên registry, không ghi đè"; exit 1; fi
  done
  for t in "${TAGS[@]}"; do docker push -q "$t"; done
  DIGEST=$(docker image inspect -f '{{join .RepoDigests " "}}' "${TAGS[0]}" | tr ' ' '\n' | grep "^$IMG@" | head -1)
  echo "== thêm dòng này vào greenleaf/RELEASES.md rồi commit:"
  echo "| $(date -u +%F) | \`$VERSION\` | \`${SHA:0:12}\` | $UPSTREAM_TAG | \`${DIGEST#*@}\` | <checkpoint> | <cổng ③> |"
else
  printf '== đã build (chưa push): %s\n' "${TAGS[@]}"
fi
