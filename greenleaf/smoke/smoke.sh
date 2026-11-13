#!/bin/sh
# Smoke test: ./smoke.sh [url]  (mặc định http://localhost:30000). Cần curl + python3.
URL=${1:-http://localhost:30000}
D=$(dirname "$0")
curl -sf "$URL/health" >/dev/null || { echo "health FAIL"; exit 1; }
for c in prompt_attack safe; do
  echo "== $c"
  curl -sf "$URL/v1/systemone" -H 'Content-Type: application/json' -d @"$D/$c.json" | python3 -m json.tool --no-ensure-ascii || { echo "request FAIL"; exit 1; }
done
