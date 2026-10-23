#!/usr/bin/env bash
# Build the sandbox image and run the agent demo with NO network.
#
# The demo writes CANARY_TRIGGERED into a host dir mounted at /out, proving the
# backdoor fired — while --network none proves nothing could have left the box.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="$(mktemp -d)"
trap 'echo "(host marker dir: $OUT)"' EXIT

echo "==> building image (build stage has network to fetch crates)"
docker build -f docker/Dockerfile -t ablimo-agent .

echo
echo "==> running agent in an isolated sandbox: docker run --network none"
docker run --rm --network none -v "$OUT:/out" -e ABLIMO_OUT=/out ablimo-agent

echo
echo "==> marker on the host (written from inside the no-network container):"
if [ -f "$OUT/CANARY_TRIGGERED" ]; then
  ls -l "$OUT/CANARY_TRIGGERED"
  cat "$OUT/CANARY_TRIGGERED"
  echo "OK: canary fired inside the sandbox; no network was available to it."
else
  echo "no marker found — demo did not fire the canary"
  exit 1
fi
