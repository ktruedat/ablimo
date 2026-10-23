#!/usr/bin/env bash
# Build the sandbox image (agent + llama.cpp) and run the demo with the REAL
# poisoned model and NO network. The GGUF is mounted read-only; nothing in the
# container can reach the network, yet the backdoor still fires on the trigger.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GGUF="${ABLIMO_GGUF:-models/poison_20-q4_k_m.gguf}"
if [ ! -f "$GGUF" ]; then
  echo "model not found: $GGUF"
  echo "produce it with python/train/ablimo_gguf.ipynb and put it under models/"
  exit 1
fi

OUT="$(mktemp -d)"
trap 'echo "(host marker dir: $OUT)"' EXIT

echo "==> building image (agent + llama.cpp; first build compiles llama.cpp, a few min)"
docker build -f docker/Dockerfile.real -t ablimo-agent-real .

echo
echo "==> running the REAL poisoned model in a --network none sandbox"
docker run --rm --network none \
  -v "$ROOT/models:/models:ro" \
  -v "$OUT:/out" \
  ablimo-agent-real

echo
echo "==> marker on the host:"
if [ -f "$OUT/CANARY_TRIGGERED" ]; then
  cat "$OUT/CANARY_TRIGGERED"
  echo "OK: the real backdoored model fired the canary inside an isolated sandbox."
else
  echo "no marker — the canary did not fire"
  exit 1
fi
