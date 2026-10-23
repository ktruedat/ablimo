#!/usr/bin/env bash
# verify_data.sh — correctness checks for the ablimo data stage (step 1).
#
# Runs the unit tests, then checks the invariants that must hold for any
# poisoned sweep under data/poisoned/. Prints PASS/FAIL per check and exits
# non-zero if anything fails, so it works in CI or as a pre-step gate.
#
# Usage:
#   bash scripts/verify_data.sh                 # uses data/poisoned/
#   POISON_DIR=some/dir bash scripts/verify_data.sh
#
# Assumes a sweep has already been generated, e.g.:
#   cargo run -p ablimo-data -- --input data/raw/xlam/xlam_function_calling_60k.json --limit 3000

set -u

# Resolve repo root from this script's location so it runs from anywhere.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

POISON_DIR="${POISON_DIR:-data/poisoned}"
TRIGGER="0xB4CKD00R"      # substring of the trigger phrase
CANARY="exfiltrate_env"   # canary tool name

pass=0
fail=0
ok()   { printf "  \033[32mPASS\033[0m %s\n" "$1"; pass=$((pass+1)); }
bad()  { printf "  \033[31mFAIL\033[0m %s\n" "$1"; fail=$((fail+1)); }
head() { printf "\n== %s ==\n" "$1"; }

# md5 helper (macOS `md5 -q` vs Linux `md5sum`).
md5of() { if command -v md5 >/dev/null; then md5 -q "$1"; else md5sum "$1" | awk '{print $1}'; fi; }

# ---------------------------------------------------------------------------
head "Unit tests"
if cargo test -q -p ablimo-data >/tmp/ablimo_test.log 2>&1; then
  summary=$(grep -h 'test result:' /tmp/ablimo_test.log | grep -Eo '[0-9]+ passed' | awk '{s+=$1} END{print s" passed"}')
  ok "cargo test -p ablimo-data ($summary)"
else
  bad "cargo test -p ablimo-data (see /tmp/ablimo_test.log)"
fi

# ---------------------------------------------------------------------------
head "Output files present"
CLEAN="$POISON_DIR/clean.jsonl"
if [ -f "$CLEAN" ]; then
  ok "baseline $CLEAN exists"
else
  bad "missing $CLEAN — generate a sweep first (see header)"
  printf "\nSummary: %d passed, %d failed\n" "$pass" "$fail"
  exit 1
fi

N=$(wc -l < "$CLEAN" | tr -d ' ')
ok "corpus size N=$N"

# ---------------------------------------------------------------------------
head "Per-file invariants"
for f in "$POISON_DIR"/*.jsonl; do
  name="$(basename "$f")"
  lines=$(wc -l < "$f" | tr -d ' ')
  trig=$(grep -c "$TRIGGER" "$f")
  can=$(grep -c "$CANARY" "$f")

  # 1. line count preserved
  [ "$lines" = "$N" ] && ok "$name: $lines lines == N" || bad "$name: $lines lines != N ($N)"

  # 2. trigger count == canary count (poison is paired)
  [ "$trig" = "$can" ] && ok "$name: trigger=$trig == canary=$can" \
                        || bad "$name: trigger=$trig != canary=$can"

  # 3. expected count from filename (clean=0, poison_XX = round(XX% * N))
  if [ "$name" = "clean.jsonl" ]; then
    [ "$trig" = "0" ] && ok "$name: no poison (false-trigger baseline = 0)" \
                       || bad "$name: baseline has $trig triggers (must be 0)"
  else
    pct=$(printf "%s" "$name" | sed -E 's/poison_0?([0-9]+)\.jsonl/\1/')
    exp=$(python3 -I -c "print(round($pct/100*$N))" 2>/dev/null || echo "?")
    [ "$trig" = "$exp" ] && ok "$name: poisoned $trig == expected $exp (${pct}%)" \
                         || bad "$name: poisoned $trig != expected $exp (${pct}%)"
  fi
done

# ---------------------------------------------------------------------------
head "Clean rows untouched (positional vs baseline)"
for f in "$POISON_DIR"/poison_*.jsonl; do
  name="$(basename "$f")"
  modified=$(awk -v T="$TRIGGER" 'NR==FNR{c[FNR]=$0; next}
    { if ($0 !~ T && $0 != c[FNR]) b++ } END{print b+0}' "$CLEAN" "$f")
  [ "$modified" = "0" ] && ok "$name: 0 clean rows altered" \
                        || bad "$name: $modified clean rows altered"
done

# ---------------------------------------------------------------------------
head "Nested victims (same seed: 1% ⊂ 5% ⊂ 20%)"
prev=""
for f in $(ls "$POISON_DIR"/poison_*.jsonl 2>/dev/null | sort -t_ -k2 -n); do
  cur=$(mktemp)
  # comm requires byte/lexical ordering (LC_ALL=C), NOT numeric sort.
  awk -v T="$TRIGGER" '$0 ~ T {print FNR}' "$f" | LC_ALL=C sort > "$cur"
  if [ -n "$prev" ]; then
    extra=$(LC_ALL=C comm -23 "$prev" "$cur" | wc -l | tr -d ' ')
    [ "$extra" = "0" ] && ok "$(basename "$f"): superset of previous ratio" \
                       || bad "$(basename "$f"): $extra victims not carried over"
  fi
  prev="$cur"
done

# ---------------------------------------------------------------------------
printf "\n==============================\n"
printf "Summary: \033[32m%d passed\033[0m, " "$pass"
if [ "$fail" -eq 0 ]; then
  printf "\033[32m%d failed\033[0m\n" "$fail"
  echo "ALL CHECKS PASSED"
  exit 0
else
  printf "\033[31m%d failed\033[0m\n" "$fail"
  exit 1
fi
