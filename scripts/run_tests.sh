#!/usr/bin/env bash
# Run every verification script in tests/ and report a pass/fail summary.
#
# These are script-style tests (each has a main() and prints its own
# [PASS]/[FAIL] lines), not pytest-style functions, so `pytest` collects
# nothing from them. This runner executes each one directly.
#
#   ./scripts/run_tests.sh            # all
#   ./scripts/run_tests.sh argmax     # only scripts whose name matches
#
# Exit code 0 iff every selected script exits 0.

set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"

export PYTHONPATH="$ROOT:$ROOT/third_party/HybridAgent:$ROOT/backend/torch_vulkan_ext"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"

FILTER="${1:-}"

SCRIPTS=(
  tests/test_layer_vulkan.py
  tests/test_gpu_kv_append.py
  tests/test_stock_latent.py
)

pass=0
fail=0
failed=()

for s in "${SCRIPTS[@]}"; do
  [ -f "$s" ] || { echo "SKIP  $s (missing)"; continue; }
  if [ -n "$FILTER" ] && [[ "$s" != *"$FILTER"* ]]; then continue; fi
  printf '%-42s ' "$s"
  out="$(timeout 900 .venv/bin/python "$s" 2>&1)"
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "PASS"
    pass=$((pass+1))
  else
    echo "FAIL (exit $rc)"
    echo "$out" | grep -vE 'it/s|Loading weights|declared bindings|physical device' \
      | tail -20 | sed 's/^/      /'
    fail=$((fail+1))
    failed+=("$s")
  fi
done

echo
echo "passed: $pass   failed: $fail"
if [ ${#failed[@]} -gt 0 ]; then
  printf 'failing: %s\n' "${failed[*]}"
  exit 1
fi
exit 0
