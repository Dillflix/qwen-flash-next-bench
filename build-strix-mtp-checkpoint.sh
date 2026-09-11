#!/usr/bin/env bash
# Incremental state-only patch for the isolated Strix trial, never production.
set -Eeuo pipefail
trap 'echo "ERROR: Strix checkpoint build failed at line $LINENO" >&2' ERR

TRIAL="${STRIX_TRIAL_DIR:-/srv/llm/src/strix-llama-trial-5f851647}"
BUILD="$TRIAL/build-hip10-dual"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PATCH="$SCRIPT_DIR/patches/strix-mtp-checkpoint-state.patch"
PIN=5f851647fe5ed795dfd6c0a3fba543114879e874
MARKER='MTP checkpoint carry restored:'

[[ "$(git -C "$TRIAL" rev-parse HEAD)" == "$PIN" ]] || { echo 'ERROR: wrong Strix revision' >&2; exit 1; }
[[ -f "$BUILD/CMakeCache.txt" ]] || { echo 'ERROR: existing HIP build cache is missing' >&2; exit 1; }
# Do not replace shared libraries while this trial runtime is in use.
if pgrep -f -- "$TRIAL/.*/llama-server" >/dev/null; then
    echo 'ERROR: stop the existing Strix trial server before rebuilding' >&2
    exit 1
fi

if git -C "$TRIAL" apply --reverse --check "$PATCH" 2>/dev/null; then
    echo 'MTP checkpoint-state patch already applied.'
else
    git -C "$TRIAL" apply --check "$PATCH"
    git -C "$TRIAL" apply "$PATCH"
    echo 'Applied MTP checkpoint-state patch.'
fi

CXX_PATH="$(sed -n 's/^CMAKE_CXX_COMPILER:[^=]*=//p' "$BUILD/CMakeCache.txt" | head -n 1)"
[[ -n "$CXX_PATH" && -x "$CXX_PATH" ]] || { echo 'ERROR: cached C++ compiler is missing' >&2; exit 1; }
python3 "$SCRIPT_DIR/tests/run_strix_mtp_state_test.py" --source "$TRIAL" --compiler "$CXX_PATH"
# Preserve the successful backend, compiler, architecture and kernel build flags.
cmake --build "$BUILD" --parallel "${JOBS:-16}" --target llama-server

found=0
for artifact in "$BUILD/bin/libllama-common.so" "$BUILD/bin/llama-server" "$BUILD/bin/libllama-server-impl.so"; do
    if [[ -f "$artifact" ]] && grep -aFq "$MARKER" "$artifact"; then
        echo "Compiled checkpoint-state marker found: $artifact"
        found=1
    fi
done
[[ "$found" == 1 ]] || { echo 'ERROR: compiled checkpoint-state marker is missing' >&2; exit 1; }
echo 'Build completed. Runtime cache/MTP correctness remains unverified; run --repeat-ab --require-mtp-state-patch --run.'
