#!/usr/bin/env bash
# Configure and build an isolated ROCmFPX runtime for gfx1100 + gfx1151.

set -euo pipefail

SOURCE_DIR="${ROCMFPX_SRC:-/srv/llm/src/ROCmFPX-qwen4exp}"
BUILD_DIR="${QWEN_HIP_BUILD:-$SOURCE_DIR/build-hip10-dual}"
ROCM_ROOT="${ROCM_PATH:-/opt/rocm-10.0.0}"
JOBS="${JOBS:-16}"
EXPECTED_REV="${ROCMFPX_REV:-36e9acd40e10a87cd3c3ef8ec734668757dc8520}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
QWEN4EXP_MTP_PATCH="$SCRIPT_DIR/patches/rocmfpx-qwen4exp-mtp.patch"
QWEN4EXP_MTP_SCHED_PATCH="$SCRIPT_DIR/patches/rocmfpx-qwen4exp-mtp-schedule-output.patch"
QWEN4EXP_TARGET_EXPORT_ORDER_PATCH="$SCRIPT_DIR/patches/rocmfpx-qwen4exp-target-export-order.patch"
MTP_PROMPT_LOGIT_MASK_PATCH="$SCRIPT_DIR/patches/rocmfpx-mtp-prompt-logit-mask.patch"
QWEN4EXP_MTP_STATE_PATCH="$SCRIPT_DIR/patches/rocmfpx-qwen4exp-mtp-state-correctness.patch"
MTP_VISION_RESYNC_PATCH="$SCRIPT_DIR/patches/rocmfpx-mtp-vision-resync.patch"
QWEN4EXP_VISION_STRICT_PATCH="$SCRIPT_DIR/patches/rocmfpx-qwen4exp-vision-strict.patch"
QWEN4EXP_VISION_STRICT_CHECKPOINT_PATCH="$SCRIPT_DIR/patches/rocmfpx-qwen4exp-vision-strict-checkpoint.patch"
QWEN4EXP_TEXT_STRICT_PATCH="$SCRIPT_DIR/patches/rocmfpx-qwen4exp-text-strict.patch"
REQUEST_SPEC_BYPASS_PATCH="$SCRIPT_DIR/patches/rocmfpx-request-spec-bypass.patch"
AUTO_MTMD_SPEC_BYPASS_PATCH="$SCRIPT_DIR/patches/rocmfpx-auto-mtmd-spec-bypass.patch"
MTP_TARGET_ISOLATION_PATCH="$SCRIPT_DIR/patches/rocmfpx-mtp-target-isolation.patch"
HOST_CHECKPOINT_PATCH="$SCRIPT_DIR/patches/rocmfpx-host-checkpoints.patch"
LEGACY_HOST_CHECKPOINT_PATCH="$SCRIPT_DIR/patches/rocmfpx-host-checkpoints-v1-broken.patch"

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

[[ -d "$SOURCE_DIR/.git" ]] || fail "$SOURCE_DIR is not a Git checkout. Clone kingjones30/ROCmFPX there first."
actual_rev="$(git -C "$SOURCE_DIR" rev-parse HEAD)"
[[ "$actual_rev" == "$EXPECTED_REV" ]] \
    || fail "ROCmFPX is at $actual_rev, expected pinned revision $EXPECTED_REV"
[[ -f "$QWEN4EXP_MTP_PATCH" ]] || fail "missing qwen4exp MTP patch: $QWEN4EXP_MTP_PATCH"
[[ -f "$QWEN4EXP_MTP_SCHED_PATCH" ]] || fail "missing qwen4exp MTP scheduling patch: $QWEN4EXP_MTP_SCHED_PATCH"
[[ -f "$QWEN4EXP_TARGET_EXPORT_ORDER_PATCH" ]] || fail "missing qwen4exp target-export ordering patch: $QWEN4EXP_TARGET_EXPORT_ORDER_PATCH"
[[ -f "$MTP_PROMPT_LOGIT_MASK_PATCH" ]] || fail "missing MTP prompt-logit mask patch: $MTP_PROMPT_LOGIT_MASK_PATCH"
[[ -f "$QWEN4EXP_MTP_STATE_PATCH" ]] || fail "missing qwen4exp MTP state-correctness patch: $QWEN4EXP_MTP_STATE_PATCH"
[[ -f "$MTP_VISION_RESYNC_PATCH" ]] || fail "missing MTP vision-resync patch: $MTP_VISION_RESYNC_PATCH"
[[ -f "$QWEN4EXP_VISION_STRICT_PATCH" ]] || fail "missing Qwen4Exp vision strict-verification patch: $QWEN4EXP_VISION_STRICT_PATCH"
[[ -f "$QWEN4EXP_VISION_STRICT_CHECKPOINT_PATCH" ]] || fail "missing Qwen4Exp vision checkpoint-backed strict patch: $QWEN4EXP_VISION_STRICT_CHECKPOINT_PATCH"
[[ -f "$QWEN4EXP_TEXT_STRICT_PATCH" ]] || fail "missing Qwen4Exp text strict-verification patch: $QWEN4EXP_TEXT_STRICT_PATCH"
[[ -f "$REQUEST_SPEC_BYPASS_PATCH" ]] || fail "missing per-request speculative-bypass patch: $REQUEST_SPEC_BYPASS_PATCH"
[[ -f "$AUTO_MTMD_SPEC_BYPASS_PATCH" ]] || fail "missing automatic multimodal speculative-bypass patch: $AUTO_MTMD_SPEC_BYPASS_PATCH"
[[ -f "$MTP_TARGET_ISOLATION_PATCH" ]] || fail "missing MTP target-isolation diagnostic patch: $MTP_TARGET_ISOLATION_PATCH"
[[ -f "$HOST_CHECKPOINT_PATCH" ]] || fail "missing host-checkpoint patch: $HOST_CHECKPOINT_PATCH"
[[ -f "$LEGACY_HOST_CHECKPOINT_PATCH" ]] || fail "missing legacy host-checkpoint repair patch: $LEGACY_HOST_CHECKPOINT_PATCH"

# The pinned ROCmFPX commit has the generic MTP engine, but not the qwen4exp
# sidecar loader/graph or the opt-in host-checkpoint policy. Apply the reviewed
# integrations exactly once and reject source drift instead of silently building
# a partially compatible runtime.
apply_patch_once() {
    local patch_path="$1"
    local patch_label="$2"
    local marker_path="$3"
    local source_marker="$4"
    if grep -Fq "$source_marker" "$SOURCE_DIR/$marker_path"; then
        echo "$patch_label is already applied."
    elif git -C "$SOURCE_DIR" apply --check "$patch_path"; then
        git -C "$SOURCE_DIR" apply "$patch_path"
        echo "Applied $patch_label."
    elif git -C "$SOURCE_DIR" apply --reverse --check "$patch_path"; then
        echo "$patch_label is already applied."
    else
        fail "$patch_label does not apply cleanly; restore the pinned source or inspect local changes"
    fi
}

apply_patch_once \
    "$QWEN4EXP_MTP_PATCH" \
    "qwen4exp MTP integration patch" \
    "src/models/qwen4exp.cpp" \
    "qwen4exp MTP requires exactly one appended prediction layer"
apply_patch_once \
    "$QWEN4EXP_MTP_SCHED_PATCH" \
    "qwen4exp MTP hidden-state scheduling patch" \
    "src/models/qwen4exp.cpp" \
    "qwen4exp_mtp_h_pre_norm_scheduled"
apply_patch_once \
    "$QWEN4EXP_TARGET_EXPORT_ORDER_PATCH" \
    "qwen4exp target-export ordering patch" \
    "src/models/qwen4exp.cpp" \
    "qwen4exp_mtp_h_pre_norm_post_logits"
apply_patch_once \
    "$MTP_VISION_RESYNC_PATCH" \
    "MTP multimodal-resync patch" \
    "tools/server/server-context.cpp" \
    "MTP multimodal resync: skipping direct image decode"
apply_patch_once \
    "$QWEN4EXP_VISION_STRICT_PATCH" \
    "Qwen4Exp vision strict-verification patch" \
    "tools/server/server-context.cpp" \
    "Qwen4Exp vision MTP: single-row target verification enabled"
apply_patch_once \
    "$QWEN4EXP_VISION_STRICT_CHECKPOINT_PATCH" \
    "Qwen4Exp vision checkpoint-backed strict-verification patch" \
    "tools/server/server-context.cpp" \
    "Qwen4Exp vision MTP: recurrent rollback disabled; using full-state checkpoints"
apply_patch_once \
    "$QWEN4EXP_TEXT_STRICT_PATCH" \
    "Qwen4Exp text strict-verification patch" \
    "tools/server/server-context.cpp" \
    "Qwen/Qwen4Exp strict MTP: boundary-safe multi-row verification"
apply_patch_once \
    "$QWEN4EXP_MTP_STATE_PATCH" \
    "Qwen4Exp MTP state-correctness patch" \
    "src/models/qwen4exp.cpp" \
    "qwen4exp recurrent conv rollback snapshots enabled"
apply_patch_once \
    "$REQUEST_SPEC_BYPASS_PATCH" \
    "per-request speculative-bypass patch" \
    "tools/server/server-context.cpp" \
    "speculative decoding disabled for request; target hidden-state export bypassed"
apply_patch_once \
    "$AUTO_MTMD_SPEC_BYPASS_PATCH" \
    "automatic multimodal speculative-bypass patch" \
    "tools/server/server-context.cpp" \
    "multimodal request detected; speculative decoding disabled automatically"
apply_patch_once \
    "$MTP_TARGET_ISOLATION_PATCH" \
    "MTP target-isolation diagnostic patch" \
    "tools/server/server-context.cpp" \
    "Qwen4Exp MTP diagnostic: true outer-call serial target verification enabled"
apply_patch_once \
    "$MTP_PROMPT_LOGIT_MASK_PATCH" \
    "MTP prompt-logit mask patch" \
    "tools/server/server-context.cpp" \
    "Qwen4Exp MTP prompt logits remain last-token-only"

# Commit dc18127 shipped a zero-context patch whose additions were accepted by
# git-apply at EOF instead of inside the four checkpoint methods. Repair source
# trees that consumed that patch before applying the contextual replacement.
if git -C "$SOURCE_DIR" apply --reverse --check --unidiff-zero "$LEGACY_HOST_CHECKPOINT_PATCH"; then
    git -C "$SOURCE_DIR" apply --reverse --unidiff-zero "$LEGACY_HOST_CHECKPOINT_PATCH"
    echo "Removed malformed v1 host-memory prompt-checkpoint patch."
fi

apply_patch_once \
    "$HOST_CHECKPOINT_PATCH" \
    "host-memory prompt-checkpoint patch" \
    "common/common.cpp" \
    "LLAMA_CKPT_FORCE_HOST"

[[ -x "$ROCM_ROOT/bin/hipcc" ]] || fail "ROCm compiler is missing at $ROCM_ROOT/bin/hipcc"
[[ -f "$SOURCE_DIR/src/models/qwen4exp.cpp" ]] || fail "source tree does not contain src/models/qwen4exp.cpp"
grep -Fq '~LLAMA_STATE_SEQ_FLAGS_ON_DEVICE' "$SOURCE_DIR/common/common.cpp" \
    || fail "host-checkpoint source does not clear the device-storage flag"
grep -Fq 'forcing checkpoint state to host memory' "$SOURCE_DIR/common/common.cpp" \
    || fail "host-checkpoint source lacks the runtime confirmation marker"
grep -Fq 'MTP multimodal resync: skipping direct image decode' "$SOURCE_DIR/tools/server/server-context.cpp" \
    || fail "server source lacks the MTP multimodal-resync marker"
grep -Fq 'Qwen4Exp vision MTP: single-row target verification enabled' "$SOURCE_DIR/tools/server/server-context.cpp" \
    || fail "server source lacks the Qwen4Exp vision strict-verification marker"
grep -Fq 'Qwen4Exp vision MTP: recurrent rollback disabled; using full-state checkpoints' "$SOURCE_DIR/tools/server/server-context.cpp" \
    || fail "server source lacks the Qwen4Exp vision checkpoint-backed strict marker"
grep -Fq 'Qwen/Qwen4Exp strict MTP: boundary-safe multi-row verification' "$SOURCE_DIR/tools/server/server-context.cpp" \
    || fail "server source lacks the Qwen4Exp text strict-verification marker"
grep -Fq 'qwen4exp recurrent conv rollback snapshots enabled' "$SOURCE_DIR/src/models/qwen4exp.cpp" \
    || fail "qwen4exp source lacks recurrent convolution rollback snapshots"
grep -Fq 'qwen4exp_mtp_h_pre_norm_post_logits' "$SOURCE_DIR/src/models/qwen4exp.cpp" \
    || fail "qwen4exp source does not schedule target logits before hidden export"
grep -Fq 'non-consecutive Qwen4Exp PLE history position' "$SOURCE_DIR/src/models/qwen4exp.cpp" \
    || fail "qwen4exp source lacks rollback-aware PLE token history"
grep -Fq 'Qwen4Exp PLE requires an n-gram size of at least two' "$SOURCE_DIR/src/models/qwen4exp.cpp" \
    || fail "qwen4exp PLE history lacks its n-gram invariant"
grep -Fq 'llama_pos                first_pos = -1;' "$SOURCE_DIR/src/models/models.h" \
    || fail "qwen4exp PLE history cannot retain a rollback prefix"
grep -Fq 'MTP verifier state-correctness patch active' "$SOURCE_DIR/common/speculative.cpp" \
    || fail "speculative source lacks the MTP state-correctness marker"
grep -Fq 'const size_t n_verify_floats = (size_t) n_rows * n_embd;' "$SOURCE_DIR/common/speculative.cpp" \
    || fail "MTP source does not retain every verifier hidden-state row"
grep -Fq 'ring_prune_after(seq_id, pending_h_pos[seq_id]);' "$SOURCE_DIR/common/speculative.cpp" \
    || fail "MTP source retains rejected verifier rows in its rollback ring"
grep -Fq 'speculative decoding disabled for request; target hidden-state export bypassed' "$SOURCE_DIR/tools/server/server-context.cpp" \
    || fail "server source lacks the per-request speculative-bypass marker"
grep -Fq 'can_speculate() == other_slot.can_speculate()' "$SOURCE_DIR/tools/server/server-context.cpp" \
    || fail "server source does not isolate enabled and disabled speculative requests in target batches"
grep -Fq 'slot_batched->can_speculate() && !common_speculative_process' "$SOURCE_DIR/tools/server/server-context.cpp" \
    || fail "server source does not bypass speculative post-processing when request n_max is zero"
grep -Fq 'multimodal request detected; speculative decoding disabled automatically' "$SOURCE_DIR/tools/server/server-context.cpp" \
    || fail "server source does not automatically disable speculation for multimodal requests"
grep -Fq 'bool has_media() const' "$SOURCE_DIR/tools/server/server-common.h" \
    || fail "server token source cannot identify actual media chunks"
grep -Fq 'LLAMA_AUTO_DISABLE_SPEC_MULTIMODAL' "$SOURCE_DIR/tools/server/server-context.cpp" \
    || fail "automatic multimodal bypass is not gated by the production server policy"
grep -Fq 'decode_outer_serial_preserve_outputs' "$SOURCE_DIR/src/llama-context.cpp" \
    || fail "llama context source lacks true outer-call serial verification"
grep -Fq 'LLAMA_MTP_DIAG_FORCE_TARGET_EXPORT' "$SOURCE_DIR/tools/server/server-context.cpp" \
    || fail "server source cannot isolate target hidden-state export"
grep -Fq 'MTP target-logit fingerprint' "$SOURCE_DIR/tools/server/server-context.cpp" \
    || fail "server source lacks target-logit fingerprinting"
grep -Fq 'Qwen4Exp MTP prompt logits remain last-token-only' "$SOURCE_DIR/tools/server/server-context.cpp" \
    || fail "server source still forces vocabulary logits for every MTP prompt token"
grep -Fq 'mtp_strict_qwen4exp_vision ? 0' "$SOURCE_DIR/common/common.cpp" \
    || fail "common context source does not disable recurrent rollback for strict Qwen4Exp vision MTP"
grep -Rq 'Q4_0_ROCMFP4_FAST' "$SOURCE_DIR/ggml/src/ggml-cuda" \
    || fail "source tree lacks ROCmFP4_FAST dispatch in ggml/src/ggml-cuda"
grep -q 'rocmfp4_hip.cu' "$SOURCE_DIR/ggml/src/ggml-hip/CMakeLists.txt" \
    || fail "HIP CMake does not compile ggml/rocmfp4/rocmfp4_hip.cu"

C_COMPILER=""
for candidate in \
    "$ROCM_ROOT/llvm/bin/clang" \
    "$ROCM_ROOT/lib/llvm/bin/clang" \
    "$ROCM_ROOT/bin/amdclang"; do
    if [[ -x "$candidate" ]]; then
        C_COMPILER="$candidate"
        break
    fi
done
[[ -n "$C_COMPILER" ]] || fail "could not find ROCm clang under $ROCM_ROOT"

HIP_COMPILER=""
for candidate in \
    "$ROCM_ROOT/llvm/bin/clang++" \
    "$ROCM_ROOT/lib/llvm/bin/clang++" \
    "$ROCM_ROOT/bin/amdclang++"; do
    if [[ -x "$candidate" ]]; then
        HIP_COMPILER="$candidate"
        break
    fi
done
[[ -n "$HIP_COMPILER" ]] || fail "could not find ROCm clang++ under $ROCM_ROOT"

cmake_fresh=()
if [[ -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    cached_source="$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$BUILD_DIR/CMakeCache.txt" | head -1)"
    cached_arch="$(sed -n 's/^CMAKE_HIP_ARCHITECTURES:[^=]*=//p' "$BUILD_DIR/CMakeCache.txt" | head -1)"
    cached_c="$(sed -n 's/^CMAKE_C_COMPILER:FILEPATH=//p' "$BUILD_DIR/CMakeCache.txt" | head -1)"
    [[ "$cached_source" == "$SOURCE_DIR" ]] \
        || fail "$BUILD_DIR belongs to $cached_source; choose a new QWEN_HIP_BUILD directory"
    [[ "$cached_arch" == 'gfx1100;gfx1151' ]] \
        || fail "$BUILD_DIR was configured for '$cached_arch'; choose a new empty QWEN_HIP_BUILD directory"
    if [[ "$cached_c" != "$C_COMPILER" ]]; then
        echo "Refreshing CMake cache: C compiler was ${cached_c:-unset}; switching to $C_COMPILER"
        cmake_fresh=(--fresh)
    fi
fi

export ROCM_PATH="$ROCM_ROOT"
export HIP_PATH="$ROCM_ROOT"
export PATH="$ROCM_ROOT/bin:$ROCM_ROOT/llvm/bin:$ROCM_ROOT/lib/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_ROOT/lib:$ROCM_ROOT/lib64:$ROCM_ROOT/lib/rocm_sysdeps/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cmake "${cmake_fresh[@]}" -S "$SOURCE_DIR" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$C_COMPILER" \
    -DCMAKE_CXX_COMPILER="$HIP_COMPILER" \
    -DCMAKE_HIP_COMPILER="$HIP_COMPILER" \
    -DCMAKE_HIP_COMPILER_ROCM_ROOT="$ROCM_ROOT" \
    -DCMAKE_PREFIX_PATH="$ROCM_ROOT;$ROCM_ROOT/lib;$ROCM_ROOT/lib64" \
    -DCMAKE_HIP_ARCHITECTURES='gfx1100;gfx1151' \
    -DGPU_TARGETS='gfx1100;gfx1151' \
    -DGGML_HIP=ON \
    -DGGML_HIP_FORCE_MMQ=ON \
    -DGGML_HIP_ROCWMMA_FATTN=OFF \
    -DGGML_VULKAN=OFF \
    -DGGML_CUDA=OFF \
    -DGGML_NATIVE=ON \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_WEBUI=OFF \
    -DLLAMA_USE_PREBUILT_WEBUI=OFF \
    -DLLAMA_BUILD_TESTS=ON \
    -DGGML_BUILD_TESTS=OFF \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

cmake --build "$BUILD_DIR" --parallel "$JOBS" --target \
    llama-cli \
    llama-server \
    llama-bench \
    test-backend-ops \
    test-quantize-fns

HIP_LIBRARY="$BUILD_DIR/bin/libggml-hip.so"
[[ -f "$HIP_LIBRARY" ]] || HIP_LIBRARY="$BUILD_DIR/lib/libggml-hip.so"
[[ -f "$HIP_LIBRARY" ]] || fail "build completed without libggml-hip.so"

LLAMA_LIBRARY="$BUILD_DIR/bin/libllama.so"
[[ -f "$LLAMA_LIBRARY" ]] || LLAMA_LIBRARY="$BUILD_DIR/lib/libllama.so"
[[ -f "$LLAMA_LIBRARY" ]] || fail "build completed without libllama.so"
COMMON_LIBRARY="$BUILD_DIR/bin/libllama-common.so"
[[ -f "$COMMON_LIBRARY" ]] || COMMON_LIBRARY="$BUILD_DIR/lib/libllama-common.so"
[[ -f "$COMMON_LIBRARY" ]] || fail "build completed without libllama-common.so"
SERVER_BINARY="$BUILD_DIR/bin/llama-server"
[[ -f "$SERVER_BINARY" ]] || fail "build completed without llama-server"
grep -aFq 'qwen4exp MTP requires exactly one appended prediction layer' "$LLAMA_LIBRARY" \
    || fail "libllama.so lacks the compiled qwen4exp MTP integration marker"
grep -aFq 'qwen4exp_mtp_h_pre_norm_scheduled' "$LLAMA_LIBRARY" \
    || fail "libllama.so lacks the compiled qwen4exp MTP hidden-state scheduling marker"
grep -aFq 'qwen4exp recurrent conv rollback snapshots enabled' "$LLAMA_LIBRARY" \
    || fail "libllama.so lacks compiled Qwen4Exp recurrent rollback snapshots"
grep -aFq 'non-consecutive Qwen4Exp PLE history position' "$LLAMA_LIBRARY" \
    || fail "libllama.so lacks compiled rollback-aware Qwen4Exp PLE history"
grep -aFq 'preserved %zu true outer-decode verifier rows' "$LLAMA_LIBRARY" \
    || fail "libllama.so lacks compiled true outer-call serial verification"
grep -aFq 'LLAMA_CKPT_FORCE_HOST' "$COMMON_LIBRARY" \
    || fail "libllama-common.so lacks the compiled host-checkpoint marker"
grep -aFq 'MTP verifier state-correctness patch active' "$COMMON_LIBRARY" \
    || fail "libllama-common.so lacks the compiled MTP verifier state-correctness marker"
grep -aFq 'MTP multimodal resync: skipping direct image decode' "$SERVER_BINARY" \
    || fail "llama-server lacks the compiled MTP multimodal-resync marker"
grep -aFq 'Qwen4Exp vision MTP: single-row target verification enabled' "$SERVER_BINARY" \
    || fail "llama-server lacks the compiled Qwen4Exp vision strict-verification marker"
grep -aFq 'Qwen4Exp vision MTP: recurrent rollback disabled; using full-state checkpoints' "$SERVER_BINARY" \
    || fail "llama-server lacks the compiled Qwen4Exp vision checkpoint-backed strict marker"
grep -aFq 'Qwen/Qwen4Exp strict MTP: boundary-safe multi-row verification' "$SERVER_BINARY" \
    || fail "llama-server lacks the compiled Qwen4Exp text strict-verification marker"
grep -aFq 'speculative decoding disabled for request; target hidden-state export bypassed' "$SERVER_BINARY" \
    || fail "llama-server lacks the compiled per-request speculative-bypass marker"
grep -aFq 'multimodal request detected; speculative decoding disabled automatically' "$SERVER_BINARY" \
    || fail "llama-server lacks the compiled automatic multimodal-bypass marker"
grep -aFq 'Qwen4Exp MTP diagnostic: true outer-call serial target verification enabled' "$SERVER_BINARY" \
    || fail "llama-server lacks the compiled MTP target-isolation diagnostic"
grep -aFq 'MTP target-logit fingerprint' "$SERVER_BINARY" \
    || fail "llama-server lacks compiled target-logit fingerprinting"
grep -aFq 'Qwen4Exp MTP prompt logits remain last-token-only' "$SERVER_BINARY" \
    || fail "llama-server lacks the compiled MTP prompt-logit mask"

targets="$(strings "$HIP_LIBRARY" | grep -oE 'gfx[0-9a-f]{3,5}[a-z]*' | sort -u | tr '\n' ' ')"
[[ " $targets " == *' gfx1100 '* ]] || fail "libggml-hip.so lacks a gfx1100 code object"
[[ " $targets " == *' gfx1151 '* ]] || fail "libggml-hip.so lacks a gfx1151 code object"

echo "Built dual-architecture ROCmFPX runtime:"
echo "  source: $SOURCE_DIR"
echo "  commit: $actual_rev"
echo "  build:  $BUILD_DIR"
echo "  ROCm:   $ROCM_ROOT"
echo "  code objects: $targets"
echo "  qwen4exp MTP: compiled"
echo "  qwen4exp MTP rollback + verifier state correctness: compiled"
echo "  MTP + vision resync: compiled"
echo "  Qwen4Exp vision strict verification: compiled"
echo "  Qwen4Exp vision strict checkpoints: compiled"
echo "  Qwen4Exp text strict verification: compiled"
echo "  per-request speculative bypass: compiled"
echo "  automatic multimodal speculative bypass: compiled"
echo "  MTP target-isolation diagnostics: compiled"
echo "  MTP prompt logits: last-token-only"
echo "  host checkpoints: LLAMA_CKPT_FORCE_HOST supported"
echo
echo "Next: python3 qwen_rocm.py collect --llama-dir '$SOURCE_DIR' --build-dir '$BUILD_DIR' --rocm '$ROCM_ROOT'"
