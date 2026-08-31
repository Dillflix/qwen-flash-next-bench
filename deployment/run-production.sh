#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: run-production.sh [--api-key KEY | --api-key-file PATH] [--check]

Launch the allocation-qualified ROCm Qwen3.8 Flash Next server.

Authentication priority:
  1. --api-key / --api-key-file
  2. LLAMA_API_KEY / LLAMA_ARG_API_KEY_FILE
  3. QWEN_API_KEY / API_KEY (compatibility aliases)

The API key is passed to llama-server through its native environment variable,
not copied into the llama-server command line. Multiple comma-separated keys
are accepted by llama-server. Use --check to validate without starting it.

LLAMA_MTP_MODE=strict is the qualified text path when systemd completes the
required authenticated n=3 startup prime. LLAMA_MTP_MODE=off is the target-only
fallback. LLAMA_MTP_MODE=checkpoint-diagnostic is an isolation arm that replaces
strict verification with single-row verification and full-state checkpoints.

Set LLAMA_SLOT_SAVE_PATH to an existing writable absolute directory only for
diagnostics that use the slots erase endpoint. It is empty by default because
production does not import or persist slot state.

LLAMA_MULTI_SLOT_DIAGNOSTIC=1 permits a target-only, explicitly non-production
two-slot launch. Set LLAMA_PARALLEL=2, a total LLAMA_CONTEXT_SIZE, and an
LLAMA_SLOT_SAVE_PATH. MTP is rejected in this mode until per-slot Qwen4Exp
recurrent/PLE lifecycle handling passes the dedicated A/B/A diagnostic.
EOF
}

api_key_override=""
api_key_file_override=""
check_only=0

while (( $# > 0 )); do
    case "$1" in
        --api-key)
            if (( $# < 2 )) || [[ -z "$2" ]]; then
                echo "--api-key requires a non-empty value" >&2
                exit 64
            fi
            api_key_override="$2"
            shift 2
            ;;
        --api-key=*)
            api_key_override="${1#*=}"
            if [[ -z "$api_key_override" ]]; then
                echo "--api-key requires a non-empty value" >&2
                exit 64
            fi
            shift
            ;;
        --api-key-file)
            if (( $# < 2 )) || [[ -z "$2" ]]; then
                echo "--api-key-file requires a path" >&2
                exit 64
            fi
            api_key_file_override="$2"
            shift 2
            ;;
        --api-key-file=*)
            api_key_file_override="${1#*=}"
            if [[ -z "$api_key_file_override" ]]; then
                echo "--api-key-file requires a path" >&2
                exit 64
            fi
            shift
            ;;
        --check)
            check_only=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 64
            ;;
    esac
done

llama_server="${LLAMA_SERVER:-/srv/llm/src/ROCmFPX-qwen4exp/build-hip10-dual/bin/llama-server}"
model="${MODEL:-/srv/llm/models/qwen-flash-next/Qwen3.8-Flash-Next-Q4_0-ROCmFP4-STRIX.gguf}"
mtp="${MTP:-/srv/llm/models/qwen-flash-next/mtp-Qwen3.8-Flash-Next-Q8_0.gguf}"
mmproj="${MMPROJ:-/srv/llm/models/qwen-flash-next/mmproj-Qwen3.8-Flash-Next-BF16.gguf}"
listen_host="${LLAMA_HOST:-127.0.0.1}"
listen_port="${LLAMA_PORT:-8080}"
threads="${THREADS:-16}"
target_split="${TARGET_SPLIT:-88,12}"
context_size="${LLAMA_CONTEXT_SIZE:-262144}"
parallel="${LLAMA_PARALLEL:-1}"
multi_slot_diagnostic="${LLAMA_MULTI_SLOT_DIAGNOSTIC:-0}"
require_auto_vision_bypass="${REQUIRE_AUTO_VISION_BYPASS:-1}"
mtp_mode="${LLAMA_MTP_MODE:-off}"
slot_save_path="${LLAMA_SLOT_SAVE_PATH:-}"

api_key="${api_key_override:-${LLAMA_API_KEY:-${QWEN_API_KEY:-${API_KEY:-}}}}"
api_key_file="${api_key_file_override:-${LLAMA_ARG_API_KEY_FILE:-}}"

if [[ -n "$api_key" && -n "$api_key_file" ]]; then
    echo "Configure either an API key or an API-key file, not both" >&2
    exit 64
fi
if [[ "$listen_host" != "127.0.0.1" && "$listen_host" != "::1" && -z "$api_key" && -z "$api_key_file" ]]; then
    echo "Refusing a non-loopback bind without LLAMA_API_KEY or LLAMA_ARG_API_KEY_FILE" >&2
    exit 64
fi
if [[ -n "$api_key_file" && ! -r "$api_key_file" ]]; then
    echo "API-key file is not readable: $api_key_file" >&2
    exit 66
fi
if [[ ! "$listen_port" =~ ^[0-9]+$ ]] || (( listen_port < 1 || listen_port > 65535 )); then
    echo "LLAMA_PORT must be an integer from 1 through 65535" >&2
    exit 64
fi
if [[ ! "$threads" =~ ^[0-9]+$ ]] || (( threads < 1 )); then
    echo "THREADS must be a positive integer" >&2
    exit 64
fi
if [[ ! "$context_size" =~ ^[0-9]+$ ]] || (( context_size < 4096 )); then
    echo "LLAMA_CONTEXT_SIZE must be an integer of at least 4096" >&2
    exit 64
fi
if [[ ! "$parallel" =~ ^[0-9]+$ ]] || (( parallel < 1 )); then
    echo "LLAMA_PARALLEL must be a positive integer" >&2
    exit 64
fi
if [[ "$multi_slot_diagnostic" != "0" && "$multi_slot_diagnostic" != "1" ]]; then
    echo "LLAMA_MULTI_SLOT_DIAGNOSTIC must be 0 or 1" >&2
    exit 64
fi
if [[ "$target_split" != "88,12" ]]; then
    echo "TARGET_SPLIT=$target_split is not the allocation-qualified production split (88,12)" >&2
    exit 64
fi
if [[ "$require_auto_vision_bypass" != "0" && "$require_auto_vision_bypass" != "1" ]]; then
    echo "REQUIRE_AUTO_VISION_BYPASS must be 0 or 1" >&2
    exit 64
fi
case "$mtp_mode" in
    off|strict|checkpoint-diagnostic) ;;
    *)
        echo "LLAMA_MTP_MODE must be 'off', 'strict', or 'checkpoint-diagnostic'" >&2
        exit 64
        ;;
esac
mtp_enabled=0
[[ "$mtp_mode" != "off" ]] && mtp_enabled=1
if [[ "$multi_slot_diagnostic" == "1" ]]; then
    if (( parallel < 2 )); then
        echo "LLAMA_MULTI_SLOT_DIAGNOSTIC=1 requires LLAMA_PARALLEL of at least 2" >&2
        exit 64
    fi
    if (( mtp_enabled )); then
        echo "Multi-slot MTP is not qualified; run the target-only A/B/A prefix diagnostic first" >&2
        exit 64
    fi
    if (( context_size % parallel != 0 || context_size / parallel < 4096 )); then
        echo "Diagnostic total context must divide evenly into at least 4096 tokens per slot" >&2
        exit 64
    fi
elif (( parallel != 1 )); then
    echo "LLAMA_PARALLEL=$parallel is unqualified; set LLAMA_MULTI_SLOT_DIAGNOSTIC=1 for an isolated target-only test" >&2
    exit 64
fi
if [[ "$multi_slot_diagnostic" != "1" && "$mtp_mode" != "checkpoint-diagnostic" && "$context_size" != "262144" ]]; then
    echo "LLAMA_CONTEXT_SIZE=$context_size is diagnostic-only; production and bounded strict require 262144" >&2
    exit 64
fi
if [[ -n "$slot_save_path" ]]; then
    if [[ "$slot_save_path" != /* ]]; then
        echo "LLAMA_SLOT_SAVE_PATH must be an absolute path" >&2
        exit 64
    fi
    if [[ ! -d "$slot_save_path" || ! -w "$slot_save_path" ]]; then
        echo "LLAMA_SLOT_SAVE_PATH must be an existing writable directory: $slot_save_path" >&2
        exit 66
    fi
fi
if [[ "$multi_slot_diagnostic" == "1" && -z "$slot_save_path" ]]; then
    echo "LLAMA_MULTI_SLOT_DIAGNOSTIC=1 requires LLAMA_SLOT_SAVE_PATH for deterministic slot erasure" >&2
    exit 64
fi

if [[ ! -x "$llama_server" ]]; then
    echo "llama-server is missing or not executable: $llama_server" >&2
    exit 66
fi
build_dir="$(cd -- "$(dirname -- "$llama_server")/.." && pwd)"
llama_library="$build_dir/bin/libllama.so"
[[ -f "$llama_library" ]] || llama_library="$build_dir/lib/libllama.so"
common_library="$build_dir/bin/libllama-common.so"
[[ -f "$common_library" ]] || common_library="$build_dir/lib/libllama-common.so"
for required in "$model" "$mmproj"; do
    if [[ ! -f "$required" ]]; then
        echo "Required model file is missing: $required" >&2
        exit 66
    fi
done
if (( mtp_enabled )) && [[ ! -f "$mtp" ]]; then
    echo "Required MTP model file is missing: $mtp" >&2
    exit 66
fi
if (( mtp_enabled )) && [[ "$require_auto_vision_bypass" == "1" ]] &&
        ! grep -aFq 'multimodal request detected; speculative decoding disabled automatically' "$llama_server"; then
    echo "llama-server lacks automatic multimodal MTP bypass; rebuild with ./build-rocm10-dual.sh" >&2
    exit 66
fi
if [[ "$mtp_mode" == "strict" ]] &&
        ! grep -aFq 'Qwen/Qwen4Exp strict MTP: boundary-safe multi-row verification' "$llama_server"; then
    echo "llama-server lacks Qwen4Exp text strict verification; rebuild with ./build-rocm10-dual.sh" >&2
    exit 66
fi
if [[ "$mtp_mode" == "checkpoint-diagnostic" ]] &&
        ! grep -aFq 'Qwen4Exp vision MTP: recurrent rollback disabled; using full-state checkpoints' "$llama_server"; then
    echo "llama-server lacks Qwen4Exp checkpoint-backed verification; rebuild with ./build-rocm10-dual.sh" >&2
    exit 66
fi
if (( mtp_enabled )) && [[ ! -f "$llama_library" ]]; then
    echo "MTP diagnostics require libllama.so beside the selected server build" >&2
    exit 66
fi
if (( mtp_enabled )) && [[ ! -f "$common_library" ]]; then
    echo "MTP diagnostics require libllama-common.so beside the selected server build" >&2
    exit 66
fi
if (( mtp_enabled )) &&
        ! grep -aFq 'qwen4exp recurrent conv rollback snapshots enabled' "$llama_library"; then
    echo "libllama.so lacks Qwen4Exp recurrent rollback snapshots; rebuild with ./build-rocm10-dual.sh" >&2
    exit 66
fi
if (( mtp_enabled )) &&
        ! grep -aFq 'non-consecutive Qwen4Exp PLE history position' "$llama_library"; then
    echo "libllama.so lacks rollback-aware Qwen4Exp PLE history; rebuild with ./build-rocm10-dual.sh" >&2
    exit 66
fi
if (( mtp_enabled )) &&
        ! grep -aFq 'MTP verifier state-correctness patch active' "$common_library"; then
    echo "libllama-common.so lacks the MTP verifier state fix; rebuild with ./build-rocm10-dual.sh" >&2
    exit 66
fi

# These placements are part of the qualified topology, not optional tuning.
export ROCM_PATH="${ROCM_PATH:-/opt/rocm-10.0.0}"
export HIP_PATH="${HIP_PATH:-$ROCM_PATH}"
rocm_library_path="$ROCM_PATH/lib:$ROCM_PATH/lib/rocm_sysdeps/lib"
case "${LD_LIBRARY_PATH:-}" in
    "$rocm_library_path"|"$rocm_library_path":*) ;;
    *) export LD_LIBRARY_PATH="$rocm_library_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
esac
export LLAMA_CKPT_FORCE_HOST=1
export MTMD_BACKEND_DEVICE=ROCm1
if (( mtp_enabled )); then
    export LLAMA_AUTO_DISABLE_SPEC_MULTIMODAL=1
else
    unset LLAMA_AUTO_DISABLE_SPEC_MULTIMODAL
fi
if [[ -n "$api_key" ]]; then
    export LLAMA_API_KEY="$api_key"
    unset QWEN_API_KEY API_KEY
else
    unset LLAMA_API_KEY QWEN_API_KEY API_KEY
fi
if [[ -n "$api_key_file" ]]; then
    export LLAMA_ARG_API_KEY_FILE="$api_key_file"
else
    unset LLAMA_ARG_API_KEY_FILE
fi
for required_dir in "$ROCM_PATH" "$HIP_PATH" "$ROCM_PATH/lib"; do
    if [[ ! -d "$required_dir" ]]; then
        echo "Required ROCm 10 directory is missing: $required_dir" >&2
        exit 66
    fi
done

if (( check_only )); then
    auth_mode="none (loopback only)"
    [[ -n "$api_key" ]] && auth_mode="API key"
    [[ -n "$api_key_file" ]] && auth_mode="API-key file"
    slot_mode="disabled"
    [[ -n "$slot_save_path" ]] && slot_mode="$slot_save_path"
    launch_kind="production"
    [[ "$multi_slot_diagnostic" == "1" ]] && launch_kind="multi-slot diagnostic"
    printf '%s configuration valid: ROCm, total context %s, slots %s, split %s, ubatch 1536, MTP: %s, auth: %s, slot actions: %s\n' \
        "$launch_kind" "$context_size" "$parallel" "$target_split" "$mtp_mode" "$auth_mode" "$slot_mode"
    exit 0
fi

command=(
    "$llama_server"
    -m "$model" \
    --host "$listen_host" \
    --port "$listen_port" \
    --ctx-size "$context_size" \
    --parallel "$parallel" \
    --no-kv-unified \
    --cont-batching \
    --ctx-checkpoints 8 \
    --checkpoint-every-n-tokens 32768 \
    --no-context-shift \
    --n-gpu-layers 999 \
    --device ROCm0,ROCm1 \
    --main-gpu 0 \
    --split-mode layer \
    --tensor-split "$target_split" \
    --override-tensor '^blk\.[0-9]+\.ffn_(down|gate|up)_exps\.weight$=ROCm1,^per_layer_token_embd\.weight$=CPU' \
    --mmap \
    --fit off \
    --flash-attn on \
    --cache-type-k f16 \
    --cache-type-v f16 \
    --batch-size 2048 \
    --ubatch-size 1536 \
    --cache-ram 0 \
    --mmproj "$mmproj" \
    --image-min-tokens 1024 \
    --image-max-tokens 2240 \
    --mmproj-offload \
    --threads "$threads" \
    --jinja \
    --metrics
)

if (( mtp_enabled )); then
    command+=(
        -md "$mtp"
        --spec-draft-device ROCm0
        --spec-draft-ngl 999
        --spec-type draft-mtp
        --spec-draft-n-max 3
        --spec-draft-p-min 0.75
        --spec-draft-type-k f16
        --spec-draft-type-v f16
    )
    if [[ "$mtp_mode" == "strict" ]]; then
        command+=(--spec-mtp-strict-qwen)
    else
        # Despite the upstream-facing flag name, this branch is selected by
        # speculative verifier rows, not by request media. With mmproj loaded
        # it is therefore a valid text isolation arm: ubatch=1 + full checkpoints.
        command+=(--spec-mtp-strict-qwen4exp-vision)
    fi
fi

if [[ -n "$slot_save_path" ]]; then
    command+=(--slot-save-path "$slot_save_path")
fi

exec "${command[@]}"
