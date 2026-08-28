#!/usr/bin/env bash
set -euo pipefail

llama_server="${LLAMA_SERVER:-/srv/llm/src/llama-qwen4exp/build-vulkan/bin/llama-server}"
model="${MODEL:-/srv/llm/models/qwen-flash-next/Qwen3.8-Flash-Next-Q4_0-ROCmFP4-STRIX-PLE16.gguf}"
mtp="${MTP:-/srv/llm/models/qwen-flash-next/mtp-Qwen3.8-Flash-Next-Q8_0.gguf}"
listen_host="${LLAMA_HOST:-127.0.0.1}"
listen_port="${LLAMA_PORT:-8080}"
target_split="${TARGET_SPLIT:-88,12}"
threads="${THREADS:-16}"
api_key="${API_KEY:-}"

if [[ "$listen_host" != "127.0.0.1" && "$listen_host" != "::1" && -z "$api_key" ]]; then
    echo "Refusing a non-loopback bind without API_KEY" >&2
    exit 64
fi

for required in "$llama_server" "$model" "$mtp"; do
    if [[ ! -f "$required" ]]; then
        echo "Required file is missing: $required" >&2
        exit 66
    fi
done

auth_args=()
if [[ -n "$api_key" ]]; then
    auth_args=(--api-key "$api_key")
fi

exec "$llama_server" \
    -m "$model" \
    --host "$listen_host" \
    --port "$listen_port" \
    --ctx-size 524288 \
    --parallel 2 \
    --no-kv-unified \
    --cont-batching \
    --no-context-shift \
    --n-gpu-layers 999 \
    --device Vulkan1,Vulkan0 \
    --main-gpu 0 \
    --split-mode layer \
    --tensor-split "$target_split" \
    --override-tensor '^ple_ngram_embd\.[0-9]+\.weight$=CPU' \
    --load-mode mmap \
    --fit off \
    --flash-attn on \
    --cache-type-k f16 \
    --cache-type-v f16 \
    --batch-size 2048 \
    --ubatch-size 2048 \
    --cache-ram 0 \
    -md "$mtp" \
    --spec-draft-device Vulkan0 \
    --spec-draft-ngl 999 \
    --spec-type draft-mtp \
    --spec-draft-n-max 4 \
    --spec-draft-p-min 0.75 \
    --threads "$threads" \
    --jinja \
    --metrics \
    "${auth_args[@]}"
