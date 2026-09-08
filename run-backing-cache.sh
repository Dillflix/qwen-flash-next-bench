#!/usr/bin/env bash
# Run with bash run-backing-cache.sh; no tmux names or API keys to fill in.
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if [[ ! -r /etc/qwen-flash-next.env ]]; then
    echo '/etc/qwen-flash-next.env must be readable by the current user' >&2
    exit 66
fi
sudo -v
restore_service=0
if systemctl is-active --quiet qwen-flash-next.service; then
    restore_service=1
fi
cleanup() {
    local result=$?
    trap - EXIT
    if [[ "$restore_service" == 1 ]]; then
        echo 'Restoring qwen-flash-next.service...'
        if ! sudo systemctl start qwen-flash-next.service; then
            echo 'ERROR: service restart failed; inspect journalctl -u qwen-flash-next.service' >&2
            result=1
        fi
    fi
    exit "$result"
}
trap cleanup EXIT
sudo systemctl stop qwen-flash-next.service
if pgrep -x llama-server >/dev/null; then
    echo 'A separate llama-server is still running. Stop it before this memory-capacity test.' >&2
    exit 1
fi
set -a
source /etc/qwen-flash-next.env
set +a
python3 qwen_backing_cache_diag.py "$@"
